#!/usr/bin/env python3
"""Unit costs of the compute pipeline on one Blackhole core: cycles per fp32 tile load, pack and SFPU tile op.

    TT_VISIBLE_DEVICES=<chip> python bench/unit/unit_costs.py [--ns 65536] [--reps 3] [--json out.json]

Runs ub_compute.cpp on core (0, 0). Every cell is NS sections of work on DEST; its cost is the wall-clock cycles of
the slowest TRISC minus the empty-section cell, per unit. AICLK is read from sysfs while the program runs. The checks
compare tiles the kernel moved or computed with numpy, so a lean path that moves the wrong bytes fails here."""
import argparse, json, os, threading, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ns", type=int, default=65536, help="sections per cell (4 units each for most cells)")
ap.add_argument("--mask", type=lambda x: int(x, 0), default=0x8007FFFF, help="cells to run, bit 31 = the checks")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--sysfs-node", type=int, default=None, help="sysfs node of the chip (default: TT_VISIBLE_DEVICES)")
ap.add_argument("--break-arm", type=int, default=0, help="1: the lean-load check loads one tile where it claims four; 2: the fused blocks store the wrong register. The checks must then fail")
ap.add_argument("--json")
a = ap.parse_args()

import torch  # noqa: E402
import ttnn  # noqa: E402

HERE = Path(__file__).resolve().parent
NV, CB_V, CB_HDR, CB_DONE, CB_H = 24, 0, 1, 2, 3
CELLS = {0: "empty section", 1: "stock load (init once)", 2: "stock load (init each)", 3: "stock pack to L1",
         4: "stock add", 5: "stock mul", 6: "stock addcmul", 7: "stock where", 8: "stock bf16 load (reconfig)", 9: "lean load x1", 10: "lean load x2",
         11: "lean load x4", 12: "fused block, issued", 13: "fused block, sfpi", 14: "fused block, MOP",
         15: "add block, MOP", 16: "scheduled block, NOPs", 17: "scheduled block, no NOPs",
         18: "pipelined add in place (load macro)"}
node = a.sysfs_node if a.sysfs_node is not None else int(os.environ["TT_VISIBLE_DEVICES"].split(",")[0])
CLK = f"/sys/class/tenstorrent/tenstorrent!{node}/tt_aiclk"


class Aiclk:
    def __enter__(self):
        self.vals, self.stop = [], threading.Event()

        def loop():
            while not self.stop.is_set():
                try:
                    self.vals.append(int(open(CLK).read().split()[0]))
                except Exception:
                    pass
                time.sleep(0.01)
        self.th = threading.Thread(target=loop, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *x):
        self.stop.set()
        self.th.join()


md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
# TT_VISIBLE_DEVICES n is not /dev/tenstorrent/n on a Galaxy (anx-glx: 16-23 <-> 24-31): prove the sampled node is ours
opened = sorted({int(os.readlink(f"/proc/self/fd/{fd}").rsplit("/", 1)[1]) for fd in os.listdir("/proc/self/fd")
                 if os.path.exists(f"/proc/self/fd/{fd}") and os.readlink(f"/proc/self/fd/{fd}").startswith("/dev/tenstorrent/")})
print("OPENED-NODES", opened, "clock sampled on node", node, flush=True)
if opened != [node]:
    ttnn.close_mesh_device(md)
    raise SystemExit(f"NODE-MISMATCH: opened {opened}, clock is sampled on node {node}")
t_up = time.time()   # a chip nobody had open idles at 800 MHz and ramps after the open: wait for the working clock
while time.time() - t_up < 30 and int(open(CLK).read().split()[0]) < 1300:
    time.sleep(0.05)
print("aiclk %s MHz %.2f s after the open" % (open(CLK).read().split()[0], time.time() - t_up), flush=True)
core = ttnn.CoreCoord(0, 0)
grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
rng = np.random.default_rng(0)
tiles = np.zeros((NV, 1024), np.float32)
tiles[:8] = rng.uniform(0.5, 2.0, (8, 1024)).astype(np.float32) * rng.choice([-1.0, 1.0], (8, 1024)).astype(np.float32)


def to_dev(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr).view(np.int32)), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def cbd(i, pages, page, fmt):
    return ttnn.CBDescriptor(total_size=pages * page, core_ranges=grid,
                             format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)])


def program(t_in, t_out):
    prog = ttnn.ProgramDescriptor()
    arena = ttnn.CBDescriptor(total_size=NV * 4096, core_ranges=grid, format_descriptors=[   # CB_H: a bf16 view of the same memory
        ttnn.CBFormatDescriptor(buffer_index=CB_V, data_format=ttnn.float32, page_size=4096),
        ttnn.CBFormatDescriptor(buffer_index=CB_H, data_format=ttnn.bfloat16, page_size=2048)])
    prog.cbs = [arena, cbd(CB_HDR, 1, 4096, ttnn.uint32), cbd(CB_DONE, 1, 64, ttnn.uint32)]
    acc = list(ttnn.TensorAccessorArgs(t_in).get_compile_time_args()) + list(ttnn.TensorAccessorArgs(t_out).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t_in.buffer_address(), t_out.buffer_address()]
    dm = ttnn.KernelDescriptor(kernel_source=str(HERE / "ub_dm.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                               core_ranges=grid, compile_time_args=[CB_V, CB_HDR, CB_DONE, NV] + acc,
                               config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0))
    dm.runtime_args = rt
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)
    um = [ttnn.UnpackToDestMode.Default] * 64
    um[CB_V] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc.unpack_to_dest_mode = um
    comp = ttnn.KernelDescriptor(kernel_source=str(HERE / "ub_compute.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                 core_ranges=grid, compile_time_args=[CB_V, CB_HDR, CB_DONE, NV, CB_H], config=cc)
    comp.common_runtime_args = [a.mask & 0xFFFFFFFF, a.ns]
    if a.break_arm:
        comp.defines = [("UB_BREAK", str(a.break_arm))]
    prog.kernels = [dm, comp]
    mpd = ttnn.MeshProgramDescriptor()
    mc = ttnn.MeshCoordinate(0, 0)
    mpd[ttnn.MeshCoordinateRange(mc, mc)] = prog
    return mpd


t_in = to_dev(tiles.reshape(1, -1).view(np.uint32))
runs = []
for r in range(a.reps + 1):   # the first run carries the JIT build and is dropped
    t_out = to_dev(np.zeros((1, 1024 + NV * 1024), np.uint32))
    with Aiclk() as clk:
        t0 = time.time()
        ttnn.generic_op([t_in, t_out], program(t_in, t_out))
        ttnn.synchronize_device(md)
        wall = time.time() - t0
    o = ttnn.to_torch(t_out).numpy().view(np.uint32).reshape(-1)
    hdr, arena = o[:1024], o[1024:].view(np.float32).reshape(NV, 1024)
    assert all((hdr[i] >> 16) == 0xC0DE for i in range(3)), "kernel did not reach its end: %s" % [hex(x) for x in hdr[:3]]
    cyc = {i: [int(x) for x in hdr[32 + 4 * i:36 + 4 * i]] for i in CELLS if a.mask >> i & 1}
    runs.append(dict(wall=wall, t0=t0, t1=t0 + wall, aiclk_min=min(clk.vals, default=None), aiclk_max=max(clk.vals, default=None),
                     aiclk_n=len(clk.vals), cyc=cyc))
    ttnn.deallocate(t_out)
    print("run %d wall %.3f s aiclk %s..%s (%d samples)%s" % (r, wall, runs[-1]["aiclk_min"], runs[-1]["aiclk_max"],
                                                              len(clk.vals), " (JIT run, dropped)" if r == 0 else ""), flush=True)

checks = {}
if a.mask >> 31 & 1:
    t = tiles.astype(np.float32)
    checks["stock load -> pack"] = bool((arena[8:12].view(np.uint32) == t[0:4].view(np.uint32)).all())
    checks["lean load x4 -> pack"] = bool((arena[12:16].view(np.uint32) == t[4:8].view(np.uint32)).all())
    two = ((t[0] * t[1] + t[2]).astype(np.float32) * t[1] + t[0]).astype(np.float32)      # every product and sum rounded
    t64 = t.astype(np.float64)
    fma = ((t64[0] * t64[1] + t64[2]).astype(np.float32).astype(np.float64) * t64[1] + t64[0]).astype(np.float32)   # each multiply-add rounded once
    for i, name in ((16, "2x stock addcmul"), (17, "fused issued"), (18, "fused sfpi"), (19, "fused MOP"),
                    (20, "scheduled, NOPs"), (21, "scheduled, no NOPs")):
        got = arena[i]
        checks[name] = dict(eq_two_roundings=int((got.view(np.uint32) == two.view(np.uint32)).sum()),
                            eq_one_rounding=int((got.view(np.uint32) == fma.view(np.uint32)).sum()),
                            eq_stock=int((got.view(np.uint32) == arena[16].view(np.uint32)).sum()),
                            max_rel=float(np.max(np.abs(got.astype(np.float64) - t64[0] - (t64[0] * t64[1] + t64[2]) * t64[1])
                                                 / np.abs(two.astype(np.float64)))), lanes=1024)
    if NV > 23:
        add = (t[0] + t[1]).astype(np.float32)
        checks["pipelined add"] = dict(eq_stock=int((arena[23].view(np.uint32) == arena[22].view(np.uint32)).sum()),
                                       eq_s1=int((arena[23].view(np.uint32) == t[1].view(np.uint32)).sum()),
                                       stock_eq_numpy=int((arena[22].view(np.uint32) == add.view(np.uint32)).sum()), lanes=1024)
keep = runs[1:]
med = lambda xs: float(np.median(xs))
rows = {}
base = med([max(r["cyc"][0][:3]) / r["cyc"][0][3] for r in keep]) if a.mask & 1 else 0.0
for i, name in CELLS.items():
    if not a.mask >> i & 1:
        continue
    per = [max(r["cyc"][i][:3]) / r["cyc"][i][3] for r in keep]            # slowest thread, per unit
    sec = [max(r["cyc"][i][:3]) / a.ns for r in keep]                       # per section
    upn = keep[0]["cyc"][i][3] // a.ns
    rows[name] = dict(cell=i, units=keep[0]["cyc"][i][3], per_unit=med(per), per_unit_net=med([(s - base) / upn for s in sec]) if i else base,
                      spread=float(max(per) - min(per)), threads=[int(np.median([r["cyc"][i][k] for r in keep])) for k in range(3)])
lo = min(r["aiclk_min"] for r in keep if r["aiclk_min"]) if any(r["aiclk_min"] for r in keep) else None
print("\ncell                         units  cyc/unit  net of empty section  [unpack math pack cycles]")
for name, v in rows.items():
    print("%-28s %5d  %8.1f  %8.1f  %s" % (name, v["units"], v["per_unit"], v["per_unit_net"], v["threads"]))
ok = all(v if isinstance(v, bool) else v["eq_stock"] == v["lanes"] for k, v in checks.items() if k != "scheduled, no NOPs")   # that one is a question, not a gate
print("checks:", json.dumps(checks))
print("CHECKS %s%s" % ("PASS" if ok else "FAIL", " (break arm %d)" % a.break_arm if a.break_arm else ""))
print("aiclk min %s MHz over %d kept runs, n=%d" % (lo, len(keep), len(keep)))
print("TIMED %.3f %.3f" % (keep[0]["t0"], keep[-1]["t1"]))   # epoch span of the kept runs, for an outside clock sampler
if a.json:
    json.dump(dict(rows=rows, runs=runs, checks=checks, checks_ok=ok, break_arm=a.break_arm, ns=a.ns, mask=a.mask, aiclk_min=lo, n=len(keep), node=node), open(a.json, "w"), indent=1)
ttnn.close_mesh_device(md)
