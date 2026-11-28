#!/usr/bin/env python3
"""Roofs of the data-movement RISC-Vs on one Blackhole chip: scalar copy, gather and scatter per word, a NoC copy
from a core's L1 to its own L1, the DRAM read path per page (one core alone and every core at once) and a ping-pong
between two cores. Kernel: roof_dm.cpp.

    TT_VISIBLE_DEVICES=<chip> python bench/roof/roof_dm.py --node <sysfs node> --aiclk-log FILE --out rows.jsonl

Every cost is a difference of two run lengths, so launch and setup are not in it. Every row is checked against numpy
on the bytes the kernel moved; --break-arm builds the kernel so each op does less than it reports and the checks
must then fail. The clock record is the sysfs AICLK log an outside process writes (run_roof.sh), cut to each row's
wall-time span: `<epoch> <MHz>` per line."""
import argparse, json, os, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--node", required=True, help="sysfs node of the visible chip")
ap.add_argument("--aiclk-log", required=True)
ap.add_argument("--out", default="roof_dm.jsonl")
ap.add_argument("--only", default="scalar,noc,dram,ping")
ap.add_argument("--break-arm", type=int, default=0)
a = ap.parse_args()

import torch  # noqa: E402
import ttnn  # noqa: E402

HERE = Path(__file__).resolve().parent
TB, NX, NI, NR, M, SW = 4096, 16, 8, 16, 16384, 65536
dev = ttnn.open_device(device_id=0)
dev.disable_and_clear_program_cache()   # every launch changes runtime arguments; the kernel binaries stay cached on disk
g = dev.compute_with_storage_grid_size()
ALL = [(x, y) for y in range(g.y) for x in range(g.x)]
nc = len(ALL)
rng = np.random.default_rng(7)
X = rng.integers(1, 2**32, (nc, NX * 1024), dtype=np.uint32)
perm = np.stack([rng.permutation(M).astype(np.uint16) for _ in range(nc)])
half = lambda role: slice(role * (M // 2), (role + 1) * (M // 2))
STR = rng.integers(1, 2**32, (nc, SW), dtype=np.uint32)


def to_dev(arr):
    return ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(arr).view(np.int32)), dtype=ttnn.uint32,
                           layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


inp = np.zeros((nc, (NX + NI) * 1024), np.uint32)
inp[:, :NX * 1024] = X
inp[:, NX * 1024:NX * 1024 + M // 2] = perm.view(np.uint32)
t_in, t_str = to_dev(inp), to_dev(STR)
t_out = to_dev(np.zeros((nc, (1 + NR) * 1024), np.uint32))
acc = [x for t in (t_in, t_out, t_str) for x in ttnn.TensorAccessorArgs(t).get_compile_time_args()]


def clock(t0, t1):
    v = [int(l.split()[1]) for l in open(a.aiclk_log) if len(l.split()) == 2 and t0 <= float(l.split()[0]) <= t1]
    return dict(min=min(v, default=None), max=max(v, default=None), n=len(v))


def launch(cores, op, reps, nbytes, nb, mask, peers=None):
    """One program on `cores`; returns (ticks[core, role], R[core, words], t0, t1)."""
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in cores])
    cbs = [ttnn.CBDescriptor(total_size=n * TB, core_ranges=grid, format_descriptors=[
        ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=TB)]) for i, n in ((0, NX + NI + NR), (1, 1))]
    sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=grid, initial_value=0) for i in (0, 1)]
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(cores):
        if peers:
            p = dev.worker_core_from_logical_core(ttnn.CoreCoord(*peers[i]))
            rt[x][y] = [ALL.index((x, y)), p.x, p.y, int(i == 0)]
        else:
            rt[x][y] = [ALL.index((x, y)), 0, 0, 0]
    ks = []
    for role, cfg in ((0, ttnn.ReaderConfigDescriptor()), (1, ttnn.WriterConfigDescriptor())):
        k = ttnn.KernelDescriptor(kernel_source=str(HERE / "roof_dm.cpp"), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                  core_ranges=grid, compile_time_args=[role, 0, 1, SW] + acc, config=cfg)
        k.runtime_args = rt
        k.common_runtime_args = [t_in.buffer_address(), t_out.buffer_address(), t_str.buffer_address(), op, reps, nbytes, nb, mask]
        k.defines = [("ROOF_BREAK", str(a.break_arm))]
        ks.append(k)
    t0 = time.time()
    ttnn.generic_op([t_in, t_out], ttnn.ProgramDescriptor(kernels=ks, semaphores=sems, cbs=cbs))
    ttnn.synchronize_device(dev)
    t1 = time.time()
    o = ttnn.to_torch(t_out).numpy().view(np.uint32).reshape(nc, -1)
    idx = [ALL.index(c) for c in cores]
    h = o[idx, :16].reshape(len(cores), 2, 8)
    for role in (0, 1):   # the kernel ran this op with these arguments (a cached program must not replay old ones)
        assert (h[:, role, 1] == (0xC0DE0000 | op)).all() and (h[:, role, 2] == reps).all() and (h[:, role, 3] == nbytes).all(), \
            "kernel saw other arguments: %s" % h[0]
    return h[:, :, 0].astype(np.int64), o[idx, 1024:], t0, t1


def expect(cores, op, reps, nbytes, nb, mask, peers):
    """R as numpy says it must be."""
    R = np.zeros((len(cores), NR * 1024), np.uint32)
    for i, c in enumerate(cores):
        ci = ALL.index(c)
        for role in (0, 1):
            if not mask >> role & 1:
                continue
            h, w = half(role), nbytes // 4
            if op == 1:
                R[i, h] = X[ci, h]
            elif op == 2:
                R[i, h] = X[ci, perm[ci, h]]
            elif op == 3:
                R[i, perm[ci, h]] = X[ci, h]
            elif op in (4, 5, 6):
                R[i, role * 8192:role * 8192 + w] = X[ci, role * 8192:role * 8192 + w]
            elif op == 7:
                npg = SW * 4 // nbytes
                for n in range(max(0, reps - nb), reps):
                    R[i, role * 8192 + (n % nb) * w:role * 8192 + (n % nb + 1) * w] = STR[ci, (n % npg) * w:(n % npg + 1) * w]
            elif op == 8:
                pi = ALL.index(peers[i])
                R[i, role * 8192:role * 8192 + w] = X[pi, role * 8192:role * 8192 + w]
    return R


rows, bad = [], 0


def cell(name, cores, op, r1, r2, nbytes=0, nb=1, mask=1, peers=None, unit=1, **kw):
    """Cost per unit = (ticks at r2 - ticks at r1) / (r2 - r1) / unit, median and worst over cores and active RISCs."""
    global bad
    launch(cores, op, r1, nbytes, nb, mask, peers)                       # JIT and warm
    ta, _, _, _ = launch(cores, op, r1, nbytes, nb, mask, peers)
    tb, R, t0, t1 = launch(cores, op, r2, nbytes, nb, mask, peers)
    tc, _, _, t1 = launch(cores, op, r2, nbytes, nb, mask, peers)
    ok = bool(np.array_equal(R, expect(cores, op, r2, nbytes, nb, mask, peers)))
    bad += not ok
    act = [r for r in (0, 1) if mask >> r & 1]
    if op == 8:
        act, tb, tc, ta = act[:1], tb[:1], tc[:1], ta[:1]                # the ping side holds the round trip
    per = (tb[:, act] - ta[:, act]) / (r2 - r1) / unit
    per2 = (tc[:, act] - ta[:, act]) / (r2 - r1) / unit
    row = dict(probe="roof_dm", cell=name, op=op, cores=len(cores), mask=mask, bytes=nbytes, nb=nb, reps=[r1, r2],
               cyc=round(float(np.median(per)), 3), cyc_worst_core=round(float(per.max()), 3), cyc_run2=round(float(np.median(per2)), 3),
               check="PASS" if ok else "FAIL", aiclk=clock(t0, t1), node=a.node, chip=os.environ.get("TT_VISIBLE_DEVICES"), grid=[g.x, g.y], **kw)
    if op == 7:   # chip rate from the slowest core, at the clock of the row
        mhz = row["aiclk"]["min"] or 1350
        row["GBs_chip"] = round(len(cores) * len(act) * nbytes * (r2 - r1) / ((tb[:, act] - ta[:, act]).max() / (mhz * 1e6)) / 1e9, 1)
        row["cyc_per_4k"] = round(row["cyc"] * 4096 / nbytes, 1)
    rows.append(row)
    print(json.dumps(row), flush=True)


ONE = [ALL[nc // 2]]
want = a.only.split(",")
try:
    if "scalar" in want:
        for op, nm in ((1, "copy"), (2, "gather"), (3, "scatter")):
            for mask in (1, 2, 3):
                cell(f"{nm} per word", ALL, op, 2, 12, mask=mask, unit=M // 2)
    if "noc" in want:
        for cores, cn in ((ONE, "one core"), (ALL, "all cores")):
            for op, nm in ((4, "NoC self read + barrier"), (6, "NoC self write + barrier"), (5, "NoC self read, one barrier")):
                for nbytes in (4, 64, 1024, 4096, 16384):
                    for mask in (1, 2):
                        cell(f"{nm}, {cn}", cores, op, 100, 600, nbytes=nbytes, mask=mask)
    if "dram" in want:
        for cores, cn in ((ONE, "one core"), (ALL, "all cores")):
            for nbytes, nbs in ((4096, (1, 2, 8)), (16384, (1, 2))):
                for nb in nbs:
                    for mask in (1, 2, 3):
                        cell(f"DRAM read stream, {cn}", cores, 7, 64, 2112, nbytes=nbytes, nb=nb, mask=mask)
    if "ping" in want:
        far = (g.x - 1, g.y - 1)
        for peer, pn in (((1, 0), "1 hop x"), ((0, 1), "1 hop y"), (far, "far corner")):
            for nbytes in (4, 64, 1024, 4096):
                for mask in (1, 2):
                    cell(f"ping-pong round trip, {pn}", [(0, 0), peer], 8, 100, 600, nbytes=nbytes, mask=mask, peers=[peer, (0, 0)])
finally:
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ttnn.close_device(dev)
print("CHECKS: %s (%d of %d rows failed)%s" % ("PASS" if not bad else "FAIL", bad, len(rows), ", BROKEN-ARM build" if a.break_arm else ""))
raise SystemExit(1 if bad else 0)
