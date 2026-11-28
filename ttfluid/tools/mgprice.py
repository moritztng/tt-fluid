"""Price mgmodel's solves on the device: what one preconditioned p iteration costs a core, against fused Jacobi PCG.

    python -m ttfluid.tools.mgprice SWEEP.jsonl [--pre target|floor] [--set name=value ...] [--hist HIST.json] [--md]

SWEEP.jsonl is mgmodel's --json output. The cost of a loop on a core is counted the way the campaign counts it:

    time = tile moves x move_cyc + tile ops x op_cyc + gathered values x g_cyc / g_thr + waits

and the reference is the fused one-reduction PCG on the Jacobi-scaled system: per fine tile an update sweep of 10
moves and 6 ops, an SpMV sweep of 15 moves and 14 ops, the gather of the SpMV's off-diagonal values, one halo
exchange, one reduction. `t0_ms` is what that iteration is taken to cost in all; whatever the counts leave of it is
waiting that every fine sweep pays again (a multigrid iteration with two fine SpMVs pays it twice).

A multigrid iteration adds, per core:
  * the preconditioned residual as a fifth resident fine vector: 2 moves and 2 ops per fine tile;
  * restriction and prolongation between the fine level and level 1: with the cells of a core ordered aggregate-major
    in planes (rp=plane) one op and one move per fine tile each way, as generic gathers (rp=gather) one gathered
    value per fine cell each way;
  * every coarse SpMV: the level's off-diagonal entries on the fullest core, padded by ell_pad, as gathered values,
    two moves and one op per slot tile, and on a level whose couplings cross cores one halo exchange (chalo_us);
  * the coarse smoother's vector work, 3 ops and 2 moves per coarse tile per step, and the transfers between coarse
    levels as gathers.
Levels whose operator does not fit L1 next to the fine vectors stream it: counted as DRAM pages (coefficients as
fp32, sources as 16 bit) and reported as reader load, the share of the iteration the page reader is busy.

Unit costs (--set name=value; --pre picks a set):
  target  the fused loop at today's kernels with the gather on four threads, t0 1.5 ms
  floor   the campaign's floor estimates of a move, an op and a gathered value, t0 0.95 ms
With --hist (a run_mesh history) the solves are weighted to the run by the device's own Jacobi iteration count per
SIMPLE iteration: the multigrid count is interpolated in the Jacobi count between the captured systems.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

PRE = dict(
    target=dict(mhz=1350.0, tiles=39.0, move_cyc=404.0, op_cyc=400.0, g_cyc=11.7, g_thr=4.0, gvals=243000.0,
                page_us=1.02, red_us=85.0, halo_us=110.0, chalo_us=30.0, t0_ms=1.5, ell_pad=1.15, rp=0.0,
                l1_kb=990.0),
    floor=dict(mhz=1350.0, tiles=39.0, move_cyc=150.0, op_cyc=75.0, g_cyc=6.0, g_thr=4.0, gvals=243000.0,
               page_us=1.02, red_us=85.0, halo_us=110.0, chalo_us=30.0, t0_ms=0.95, ell_pad=1.15, rp=0.0,
               l1_kb=990.0),
)
CFG = dict(agg="3", mode="post", k=1, cmode="sym", ck=2, kc=16, local=99, glob=99, gamma=1, depth=None)
M_UPD, O_UPD, M_SPMV, O_SPMV = 10, 6, 15, 14                            # per fine tile, the fused reference


def cfg_of(name):
    c = dict(CFG)
    for kv in name.split():
        k, v = kv.split("=")
        if k in c:
            c[k] = v if k in ("agg", "mode", "cmode") else int(v)
    return c


def reference(u):
    """The fused Jacobi PCG iteration on a core: its SpMV sweep, its update sweep, the waiting left of t0 (ms)."""
    cyc = 1e-3 / u["mhz"]
    spmv = ((M_SPMV * u["move_cyc"] + O_SPMV * u["op_cyc"]) * u["tiles"] + u["gvals"] * u["g_cyc"] / u["g_thr"]) * cyc \
        + u["halo_us"] * 1e-3
    upd = (M_UPD * u["move_cyc"] + O_UPD * u["op_cyc"]) * u["tiles"] * cyc
    wait = max(u["t0_ms"] - spmv - upd - u["red_us"] * 1e-3, 0.0)
    return spmv, upd, wait


def solve_ms(row, u):
    """One iteration of a mgmodel solve on a core, ms, split into its parts, with its syncs, pages and L1 bytes."""
    c, its = cfg_of(row["cfg"]), max(row["its"], 1)
    cyc = 1e-3 / u["mhz"]
    g = u["g_cyc"] / u["g_thr"] * cyc
    spmv0, upd, wait = reference(u)
    red = u["red_us"] * 1e-3
    if "mode=jac" in row["cfg"]:
        return dict(total=spmv0 + upd + wait + red, fine=spmv0 + upd + wait, rp=0.0, coarse=0.0, halo=0.0, red=red,
                    sync=2.0, pages=0.0, l1=4 * u["tiles"] * 4.096, stream=[])
    f0 = row["spmv"][0] / its
    steps0 = c["k"] * (2 if c["mode"] == "sym" else 1)
    fine = f0 * (spmv0 + wait) + upd + (2 * u["move_cyc"] + (2 + 3 * (steps0 - 1)) * u["op_cyc"]) * u["tiles"] * cyc
    n0 = row["levels"][0]["rows_core"][2]
    nrp = (row["R"][0] + row["P"][0]) / its if row["R"] else 0.0
    rp = nrp * (n0 * g if u["rp"] else (u["move_cyc"] + u["op_cyc"]) * u["tiles"] * cyc)
    coarse = halo = pages = 0.0
    sync = 1.0 + f0
    l1 = 5 * u["tiles"] * 4.096                                         # r, the preconditioned residual, p, s, w
    stream = []
    for l, st in enumerate(row["levels"]):
        if l == 0:
            continue
        rows, nnz = st["rows_core"][2], st["nnz_core_max"]
        tl = math.ceil(rows / 1024)
        off = (nnz - rows) * u["ell_pad"]
        s = row["spmv"][l] / its
        last = l == len(row["levels"]) - 1
        steps = c["kc"] if last else c["ck"] * (2 if c["cmode"] == "sym" else 1)
        slots = off / rows
        coarse += s * (off * g + (2 * u["move_cyc"] + u["op_cyc"]) * slots * tl * cyc)
        coarse += steps * (2 * u["move_cyc"] + 3 * u["op_cyc"]) * tl * cyc
        if l < len(row["R"]):
            coarse += (row["R"][l] + row["P"][l]) / its * rows * g
        if l < c["local"] or l >= c["glob"]:
            halo += s * u["chalo_us"] * 1e-3
            sync += s
        vec, mat = 4 * tl * 4.096, off * 6 / 1024
        l1 += vec
        if l1 + mat <= u["l1_kb"]:
            l1 += mat
        else:                                                           # the operator of this level is streamed
            stream.append(l)
            pages += s * off * 6 / 4096
    return dict(total=fine + rp + coarse + halo + red, fine=fine, rp=rp, coarse=coarse, halo=halo, red=red, sync=sync,
                pages=pages, l1=l1, stream=stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--pre", default="target", choices=sorted(PRE))
    ap.add_argument("--hist", help="run_mesh history json: its3 is the device's Jacobi PCG count per SIMPLE iteration")
    ap.add_argument("--its", type=int, default=800, help="SIMPLE iterations of the run the weighting covers")
    ap.add_argument("--set", action="append", default=[], help="name=value, repeatable")
    ap.add_argument("--md", action="store_true", help="markdown table")
    ap.add_argument("--only", help="keep the configurations that contain this text")
    a = ap.parse_args()
    u = dict(PRE[a.pre])
    for kv in a.set:
        k, v = kv.split("=")
        assert k in u, k
        u[k] = float(v)
    rows = {}
    for line in open(a.sweep):
        r = json.loads(line)
        rows[(r["cfg"], r["sys"])] = r                                  # a rerun replaces the earlier row
    systems = sorted({s for _, s in rows}, key=lambda s: (s != "pf", int(s[2:]) if s[2:].isdigit() else 0))
    jac = {s: rows[("mode=jac", s)]["its"] for s in systems if ("mode=jac", s) in rows}
    spmv0, upd, wait = reference(u)
    t_j = spmv0 + upd + wait + u["red_us"] * 1e-3
    hist = None
    if a.hist:
        h = json.load(open(a.hist))
        hist = np.array(h["its3"] if isinstance(h, dict) else [x["its3"] for x in h], float)[:a.its]
    print(f"{a.pre}: fused Jacobi PCG iteration {t_j:.3f} ms on a core = SpMV sweep {spmv0:.3f} + update {upd:.3f} + "
          f"reduction {u['red_us'] * 1e-3:.3f} + waiting {wait:.3f}; coarse halo exchange {u['chalo_us']:.0f} us, "
          f"transfers {'gathers' if u['rp'] else 'planes'}; Jacobi its {jac}"
          + (f"; device mean over {len(hist)} its {hist.mean():.1f}" if hist is not None else ""))
    by = defaultdict(dict)
    for (cfg, s), r in rows.items():
        by[cfg][s] = r
    bar = "| " if a.md else ""
    print(f"{bar}{'cycle':52s} | " + " | ".join(f"{s} (Jacobi {jac.get(s, '-')})" for s in systems)
          + " | run: fine passes, gain | one iteration on a core | syncs | L1 KB, streamed" + (" |" if a.md else ""))
    if a.md:
        print("|" + "---|" * (len(systems) + 5))
    out = []
    for cfg, d in by.items():
        if cfg == "mode=jac" or (a.only and a.only not in cfg):
            continue
        cells, gains = [], {}
        for s in systems:
            if s not in d or s not in jac:
                cells.append("-")
                continue
            r, t = d[s], solve_ms(d[s], u)
            ok = r["its"] < 1000
            gains[s] = (jac[s] * t_j) / (r["its"] * t["total"]) if ok else 0.0
            cells.append(f"{r['its']} its x {t['total'] / t_j:.2f} = {gains[s]:.2f}x")
        t = solve_ms(next(iter(d.values())), u)
        w = ""
        if hist is not None and len(gains) >= 2:
            js = np.array([jac[s] for s in gains], float)
            o = np.argsort(js)
            n_mg = np.interp(hist, js[o], np.array([d[s]["its"] for s in gains], float)[o])
            cost = np.interp(hist, js[o], np.array([solve_ms(d[s], u)["total"] for s in gains])[o])
            fpi = np.interp(hist, js[o], np.array([d[s]["spmv"][0] / max(d[s]["its"], 1) for s in gains])[o])
            gain = (hist * t_j).sum() / (n_mg * cost).sum()
            w = f"{(n_mg * fpi).mean():.1f}, {gain:.2f}x"
            out.append((gain, cfg))
        print(f"{bar}{cfg:52s} | " + " | ".join(cells) + f" | {w} | {t['total']:.2f} ms = fine {t['fine']:.2f} + transfers "
              f"{t['rp']:.2f} + coarse {t['coarse']:.2f} + coarse halo {t['halo']:.2f} + reduction {t['red']:.2f} | "
              f"{t['sync']:.0f} | {t['l1']:.0f}, levels {t['stream']} at {t['pages']:.0f} pages = reader "
              f"{100 * (527 + t['pages']) * u['page_us'] * 1e-3 / t['total']:.0f} %" + (" |" if a.md else ""))
    if out:
        g, cfg = max(out)
        print(f"best weighted: {g:.2f}x [{cfg}]")


if __name__ == "__main__":
    main()
