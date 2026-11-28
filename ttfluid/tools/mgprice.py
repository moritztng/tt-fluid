"""Price mgmodel's solves on the device: what one preconditioned iteration costs a core, against fused Jacobi PCG.

    python -m ttfluid.tools.mgprice SWEEP.jsonl [--hist HIST.json] [--set name=value ...]

SWEEP.jsonl is mgmodel's --json output. Every unit cost is a named parameter (--set), with the estimates of the fused
p iteration as defaults; a measured value replaces its default without touching the model:

  mhz 1350        core clock
  tiles 39        tiles of one fine vector on a core
  slots 7         coefficient pages per fine tile per SpMV, streamed from DRAM
  page_us 1.02    one 4 KB DRAM page
  op_cyc 300      one fused SFPU tile op
  gvals 276000    values gathered per fine SpMV on a core
  g_cyc 6, g_thr 4   gather cycles per value on one thread, threads sharing it
  ops_spmv 16     tile ops per tile of an SpMV (two per slot, the diagonal)
  ops_pcg 9       tile ops per tile of the outer iteration's dots and updates
  ops_flex 3      more of those for flexible PCG (the old residual is kept and enters one more dot)
  ops_cheb 4      tile ops per tile of one smoother step
  ops_rp 1        tile ops per fine tile of one restriction or one prolongation (sums and copies over runs)
  red_us 100, reds 2   one reduction over all cores, reductions per outer iteration
  halo_us 30      the wait of one halo exchange (every SpMV on a level whose couplings cross cores)
  cpage 0         1 = coarse coefficients streamed from DRAM instead of held in L1

A coarse level's SpMV gathers the level's own off-diagonal values on the fullest core and runs ops_spmv per coarse tile
scaled by the level's mean row length over the fine one. The result per solve is its cost against the Jacobi PCG solve
of the same system; with --hist (a run_mesh history) the solves are weighted to the run by the device's own Jacobi
iteration count per SIMPLE iteration.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

DEF = dict(mhz=1350.0, tiles=39, slots=7, page_us=1.02, op_cyc=300.0, gvals=276000, g_cyc=6.0, g_thr=4, ops_spmv=16,
           ops_pcg=9, ops_flex=3, ops_cheb=4, ops_rp=1, red_us=100.0, reds=2, halo_us=30.0, cpage=0)
CFG = dict(agg="3", mode="post", k=1, cmode="sym", ck=2, kc=16, local=99, gamma=1)


def cfg_of(name):
    c = dict(CFG)
    for kv in name.split():
        k, v = kv.split("=")
        if k in c:
            c[k] = v if k in ("agg", "mode", "cmode") else int(v)
    return c


def jacobi_ms(u):
    """One fused Jacobi PCG iteration on a core, ms, and the fine SpMV inside it."""
    op = u["op_cyc"] / u["mhz"] * 1e-3
    spmv = (u["slots"] * u["tiles"] * u["page_us"] * 1e-3 + u["gvals"] * u["g_cyc"] / u["g_thr"] / u["mhz"] * 1e-3
            + u["ops_spmv"] * u["tiles"] * op + u["halo_us"] * 1e-3)
    return spmv + u["ops_pcg"] * u["tiles"] * op + u["reds"] * u["red_us"] * 1e-3, spmv


def solve_ms(row, u):
    """One iteration of a mgmodel solve on a core, ms, split into its parts."""
    c, its = cfg_of(row["cfg"]), max(row["its"], 1)
    op = u["op_cyc"] / u["mhz"] * 1e-3
    g = u["g_cyc"] / u["g_thr"] / u["mhz"] * 1e-3
    t_j, spmv0 = jacobi_ms(u)
    if c["mode"] == "jac" or "mode=jac" in row["cfg"]:
        return dict(total=t_j, fine=spmv0, vec=t_j - spmv0 - u["reds"] * u["red_us"] * 1e-3, coarse=0.0, halo=0.0,
                    red=u["reds"] * u["red_us"] * 1e-3, l1=0)
    steps0 = c["k"] * (2 if c["mode"] == "sym" else 1)
    fine = row["spmv"][0] / its * spmv0
    vec = (u["ops_pcg"] + u["ops_flex"] + steps0 * u["ops_cheb"]) * u["tiles"] * op
    coarse = halo = 0.0
    l1 = 0
    nnz0 = row["levels"][0]["nnz_row"][0]
    for l, st in enumerate(row["levels"]):
        rows, nnz = st["rows_core"][2], st["nnz_core_max"]
        tl = math.ceil(rows / 1024)
        if l < len(row["R"]):                                           # this level restricts to and prolongs from the next
            vec_or_coarse = (row["R"][l] + row["P"][l]) / its * u["ops_rp"] * (u["tiles"] if l == 0 else tl) * op
            if l == 0:
                vec += vec_or_coarse
            else:
                coarse += vec_or_coarse
        if l == 0:
            continue
        s = row["spmv"][l] / its
        last = l == len(row["levels"]) - 1
        steps = c["kc"] if last else c["ck"] * (2 if c["cmode"] == "sym" else 1) * (c["gamma"] ** l)
        per = (nnz - rows) * g + u["ops_spmv"] * tl * op * st["nnz_row"][0] / nnz0
        if u["cpage"]:
            per += u["slots"] * st["nnz_row"][0] / nnz0 * tl * u["page_us"] * 1e-3
        coarse += s * per + steps * u["ops_cheb"] * tl * op
        if l < c["local"]:
            halo += s * u["halo_us"] * 1e-3
        l1 += 4 * rows * 4 + nnz * 6                                    # four vectors, fp32 coefficients, 16-bit sources
    red = u["reds"] * u["red_us"] * 1e-3
    return dict(total=fine + vec + coarse + halo + red, fine=fine, vec=vec, coarse=coarse, halo=halo, red=red, l1=l1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--hist", help="run_mesh history json: its3 is the device's Jacobi PCG count per SIMPLE iteration")
    ap.add_argument("--its", type=int, default=800, help="SIMPLE iterations of the run the weighting covers")
    ap.add_argument("--set", action="append", default=[], help="name=value, repeatable")
    ap.add_argument("--md", action="store_true", help="markdown table")
    a = ap.parse_args()
    u = dict(DEF)
    for kv in a.set:
        k, v = kv.split("=")
        assert k in u, k
        u[k] = type(DEF[k])(float(v))
    rows = {}
    for line in open(a.sweep):
        r = json.loads(line)
        rows[(r["cfg"], r["sys"])] = r                                  # a rerun replaces the earlier row
    systems = sorted({s for _, s in rows}, key=lambda s: (s != "pf", int(s[2:]) if s[2:].isdigit() else 0))
    jac = {s: rows[("mode=jac", s)]["its"] for s in systems if ("mode=jac", s) in rows}
    t_j, spmv0 = jacobi_ms(u)
    hist = None
    if a.hist:
        h = json.load(open(a.hist))
        hist = np.array(h["its3"] if isinstance(h, dict) else [x["its3"] for x in h], float)[:a.its]
    print(f"fused Jacobi PCG iteration {t_j:.3f} ms on a core (fine SpMV {spmv0:.3f}); Jacobi its per system {jac}"
          + (f"; device mean over {len(hist)} its {hist.mean():.1f}" if hist is not None else ""))
    by = defaultdict(dict)
    for (cfg, s), r in rows.items():
        by[cfg][s] = r
    out = []
    for cfg, d in by.items():
        if cfg == "mode=jac":
            continue
        cells, gains = [], {}
        for s in systems:
            if s not in d or s not in jac:
                cells.append("-")
                continue
            r, t = d[s], solve_ms(d[s], u)
            conv = r["final"] <= max(1e-7, 0.0100001 * r["r0"]) or r["its"] < 1000
            gains[s] = (jac[s] * t_j) / (r["its"] * t["total"]) if conv and r["its"] < 1000 else 0.0
            cells.append(f"{r['its']} its x {t['total'] / t_j:.2f} = {gains[s]:.2f}x")
        t = solve_ms(next(iter(d.values())), u)
        w = ""
        if hist is not None and len(gains) >= 2:
            js = np.array([jac[s] for s in gains], float)
            o = np.argsort(js)
            ratio = np.array([d[s]["its"] / jac[s] for s in gains])[o]
            cost = np.array([solve_ms(d[s], u)["total"] for s in gains])[o]
            fpi = np.array([d[s]["spmv"][0] / max(d[s]["its"], 1) for s in gains])[o]
            m = hist * np.interp(hist, js[o], ratio)
            passes = (m * np.interp(hist, js[o], fpi)).mean()
            gain = (hist * t_j).sum() / (m * np.interp(hist, js[o], cost)).sum()
            w = f"{passes:.1f} passes, {gain:.2f}x"
            out.append((gain, cfg))
        print(f"{'| ' if a.md else ''}{cfg:58s} | " + " | ".join(cells) + f" | {w} | fine {t['fine']:.2f} vec {t['vec']:.2f} "
              f"coarse {t['coarse']:.2f} halo {t['halo']:.2f} red {t['red']:.2f} ms, L1 {t['l1'] / 1024:.0f} KB" + (" |" if a.md else ""))
    if out:
        g, cfg = max(out)
        print(f"best weighted: {g:.2f}x [{cfg}]")


if __name__ == "__main__":
    main()
