"""Price mgmodel's solves on the device: what one preconditioned p iteration costs a core, against fused Jacobi PCG.

    python -m ttfluid.tools.mgprice SWEEP.jsonl [--pre meas|lean] [--set name=value ...] [--hist HIST.json] [--md]

SWEEP.jsonl is mgmodel's --json output for ONE partition (cells per core set the tiles, so k2, k4 and k8 are priced
from their own files). A loop on a core is counted in the units the campaign measured:

    time = ring pages x page + gathered values x g + L1 loads x load + L1 stores x pack + op-tiles x op
           + reductions x red_us + halo exchanges x halo_us (fine) or chalo_us (coarse)

summed serially, as if the reader, the compute threads and the gather threads never overlapped. That is pessimistic
for both sides and more so for multigrid, whose coarse work is gather on threads the fine sweep leaves idle.

What lives in L1 is decided the same way for both solvers: the loop's operands are sorted by ring pages saved per
KB and take the `l1_kb` the p loop has; the rest streams, a page per tile per use. The reference is the fused
one-reduction PCG on the Jacobi-scaled system (flx-pfuse's design): r, p, s, w and x, `cslots` coefficient tiles and
`tabs` gather-table tiles per vector tile, 20 L1 loads, 5 stores and 25 op-tiles per fine tile, the SpMV's
off-diagonal entries as gathered values, one halo exchange, one reduction.

A multigrid iteration adds, per core:
  * the correction as one more fine vector (4 loads, 2 stores, 4 op-tiles per fine tile);
  * restriction and prolongation between the fine level and level 1: with a core's cells ordered aggregate-major in
    planes (rp=0) one load, one op and one store per fine tile each way, as generic gathers (rp=1) one gathered
    value per fine cell each way;
  * every coarse SpMV: the level's off-diagonal entries on the fullest core, padded by ell_pad, as gathered values,
    two loads and one op per slot tile, its operator as ring pages unless it is resident (6 B an entry), and on a
    level whose couplings cross cores one halo exchange;
  * the coarse smoother's vector work (2 loads, 1 store, 3 op-tiles per coarse tile per step) and the transfers
    between coarse levels as gathers.
The coarse operators are rebuilt once per SIMPLE iteration (the p matrix changes): every entry of every level but
the last is gathered once into the next level's sums, and the run weighting charges that.

Unit costs (--pre picks a set, --set name=value overrides; cycles unless named _us):
  meas  ring page 1,280 (flx-pfuse's slope on the p loop; 1,620 on assembly), gathered value 2.5 wall on three
        RISC-Vs (flx-gather's dense 16-bit table; 6.45 on one), stock L1 load 125, pack 42, op-tile 227 (flx-kernel)
  lean  the same with the lean load (58) and generated blocks (80 an op-tile)
With --hist (a run_mesh history) the solves are weighted to the run by the device's own Jacobi iteration count per
SIMPLE iteration: the multigrid count is interpolated in the Jacobi count between the captured systems.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

PRE = dict(
    meas=dict(mhz=1350.0, page=1280.0, g=2.5, load=125.0, pack=42.0, op=227.0, red_us=72.0, halo_us=110.0,
              chalo_us=30.0, cslots=6.46, tabs=2.9, ell_pad=1.15, rp=0.0, l1_kb=1000.0),
    lean=dict(mhz=1350.0, page=1280.0, g=2.5, load=58.0, pack=42.0, op=80.0, red_us=72.0, halo_us=110.0,
              chalo_us=30.0, cslots=6.46, tabs=2.9, ell_pad=1.15, rp=0.0, l1_kb=1000.0),
)
CFG = dict(agg="3", mode="post", k=1, cmode="sym", ck=2, kc=16, local=99, glob=99, gamma=1, depth=None)
LOADS, STORES, OPS = 20, 5, 25                                           # per fine tile, the fused reference
KB = 4.096                                                               # one fp32 tile


def cfg_of(name):
    c = dict(CFG)
    for kv in name.split():
        k, v = kv.split("=")
        if k in c:
            c[k] = v if k in ("agg", "mode", "cmode") else int(v)
    return c


def place(items, room):
    """items: (name, KB, ring pages per iteration if streamed). The best pages per KB take the room, the last one
    in part. Returns (pages left on the ring, KB resident, {name: resident share})."""
    pages, used, share = 0.0, 0.0, {}
    for name, kb, pg in sorted(items, key=lambda x: -(x[2] / max(x[1], 1e-9))):
        f = max(0.0, min(1.0, (room - used) / kb)) if kb > 0 else 1.0
        used += f * kb
        pages += (1.0 - f) * pg
        share[name] = f
    return pages, used, share


def solve_ms(row, u):
    """One iteration of a mgmodel solve on a core, ms, split into its parts, with its syncs, pages and L1 plan."""
    c, its = cfg_of(row["cfg"]), max(row["its"], 1)
    cyc = 1e-3 / u["mhz"]
    lv = row["levels"]
    n0 = lv[0]["rows_core"][2]
    T = math.ceil(n0 / 1024)
    gv0 = (lv[0]["nnz_core_max"] - n0) * 0.9                             # the SpMV's gathered values (boundary lanes read zero)
    jac = "mode=jac" in row["cfg"]
    f0 = 1.0 if jac else row["spmv"][0] / its
    # ---- operands and where they live
    items = [("vec", 4 * T * KB, 4 * 3 * T), ("x", T * KB, 2 * T), ("coef", u["cslots"] * T * KB, f0 * u["cslots"] * T),
             ("tab", u["tabs"] * T * KB, f0 * u["tabs"] * T)]
    loads, stores, ops = LOADS * T, STORES * T, OPS * T
    gv = f0 * gv0
    halo = red = 1.0
    chalo = 0.0
    forced = 0.0                                                         # coarse vectors: resident or the design is void
    if not jac:
        steps0 = c["k"] * (2 if c["mode"] == "sym" else 1)
        items.append(("corr", T * KB, 3 * T))
        loads += (4 + (f0 - 1) * LOADS + 2 * (steps0 - 1)) * T
        stores += (2 + (f0 - 1) * STORES + (steps0 - 1)) * T
        ops += (4 + (f0 - 1) * OPS + 3 * (steps0 - 1)) * T
        halo = f0
        nrp = (row["R"][0] + row["P"][0]) / its if row["R"] else 0.0
        if u["rp"]:
            gv += nrp * n0
        else:
            loads, stores, ops = loads + nrp * T, stores + nrp * T, ops + nrp * T
        for l, st in enumerate(lv):
            if l == 0:
                continue
            rows, nnz = st["rows_core"][2], st["nnz_core_max"]
            tl = math.ceil(rows / 1024)
            off = (nnz - rows) * u["ell_pad"]
            s = row["spmv"][l] / its
            last = l == len(lv) - 1
            steps = c["kc"] if last else c["ck"] * (2 if c["cmode"] == "sym" else 1)
            slots = off / rows
            gv += s * off
            loads += s * 2 * slots * tl + steps * 2 * tl
            stores += steps * tl
            ops += s * slots * tl + steps * 3 * tl
            if l < len(row["R"]):
                gv += (row["R"][l] + row["P"][l]) / its * rows
            if l < c["local"] or l >= c["glob"]:
                chalo += s
            forced += 4 * tl * KB
            items.append((f"A{l}", off * 6 / 1024, s * off * 6 / 4096))
    pages, used, share = place(items, u["l1_kb"] - forced)
    t = dict(pages=pages * u["page"] * cyc, gather=gv * u["g"] * cyc,
             l1=(loads * u["load"] + stores * u["pack"] + ops * u["op"]) * cyc,
             sync=(red * u["red_us"] + halo * u["halo_us"] + chalo * u["chalo_us"]) * 1e-3)
    setup = 0.0 if jac else sum(st["nnz_core_max"] for st in lv[:-1]) * u["g"] * cyc
    return dict(total=sum(t.values()), **t, npages=pages, gv=gv, nsync=red + halo + chalo, kb=used + forced,
                share=share, setup=setup, T=T, cells=n0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep")
    ap.add_argument("--pre", default="meas", choices=sorted(PRE))
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
    jac = {s: rows[("mode=jac", s)] for s in systems if ("mode=jac", s) in rows}
    assert jac, "the sweep needs its mode=jac rows"
    lv0 = next(r["levels"][0] for r in rows.values() if r.get("levels"))  # the fine level is the partition's, any row has it
    for r in jac.values():
        r["levels"] = [lv0]
    tj = solve_ms(next(iter(jac.values())), u)
    hist = None
    if a.hist:
        h = json.load(open(a.hist))
        hist = np.array(h["its3"] if isinstance(h, dict) else [x["its3"] for x in h], float)[:a.its]
    fmt = lambda t: (f"{t['total']:.2f} ms = pages {t['pages']:.2f} ({t['npages']:.0f}) + gather {t['gather']:.2f} "
                     f"({t['gv'] / 1e3:.0f}k) + L1 moves and ops {t['l1']:.2f} + syncs {t['sync']:.2f} ({t['nsync']:.0f})")
    print(f"{a.pre}: {tj['cells']} cells a core, {tj['T']} tiles; fused Jacobi PCG iteration {fmt(tj)}; resident "
          + ", ".join(f"{k} {100 * v:.0f} %" for k, v in tj["share"].items())
          + f"; Jacobi its {({s: r['its'] for s, r in jac.items()})}"
          + (f"; device mean over {len(hist)} its {hist.mean():.1f}, p solve {hist.sum() * tj['total'] * 1e-3:.0f} s"
             if hist is not None else ""))
    by = defaultdict(dict)
    for (cfg, s), r in rows.items():
        by[cfg][s] = r
    bar = "| " if a.md else ""
    print(f"{bar}{'cycle':66s} | " + " | ".join(f"{s} ({jac[s]['its'] if s in jac else '-'})" for s in systems)
          + " | run: fine passes, p solve s, gain | one iteration on a core | streamed" + (" |" if a.md else ""))
    if a.md:
        print("|" + "---|" * (len(systems) + 4))
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
            gains[s] = (jac[s]["its"] * tj["total"]) / (r["its"] * t["total"]) if r["its"] < 1000 else 0.0
            cells.append(f"{r['its']} x {t['total'] / tj['total']:.2f} = {gains[s]:.2f}x")
        t = solve_ms(next(iter(d.values())), u)
        w = ""
        if hist is not None and len(gains) >= 2:
            js = np.array([jac[s]["its"] for s in gains], float)
            o = np.argsort(js)
            n_mg = np.interp(hist, js[o], np.array([d[s]["its"] for s in gains], float)[o])
            cost = np.interp(hist, js[o], np.array([solve_ms(d[s], u)["total"] for s in gains])[o])
            fpi = np.interp(hist, js[o], np.array([d[s]["spmv"][0] / max(d[s]["its"], 1) for s in gains])[o])
            sec = (n_mg * cost + t["setup"]).sum() * 1e-3
            gain = hist.sum() * tj["total"] * 1e-3 / sec
            w = f"{(n_mg * fpi).mean():.1f}, {sec:.0f} s, {gain:.2f}x"
            out.append((gain, cfg))
        print(f"{bar}{cfg:66s} | " + " | ".join(cells) + f" | {w} | {fmt(t)} | "
              + (", ".join(f"{k} {100 * (1 - v):.0f} %" for k, v in t["share"].items() if v < 0.995) or "nothing")
              + f"; L1 {t['kb']:.0f} KB" + (" |" if a.md else ""))
    if out:
        g, cfg = max(out)
        print(f"best weighted: {g:.2f}x [{cfg}]")


if __name__ == "__main__":
    main()
