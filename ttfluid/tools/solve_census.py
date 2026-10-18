"""Static DRAM traffic of one pass of every inner Krylov loop (U/k/omega PBiCGStab, p PCG), and where the loop's vector
registers live: the k2-vs-k4 question (why a segregated solve sweep costs 8x at T 39 against T 20).

    solve_census.py MCACHE [--core C] [--gcache 6] [--json out.json]

MCACHE is run_mesh's --mcache (LOWER, prog, m). A loop body's MIR runs once per pass (its exit branch is part of it, so
a pass is counted whole). Kinds, in 4 KB pages: const (ldr of an in-place constant), vec (ldr of a DRAM-homed vector),
store (a pack into a DRAM home), gmiss (DRAM source tiles a gather or a reader-patched page fetches through the
round-robin cache: ttfluid.resident.census._misses over the core's entries in table order), tab (gather table bytes / 4096).
Dest round trips, in tiles: blocks (acquires), loads (ld/ldr), reloads (a load of a value an earlier block of the same
straight block run stored: what more Dest slots per block could keep in Dest, an upper bound).
Homes of the loop's vector registers: dram, ovl (wplace 2: on the one-tile pool), l1 (an L1 colour), const.
"""
import argparse
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ttfluid.resident.census import _misses, region_of  # noqa: E402
from ttfluid.resident.machine import DBASE, KBASE, _mir_defs, _mir_uses  # noqa: E402
from ttfluid.resident.program import PHASES  # noqa: E402


def load(path):
    import cloudpickle
    f = open(path, "rb")
    if path.endswith(".zst"):
        import zstandard
        f = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True, closefd=True), 1 << 24)
    with f:
        return cloudpickle.load(f)


def census(prog, m, core, gcache=6):
    """{loop region: dict(phase, pages by kind, op_tiles, ops, homes)} for one core, per loop pass."""
    mir, reg = m.mir, region_of(m.mir)
    labels_ph = {L: PHASES[i] for L, i in getattr(prog, "phase_at", {}).items()}
    ovl = getattr(m, "ovl", {})

    def home(r):
        if r in ovl:
            return "ovl"
        if r in m.dram:
            return "const" if m.phys[r] >= KBASE else "dram"
        return "l1"

    gm = {}

    def gmiss(mid, parts, key):
        if key not in gm:
            t = m.maps.tab[mid]
            sel = t["core"] == core
            order = np.lexsort((t["lane"][sel], t["part"][sel]))
            kind, part, idx, lane = t["kind"][sel][order], t["part"][sel][order], t["idx"][sel][order], t["lane"][sel][order]
            tot = ts = 0
            for k, (s, _) in enumerate(parts):
                if m.phys[s] >= DBASE and s not in ovl:
                    e = (part == k) & (kind == 0)
                    tot += _misses((idx[e] >> 10).tolist(), gcache)
                    # --tsort 1: each destination tile's entries walked by source tile (same values written)
                    q = np.lexsort((idx[e], lane[e] >> 10))
                    ts += _misses((idx[e][q] >> 10).tolist(), gcache)
            gm[key] = (tot, 4 * (int(sel.sum()) + int((t["zc"] == core).sum())) / 4096, ts)
        return gm[key]

    pg = defaultdict(Counter)         # region -> kind -> pages per pass
    regs = defaultdict(dict)          # region -> register -> (home, size)
    phase = {}
    ph = "top"
    stored = set()
    for j, ins in enumerate(mir):
        if ins[0] == "label" and ins[1] in labels_ph:
            ph = labels_ph[ins[1]]
        rg = reg[j]
        if not rg.startswith("solve") and rg != "outer":
            continue
        key = rg if rg != "outer" else "outer|" + ph
        phase.setdefault(key, ph)
        c = pg[key]
        if rg != "outer":
            for r in set(_mir_uses(ins)) | set(_mir_defs(ins)):
                if r != "__anchor__" and not isinstance(r, tuple) and r in m.phys and m.is_vec(r):
                    regs[rg][r] = (home(r), m.size(r))
        items = ins[1] if ins[0] == "seg" else [ins]
        blks = [x for x in items if x[0] == "blk"]
        for x in items:   # reload: a load of a value an earlier block of the same uninterrupted block run stored, the
            if x[0] != "blk":   # round trip a block with more Dest slots could keep in Dest (the price of fewer acquires)
                stored.clear()
                continue
            bt = m.tiles([x])
            c["blocks"] += bt
            for u in x[1]:
                if u[0] in ("ld", "ldr") and u[2] != "__anchor__" and u[2] is not None:
                    c["loads"] += bt
                    if u[2] in stored:
                        c["reload_dram" if u[0] == "ldr" else "reload_l1"] += bt
            stored.update(r for _, r in x[2])
        for b in blks:
            bt = m.tiles([b])
            for u in b[1]:
                if u[0] == "ldr" and u[2] is not None:
                    c["const" if m.phys[u[2]] >= KBASE else "vec"] += bt
                    if u[4] is not None:
                        g, tb, gs = gmiss(u[4], u[5], ("p", u[4], tuple(x for x, _ in u[5])))
                        c["gmiss"] += g
                        c["gmiss_tsort"] += gs
                        c["tab"] += tb
                        c["patch_ops"] += 1
                elif u[0] == "op":
                    c["op_tiles"] += bt
            for _, x in b[2]:
                if x in m.dram:
                    c["store"] += bt
        if ins[0] == "gather" and (len(ins) < 5 or ins[4] in (1, 3, 4)):
            g, tb, gs = gmiss(ins[2], ins[3], ("g", ins[2], tuple(x for x, _ in ins[3])))
            c["gmiss"] += g
            c["gmiss_tsort"] += gs
            c["tab"] += tb
            c["gather_ops"] += 1
        if ins[0] in ("red", "red2", "sync", "halo"):
            c[ins[0]] += 1
    out = {}
    for rg, c in pg.items():
        r0 = rg.split("|")[0]
        h, hs = Counter(), Counter()
        for hm, n in regs.get(r0, {}).values():
            h[hm] += 1
            hs[hm] += n
        out[rg] = dict(phase=phase[rg], pages={k: c[k] for k in KINDS}, gmiss_tsort=c["gmiss_tsort"], op_tiles=c["op_tiles"],
                       ops={k: c[k] for k in ("gather_ops", "patch_ops", "red", "red2", "sync", "halo")},
                       dst={k: c[k] for k in ("blocks", "loads", "reload_l1", "reload_dram")}, homes=dict(h), home_tiles=dict(hs))
    return out


KINDS = ("const", "vec", "store", "gmiss", "tab")


def report(prog, m, cores, gcache=6, log=print):
    cnt = m.lay.count[prog.cells.name]
    res = {}
    for core in cores:
        out = census(prog, m, core, gcache)
        res[core] = dict(cells=int(cnt[core]), loops=out)
        log(f"core {core}: {int(cnt[core])} cells, T {m.T}, P {m.P}, one-tile pool {m.nsmall}, gcache {gcache}; pages per pass "
            "(loops) / per outer it (outer|phase)")
        log("%-20s %-12s " % ("region", "phase") + " ".join("%8s" % k for k in KINDS) + " %8s %8s | tsort            | gathers patch red sync halo | homes" % (
            "pages", "op_tiles"))
        for rg in sorted(out, key=lambda x: (x.startswith("outer"), x)):
            d = out[rg]
            log("%-20s %-12s " % (rg, d["phase"]) + " ".join("%8.0f" % d["pages"][k] for k in KINDS) + " %8.0f %8d | tsort gmiss %6.0f | %s | %s" % (
                sum(d["pages"].values()), d["op_tiles"], d["gmiss_tsort"], " ".join(str(d["ops"][k]) for k in ("gather_ops", "patch_ops", "red", "sync", "halo")),
                " ".join("%s %d (%d tiles)" % (k, d["homes"][k], d["home_tiles"][k]) for k in sorted(d["homes"]))) +
                " | blocks %(blocks)d loads %(loads)d reload l1 %(reload_l1)d dram %(reload_dram)d" % d["dst"])
    return dict(T=m.T, P=m.P, nsmall=m.nsmall, cores=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcache")
    ap.add_argument("--cores", default=None, help="comma list; default: the core with the most cells")
    ap.add_argument("--gcache", type=int, default=6)
    ap.add_argument("--json")
    a = ap.parse_args()
    LOWER, prog, m = load(a.mcache)
    cnt = m.lay.count[prog.cells.name]
    cores = [int(x) for x in a.cores.split(",")] if a.cores else [int(np.argmax(cnt))]
    res = report(prog, m, cores, a.gcache)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
