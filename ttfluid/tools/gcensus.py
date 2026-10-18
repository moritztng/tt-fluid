"""Gather-cache misses of a lowered Machine, split by gather op and source, and their price under other lane orders.

    run_mesh.py ... --gdump OUT.npz      lower, dump every gather op of the program and the local entries of
                                         --gdump-cores (default: every 11th core of chip 0), exit before encoding
    gcensus.py OUT.npz MESH.npz [--orders sliced,sliced+hilbert,...] [--ngc 6,8]

A gather op is a `gather` MIR instruction (counted once per split, as ttfluid.resident.census) or a reader-patched page (an ldr
with a patch map, --pseg). Static count: an op in the outer body runs once per outer iteration, an op in a Krylov loop
once per Krylov iteration (region names from ttfluid.resident.census.region_of). A part's local entries (kind 0) whose source is
DRAM-homed are walked in destination-lane order through NGC round-robin 4 KB tile slots, reset every op (vm_ncrisc
gather cache; ttfluid.resident.census._misses). `bysrc` walks each destination tile's entries by source tile instead (an encode
order change: same values written). Other lane orders permute the cells space's lanes inside each core (both the
destination lanes and the source idx of cells-space maps), re-sorted within each face-count run so --sliced's prefix
property holds; MESH.npz (gprice.py mesh) supplies face counts and centres.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

def dump(m, prog, path, cores):
    from ttfluid.resident.census import region_of
    from ttfluid.resident.machine import DBASE
    from ttfluid.resident.program import PHASES
    reg = region_of(m.mir)
    labels_ph = {L: PHASES[i] for L, i in getattr(prog, "phase_at", {}).items()}
    ops, ph = [], "top"
    for j, ins in enumerate(m.mir):
        if ins[0] == "label" and ins[1] in labels_ph:
            ph = labels_ph[ins[1]]
        got = []
        if ins[0] == "gather" and (len(ins) < 5 or ins[4] in (1, 3, 4)):
            got.append((ins[2], ins[3], "gather"))
        blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"] if ins[0] == "seg" else []
        for b in blks:
            for u in b[1]:
                if u[0] == "ldr" and u[4] is not None:
                    got.append((u[4], u[5], "patch"))
        for mid, parts, kind in got:
            ops.append(dict(j=j, region=reg[j], phase=ph, mid=int(mid), kind=kind,
                            dram=[bool(m.phys[s] >= DBASE) for s, _ in parts]))
    maps = []
    for mid, (dsp, parts) in enumerate(m.maps.maps):
        maps.append(dict(dst=dsp.name, src=[s.name for s, _, _ in parts], lanes=m.maps.lanes[mid]))
    cells = prog.cells.name
    out = dict(meta=json.dumps(dict(T=m.T, P=m.P, cells=cells, ops=ops, maps=maps, cores=list(cores))))
    own, lane = m.lay.owner[cells], m.lay.lane[cells]
    for c in cores:
        ids = np.nonzero(own == c)[0]
        out[f"cell_{c}"] = ids[np.argsort(lane[ids])]        # cell id at each lane
        out[f"lane_{c}"] = np.sort(lane[ids])
        for mid, t in enumerate(m.maps.tab):
            sel = (t["core"] == c) & (t["kind"] == 0)
            if sel.any():
                out[f"t{mid}_{c}"] = np.stack([t["lane"][sel], t["part"][sel].astype(np.int32), t["idx"][sel]])
    np.savez_compressed(path, **out)
    print("gdump: %d gather ops, %d maps, cores %s -> %s" % (len(ops), len(maps), list(cores), path), flush=True)


def misses(tiles, ngc):
    tags, nxt, miss = [-1] * ngc, 0, 0
    for t in tiles:
        if t not in tags:
            tags[nxt] = t
            nxt = (nxt + 1) % ngc
            miss += 1
    return miss


def perm(how, cells, fc, xyz):
    """New lane of each current lane (cells = cell id at each current lane of one core)."""
    from gprice import core_lanes
    if how == "sliced":
        return np.arange(len(cells))
    if how == "index":                      # no --sliced: rank of the global index inside the core
        return np.argsort(np.argsort(cells, kind="stable"), kind="stable")
    return core_lanes(cells, fc[cells], xyz[cells], how)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("mesh")
    ap.add_argument("--orders", default="sliced,sliced+hilbert,sliced+rcbleaf")
    ap.add_argument("--ngc", default="6,8")
    ap.add_argument("--json")
    ap.add_argument("--slotl", default="20,20,20,20,20,20,3,3,3" + ",1" * 12, help="Machine.slotL (lowering log)")
    a = ap.parse_args()
    import gprice
    gprice.SLOTL = [int(x) for x in a.slotl.split(",")]
    D = np.load(a.dump)
    meta = json.loads(str(D["meta"]))
    ngcs = [int(x) for x in a.ngc.split(",")]
    M = None
    fc = xyz = None
    if a.orders != "sliced":
        from gprice import ell
        M = dict(np.load(a.mesh))
        _, fc = ell(M)
        xyz = M["C"]
    cells = meta["cells"]
    ops = meta["ops"]
    nops = Counter((o["region"], o["phase"], o["kind"]) for o in ops)
    print("gather ops by region/phase/kind:", dict(nops))
    res = defaultdict(Counter)       # (order, walk, ngc) -> (region|phase) -> misses per execution
    per_op = defaultdict(Counter)    # (order, walk, ngc) -> mid -> misses
    per_core = defaultdict(Counter)  # (order, walk, ngc) -> core -> outer misses
    for c in meta["cores"]:
        cid = D[f"cell_{c}"]
        for how in a.orders.split(","):
            p = perm(how, cid, fc, xyz)          # current lane -> new lane (cells space)
            cache = {}
            for o in ops:
                mid = o["mid"]
                key = f"t{mid}_{c}"
                if key not in D.files:
                    continue
                mp = meta["maps"][mid]
                for walk in ("bylane", "bysrc"):
                    ck = (mid, walk, tuple(o["dram"]))
                    if ck not in cache:
                        ln, pt, ix = D[key]
                        if mp["dst"] == cells:
                            ln = p[ln]
                        tot = Counter()
                        for k, sname in enumerate(mp["src"]):
                            if not o["dram"][k]:
                                continue
                            e = pt == k
                            dl, si = ln[e], ix[e]
                            if sname == cells:
                                si = p[si]
                            q = np.lexsort((si, dl >> 10)) if walk == "bysrc" else np.argsort(dl, kind="stable")
                            seq = (si[q] >> 10).tolist()
                            for g in ngcs:
                                tot[g] += misses(seq, g)
                        cache[ck] = tot
                    for g in ngcs:
                        res[(how, walk, g)][o["region"] + "|" + o["phase"]] += cache[ck][g]
                        res[(how, walk, g)]["_all_" + o["region"]] += cache[ck][g]
                        per_op[(how, walk, g)][mid] += cache[ck][g]
                        if o["region"] == "outer":
                            per_core[(how, walk, g)][c] += cache[ck][g]
    nc = len(meta["cores"])
    print("cores %s, T %d; misses per execution of the region, mean over cores (pages of 4 KB)" % (meta["cores"], meta["T"]))
    rows = sorted({k for v in res.values() for k in v})
    for key in sorted(res):
        v = res[key]
        print("%-16s %-6s ngc%d  outer %8.0f  | " % (key + (), ) if False else "%-16s %-6s ngc%d  outer %8.0f  | " % (
            key[0], key[1], key[2], v["_all_outer"] / nc) + " ".join("%s %.0f" % (r, v[r] / nc) for r in rows
                                                                    if not r.startswith("_") and v[r]), flush=True)
    for key in sorted(per_core):
        print("%-16s %-6s ngc%d  outer by core: " % key + " ".join("%d:%d" % kv for kv in sorted(per_core[key].items())))
    if a.json:
        Path(a.json).write_text(json.dumps({"|".join(map(str, k)): {r: x / nc for r, x in v.items()} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
