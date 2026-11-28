"""Residue census: how many neighbour lanes a constant lane shift delivers under a lane order, from the mesh alone.

    python3 -m ttfluid.tools.residcensus MESH.npz [--own own_cut] [--orders sliced+index,cls+rcm,cls+boxx,...]
                                         [--cores all] [--json OUT] [--jobs 6]

A neighbour gather is one map per ELL slot. The machine splits a map into a bulk read of the source at one lane shift
(Machine._split_gathers: the commonest shift that is a multiple of 4 lanes, one for all cores) and the residue, the
entries a scalar loop of the reader patches in. An entry at the shift costs nothing, so the lane order and the slot a
face takes decide how many residue entries exist. MESH.npz and the CLASS+KEY orders are ordercensus's. Added keys:

    box<f>  lexicographic on the octree lattice, axis f (x, y, z) fastest; boxa: per piece the fastest axis is the one
            with the most cells in rows of 8 or more. The lattice: a cell with six faces, one per
            direction, is regular; regular cells joined by a face form a piece of one refinement level, and walking the
            faces gives every cell of a piece integer coordinates. Pieces by size, the other cells behind them (rcm).
A class may carry q<Q> (clsq256+boxa): every stretch of 4Q lanes of a class is split into four interleaved quarters
(machine._quarter; q0 the whole class, qp each lattice piece), so a neighbour s places on sits 4s lanes on and the
shift is one the unpacker and the NoC can address (16 B).

Slots: `face` (today: the k-th face of a cell in face order) and `dir` (the first face of direction c takes slot c,
further faces of a direction take slots 6 and up).

Per order and slotting, over the local entries (the neighbour is on the same core), with any shift and with shifts
that are multiples of 4 lanes:
    map    entries at the commonest shift of their slot over all cores (the machine today)
    core   one shift per slot and core
    tile   one shift per slot and destination tile; tile24 counts a tile's shift only from 24 entries (a bulk read
           costs about 24 scalar entries)
    run8   entries in runs of 8 or more consecutive destination lanes of one tile at one shift (run4, run32: 4, 32),
           and the runs per cell: what a reader that copies (start, length, shift) runs would not patch one by one
    free8  the same with the slot of a face free: runs of one shift whatever slot the entries sit in. Runs of one
           shift are intervals of lanes and a lane is in at most as many as its cell has faces, so a slot assignment
           that keeps every run in one slot exists
Not orders, but layouts with idle lanes, from the lattice alone:
    bbox   every piece padded to its bounding box: every face between two regular cells is at a constant shift
    brick  pieces cut into columns of 32 x 32 cells, one tile per layer: shifts 1, 32 and 1024 lanes for every core;
           faces that leave a column sideways are residue. `fill` is cells per lane.
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

from . import ordercensus as oc
from .ordercensus import G, LANES
from ..resident.lattice import AX, BIG, box_order, rows8, pieces as lattice
from ..resident.machine import _quarter

def dir_slots(row, dc, fid, first):
    """The first face of direction c takes slot c, a cell's further faces of a direction slots 6 and up."""
    od = np.lexsort((fid, dc, row))
    r, d = row[od], dc[od].astype(np.int64)
    head = np.r_[True, (r[1:] != r[:-1]) | (d[1:] != d[:-1])]
    cs = np.r_[0, np.cumsum(~head)]
    s = np.empty(len(row), np.int64)
    s[od] = np.where(head, d, 5 + cs[1:] - cs[first[r]])
    return s


def cover(lane, row, oth, loc, slot, nslot):
    """Local entries a constant shift delivers, per kind of shift (any, a multiple of 4 lanes)."""
    dl = lane[row[loc]]
    d, s, t = lane[oth[loc]] - dl, slot[loc], dl >> 10
    out = {}
    for name, ok in (("any", slice(None)), ("m4", d % 4 == 0)):
        dd, ss, tt = d[ok] + BIG // 2, s[ok], t[ok]
        u, cnt = np.unique(ss * BIG + dd, return_counts=True)
        us = u // BIG
        best = np.zeros(nslot, np.int64)
        np.maximum.at(best, us, cnt)
        top = {}
        for k in range(nslot):
            m = np.flatnonzero(us == k)
            m = m[np.argsort(-cnt[m])[:8]]
            top[k] = {int(u[i] % BIG - BIG // 2): int(cnt[i]) for i in m}
        u2, c2 = np.unique((ss * 64 + tt) * BIG + dd, return_counts=True)
        bt = np.zeros(nslot * 64, np.int64)
        np.maximum.at(bt, u2 // BIG, c2)
        out[name] = dict(core=int(best.sum()), tile=int(bt.sum()), tile24=int(bt[bt >= 24].sum()),
                         ntile24=int((bt >= 24).sum()), top=top)
        for pre, by in (("run", ss), ("free", dd)):       # runs inside a slot; runs of one shift whatever the slot
            q = np.lexsort((dl[ok], by))
            a, b, e = by[q], dl[ok][q], dd[q]
            cont = (a[1:] == a[:-1]) & (b[1:] == b[:-1] + 1) & (b[1:] >> 10 == b[:-1] >> 10) & (e[1:] == e[:-1])
            rl = np.diff(np.r_[np.flatnonzero(np.r_[True, ~cont]), len(q)])
            for m in (4, 8, 32):
                out[name]["%s%d" % (pre, m)], out[name]["n%s%d" % (pre, m)] = int(rl[rl >= m].sum()), int((rl >= m).sum())
    return out


def core(c):
    R, O, FID, DC, L = (G[k] for k in ("R", "O", "FID", "DC", "L"))
    lo, hi = G["start"][c], G["start"][c + 1]
    cid = np.flatnonzero(G["own"] == c)
    n = len(cid)
    row = np.searchsorted(cid, R[lo:hi])
    o = O[lo:hi].astype(np.int64)
    loc = o >= 0
    loc[loc] = G["own"][o[loc]] == c
    halo = (o >= 0) & ~loc
    oth = np.where(loc, np.searchsorted(cid, np.where(loc, o, cid[0])), 0)
    fc = np.bincount(row, minlength=n)
    first = np.zeros(n + 1, np.int64)
    np.cumsum(fc, out=first[1:])
    N = hi - lo
    dc = DC[lo:hi]
    slots = dict(face=np.arange(N) - first[row], dir=dir_slots(row, dc, FID[lo:hi], first))
    nslot = int(max(s.max() for s in slots.values())) + 1
    sender = np.zeros(n, bool)
    sender[row[halo]] = True
    K, _ = oc.keys(n, row, oth, loc, G["C"][cid], sender)
    reg, comp, xyz, size, dims, good = lattice(n, row, oth, loc, dc, fc)
    rows = rows8(comp, xyz, len(size))
    for f in AX:
        K["box" + f] = oc.rank(box_order(comp, xyz, size, K["rcm"], f))
    K["boxa"] = oc.rank(box_order(comp, xyz, size, K["rcm"]))
    big = size > 1
    inbox = big[comp]
    out = dict(core=c, n=n, entries=int(N), local=int(loc.sum()), halo=int(halo.sum()), regular=int(reg.sum()),
               rows8=int(rows.max(1).sum()),
               inbox=int(inbox.sum()), pieces=int(big.sum()), piece_max=int(size.max()),
               in1000=int(size[size >= 1000].sum()), regular_entries=int((loc & reg[row] & reg[oth]).sum()),
               good=int(good.sum()), bbox_lanes=int(dims[big].prod(1).sum() + (~inbox).sum()), orders={}, brick={})
    for f, (a0, a1, a2) in AX.items():
        col = (comp * 64 + xyz[:, a0] // 32) * 64 + xyz[:, a1] // 32
        tiles = len(np.unique((col * 1024 + xyz[:, a2])[inbox]))
        side = good & (dc // 2 != a2) & (col[row] != col[oth])
        out["brick"][f] = dict(tiles=tiles, lanes=int(tiles * LANES + (~inbox).sum()), covered=int(good.sum() - side.sum()))
    for how in G["orders"]:
        cls, _, key = how.rpartition("+")
        cls, _, quart = cls.partition("q")
        cl = -fc if cls == "sliced" else L[fc - 1] if cls == "cls" else np.zeros(n, np.int64)
        lane = np.empty(n, np.int64)
        if cls.startswith("ch"):
            cl = L[fc - 1]
            lane[np.lexsort((K[key], cl, K[key] // (int(cls[2:]) * LANES)))] = np.arange(n)
        else:
            lane[np.lexsort((K[key], cl))] = np.arange(n)
        if quart:       # q<Q>: stretches of 4Q lanes of a class quartered (0: the whole class); qp: each lattice piece
            od = np.argsort(lane)
            run_of = cl[od] * len(size) + comp[od] if quart == "p" else cl[od]
            start = np.flatnonzero(np.r_[True, run_of[1:] != run_of[:-1]])
            run = np.diff(np.r_[start, n])
            m = np.arange(n) - np.repeat(start, run)
            lane[od] = np.repeat(start, run) + _quarter(m, np.repeat(run, run), BIG if quart in "0p" else int(quart))
            assert len(np.unique(lane)) == n
        if G.get("shifts") is None:
            out["orders"][how] = {s: cover(lane, row, oth, loc, slots[s], nslot) for s in slots}
        else:
            out["orders"][how] = {s: gmiss(lane, row, oth, loc, slots[s], G["shifts"][how][s], G["ngc"]) for s in slots}
    return out


def gmiss(lane, row, oth, loc, slot, shift, ngcs):
    """ordercensus's page model (per slot, the (destination tile, source tile) pairs of the local entries walked by
    destination tile through ngc round-robin 4-KB slots) for every local entry (`all`) and for the residue the map's
    shift leaves (`res`, shift: slot -> lanes); `floor` is one page per source tile and slot, `at` the entries at
    the shift."""
    dl, sl, s = lane[row[loc]], lane[oth[loc]], slot[loc]
    at = sl - dl == np.array([shift.get(k, BIG) for k in range(int(s.max()) + 1)])[s]
    out = dict(at=int(at.sum()))
    for name, keep in (("all", np.ones(len(dl), bool)), ("res", ~at)):
        r = dict(floor=0, **{str(g): 0 for g in ngcs})
        for k in np.unique(s[keep]):
            e = keep & (s == k)
            u = np.unique((dl[e] >> 10) * 64 + (sl[e] >> 10))
            r["floor"] += len(np.unique(u % 64))
            for g in ngcs:
                r[str(g)] += oc.misses((u % 64).tolist(), g)
        out[name] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--own", default="own_cut")
    ap.add_argument("--orders", default="sliced+index,cls+rcm,ch8+rcm,cls+boxx,cls+boxy,cls+boxz,boxx")
    ap.add_argument("--cores", default="all")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--json")
    ap.add_argument("--misses", help="gather-cache sizes in tiles (14,40): add the page model, with and without the map's shift")
    ap.add_argument("--sweeps", type=float, default=57.0, help="slot sweeps over DRAM-homed sources per core-it (ordercensus)")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print("[%6.1f s]" % (time.time() - t0), *x, flush=True)
    P, nC, S = oc.load(a.mesh, a.own, log)
    cores = list(range(P)) if a.cores == "all" else [int(x) for x in a.cores.split(",")]
    G.update(orders=a.orders.split(","))
    res = []
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(core, cores):
            res.append(r)
            if len(res) % 10 == 0 or len(res) == len(cores):
                log("%d of %d cores" % (len(res), len(cores)))
    res.sort(key=lambda r: r["core"])
    tot = {k: sum(r[k] for r in res) for k in ("n", "entries", "local", "halo", "regular", "inbox", "in1000", "regular_entries", "good", "bbox_lanes", "pieces")}
    n, loc = tot["n"], tot["local"]
    print("cells %d on %d cores; entries per cell %.3f = local %.3f + other core %.3f + patch %.3f" % (
        n, len(res), tot["entries"] / n, loc / n, tot["halo"] / n, (tot["entries"] - loc - tot["halo"]) / n))
    print("regular cells %.1f %%, in a piece %.1f %% (pieces of >= 1000 cells %.1f %%), %.1f pieces a core; faces between regular cells "
          "%.1f %% of the local entries (%.2f %% of them off the lattice)" % (
              100 * tot["regular"] / n, 100 * tot["inbox"] / n, 100 * tot["in1000"] / n, tot["pieces"] / len(res),
              100 * tot["regular_entries"] / loc, 100 * (1 - tot["good"] / max(tot["regular_entries"], 1))))
    summ = dict(totals=tot, layouts={}, orders={})
    summ["layouts"]["bbox"] = dict(fill=n / tot["bbox_lanes"], covered=tot["good"] / loc)
    best = [min(r["brick"].values(), key=lambda b: b["lanes"]) for r in res]
    summ["layouts"]["brick"] = dict(fill=n / sum(b["lanes"] for b in best), covered=sum(b["covered"] for b in best) / loc,
                                    tiles=sum(b["tiles"] for b in best) / len(res))
    for name, s in summ["layouts"].items():
        print("layout %-5s | cells per lane %.3f | at a shift %.1f %% of local entries | residue %.3f lanes per cell and sweep (with other-core %.3f)" % (
            name, s["fill"], 100 * s["covered"], loc / n * (1 - s["covered"]), (loc * (1 - s["covered"]) + tot["halo"]) / n))
    print("order         slots kind | at a shift, %% of local entries: map core tile tile24 | residue lanes per cell and sweep: map core tile24 | (slot, tile) shifts a core | in slot runs >= 4, 8, 32, runs per cell | in free-slot runs >= 4, 8, 32, runs per cell | map shifts")
    for how in G["orders"]:
        for sl in ("face", "dir"):
            for kind in ("any", "m4"):
                rs = [r["orders"][how][sl][kind] for r in res]
                glob = {}
                for r in rs:
                    for k, top in r["top"].items():
                        for d, cnt in top.items():
                            glob[(int(k), int(d))] = glob.get((int(k), int(d)), 0) + cnt
                shifts = {}
                for (k, d), cnt in glob.items():
                    if cnt > shifts.get(k, (0, 0))[1]:
                        shifts[k] = (d, cnt)
                s = dict(map=sum(v[1] for v in shifts.values()) / loc, core=sum(r["core"] for r in rs) / loc,
                         tile=sum(r["tile"] for r in rs) / loc, tile24=sum(r["tile24"] for r in rs) / loc,
                         ntile24=sum(r["ntile24"] for r in rs) / len(res), map_shifts={k: v[0] for k, v in sorted(shifts.items())},
                         **{k: sum(r[k] for r in rs) / loc for k in ("run4", "run8", "run32", "free4", "free8", "free32")},
                         **{k: sum(r[k] for r in rs) / n for k in ("nrun4", "nrun8", "nrun32", "nfree4", "nfree8", "nfree32")})
                s["residue"] = {k: loc / n * (1 - s[k]) for k in ("map", "core", "tile24", "run4", "run8", "run32", "free4", "free8")}
                summ["orders"]["%s/%s/%s" % (how, sl, kind)] = s
                print("%-13s %-4s %-3s | %5.1f %5.1f %5.1f %5.1f | %.3f %.3f %.3f | %6.1f | %5.1f %5.1f %5.1f %% %.3f %.3f %.3f | %5.1f %5.1f %5.1f %% %.3f %.3f %.3f | %s" % (
                    how, sl, kind, 100 * s["map"], 100 * s["core"], 100 * s["tile"], 100 * s["tile24"], s["residue"]["map"],
                    s["residue"]["core"], s["residue"]["tile24"], s["ntile24"], 100 * s["run4"], 100 * s["run8"], 100 * s["run32"],
                    s["nrun4"], s["nrun8"], s["nrun32"], 100 * s["free4"], 100 * s["free8"], 100 * s["free32"], s["nfree4"],
                    s["nfree8"], s["nfree32"], list(s["map_shifts"].values())[:6]), flush=True)
    if a.misses:
        G.update(ngc=[int(g) for g in a.misses.split(",")],
                 shifts={how: {sl: {int(k): d for k, d in summ["orders"]["%s/%s/m4" % (how, sl)]["map_shifts"].items()}
                               for sl in ("face", "dir")} for how in G["orders"]})
        with Pool(a.jobs) as pool:
            mres = sorted(pool.imap_unordered(core, cores), key=lambda r: r["core"])
        log("page model done")
        print("gather pages per slot sweep, mean of %d cores (x %.1f sweeps = pages per core-it on today's order)" % (len(res), a.sweeps))
        print("order         slots | entries at the map's shift | every local entry gathered: floor, " + ", ".join("gcache %s" % g for g in G["ngc"]) +
              " | residue of the shift only: floor, " + ", ".join("gcache %s" % g for g in G["ngc"]))
        summ["pages"] = {}
        for how in G["orders"]:
            for sl in ("face", "dir"):
                rs = [r["orders"][how][sl] for r in mres]
                m = {w: {k: sum(r[w][k] for r in rs) / len(rs) for k in rs[0][w]} for w in ("all", "res")}
                m["at"] = sum(r["at"] for r in rs) / loc
                summ["pages"]["%s/%s" % (how, sl)] = m
                print("%-13s %-4s | %5.1f %% | %s | %s" % (how, sl, 100 * m["at"], *(" ".join("%7.1f" % m[w][k] for k in m[w]) for w in ("all", "res"))), flush=True)
        for r, mr in zip(res, mres):
            r["pages"] = mr["orders"]
    if a.json:
        for r in res:
            for o in r["orders"].values():
                for sl in o.values():
                    for kind in sl.values():
                        del kind["top"]
        with open(a.json, "w") as f:
            json.dump(dict(mesh=a.mesh, own=a.own, P=P, cells=nC, cores=cores, summary=summ, per_core=res), f)
        log("wrote", a.json)


if __name__ == "__main__":
    main()
