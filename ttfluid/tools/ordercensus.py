"""In-core cell order census: what a lane order inside each core does to the neighbour gather, from the mesh alone.

    python3 -m ttfluid.tools.ordercensus MESH.npz [--own own_cut] [--orders sliced,cls+rcm,...] [--cores all]
                                         [--ngc 4,8,14] [--json OUT] [--jobs 6]

MESH.npz is the cache the regularity census reads (cell centres C, owner, neighbour, face normal axis `ax` and sign
`sg`, patch ranges, one core owner per cell). No lowering is needed: the neighbour gather of every assembly and Krylov
op goes through the ELL slots, slot k of a cell being its k-th face in face order.

An order is CLASS+KEY or KEY. CLASS keeps what --sliced gives the lowering:
    sliced  face count, widest first (machine.sliced_lanes): slot k's entries are a lane prefix
    cls     slot class: cells whose last slot spans the fewest tiles first. Every slot keeps its tile extent L_k, so
            op-tile counts do not change, and the cells are cut into as few blocks as there are distinct L_k
    ch<B>   KEY cut into chunks of B tiles, slot classes inside each chunk: slot k is a set of tiles (`mask`), no
            longer a prefix. ch<B>g<G>: the classes with L_k <= G stay one prefix for the whole core
    (none)  no grouping: slots lose their short extents. A bound, not a candidate
KEY orders the cells inside a class:
    index   OpenFOAM cell id (today)
    rcm     reverse Cuthill-McKee on the core's cell graph (one sweep through the whole core, classes share it)
    hilbert 3D Hilbert curve on the centroids, 1024 steps over the core's longest extent
    axis    centroid along the core's longest extent
    bfs     breadth-first level from the cells that face another core, then rcm inside a level

Per order, over the local slot entries (the neighbour is on the same core) of every core:
    BAND    |tile of the destination - tile of the source| (p50, p99, max), also in lanes
    MISSES  4 KB pages a round-robin gather cache of N tiles reads per slot sweep (one gather of every slot, cache
            reset per slot: ttfluid.resident.census._misses), entries of a destination tile walked by source tile
            (--tsort 1, the kept build) or by lane; `floor` is the distinct source tiles per slot, which any cache pays.
            `lru` and `opt` are the same walk under least-recently-used and under the best possible replacement
    WSET    distinct source tiles one destination tile reads in one slot (p50, p99, max): the cache a walk needs
    RUNS    share of all slot entries in runs of >= 8 and >= 32 consecutive destination lanes with one lane shift, for
            slots in face order and by face direction
    HALO    lane runs and tiles the cells with a face on another core take (one run: the halo push is a block copy)
"""
import argparse
import json
import time
from multiprocessing import Pool

import numpy as np

LANES = 1024
G = {}


def hilbert_key(xyz, bits=10):
    """3D Hilbert index (Skilling) of points on a 2^bits grid over the longest extent."""
    lo = xyz.min(0)
    X = ((xyz - lo) / max(float((xyz.max(0) - lo).max()), 1e-30) * ((1 << bits) - 1)).astype(np.int64).T.copy()
    M = 1 << (bits - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(3):
            s = (X[i] & Q) != 0
            t = np.where(s, 0, (X[0] ^ X[i]) & P)
            X[0] = np.where(s, X[0] ^ P, X[0] ^ t)
            X[i] ^= t
        Q >>= 1
    for i in range(1, 3):
        X[i] ^= X[i - 1]
    t = np.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = np.where((X[2] & Q) != 0, t ^ (Q - 1), t)
        Q >>= 1
    X ^= t
    key = np.zeros_like(X[0])
    for b in range(bits - 1, -1, -1):
        for i in range(3):
            key = (key << 1) | ((X[i] >> b) & 1)
    return key


def rank(key):
    r = np.empty(len(key), np.int64)
    r[np.argsort(key, kind="stable")] = np.arange(len(key))
    return r


def misses(seq, ngc):
    tags, nxt, miss = [-1] * ngc, 0, 0
    for t in seq:
        if t not in tags:
            tags[nxt] = t
            nxt = (nxt + 1) % ngc
            miss += 1
    return miss


def misses_opt(seq, ngc):
    """The fewest misses any replacement rule can reach (Belady: evict the tile used farthest ahead). The walk is known
    at lowering time, so a table can carry this schedule and the kernel needs no rule of its own."""
    n = len(seq)
    nxt, last = [0] * n, {}
    for i in range(n - 1, -1, -1):
        nxt[i] = last.get(seq[i], n)
        last[seq[i]] = i
    cache, miss = {}, 0
    for i, t in enumerate(seq):
        if t not in cache:
            miss += 1
            if len(cache) >= ngc:
                del cache[max(cache, key=cache.get)]
        cache[t] = nxt[i]
    return miss


def misses_lru(seq, ngc):
    cache, miss = {}, 0
    for t in seq:
        if t in cache:
            del cache[t]
        else:
            miss += 1
            if len(cache) >= ngc:
                del cache[next(iter(cache))]
        cache[t] = 1
    return miss


def keys(n, row, oth, loc, xyz, sender):
    """KEY name -> rank of each of the core's cells."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    A = csr_matrix((np.ones(int(loc.sum()), np.int8), (row[loc], oth[loc])), shape=(n, n))
    rcm = np.empty(n, np.int64)
    rcm[reverse_cuthill_mckee(A, symmetric_mode=True)] = np.arange(n)
    lev = np.where(sender, 0, -1)
    x, l = sender.astype(np.int8), 0
    while True:
        l += 1
        new = (A.dot(x) > 0) & (lev < 0)
        if not new.any():
            break
        lev[new] = l
        x = new.astype(np.int8)
    lev[lev < 0] = l
    ext = xyz.max(0) - xyz.min(0)
    return dict(index=np.arange(n), rcm=rcm, hilbert=rank(hilbert_key(xyz)), axis=rank(xyz[:, int(np.argmax(ext))]),
                bfs=rank(lev * n + rcm)), int(lev.max())


def core(c):
    R, O, FID, DC, L, ngcs, orders, bylane = (G[k] for k in ("R", "O", "FID", "DC", "L", "ngc", "orders", "bylane"))
    lo, hi = G["start"][c], G["start"][c + 1]
    cid = np.flatnonzero(G["own"] == c)
    n = len(cid)
    row = np.searchsorted(cid, R[lo:hi])
    o = O[lo:hi].astype(np.int64)
    loc = o >= 0
    loc[loc] = G["own"][o[loc]] == c
    halo = (o >= 0) & ~loc
    oth = np.where(loc, np.searchsorted(cid, np.where(loc, o, cid[0])), -1)
    fc = np.bincount(row, minlength=n)
    first = np.zeros(n + 1, np.int64)
    np.cumsum(fc, out=first[1:])
    N = hi - lo
    slot = np.arange(N) - first[row]                       # entries are sorted by (row, face id): face order
    od = np.lexsort((FID[lo:hi], DC[lo:hi], row))
    slot_dir = np.empty(N, np.int64)
    slot_dir[od] = np.arange(N) - first[row[od]]
    sender = np.zeros(n, bool)
    sender[row[halo]] = True
    K, depth = keys(n, row, oth, loc, G["C"][cid], sender)
    T = -(-n // LANES)
    out = dict(core=c, n=n, T=T, entries=int(N), local=int(loc.sum()), halo=int(halo.sum()), senders=int(sender.sum()),
               bfs_depth=depth, orders={})
    for how in orders:
        cls, _, key = how.rpartition("+")
        cl = -fc if cls == "sliced" else L[fc - 1] if cls == "cls" else np.zeros(n, np.int64)
        lane = np.empty(n, np.int64)
        if cls.startswith("ch"):                             # chunks of B tiles along KEY, slot classes inside a chunk
            B, _, Gl = cls[2:].partition("g")                # g<G>: classes of L_k <= G stay one prefix for the core
            cl = L[fc - 1]
            ch = np.where(cl <= int(Gl or 0), -1, K[key] // (int(B) * LANES))
            lane[np.lexsort((K[key], cl, ch))] = np.arange(n)
            cl = np.where(cl == cl.max(), 1, 0)
        else:
            lane[np.lexsort((K[key], cl))] = np.arange(n)
        dl, sl = lane[row], np.where(loc, lane[np.where(loc, oth, 0)], 0)
        d = np.abs((dl >> 10) - (sl >> 10))[loc]
        r = dict(band=np.bincount(d, minlength=64).tolist(),
                 band_lanes=np.bincount(np.abs(dl - sl)[loc] >> 7, minlength=8 * 64).tolist(),
                 ext=[int(dl[slot == k].max() >> 10) + 1 for k in range(int(slot.max()) + 1)])
        ms = {f"{w}{g}": 0 for w in ("src", "lane", "lru", "opt") for g in ngcs}
        floor = pairs = 0
        wslot, ws = [], np.zeros(64, np.int64)
        main = cl == cl.max()                                # the class that spans every tile
        r["band_main"] = np.bincount(d[(main[row] & main[np.where(loc, oth, 0)])[loc]], minlength=64).tolist()
        r["mask"] = [int(len(np.unique(dl[slot == k] >> 10))) for k in range(int(slot.max()) + 1)]
        for k in range(int(slot.max()) + 1):
            e = loc & (slot == k)
            if not e.any():
                wslot.append(0)
                continue
            dt, st = dl[e] >> 10, sl[e] >> 10
            wslot.append(int(np.abs(dt - st).max()))
            u = np.unique(dt * 64 + st)
            floor += len(np.unique(st))
            pairs += len(u)
            seq = (u % 64).tolist()
            ws += np.bincount(np.bincount(u // 64), minlength=64)[:64]
            for g in ngcs:
                ms[f"src{g}"] += misses(seq, g)
                ms[f"lru{g}"] += misses_lru(seq, g)
                ms[f"opt{g}"] += misses_opt(seq, g)
            if bylane:
                s2 = st[np.argsort(dl[e], kind="stable")]
                seq = s2[np.r_[True, s2[1:] != s2[:-1]]].tolist()
                for g in ngcs:
                    ms[f"lane{g}"] += misses(seq, g)
        ws[0] = 0
        r.update(misses=ms, floor=floor, pairs=pairs, wslot=wslot, wset=ws.tolist())
        for name, sl_ in (("face", slot), ("dir", slot_dir)):
            q = np.lexsort((dl, sl_))
            a, b, lc, de = sl_[q], dl[q], loc[q], (sl - dl)[q]
            cont = np.zeros(N, bool)
            cont[1:] = (a[1:] == a[:-1]) & (b[1:] == b[:-1] + 1) & (b[1:] >> 10 == b[:-1] >> 10) & lc[1:] & lc[:-1] & (de[1:] == de[:-1])
            st0 = np.flatnonzero(~cont)
            rl = np.diff(np.r_[st0, N])[lc[st0]]
            r[f"runs_{name}"] = [int(rl[rl >= m].sum()) for m in (8, 32)]
        hl = np.sort(lane[sender])
        r["halo_runs"] = int((np.diff(hl) != 1).sum()) + 1 if len(hl) else 0
        r["halo_tiles"] = int(len(np.unique(hl >> 10)))
        out["orders"][how] = r
    return out


def pct(h, p):
    c = np.cumsum(h)
    return int(np.searchsorted(c, p / 100 * c[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--own", default="own_cut")
    ap.add_argument("--orders", default="sliced+index,sliced+rcm,sliced+hilbert,cls+index,cls+rcm,cls+hilbert,cls+axis,cls+bfs,rcm,hilbert")
    ap.add_argument("--cores", default="all")
    ap.add_argument("--ngc", default="4,8,14")
    ap.add_argument("--bylane", type=int, default=0, help="1: also walk each slot in destination-lane order (no --tsort)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--json")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print("[%6.1f s]" % (time.time() - t0), *x, flush=True)
    D = np.load(a.mesh)
    own = D[a.own].astype(np.int64)
    P, nC = int(own.max()) + 1, len(own)
    o, nb = D["owner"].astype(np.int64), D["neighbour"].astype(np.int64)
    nI = len(nb)
    b = np.concatenate([np.arange(s, s + z) for t, s, z in zip(D["ptype"], D["pstart"], D["psize"]) if t != "empty"])
    ax, sg = D["ax"].astype(np.int8), D["sg"]
    R = np.concatenate([o[:nI], nb, o[b]]).astype(np.int32)
    O = np.concatenate([nb, o[:nI], np.full(len(b), -1)]).astype(np.int32)
    FID = np.concatenate([np.arange(nI), np.arange(nI), b]).astype(np.int32)
    DC = np.concatenate([2 * ax[:nI] + sg[:nI], 2 * ax[:nI] + ~sg[:nI], 2 * ax[b] + sg[b]]).astype(np.int8)
    del o, nb, ax, sg
    q = np.lexsort((FID, R, own[R]))
    R, O, FID, DC = R[q], O[q], FID[q], DC[q]
    del q
    fc = np.bincount(R, minlength=nC)
    start = np.zeros(P + 1, np.int64)
    np.cumsum(np.bincount(own, weights=fc, minlength=P).astype(np.int64), out=start[1:])
    S = int(fc.max())
    nk = np.array([np.bincount(own[fc > k], minlength=P).max() for k in range(S)])
    L = -(-nk // LANES)
    log("mesh %d cells, %d parts, %d entries, face counts %s" % (nC, P, len(R), dict(zip(*map(lambda v: v.tolist(), np.unique(fc, return_counts=True))))))
    log("slot tiles L_k", L.tolist())
    cores = list(range(P)) if a.cores == "all" else [int(x) for x in a.cores.split(",")]
    orders = a.orders.split(",")
    G.update(own=own, R=R, O=O, FID=FID, DC=DC, C=D["C"], L=L, start=start, ngc=[int(x) for x in a.ngc.split(",")],
             orders=orders, bylane=a.bylane)
    res = []
    with Pool(a.jobs) as pool:
        for r in pool.imap_unordered(core, cores):
            res.append(r)
            if len(res) % 10 == 0 or len(res) == len(cores):
                log("%d of %d cores" % (len(res), len(cores)))
    res.sort(key=lambda r: r["core"])
    nc = len(res)
    N = sum(r["entries"] for r in res)
    summ = {}
    print("order            | band tiles p50 p99 max | lanes p99 | misses/sweep/core by source tile " + " ".join("ngc%d" % g for g in G["ngc"])
          + " | floor | worst core ngc%d | runs>=8 >=32 face | dir | halo runs tiles | slot tiles" % G["ngc"][1])
    for how in orders:
        rs = [r["orders"][how] for r in res]
        h = np.sum([r["band"] for r in rs], 0)
        hl = np.sum([r["band_lanes"] for r in rs], 0)
        m = {k: float(np.mean([r["misses"][k] for r in rs])) for k in rs[0]["misses"]}
        g1 = "src%d" % G["ngc"][1]
        s = dict(band_p50=pct(h, 50), band_p99=pct(h, 99), band_p999=pct(h, 99.9), band_max=int(np.flatnonzero(h).max()),
                 lanes_p99=(pct(hl, 99) + 1) * 128, misses=m, floor=float(np.mean([r["floor"] for r in rs])),
                 pairs=float(np.mean([r["pairs"] for r in rs])),
                 worst=max(r["misses"][g1] for r in rs), wset=[pct(np.sum([r["wset"] for r in rs], 0), q) for q in (50, 99, 100)],
                 band_main=[pct(np.sum([r["band_main"] for r in rs], 0), q) for q in (50, 99, 100)],
                 mask_tiles=float(np.mean([sum(r["mask"]) for r in rs])), band_p99_worst_core=max(pct(np.array(r["band"]), 99) for r in rs),
                 w_per_slot=np.max([r["wslot"] + [0] * (S - len(r["wslot"])) for r in rs], 0).tolist(),
                 runs_face=[sum(r["runs_face"][i] for r in rs) / N for i in (0, 1)],
                 runs_dir=[sum(r["runs_dir"][i] for r in rs) / N for i in (0, 1)],
                 halo_runs=float(np.mean([r["halo_runs"] for r in rs])), halo_tiles=float(np.mean([r["halo_tiles"] for r in rs])),
                 slot_tiles=int(np.sum(np.max([r["ext"] + [0] * (S - len(r["ext"])) for r in rs], 0))))
        summ[how] = s
        print("%-16s | %2d %2d %2d | %6d | %s | %6.0f | %6d | %5.1f %5.1f %% | %5.1f %5.1f %% | %7.0f %5.1f | %d | %s | %s | %s | %.0f" % (
            how, s["band_p50"], s["band_p99"], s["band_max"], s["lanes_p99"],
            " ".join("%7.0f" % m["src%d" % g] for g in G["ngc"]), s["floor"], s["worst"], 100 * s["runs_face"][0],
            100 * s["runs_face"][1], 100 * s["runs_dir"][0], 100 * s["runs_dir"][1], s["halo_runs"], s["halo_tiles"], s["slot_tiles"],
            " ".join("%.0f" % m["lru%d" % g] for g in G["ngc"]), " ".join("%.0f" % m["opt%d" % g] for g in G["ngc"]),
            " ".join(map(str, s["wset"] + s["band_main"])), s["mask_tiles"]), flush=True)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(mesh=a.mesh, own=a.own, P=P, cells=nC, entries=int(N), L=L.tolist(), cores=cores, ngc=G["ngc"],
                           summary=summ, per_core=res), f)
        log("wrote", a.json)


if __name__ == "__main__":
    main()
