"""Gather regularity census: how much of a mesh's neighbour gather is `src = lane + shift`, per lane and slot order.

    python3 -m ttfluid.tools.regcensus MESH.npz [--own own_cut] [--orders face/sliced,dir/geo,dir/topo] [--json OUT]

MESH.npz: cell centres C, volumes V, owner, neighbour, the dominant axis `ax` and sign `sg` of every face normal, patch
ranges and types, and one core owner per cell (`own_cut`: min-cut rcb, `own_long`: longest-axis rcb), all from the
polyMesh alone, so the census needs no lowering. An order is SLOTS/LANES:

    slots  face   a row's faces in face-index order (what --layout rcb lowers today)
           dir    by the direction of the outward normal, -x +x -y +y -z +z (Ell(order="dir"))
    lanes  sliced a core's cells by face count, widest first, then cell index (machine.sliced_lanes)
           geo    face count, then refinement level and integer position of the cell centre on that level's grid
           topo   face count, then coordinates propagated through opposite faces of cells that have exactly one face per
                  direction, each connected piece ordered with its longest extent fastest
           box[q][:FILL[:NMIN]]  the pieces of topo cut into boxes filled to FILL (0.9), each laid out as the full
                  lattice of its bounding box with unused lanes at the holes; boxq deals a box out as four interleaved
                  quarters so that its shifts are multiples of 4 lanes

A slot entry is one (cell, slot): local (the neighbour is on the same core), halo, or boundary. For every core, slot and
destination tile the census takes the most common shift `src lane - destination lane` of its local entries (REGULAR),
and the runs of consecutive destination lanes that share one shift (a run never leaves its destination tile). `% 4`
counts only shifts that are a multiple of 4 lanes, what a NoC copy inside L1 or an unpacker read at an offset needs.
"""
import argparse
import json
import time

import numpy as np

LANES = 1024


def entries(D):
    """Every (cell, face) pair of the ELL layout: row cell, the cell across the face (-1: boundary), face order key,
    direction code 2 * axis + (outward normal positive)."""
    o, n = D["owner"].astype(np.int64), D["neighbour"].astype(np.int64)
    nI = len(n)
    b = [np.arange(s, s + z) for t, s, z in zip(D["ptype"], D["pstart"], D["psize"]) if t != "empty"]
    b = np.concatenate(b) if b else np.zeros(0, np.int64)
    ax, sg = D["ax"].astype(np.int64), D["sg"]
    rows = np.concatenate([o[:nI], n, o[b]])
    other = np.concatenate([n, o[:nI], np.full(len(b), -1)])
    fid = np.concatenate([np.arange(nI), np.arange(nI), nI + np.arange(len(b))])
    dc = np.concatenate([2 * ax[:nI] + sg[:nI], 2 * ax[:nI] + ~sg[:nI], 2 * ax[b] + sg[b]])
    return rows, other, fid, dc


def slots(rows, fid, dc, how, nC):
    """Slot of every entry: its rank in the row under the slot order."""
    order = np.lexsort((fid, rows) if how == "face" else (fid, dc, rows))
    start = np.zeros(nC + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=nC), out=start[1:])
    slot = np.empty(len(rows), np.int16)
    slot[order] = np.arange(len(rows)) - start[rows[order]]
    return slot


def lanes_of(own, keys, P):
    """Lane of every cell: its rank inside its core under the keys (most significant first)."""
    order = np.lexsort(tuple(keys[::-1]) + (own,))
    cnt = np.bincount(own, minlength=P)
    lane = np.empty(len(own), np.int64)
    lane[order] = np.arange(len(own)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    return lane


def geo_keys(D):
    """Refinement level from the cell volume and the integer position of the centre on that level's grid."""
    C, h = D["C"], np.cbrt(D["V"])
    L = np.round(np.log2(h.max() / h)).astype(np.int64)
    q = np.floor((C - C.min(0)) / (h.max() / 2.0 ** L)[:, None]).astype(np.int64)
    return [L, q[:, 2], q[:, 1], q[:, 0]]


def topo_pieces(rows, other, dc, own, nC, P, log, rounds=4000):
    """Coordinates through opposite faces. A cell with exactly one face per direction links to the cell across each
    face when that one is such a cell on the same core. Every connected piece gets integer coordinates from a breadth
    first walk (+-1 along the direction crossed). Returns the piece of every cell (-1: none), its coordinates, the
    number of pieces."""
    cnt = np.zeros((nC, 6), np.int8)
    np.add.at(cnt, (rows, dc), 1)
    hex6 = (cnt == 1).all(1)
    del cnt
    nb = np.full((nC, 6), -1, np.int32)
    ok = hex6[rows] & (other >= 0)
    ok[ok] = hex6[other[ok]] & (own[other[ok]] == own[rows[ok]])
    nb[rows[ok], dc[ok]] = other[ok]
    del ok
    comp = np.full(nC, -1, np.int64)
    xyz = np.zeros((nC, 3), np.int32)
    step = np.zeros((6, 3), np.int32)
    for d in range(6):
        step[d, d // 2] = 1 if d & 1 else -1
    todo = hex6.copy()
    ncomp = 0
    first = np.full(P, -1, np.int64)
    for r in range(rounds):
        idx = np.flatnonzero(todo)
        if not len(idx):
            break
        first[:] = -1
        first[own[idx[::-1]]] = idx[::-1]          # one unvisited cell per core
        front = first[first >= 0]
        comp[front] = ncomp + np.arange(len(front))
        ncomp += len(front)
        todo[front] = False
        while len(front):
            nxt, cid, pos = [], [], []
            for d in range(6):
                t = nb[front, d]
                k = t >= 0
                k[k] = todo[t[k]]
                nxt.append(t[k]); cid.append(comp[front[k]]); pos.append(xyz[front[k]] + step[d])
            nxt = np.concatenate(nxt)
            if not len(nxt):
                break
            u, i = np.unique(nxt, return_index=True)     # reached from several cells: the first assignment stays
            comp[u] = np.concatenate(cid)[i]
            xyz[u] = np.concatenate(pos)[i]
            todo[u] = False
            front = u
    log(f"topo: {hex6.mean():.4f} of cells have one face per direction, {ncomp} pieces, "
        f"{int(todo.sum())} such cells left after {rounds} rounds")
    return comp, xyz, ncomp


def topo_keys(rows, other, dc, own, nC, P, log):
    """Lane keys from topo_pieces: larger pieces first, each ordered with its longest extent fastest. Cells in no
    piece keep their index order after the pieces."""
    comp, xyz, ncomp = topo_pieces(rows, other, dc, own, nC, P, log)
    # per piece: extents, axis order (longest fastest), coordinates from the piece's minimum
    inp = comp >= 0
    lo = np.full((ncomp, 3), np.iinfo(np.int32).max, np.int64)
    hi = np.full((ncomp, 3), np.iinfo(np.int32).min, np.int64)
    np.minimum.at(lo, comp[inp], xyz[inp])
    np.maximum.at(hi, comp[inp], xyz[inp])
    size = np.bincount(comp[inp], minlength=ncomp)
    axo = np.argsort(-(hi - lo), axis=1, kind="stable")     # axo[:, 0]: the longest extent
    rel = np.zeros((nC, 3), np.int64)
    rel[inp] = xyz[inp] - lo[comp[inp]]
    fast = np.take_along_axis(rel[inp], axo[comp[inp], 0:1], 1)[:, 0]
    mid = np.take_along_axis(rel[inp], axo[comp[inp], 1:2], 1)[:, 0]
    slow = np.take_along_axis(rel[inp], axo[comp[inp], 2:3], 1)[:, 0]
    big = np.argsort(np.argsort(-size, kind="stable"), kind="stable")     # larger pieces first inside a core
    k0 = np.full(nC, ncomp, np.int64)
    k0[inp] = big[comp[inp]]
    k1, k2, k3 = (np.zeros(nC, np.int64) for _ in range(3))
    k1[inp], k2[inp], k3[inp] = slow, mid, fast
    k3[~inp] = np.arange(nC)[~inp]
    sz = np.sort(size)[::-1]
    log(f"topo: cells in pieces {inp.mean():.4f}; in pieces of >= 4096 cells {size[size >= 4096].sum() / nC:.4f}, "
        f">= 512 {size[size >= 512].sum() / nC:.4f}, >= 64 {size[size >= 64].sum() / nC:.4f}; largest {sz[:5].tolist()}")
    return [k0, k1, k2, k3]


def boxes(X, fill, nmin):
    """Split integer points X (n, 3) into boxes: a set whose tight bounding box is filled to `fill` is a box, any other
    is cut by the plane (any axis) that leaves the two smallest bounding boxes, the median plane of the longest axis
    when no plane shrinks them. Sets under nmin points that do not fill their box are left loose. Returns a list of
    (index array, lo, dims) and the loose indices."""
    out, loose, stack = [], [], [np.arange(len(X))]
    while stack:
        idx = stack.pop()
        x = X[idx]
        lo, hi = x.min(0), x.max(0)
        dims = hi - lo + 1
        vol = int(dims.prod())
        if len(idx) >= fill * vol:
            out.append((idx, lo, dims))
            continue
        if len(idx) < nmin:
            loose.append(idx)
            continue
        best = None
        for a in range(3):
            if dims[a] < 2:
                continue
            o = np.argsort(x[:, a], kind="stable")
            xs = x[o]
            cut = np.flatnonzero(xs[1:, a] != xs[:-1, a]) + 1          # left = xs[:c], right = xs[c:]
            if not len(cut):
                continue
            pmin, pmax = np.minimum.accumulate(xs, 0), np.maximum.accumulate(xs, 0)
            smin, smax = np.minimum.accumulate(xs[::-1], 0)[::-1], np.maximum.accumulate(xs[::-1], 0)[::-1]
            v = (pmax[cut - 1] - pmin[cut - 1] + 1).prod(1) + (smax[cut] - smin[cut] + 1).prod(1)
            k = int(np.argmin(v))
            if best is None or v[k] < best[0]:
                best = (int(v[k]), o, int(cut[k]))
        if best is None or best[0] >= vol:      # a hole inside: halve the longest axis
            a = int(np.argmax(dims))
            o = np.argsort(x[:, a], kind="stable")
            xs = x[o, a]
            cut = np.flatnonzero(xs[1:] != xs[:-1]) + 1
            c = int(cut[np.argmin(np.abs(cut - len(idx) // 2))])
            best = (vol, o, c)
        _, o, c = best
        stack.append(idx[o[:c]])
        stack.append(idx[o[c:]])
    return out, (np.concatenate(loose) if loose else np.zeros(0, np.int64))


def box_lanes(comp, xyz, own, fc, P, log, fill=0.9, nmin=64, quarter=False):
    """Lanes of a padded box layout. Per core: the cells with more than six faces first (widest first), then every
    box of every piece as a full lattice of its bounding box, longest extent fastest, a hole being a lane no cell
    uses, then the other cells. Inside a box every neighbour along an axis is one constant lane shift away.
    quarter: a box's positions are dealt out as four interleaved quarters (lane 4 * (m % q) + m // q), so every shift
    inside a quarter is a multiple of 4 lanes. Returns the lane of every cell and the lanes used per core."""
    nC = len(own)
    lane = np.full(nC, -1, np.int64)
    used = np.zeros(P, np.int64)
    order = np.lexsort((np.arange(nC), -fc, own))
    start = np.r_[0, np.cumsum(np.bincount(own, minlength=P))]
    nbox = vbox = cbox = ndup = 0
    hist = np.zeros(5, np.int64)           # cells in boxes of < 256, < 1024, < 4096, < 16384, more positions
    for c in range(P):
        cells = order[start[c]:start[c + 1]]
        wide = cells[fc[cells] > 6]
        lane[wide] = np.arange(len(wide))
        at = len(wide)
        rest = cells[fc[cells] <= 6]
        inp = rest[comp[rest] >= 0]
        o = np.argsort(comp[inp], kind="stable")
        inp = inp[o]
        cuts = np.flatnonzero(np.r_[True, comp[inp][1:] != comp[inp][:-1], True])
        loose = [rest[comp[rest] < 0]]
        got = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            ids = inp[a:b]
            # a walk that closes around the body can give two cells of one piece the same (i, j, k): only the first
            # keeps its place in a box, or two cells would share a lane
            x = xyz[ids].astype(np.int64)
            x -= x.min(0)
            first = np.unique((x[:, 0] << 42) | (x[:, 1] << 21) | x[:, 2], return_index=True)[1]
            if len(first) < len(ids):
                dup = np.ones(len(ids), bool)
                dup[first] = False
                loose.append(ids[dup])
                ndup += int(dup.sum())
                ids = ids[np.sort(first)]
            bx, ls = boxes(xyz[ids].astype(np.int64), fill, nmin)
            loose.append(ids[ls])
            got += [(ids[i], lo, d) for i, lo, d in bx]
        got.sort(key=lambda g: -int(g[2].prod()))
        for ids, lo, d in got:
            ax = np.argsort(-d, kind="stable")
            r = xyz[ids].astype(np.int64) - lo
            m = r[:, ax[0]] + d[ax[0]] * (r[:, ax[1]] + d[ax[1]] * r[:, ax[2]])
            vol = int(d.prod())
            if quarter and vol >= 16:
                q = -(-vol // 4)
                m = 4 * (m % q) + m // q
                vol = 4 * q
            lane[ids] = at + m
            at += vol
            nbox += 1; vbox += vol; cbox += len(ids)
            hist[np.searchsorted([256, 1024, 4096, 16384], vol, side="right")] += len(ids)
        ls = np.concatenate(loose)
        ls = ls[np.lexsort((ls, -fc[ls]))]
        lane[ls] = at + np.arange(len(ls))
        used[c] = at + len(ls)
    assert (lane >= 0).all() and len(np.unique(own * (1 << 20) + lane)) == nC, "two cells share a lane"
    log(f"boxes (fill {fill}, nmin {nmin}, quarter {quarter}): {nbox / P:.1f} per core, {cbox / nC:.4f} of cells in boxes, "
        f"padding {100 * (vbox - cbox) / nC:.2f} % of cells, {100 * ndup / nC:.2f} % left out as repeated (i, j, k); lanes per core mean {used.mean():.0f} max {used.max()} "
        f"(T {-(-int(used.max()) // LANES)}); cells by box size <256/<1024/<4096/<16384/more: "
        + "/".join(f"{100 * x / nC:.1f}" for x in hist))
    return lane, used


def census(rows, other, slot, own, lane, P, log):
    nC = len(own)
    S = int(slot.max()) + 1
    ec, dl = own[rows], lane[rows]
    loc = other >= 0
    loc[loc] = own[other[loc]] == ec[loc]
    halo = (other >= 0) & ~loc
    N = len(rows)
    T = int(lane.max()) // LANES + 1
    # slot extents: tiles each slot spans on its widest core (what --sliced sizes its registers by)
    ext = np.zeros(S, np.int64)
    np.maximum.at(ext, slot, dl // LANES + 1)
    delta = np.where(loc, lane[np.where(loc, other, 0)] - dl, 0)
    order = np.lexsort((dl, slot, ec))
    ec, dl, sl, lc, de = ec[order], dl[order], slot[order].astype(np.int64), loc[order], delta[order]
    grp = (ec * S + sl) * T + dl // LANES
    out = dict(entries=N, per_cell=N / nC, local=float(loc.mean()), halo=float(halo.mean()),
               boundary=float((other < 0).mean()), T=T, slot_tiles=int(ext.sum()), slot_ext=ext.tolist())
    # the most common shift of each (core, slot, destination tile)
    for name, sel in (("any", lc), ("mod4", lc & (de % 4 == 0))):
        key = grp[sel] * (1 << 18) + (de[sel] + (1 << 17))
        assert np.abs(de[sel]).max(initial=0) < 1 << 17
        u, c = np.unique(key, return_counts=True)
        g = u >> 18
        last = np.r_[g[1:] != g[:-1], True]
        o2 = np.lexsort((c, g))
        out[f"regular_{name}"] = float(c[o2][last].sum() / N)
    # runs: consecutive destination lanes of one (core, slot, tile), local, one shift
    cont = np.zeros(N, bool)
    cont[1:] = (grp[1:] == grp[:-1]) & (dl[1:] == dl[:-1] + 1) & lc[1:] & lc[:-1] & (de[1:] == de[:-1])
    start = np.flatnonzero(~cont)
    rl = np.diff(np.r_[start, N])
    rloc, rde = lc[start], de[start]
    for name, sel in (("any", rloc), ("mod4", rloc & (rde % 4 == 0))):
        r = rl[sel]
        out[f"runs_{name}"] = {str(m): float(r[r >= m].sum() / N) for m in (2, 4, 8, 16, 32, 64, 128, 256)}
        out[f"nruns_{name}"] = {str(m): float((r >= m).sum() / P) for m in (4, 16, 64)}
    r = rl[rloc]
    out["values_per_run"] = float(r.sum() / len(r))
    h = np.bincount(np.minimum(np.floor(np.log2(r)).astype(np.int64), 10), weights=r, minlength=11) / N
    out["run_hist_log2"] = [float(x) for x in h]
    return out


def show(tag, o, log):
    log(f"{tag}: {o['per_cell']:.3f} entries/cell (local {100 * o['local']:.2f} %, halo {100 * o['halo']:.2f} %, boundary "
        f"{100 * o['boundary']:.2f} %); T {o['T']}, slot-tiles {o['slot_tiles']} {o['slot_ext']}")
    log(f"   REGULAR (one shift per core, slot, tile): {100 * o['regular_any']:.2f} % of entries, "
        f"{100 * o['regular_mod4']:.2f} % with shift % 4 == 0")
    for k in ("any", "mod4"):
        log(f"   runs {k}: " + "  ".join(f">={m}: {100 * v:.2f} %" for m, v in o[f"runs_{k}"].items())
            + " | runs per core >=4/16/64: " + "/".join(f"{v:.0f}" for v in o[f"nruns_{k}"].values()))
    log(f"   values per run {o['values_per_run']:.3f}; share of entries by run length 2^k..2^(k+1)-1: "
        + " ".join(f"{100 * x:.1f}" for x in o["run_hist_log2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--own", default="own_cut")
    ap.add_argument("--orders", default="face/sliced,dir/sliced,dir/geo,dir/topo")
    ap.add_argument("--json")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda *x: print(f"[{time.time() - t0:6.1f}s]", *x, flush=True)
    D = np.load(a.mesh)
    own = D[a.own].astype(np.int64)
    P, nC = int(own.max()) + 1, len(own)
    rows, other, fid, dc = entries(D)
    fc = np.bincount(rows, minlength=nC)
    log(f"{nC} cells, {P} cores ({a.own}), {len(rows)} slot entries")
    res, lane_cache, slot_cache, pieces = {}, {}, {}, None
    for od in a.orders.split(","):
        so, lo = od.split("/")
        if so not in slot_cache:
            slot_cache[so] = slots(rows, fid, dc, so, nC)
        if lo not in lane_cache and lo.startswith("box"):      # box[q][:fill[:nmin]]
            if pieces is None:
                pieces = topo_pieces(rows, other, dc, own, nC, P, log)
            f = lo.split(":")
            lane_cache[lo] = box_lanes(pieces[0], pieces[1], own, fc, P, log, fill=float(f[1]) if len(f) > 1 else 0.9,
                                       nmin=int(f[2]) if len(f) > 2 else 64, quarter=f[0] == "boxq")[0]
        elif lo not in lane_cache:
            keys = [] if lo == "sliced" else geo_keys(D) if lo == "geo" else topo_keys(rows, other, dc, own, nC, P, log)
            lane_cache[lo] = lanes_of(own, [-fc] + keys + [np.arange(nC)], P)
        res[od] = census(rows, other, slot_cache[so], own, lane_cache[lo], P, log)
        show(od, res[od], log)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(mesh=a.mesh, own=a.own, cells=nC, cores=P, orders=res), f, indent=1)


if __name__ == "__main__":
    main()
