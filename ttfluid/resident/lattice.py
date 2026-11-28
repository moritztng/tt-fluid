"""Lattice layout: put each core's cells on a box so that a neighbour gather is one offset load.

A hex mesh is a lattice topologically: walk the internal faces and every cell gets integer coordinates (a face whose
normal points +x leads to the cell one step along x). Cells are split over cores by coordinate bisection, and each
core stores its cells in a box of the same shape on every core, x split into 4 interleaved quarters:

    lane = 4 * (ii + w * (j + H * k)) + m,     i - i0 = m * w + ii

so one step along x is 4 lanes (16 bytes, the unpacker's address unit) and one step along y is 4w lanes, the same on
every core. A gather whose entries are mostly such steps is then an offset load plus a sparse residue (quarter seams,
box edges, other cores, boundary values), see machine.Maps.split.
"""
from collections import deque

import numpy as np

from .machine import rcb


def lattice_coords(mesh):
    """Integer (i, j, k) per cell from the internal faces' dominant normal directions; None if they disagree."""
    nC, nI = mesh.nCells, mesh.nInternalFaces
    o, n, S = mesh.owner[:nI], mesh.neighbour, mesh.Sf[:nI]
    d = np.argmax(np.abs(S), axis=1)
    step = np.zeros((nI, 3), np.int64)
    step[np.arange(nI), d] = np.where(S[np.arange(nI), d] > 0, 1, -1)
    adj = [[] for _ in range(nC)]
    for f in range(nI):
        adj[o[f]].append((n[f], f, 1))
        adj[n[f]].append((o[f], f, -1))
    X = np.full((nC, 3), np.iinfo(np.int64).min)
    for seed in range(nC):
        if X[seed, 0] != np.iinfo(np.int64).min:
            continue
        X[seed] = 0 if seed == 0 else X[:seed].max(0) + 2   # disconnected pieces side by side (never for one case)
        q = deque([seed])
        while q:
            c = q.popleft()
            for nb, f, sgn in adj[c]:
                x = X[c] + sgn * step[f]
                if X[nb, 0] == np.iinfo(np.int64).min:
                    X[nb] = x
                    q.append(nb)
                elif not np.array_equal(X[nb], x):
                    return None
    return X - X.min(0)


def lattice_lanes(X, P, chips=None):
    """owner, lane per cell and the box (w, H, D): uniform shape on every core, x in 4 interleaved quarters. The box
    may span several tiles; a vector's tiles are contiguous, so offsets cross tile boundaries freely."""
    own = rcb(X.astype(np.float64), P, chips)
    lo = np.zeros((P, 3), np.int64)
    ext = np.zeros(3, np.int64)
    for c in range(P):
        Xi = X[own == c]
        lo[c] = Xi.min(0)
        ext = np.maximum(ext, Xi.max(0) - lo[c] + 1)
    w = -(-ext[0] // 4)
    H, D = int(ext[1]), int(ext[2])
    r = X - lo[own]
    m, ii = r[:, 0] // w, r[:, 0] % w
    lane = 4 * (ii + w * (r[:, 1] + H * r[:, 2])) + m
    return own, lane, (int(w), H, D)


def strip_owner(X, P, K):
    """K strips along x of equal cell count, each cut along y (then z) into P / K parts of equal cell count."""
    N = len(X)
    own = np.zeros(N, np.int64)
    o = np.lexsort((X[:, 2], X[:, 1], X[:, 0]))
    for k, st in enumerate(np.array_split(o, K)):
        st = st[np.lexsort((X[st, 0], X[st, 2], X[st, 1]))]
        for j, part in enumerate(np.array_split(st, P // K)):
            own[part] = k * (P // K) + j
    return own


def row_lanes(X, own, P):
    """Lanes with one row width on every core and each core's own row count: x in 4 interleaved quarters of w, so x +- 1
    is 4 lanes and y +- 1 is 4w lanes on every core (both one global shift), while a short core pads only its rows'
    ends, not a box as tall as the tallest core's (lattice_lanes)."""
    lo = np.zeros((P, 3), np.int64)
    ext = np.zeros((P, 3), np.int64)
    for c in range(P):
        Xi = X[own == c]
        lo[c] = Xi.min(0)
        ext[c] = Xi.max(0) - lo[c] + 1
    w = -(-int(ext[:, 0].max()) // 4)
    r = X - lo[own]
    m, ii = r[:, 0] // w, r[:, 0] % w
    return 4 * (ii + w * (r[:, 1] + ext[own, 1] * r[:, 2])) + m


def grid_lanes(X, P, owner=None, neighbour=None):
    """Like lattice_lanes, but the cores own the cells of a gx x gy x gz grid of equal boxes over the lattice's bounding
    box, so the box every core stores is exactly its part (rcb parts of a stepped domain differ in extent and the
    common box is their maximum). The interleaved axis may be any of the three. Picks the grid with the fewest tiles,
    then the fewest residue entries (quarter seams + faces between cores) when the internal faces are given; every
    core must own a cell. Returns owner, lane, (w, H, D) and (grid, axis order)."""
    E = X.max(0) + 1
    best = None
    for gx in range(1, P + 1):
        for gy in range(1, P // gx + 1):
            if P % (gx * gy):
                continue
            g = np.array([gx, gy, P // (gx * gy)])
            b = -(-E // g)
            cid = X // b
            own = (cid[:, 0] * g[1] + cid[:, 1]) * g[2] + cid[:, 2]
            if np.bincount(own, minlength=P).min() == 0:
                continue
            for ax in range(3):
                order = [ax] + [a for a in range(3) if a != ax]
                w = -(-int(b[ax]) // 4)
                T = -(-4 * w * int(b[order[1]]) * int(b[order[2]]) // 1024)
                res = 0
                if owner is not None:
                    r = X - cid * b
                    m = r[:, ax] // w
                    d = np.argmax(np.abs(X[neighbour] - X[owner]), 1)
                    res = int(np.sum(own[owner] != own[neighbour]) +
                              np.sum((d == ax) & (own[owner] == own[neighbour]) & (m[owner] != m[neighbour])))
                key = (T, res)
                if best is None or key < best[0]:
                    best = (key, g, b, order, own)
    _, g, b, order, own = best
    r = X - (X // b) * b
    w = -(-int(b[order[0]]) // 4)
    m, ii = r[:, order[0]] // w, r[:, order[0]] % w
    H, D = int(b[order[1]]), int(b[order[2]])
    lane = 4 * (ii + w * (r[:, order[1]] + H * r[:, order[2]])) + m
    return own, lane, (w, H, D), (tuple(int(x) for x in g), tuple(order))


BIG = 1 << 20
AX = {"x": (0, 1, 2), "y": (1, 2, 0), "z": (2, 0, 1)}     # fastest, middle, slowest


def pieces(n, row, oth, loc, dc, fc):
    """The octree lattice inside one core, from its n cells' faces: entry e is a face of cell row[e] in direction dc[e]
    (2 * axis + sign of the outward normal), loc[e] says the cell behind it, oth[e], is on the core too; fc counts a
    cell's faces. A cell with six faces, one per direction, is regular; regular cells joined by a face form a piece
    of one refinement level, and walking the faces gives every cell of a piece integer coordinates.
    Returns (regular, piece of each cell, xyz, piece sizes, piece dims, entries the coordinates agree with)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    nd = np.zeros((n, 6), np.int64)
    np.add.at(nd, (row, dc), 1)
    reg = (fc == 6) & (nd == 1).all(1)
    e = loc & reg[row] & reg[np.where(loc, oth, 0)]
    a, b, d = row[e], oth[e], dc[e].astype(np.int64)
    nco, comp = connected_components(csr_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)), directed=False)
    step = np.zeros((6, 3), np.int64)
    step[np.arange(6), np.arange(6) // 2] = np.where(np.arange(6) % 2, 1, -1)
    xyz, known = np.zeros((n, 3), np.int64), np.zeros(n, bool)
    known[np.unique(comp, return_index=True)[1]] = True
    A, B, D = a, b, d
    while len(A):
        m = known[A] & ~known[B]
        if not m.any():
            break
        xyz[B[m]] = xyz[A[m]] + step[D[m]]
        known[B[m]] = True
        keep = ~known[B]
        A, B, D = A[keep], B[keep], D[keep]
    lo = np.full((nco, 3), BIG)
    hi = np.full((nco, 3), -BIG)
    for i in range(3):
        np.minimum.at(lo[:, i], comp, xyz[:, i])
        np.maximum.at(hi[:, i], comp, xyz[:, i])
    xyz -= lo[comp]
    good = np.zeros(len(row), bool)
    good[np.flatnonzero(e)] = (xyz[b] - xyz[a] == step[d]).all(1)
    return reg, comp, xyz, np.bincount(comp, minlength=nco), hi - lo + 1, good


def rows8(comp, xyz, npieces):
    """Per piece and axis: the cells in rows of 8 or more along that axis."""
    rows = np.zeros((npieces, 3), np.int64)
    for a0, a1, a2 in AX.values():
        _, inv, cnt = np.unique((comp * 4096 + xyz[:, a1]) * 4096 + xyz[:, a2], return_inverse=True, return_counts=True)
        np.add.at(rows[:, a0], comp, cnt[inv] >= 8)
    return rows


def box_order(comp, xyz, size, tie, fast=None):
    """Position of every cell of a core in lattice order (pieces as returned by `pieces`): pieces by size, a piece's
    cells lexicographic in its coordinates, the cells in no piece behind them in the order of `tie`. fast: the axis
    that runs fastest (x, y, z); None picks per piece the axis with the most cells in rows of 8 or more."""
    crank = np.empty(len(size), np.int64)
    crank[np.argsort(-size, kind="stable")] = np.arange(len(size))
    crank = np.where(size[comp] > 1, crank[comp], len(size))
    pc = xyz[:, AX[fast]] if fast else np.take_along_axis(xyz, np.array(list(AX.values()))[rows8(comp, xyz, len(size)).argmax(1)][comp], 1)
    return np.lexsort((tie, pc[:, 0], pc[:, 1], pc[:, 2], crank)).argsort()
