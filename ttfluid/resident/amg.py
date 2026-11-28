"""Aggregation multigrid for the pressure PCG, written in the ELL primitives the resident program traces.

Setup (host, once per mesh; numpy only): aggregates fixed from the matrix graph (amg/study.py grades the choice),
per coarse level the member table, the coarse ELL graph, and where every fine coefficient lands in the coarse
operator. Per outer iteration the coarse operators are recomputed from the fine coefficients by summation
(unsmoothed aggregation: A_c = P^T A P with P a 0/1 aggregate indicator, OpenFOAM GAMG's model): gathers, masked
multiply-adds, no host. Per PCG iteration one V-cycle: Jacobi-scaled Chebyshev(3) pre and post smoothing, restriction
= sum over members, prolongation = gather from the aggregate, Chebyshev(COARSE_DEG) on the coarsest level.

Every function takes plain arrays: d (n,), off (n, S), nbr (n, S) with padding slots off = 0, nbr = row. Only
indexing by constant integer arrays, elementwise arithmetic and sums over the trailing axis are used, so the same code
runs on numpy (the CPU twin) and on traced arrays.
"""
import os

import numpy as np

COARSE_DEG = 8
DEPTH = int(os.environ.get("CFD_AMG_DEPTH", 99))   # bisection: V-cycle stops (coarsest smoother) at this level
# a capped-away coarse pair: -1 discards its coefficients (the row only gets more diagonally dominant: always SPD),
# anything else sums them into the diagonal (keeps row sums, but a row that loses every pair gets a zero diagonal)
DROP = int(os.environ.get("CFD_AMG_DROP", -1))


class Level:
    """Static structure of one coarse level (host arrays)."""

    def __init__(self, nbr, mask, agg, w=None, cap=None, owner=None, fmax=None, chip=None, sort_deg=False):
        """w (n, S): edge weights of the finer graph (strength); cap: at most this many coarse neighbours per row,
        kept by strength symmetrically (a pair survives only if it is in the top `cap` of BOTH rows); a dropped
        pair's coefficients are discarded (DROP = -1: symmetric, more diagonally dominant, lmax = 2 still bounds
        D^-1 A). Lumping them into the diagonal instead gave NaN at cap <= 16 on motorBike (rows left empty).
        owner (n,), fmax: the core of every finer row and the most lanes one core may hold of a flat coefficient
        space; the coarse coefficients then go into several flat spaces (slot ranges), none wider than fmax.
        chip (n,): the chip of every finer row. A coarse pair across two chips is dropped like a capped one, so the
        coarse halo exchange stays inside a chip (the fine level keeps its couplings across chips).
        sort_deg: relabel coarse aggregates by descending coarse degree so every core's lanes are widest-first
        (slot k is a lane prefix, letting --sliced and tabstream truncate the sparse tail slots)."""
        n, S = nbr.shape
        if w is None:
            w = mask.astype(np.float64)
        nc = int(agg.max()) + 1
        self.n, self.nc = n, nc
        # coarse graph: for every fine slot (i, s) the aggregate pair (agg[i], agg[nbr[i, s]])
        rows = np.repeat(np.arange(n), S)
        cols = nbr.ravel()
        valid = mask.ravel()
        I, J = agg[rows], agg[cols]
        off_pair = valid & (I != J)
        self.cut = 0
        if chip is not None:
            same = np.asarray(chip)[rows] == np.asarray(chip)[cols]
            self.cut = int((off_pair & ~same).sum())
            off_pair &= same
        pairs, inv = np.unique(np.c_[I[off_pair], J[off_pair]], axis=0, return_inverse=True)
        pw = np.zeros(len(pairs))
        np.add.at(pw, inv.ravel(), np.abs(np.asarray(w, np.float64).ravel()[off_pair]))
        if cap is not None and len(pairs):
            o = np.lexsort((-pw, pairs[:, 0]))                       # by row, strongest first
            rank = np.empty(len(pairs), np.int64)
            first = np.r_[0, np.cumsum(np.bincount(pairs[:, 0], minlength=nc))[:-1]]
            rank[o] = np.arange(len(pairs)) - first[pairs[o, 0]]
            top = rank < cap
            key0 = pairs[:, 0].astype(np.int64) * nc + pairs[:, 1]
            keyT = pairs[:, 1].astype(np.int64) * nc + pairs[:, 0]
            srt = np.argsort(key0)
            tpos = srt[np.searchsorted(key0[srt], keyT)]
            keep = top & top[tpos]
            self.dropped = int((~keep).sum())
            pairs, pw = pairs[keep], pw[keep]
        ccnt = np.bincount(pairs[:, 0], minlength=nc)
        self.perm = None
        if sort_deg and nc > 1:
            perm = np.argsort(-ccnt, kind="stable")
            inv_id = np.empty(nc, np.int64)
            inv_id[perm] = np.arange(nc)
            agg = inv_id[agg]
            if len(pairs):
                pairs = inv_id[pairs]
                o = np.lexsort((pairs[:, 1], pairs[:, 0]))
                pairs, pw = pairs[o], pw[o]
            ccnt = ccnt[perm]
            self.perm = perm
        self.ccnt = ccnt
        # members: (nc, m) fine rows, padded with the aggregate's first member and weight 0
        order = np.argsort(agg, kind="stable")
        cnt = np.bincount(agg, minlength=nc)
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        m = int(cnt.max())
        pos = np.arange(n) - start[agg[order]]
        self.M = np.repeat(order[start][:, None], m, axis=1)
        self.M[agg[order], pos] = order
        self.Mw = np.zeros((nc, m))
        self.Mw[agg[order], pos] = 1.0
        self.S = Sc = int(ccnt.max()) if len(pairs) else 1
        cstart = np.r_[0, np.cumsum(ccnt)[:-1]]
        cslot = np.arange(len(pairs)) - cstart[pairs[:, 0]]
        self.nbr = np.tile(np.arange(nc)[:, None], (1, Sc))
        self.nbr[pairs[:, 0], cslot] = pairs[:, 1]
        self.mask = np.zeros((nc, Sc), bool)
        self.mask[pairs[:, 0], cslot] = True
        self.cw = np.zeros((nc, Sc))
        self.cw[pairs[:, 0], cslot] = pw
        # target of fine slot s of member j of aggregate I: coarse slot k, Sc for the diagonal, -1 for none
        key = {(int(a), int(b)): int(k) for (a, b), k in zip(pairs, cslot)}
        self.tgt = np.full((nc, m, S), -1, np.int64)
        for j in range(m):
            rows_j = self.M[:, j]
            live = self.Mw[:, j] > 0
            for s in range(S):
                c = nbr[rows_j, s]
                ok = live & mask[rows_j, s]
                Jc = agg[c]
                same = ok & (Jc == np.arange(nc))
                self.tgt[same, j, s] = Sc
                for I in np.nonzero(ok & ~same)[0]:
                    self.tgt[I, j, s] = key.get((int(I), int(Jc[I])), DROP if DROP == -1 else Sc)
        self.agg = agg
        # Galerkin without masks (every list below is fixed at setup; per outer iteration only gathers and adds):
        # 1. q = d + the slots that stay inside the aggregate (tgt == Sc): one same-space gather per fine slot s,
        #    listing only those rows (a gather's unlisted lanes read 0).
        # 2. coarse diagonal = sum over members j of q[M[:, j]]: one gather per member, live aggregates only.
        # 3. off-diagonals in one flat space F (coefficient (I, k) at cstart[I] + k, on I's core; only the couplings
        #    that exist, so F is as wide as the coarse matrix's off-diagonal count): contributor t of every
        #    coefficient goes into layer t, one multi-source gather per layer (sources off[:, s]), F = sum of layers.
        # 4. slot vector k = F[I * Sc + k]: one gather per coarse slot.
        # Everything stays on one core (aggregates are core-local), so none of it needs a halo.
        rowJ = np.zeros(n, np.int64)
        rowJ[self.M[self.Mw > 0]] = np.nonzero(self.Mw > 0)[1]
        rowI = agg
        self.inner = []                                     # (s, fine rows) for step 1
        for s in range(S):
            r = np.nonzero(self.tgt[rowI, rowJ, s] == Sc)[0]
            if len(r):
                self.inner.append((s, r))
        self.members = [(np.nonzero(self.Mw[:, j] > 0)[0], self.M[self.Mw[:, j] > 0, j]) for j in range(m)]
        I, j, s = np.nonzero((self.tgt >= 0) & (self.tgt < Sc))
        f = cstart[I] + self.tgt[I, j, s]
        o = np.argsort(f, kind="stable")
        f, rows, s = f[o], self.M[I[o], j[o]], s[o]
        first = np.r_[0, np.nonzero(np.diff(f))[0] + 1] if len(f) else np.zeros(0, np.int64)
        rank = np.arange(len(f)) - np.repeat(first, np.diff(np.r_[first, len(f)]))
        # F chunks: slot ranges [fb[c], fb[c + 1]) of the coarse rows, each within fmax lanes on every core, so a flat
        # space is never wider than the cells (the widest vector space sets the machine's tile count and L1 slot size)
        fb = [0]
        if fmax is not None and owner is not None and len(pairs):
            pc = np.asarray(owner)[self.M[pairs[:, 0], 0]]
            cntk = np.zeros((int(pc.max()) + 1, Sc), np.int64)
            np.add.at(cntk, (pc, cslot), 1)
            acc = np.zeros(len(cntk), np.int64)
            for k in range(Sc):
                if k > fb[-1] and (acc + cntk[:, k]).max() > fmax:
                    fb.append(k)
                    acc[:] = 0
                acc += cntk[:, k]
        fb.append(Sc)
        nch = len(fb) - 1
        fchunk = np.searchsorted(fb, cslot, side="right") - 1    # chunk and lane in it of every coefficient
        flane = np.zeros(len(pairs), np.int64)
        for c in range(nch):
            flane[fchunk == c] = np.arange(int((fchunk == c).sum()))
        self.layers = []                                    # per chunk, per layer: [(s, lanes of the chunk, fine rows)]
        for c in range(nch):
            lay = []
            for t in range(int(rank.max()) + 1 if len(f) else 0):
                sel = (rank == t) & (fchunk[f] == c)
                if sel.any():
                    lay.append([(int(q), flane[f[sel][s[sel] == q]], rows[sel][s[sel] == q]) for q in np.unique(s[sel])])
            self.layers.append(lay)
        self.nF = [int((fchunk == c).sum()) for c in range(nch)]
        self.fcells = [pairs[fchunk == c, 0].copy() for c in range(nch)]   # owner aggregate of every lane (attach -> leader cell)
        self.fslot = np.searchsorted(fb, np.arange(Sc), side="right") - 1  # the chunk of slot k
        self.flat = [(np.nonzero(ccnt > k)[0], flane[cstart[ccnt > k] + k]) for k in range(Sc)]   # slot k: (rows, lanes)


def pairwise(nbr, mask, w, part, rounds=3, max_coarse=200, min_ratio=0.7):
    """Aggregates per level for a graph given as ELL (nbr, mask) with edge weights w: `rounds` passes of heavy-edge
    pairwise matching per level (aggregates <= 2**rounds), never across `part` (cells of one core stay together, so
    restriction and prolongation are core-local), leftovers join their strongest neighbour. Stops when a level
    shrinks by less than 1 - min_ratio or reaches max_coarse rows."""
    import scipy.sparse as sp
    n, S = nbr.shape
    r = np.repeat(np.arange(n), S)[mask.ravel()]
    c = nbr.ravel()[mask.ravel()]
    G = sp.csr_matrix((np.abs(w.ravel()[mask.ravel()]), (r, c)), shape=(n, n))
    G = (G + G.T).tocsr() * 0.5
    aggs = []
    part = np.asarray(part)
    while G.shape[0] > max_coarse:
        H = G.tocoo()
        H.data[part[H.row] != part[H.col]] = 0
        Gl = H.tocsr()
        Gl.eliminate_zeros()
        agg = np.arange(G.shape[0])
        for _ in range(rounds):
            m = Gl.shape[0]
            mate = -np.ones(m, np.int64)
            for _ in range(4):
                free = mate < 0
                Hc = Gl.tocoo()
                keep = free[Hc.row] & free[Hc.col] & (Hc.row != Hc.col)
                rr, cc, ww = Hc.row[keep], Hc.col[keep], Hc.data[keep]
                if rr.size == 0:
                    break
                o = np.lexsort((-ww, rr))
                rr, cc = rr[o], cc[o]
                first = np.r_[True, rr[1:] != rr[:-1]]
                pick = -np.ones(m, np.int64)
                pick[rr[first]] = cc[first]
                ii = np.nonzero(pick >= 0)[0]
                mutual = ii[pick[pick[ii]] == ii]
                mate[mutual] = pick[mutual]
            # finish with a greedy heavy-edge pass over what the handshakes left free: on graded meshes the
            # strongest-neighbour picks form monotone chains and handshakes only match their ends (pitz_r4 coarsened
            # 195,600 -> 117,269 per level without this, 1.7x instead of ~8x)
            Hc = Gl.tocoo()
            keep = (mate[Hc.row] < 0) & (mate[Hc.col] < 0) & (Hc.row < Hc.col)
            rr, cc, ww = Hc.row[keep], Hc.col[keep], Hc.data[keep]
            for k in np.argsort(-ww, kind="stable"):
                a, b = rr[k], cc[k]
                if mate[a] < 0 and mate[b] < 0:
                    mate[a], mate[b] = b, a
            lead = np.where(mate >= 0, np.minimum(np.arange(m), mate), np.arange(m))
            Hc = Gl.tocoo()
            single = mate < 0
            keep = single[Hc.row] & (Hc.row != Hc.col)
            rr, cc, ww = Hc.row[keep], Hc.col[keep], Hc.data[keep]
            if rr.size:
                o = np.lexsort((-ww, rr))
                rr, cc = rr[o], cc[o]
                first = np.r_[True, rr[1:] != rr[:-1]]
                lead[rr[first]] = lead[cc[first]]
            _, coarse = np.unique(lead, return_inverse=True)
            Pm = sp.csr_matrix((np.ones(m), (np.arange(m), coarse)), shape=(m, coarse.max() + 1))
            Gl = (Pm.T @ Gl @ Pm).tocsr()
            agg = coarse[agg]
        nc = agg.max() + 1
        if nc > min_ratio * G.shape[0]:
            break
        P = sp.csr_matrix((np.ones(len(agg)), (np.arange(len(agg)), agg)), shape=(len(agg), nc))
        G = (P.T @ G @ P).tocsr()
        G.setdiag(0)
        G.eliminate_zeros()
        cp = np.zeros(nc, np.int64)
        cp[agg] = part
        part = cp
        aggs.append(agg)
    return aggs


def handshake(A, core, size, cap, q=4.0, rounds=40):
    """One matching pass inside each core. An edge is a candidate while both ends are free and the pair would hold at
    most cap cells. Every node takes its candidate of highest priority and mutual choices pair. The priority is the
    coupling in steps of 2^(1/q) with a hash of the edge inside a step: with the bare coupling, a mesh whose
    coefficients grow smoothly in one direction pairs one cell per round along each chain and the pass stalls.
    An edge to a node with no other candidate comes first: such a node (a cell on a ragged core boundary) otherwise
    loses every handshake and is carried down the levels as a row of its own.
    Returns the aggregate of every node and their count."""
    n = A.shape[0]
    i = np.repeat(np.arange(n, dtype=np.int32), np.diff(A.indptr))
    j = A.indices
    keep = (A.data < 0) & (core[i] == core[j]) & (size[i] + size[j] <= cap)
    i, j, w = i[keep], j[keep], -A.data[keep].astype(np.float64)
    lo, hi = np.minimum(i, j).astype(np.uint64), np.maximum(i, j).astype(np.uint64)
    h = (lo * np.uint64(0x9E3779B97F4A7C15)) ^ (hi * np.uint64(0xC2B2AE3D27D4EB4F))
    h = ((h ^ (h >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9)) >> np.uint64(40)
    w = np.floor(q * np.log2(w)) + h.astype(np.float64) / float(1 << 24)
    mate = np.full(n, -1, np.int64)
    for _ in range(rounds):
        if not len(i):
            break
        start = np.flatnonzero(np.r_[True, i[1:] != i[:-1]])           # the entries stay grouped by row
        count = np.diff(np.r_[start, len(i)])
        last = np.zeros(n, bool)                                       # nodes with one candidate left: taken first,
        last[i[start[count == 1]]] = True                              # or their only partner pairs elsewhere
        v = w + 1e6 * (last[i] | last[j])
        hit = np.flatnonzero(v == np.repeat(np.maximum.reduceat(v, start), count))
        hit = hit[np.r_[True, i[hit][1:] != i[hit][:-1]]]              # one per row
        bi, bj = i[hit], j[hit]
        best = np.full(n, -1, np.int64)
        best[bi] = bj
        mutual = bi[best[bj] == bi]
        if 2000 * len(mutual) < len(bi):                               # the tail: chains that pair one cell per round
            break
        mate[mutual] = best[mutual]
        free = (mate[i] < 0) & (mate[j] < 0)
        i, j, w = i[free], j[free], w[free]
    head = np.where((mate < 0) | (np.arange(n) < mate), np.arange(n), mate)
    ids, agg = np.unique(head, return_inverse=True)
    return agg.astype(np.int64), len(ids)


def csr_galerkin(A, agg, nc):
    """P^T A P for the piecewise-constant P of agg."""
    n = A.shape[0]
    i = agg[np.repeat(np.arange(n), np.diff(A.indptr))]
    import scipy.sparse as sp
    Ac = sp.csr_matrix((A.data, (i, agg[A.indices])), shape=(nc, nc))
    Ac.sum_duplicates()
    return Ac


def coarsen(A, core, passes, extra=2, q=4.0):
    """One level for the SPD matrix A (CSR): `passes` passes of handshake, so aggregates of at most 2**passes rows
    inside a core, and `extra` more that merge what was left over. Returns the aggregate of every row, the Galerkin
    matrix by summation and the core of every aggregate."""
    agg, Ac, cc, size = np.arange(A.shape[0]), A, core, np.ones(A.shape[0], np.int64)
    for _ in range(passes + extra):
        a, nc = handshake(Ac, cc, size, 1 << passes, q)
        if nc == len(a):
            break
        first = np.zeros(nc, np.int64)
        first[a] = np.arange(len(a))
        Ac, cc, agg, size = csr_galerkin(Ac, a, nc), cc[first], a[agg], np.bincount(a, size, nc).astype(np.int64)
    return agg, Ac, cc


def matched(nbr, mask, w, part, passes=3, max_coarse=4000, extra=2, q=4.0, min_ratio=2.0, depth=None):
    """Aggregates per level like `pairwise` returns them, by `coarsen` on the graph Laplacian of the ELL weights w:
    never across `part`, levels until max_coarse rows are left, a level shrinks by less than min_ratio or `depth`
    levels exist. tools/mgmodel.py runs the same matching on the captured p systems."""
    import scipy.sparse as sp
    n, S = nbr.shape
    k = np.asarray(mask, bool).ravel()
    r = np.repeat(np.arange(n), S)[k]
    W = np.abs(np.asarray(w, np.float64).ravel()[k])
    A = sp.csr_matrix((np.r_[np.bincount(r, W, n) + 1e-12 * W.max(), -W],
                       (np.r_[np.arange(n), r], np.r_[np.arange(n), np.asarray(nbr).ravel()[k]])), shape=(n, n))
    A.sum_duplicates()
    A.sort_indices()
    core, aggs = np.asarray(part), []
    while A.shape[0] > max_coarse and (depth is None or len(aggs) < depth):
        agg, Ac, cc = coarsen(A, core, passes, extra, q)
        if Ac.shape[0] * min_ratio > A.shape[0]:
            break
        aggs.append(agg)
        A, core = Ac, cc
    return aggs


CAP = int(os.environ.get("CFD_AMG_CAP", 24)) or None    # coarse-row width cap (0: exact Galerkin); motorBike p:
# cap none/24/16/12/8 -> PCG 11/11/13/16/21 it, coarse widths sum 445/144/96/72/48 (amg/cap_study.py)


class Levels(list):
    """The levels of one hierarchy and the cycle Ops runs on them (shape None: the V(3,3) cycle)."""
    shape = None


def build(nbr, mask, aggs, w=None, cap=CAP, shape=None, owner=None, chip_cores=None):
    """aggs: one fine->coarse map per level (level l's map indexes level l-1's rows); w: fine edge weights (n, S),
    cap: coarse ELL width limit (Level), shape: the cycle Ops runs (Ops.cycle), owner: the core of every fine row
    (the flat coefficient spaces then stay within the fullest core's cells, Level), chip_cores: cores per chip (the
    coarse levels are then cut at the chip boundary, Level)."""
    levels = Levels()
    levels.shape = shape
    fmax = int(np.bincount(np.asarray(owner)).max()) if owner is not None else None
    prev_perm = None
    for agg in aggs:
        agg = np.asarray(agg)
        if prev_perm is not None:
            agg = agg[prev_perm]
        chip = np.asarray(owner) // chip_cores if owner is not None and chip_cores else None
        L = Level(nbr, mask, agg, w, cap, owner, fmax, chip, sort_deg=(owner is not None))
        levels.append(L)
        nbr, mask, w = L.nbr, L.mask, L.cw
        prev_perm = L.perm
        if owner is not None:
            owner = np.asarray(owner)[L.M[:, 0]]
    return levels


# ---- per outer iteration -------------------------------------------------------------------------------------------
def _traced(x):
    return hasattr(x, "tr") and hasattr(x, "space")


def attach(tr, levels, cells, stack=False, owner=None):
    """Make `levels` traceable on tracer `tr`: one Space per coarse level (an aggregate lives on its leader's core,
    the leader being its first member's fine cell), every constant index array registered with its result space.
    stack (owner: the core of every fine cell): the coarse SpMV and the member sum of every level run on stacked
    spaces (sa._chunks: slot k of row I at lane k * w + lane(I)), so each is one gather and one block sum
    (trace.Tracer.bsum) instead of one gather and one block per slot, and the coarse coefficients are stacked
    vectors, not one register per slot."""
    leader = np.arange(cells.n)
    fine = cells
    from .trace import Space
    for l, L in enumerate(levels):
        leader = leader[L.M[:, 0]]
        L.space = Space(f"amg{l + 1}", L.nc, cells=leader.copy())
        if getattr(L, "perm", None) is not None:
            L.space.slot_counts = L.ccnt
        L.leader, L.cells = leader.copy(), cells
        L.fine_space = fine
        L.fspaces = [Space(f"amg{l + 1}f" + (str(c) if len(L.nF) > 1 else ""), n, cells=leader[fc])
                     for c, (n, fc) in enumerate(zip(L.nF, L.fcells))]
        L.gchunks = L.mchunks = None
        if stack:
            from .sa import _chunks
            sI, sk = np.nonzero(L.mask)
            L.gchunks, L.C = _chunks(f"amg{l + 1}g", L.space, leader, np.asarray(owner), sI, sk, L.nbr[sI, sk])
            mI, mk = np.nonzero(L.Mw > 0)
            L.mchunks, _ = _chunks(f"amg{l + 1}m", L.space, leader, np.asarray(owner), mI, mk, L.M[mI, mk])
        L.Mcols = [np.ascontiguousarray(L.M[:, j]) for j in range(L.M.shape[1])]
        for a in [L.M, L.nbr] + L.Mcols:
            tr.index_space[id(a)] = L.space
        tr.index_space[id(L.agg)] = fine
        fine = L.space
    return levels


def galerkin(L, d, off, dtype):
    """Coarse (d, off) of level L from the next finer level's (d, off) by summation over the fixed aggregates, as
    gathers and adds only (Level.__init__ lists them)."""
    tr = d.tr if _traced(d) else None
    if tr is not None:
        from .trace import T, _zero_vec

    def gather(space, n, parts):
        """parts: [(source, dst lanes, src lanes)], at most 8 sources per gather: the DRAM reader patches a DRAM-homed
        destination with 8 part cursors (vm_ncrisc.cpp); an L1 gather could take 15."""
        if tr is None:
            g = np.zeros(n, dtype)
            for src, dst, pos in parts:
                g[dst] = src[pos]
            return g
        out = None
        for c in range(0, len(parts), 8):
            g = T(tr, space, np.array(tr.gather(space, [(src.c[()], dst, pos) for src, dst, pos in parts[c:c + 8]]),
                                      object))
            out = g if out is None else out + g
        return out

    def total(xs):
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out

    def col(s, r):
        """slot s of the finer level's coefficients at its rows r, as a gather source: (vector, lanes)"""
        if isinstance(off, Stacked):
            return off[s // off.C], (s % off.C) * off.n + r
        return off[:, s], r

    fine = getattr(L, "fine_space", None)
    q = total([d] + [gather(fine, L.n, [col(s, r)[:1] + (r, col(s, r)[1])]) for s, r in L.inner])
    dc = total([gather(getattr(L, "space", None), L.nc, [(q, live, rows)]) for live, rows in L.members])
    F = [total([gather(sp_, n, [col(s, r)[:1] + (f, col(s, r)[1]) for s, f, r in layer]) for layer in lay]) if lay
         else None for sp_, n, lay in zip(getattr(L, "fspaces", [None] * len(L.nF)), L.nF, L.layers)]
    if tr is not None and getattr(L, "gchunks", None) is not None:
        # stacked: per chunk of C slots one gather, a part per flat space its slots read (slot k of row I at
        # (k - k0) * nc + I, sa._chunks)
        out = Stacked()
        out.C, out.n = L.C, L.nc
        for j, (g, _, _, _) in enumerate(L.gchunks):
            byf = {}
            for k in range(j * L.C, min((j + 1) * L.C, L.S)):
                rows, lanes = L.flat[k]
                if F[L.fslot[k]] is not None and len(rows):
                    byf.setdefault(int(L.fslot[k]), []).append(((k - j * L.C) * L.nc + rows, lanes))
            parts = [(F[c], np.concatenate([a for a, _ in v]), np.concatenate([b for _, b in v])) for c, v in
                     sorted(byf.items())]
            out.append(gather(g, g.n, parts) if parts else T(tr, g, np.array(_zero_vec(tr, g), object)))
        return dc, out
    cols = [gather(getattr(L, "space", None), L.nc, [(F[L.fslot[k]], *L.flat[k])]) if F[L.fslot[k]] is not None
            else (dc * 0) for k in range(L.S)]
    if tr is not None:
        return dc, T(tr, L.space, np.array([c.c[()] for c in cols], object))
    return dc, np.stack(cols, axis=1)


class Stacked(list):
    """A level's coefficients as stacked vectors: chunk j holds slots [j C, (j + 1) C), slot k of row I at entry
    (k - j C) * n + I of its space (attach(stack=True))."""
    C = n = None


def _ssum(x, chunks, base, weights=None, init=None):
    """Sum over the slots of a stacked map (traced only): per chunk one gather of x and one block sum, of weight *
    gathered if the chunk has a weight vector. A gather's unlisted lanes read 0, so a slot a row does not have adds
    nothing."""
    from .trace import T
    y = init
    for j, (g, dst, pos, _) in enumerate(chunks):
        xg = T(x.tr, g, np.array(x.tr.gather(g, [(x.c[()], dst, pos)]), object))
        if weights is not None:
            xg = xg * weights[j]
        t = xg if g is base else T(x.tr, base, np.array(x.tr.bsum(xg.c[()]), object))
        y = t if y is None else y + t
        if weights is not None and g is base and init is not None and getattr(base, "slot_counts", None) is not None:
            x.tr.slotfold[y.c[()]] = (base.name, j)
    return y


def gershgorin(d, off):
    """Upper bound of the spectrum of D^-1 A: max row sum of |D^-1 A| (one reduction)."""
    return ((np.abs(d) + np.abs(off).sum(axis=1)) / np.abs(d)).max()   # D^-1 A has a positive diagonal either sign


class Ops:
    """The per-outer-iteration operator set of every level."""

    def __init__(self, levels, d, off, nbr, dtype=np.float32, shape=None):
        """shape: None for the V(3,3) cycle with Chebyshev(COARSE_DEG) on the last level, or the cycle of
        tools/mgmodel.py as a dict: fine / coarse = "add" (smoother and coarse correction from the same residual),
        "post" or "sym"; k / ck smoothing steps on [lmax / a, lmax] / [lmax / ca, lmax]; kc steps on [lmax / ac,
        lmax] on the last level; w weights every coarse correction, g visits per coarse level from level gf down."""
        self.dtype = dtype
        self.shape = shape if shape is not None else getattr(levels, "shape", None)
        self.A = [(d, off, nbr)]
        for L in levels:
            d, off = galerkin(L, d, off, dtype)
            self.A.append((d, off, L.nbr))
        self.levels = levels
        self.cheb = []
        for d, off, _ in self.A:
            # OpenFOAM's p matrix is weakly diagonally dominant (diag = -sum off, plus boundary terms) and Galerkin
            # by summation keeps that, so the Gershgorin bound of D^-1 A is <= 2: a constant, no max-reduction (the
            # machine lowers sum reductions only)
            lmax = 2.0
            a, b = lmax / 30.0, lmax
            self.cheb.append(((b + a) / 2, (b - a) / 2, 1.0 / d))

    def spmv(self, l, x):
        d, off, nbr = self.A[l]
        if isinstance(off, Stacked):                # one gather and one block sum per chunk of slots
            L = self.levels[l - 1]
            return _ssum(x, L.gchunks, L.space, off, init=d * x)
        return d * x + (off * x[nbr]).sum(axis=1)   # a coarse level reads other cores through its own halo plan

    def restrict(self, l, res):
        """Residual of level l summed over the members of level l + 1's aggregates."""
        L = self.levels[l]
        if _traced(res) and getattr(L, "mchunks", None) is not None:
            return _ssum(res, L.mchunks, L.space)
        return (L.Mw.astype(self.dtype) * res[L.M]).sum(axis=1)

    def smooth(self, l, b, x, deg, ratio=None):
        th, de, dinv = self.cheb[l]
        if ratio is not None:                       # [lmax / ratio, lmax] with lmax = 2
            th, de = 1.0 + 1.0 / ratio, 1.0 - 1.0 / ratio
        r = b if x is None else b - self.spmv(l, x)
        dd = dinv * r / th
        s1 = th / de
        rho = 1 / s1
        x = dd if x is None else x + dd
        for _ in range(deg - 1):
            r = r - self.spmv(l, dd)
            rn = 1 / (2 * s1 - rho)
            dd = (rn * rho) * dd + (2 * rn / de) * dinv * r
            x = x + dd
            rho = rn
        return x

    def cycle(self, r, l=0):
        """The shaped cycle: the correction of level l for the residual r."""
        c = self.shape
        if l == len(self.levels):
            return self.smooth(l, r, None, c["kc"], c["ac"])
        L = self.levels[l]
        if l == 0:
            mode, k, a = c["fine"], c["k"], c["a"]
        elif l == 1 and "l1" in c:
            mode, k, a = c["l1"], c.get("k1", c["ck"]), c.get("a1", c["ca"])
        else:
            mode, k, a = c["coarse"], c["ck"], c["ca"]

        def corr(res):
            rc = self.restrict(l, res)
            lc = l + 1
            if c.get("ktype") == "amli" and c.get("kf", 1) <= lc <= c.get("kmax", len(self.levels) - 1):
                r1, r2 = self.dtype(c.get("r1", 1.15)), self.dtype(c.get("r2", 2.4))
                e1 = r1 * self.cycle(rc, lc)
                e = e1 + r2 * self.cycle(rc - self.spmv(lc, e1), lc)
            else:
                e = self.cycle(rc, lc)
                if c.get("gf", 1) <= lc < len(self.levels):      # W from level gf down: g visits, each on the residual
                    for _ in range(c.get("g", 1) - 1):
                        e = e + self.cycle(rc - self.spmv(lc, e), lc)
            w = c.get("wf", c.get("w", 1.0)) if l == 0 else c.get("w", 1.0)   # piecewise-constant aggregates under-correct
            return (e if w == 1.0 else self.dtype(w) * e)[L.agg]
        if mode == "add":
            return self.smooth(l, r, None, k, a) + corr(r)
        if mode == "post":
            return self.smooth(l, r, corr(r), k, a)
        assert mode == "sym", mode
        z = self.smooth(l, r, None, k, a)
        return self.smooth(l, r, z + corr(r - self.spmv(l, z)), k, a)

    def vcycle(self, r, l=0):
        if self.shape is not None:
            return self.cycle(r)
        if l == min(len(self.levels), DEPTH):
            return self.smooth(l, r, None, COARSE_DEG)
        L = self.levels[l]
        x = self.smooth(l, r, None, 3)
        res = r - self.spmv(l, x)
        rc = self.restrict(l, res)
        ec = self.vcycle(rc, l + 1)
        x2 = x + ec[L.agg]
        out = self.smooth(l, r, x2, 3)
        if getattr(self, "steps", None) is not None and l == 0:    # debugging: the probe stores each step
            self.steps.update(x1=x, res=res, rc=rc, ec=ec, x2=x2, out=out)
        return out
