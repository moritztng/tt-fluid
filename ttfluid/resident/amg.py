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

    def __init__(self, nbr, mask, agg, w=None, cap=None):
        """w (n, S): edge weights of the finer graph (strength); cap: at most this many coarse neighbours per row,
        kept by strength symmetrically (a pair survives only if it is in the top `cap` of BOTH rows); a dropped
        pair's coefficients are discarded (DROP = -1: symmetric, more diagonally dominant, lmax = 2 still bounds
        D^-1 A). Lumping them into the diagonal instead gave NaN at cap <= 16 on motorBike (rows left empty)."""
        n, S = nbr.shape
        if w is None:
            w = mask.astype(np.float64)
        nc = int(agg.max()) + 1
        self.n, self.nc = n, nc
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
        # coarse graph: for every fine slot (i, s) the aggregate pair (agg[i], agg[nbr[i, s]])
        rows = np.repeat(np.arange(n), S)
        cols = nbr.ravel()
        valid = mask.ravel()
        I, J = agg[rows], agg[cols]
        off_pair = valid & (I != J)
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
        self.layers = []                                    # per layer: [(s, F lanes, fine rows)]
        for t in range(int(rank.max()) + 1 if len(f) else 0):
            sel = rank == t
            self.layers.append([(int(q), f[sel][s[sel] == q], rows[sel][s[sel] == q]) for q in np.unique(s[sel])])
        self.nF = len(pairs)
        self.fcells = pairs[:, 0].copy()                    # owner aggregate of every F lane (attach -> leader cell)
        self.flat = [(np.nonzero(ccnt > k)[0], cstart[ccnt > k] + k) for k in range(Sc)]   # slot k: (rows, F lanes)


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


CAP = int(os.environ.get("CFD_AMG_CAP", 24)) or None    # coarse-row width cap (0: exact Galerkin); motorBike p:
# cap none/24/16/12/8 -> PCG 11/11/13/16/21 it, coarse widths sum 445/144/96/72/48 (amg/cap_study.py)


def build(nbr, mask, aggs, w=None, cap=CAP):
    """aggs: one fine->coarse map per level (level l's map indexes level l-1's rows); w: fine edge weights (n, S),
    cap: coarse ELL width limit (Level)."""
    levels = []
    for agg in aggs:
        L = Level(nbr, mask, np.asarray(agg), w, cap)
        levels.append(L)
        nbr, mask, w = L.nbr, L.mask, L.cw
    return levels


# ---- per outer iteration -------------------------------------------------------------------------------------------
def _traced(x):
    return hasattr(x, "tr") and hasattr(x, "space")


def attach(tr, levels, cells):
    """Make `levels` traceable on tracer `tr`: one Space per coarse level (an aggregate lives on its leader's core,
    the leader being its first member's fine cell), every constant index array registered with its result space."""
    leader = np.arange(cells.n)
    fine = cells
    from .trace import Space
    for l, L in enumerate(levels):
        leader = leader[L.M[:, 0]]
        L.space = Space(f"amg{l + 1}", L.nc, cells=leader.copy())
        L.leader, L.cells = leader.copy(), cells
        L.fine_space = fine
        L.fspace = Space(f"amg{l + 1}f", L.nF, cells=leader[L.fcells])
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
        from .trace import T

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

    fine = getattr(L, "fine_space", None)
    q = total([d] + [gather(fine, L.n, [(off[:, s], r, r)]) for s, r in L.inner])
    dc = total([gather(getattr(L, "space", None), L.nc, [(q, live, rows)]) for live, rows in L.members])
    F = total([gather(getattr(L, "fspace", None), L.nF, [(off[:, s], f, r) for s, f, r in layer])
               for layer in L.layers]) if L.layers else None
    cols = [gather(getattr(L, "space", None), L.nc, [(F, *L.flat[k])]) if F is not None
            else (dc * 0) for k in range(L.S)]
    if tr is not None:
        return dc, T(tr, L.space, np.array([c.c[()] for c in cols], object))
    return dc, np.stack(cols, axis=1)


def gershgorin(d, off):
    """Upper bound of the spectrum of D^-1 A: max row sum of |D^-1 A| (one reduction)."""
    return ((np.abs(d) + np.abs(off).sum(axis=1)) / np.abs(d)).max()   # D^-1 A has a positive diagonal either sign


class Ops:
    """The per-outer-iteration operator set of every level."""

    def __init__(self, levels, d, off, nbr, dtype=np.float32):
        self.dtype = dtype
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
        return d * x + (off * x[nbr]).sum(axis=1)   # a coarse level reads other cores through its own halo plan

    def smooth(self, l, b, x, deg):
        th, de, dinv = self.cheb[l]
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

    def vcycle(self, r, l=0):
        if l == min(len(self.levels), DEPTH):
            return self.smooth(l, r, None, COARSE_DEG)
        L = self.levels[l]
        x = self.smooth(l, r, None, 3)
        res = r - self.spmv(l, x)
        rc = (L.Mw.astype(self.dtype) * res[L.M]).sum(axis=1)
        ec = self.vcycle(rc, l + 1)
        x2 = x + ec[L.agg]
        out = self.smooth(l, r, x2, 3)
        if getattr(self, "steps", None) is not None and l == 0:    # debugging: the probe stores each step
            self.steps.update(x1=x, res=res, rc=rc, ec=ec, x2=x2, out=out)
        return out
