"""Smoothed aggregation for the pressure PCG with the prolongator frozen at setup (CPU twin; numpy + scipy, no pyamg).

Setup (host, once per mesh, from one p matrix): per level, aggregates by standard aggregation (a root and its
neighbours, then leftovers join a neighbour's aggregate, then the rest form their own) on the level's graph with every
edge between two cores removed, so an aggregate never spans cores; the prolongator P = S T with T the 0/1 aggregate
basis and S = I - w D^-1 A (w = 4/3 / rho(D^-1 A)), both frozen; the coarse graph = the pattern of P^T A P.

Per outer iteration only the coefficients move, and with P frozen the Galerkin product is LINEAR in them with fixed
weights: AP = G1 [d, off] (one value per nonzero of A P), then A_c = G2 AP. Each G is a fixed sparse matrix; on device
it becomes layered gathers times constant weights. Per V-cycle: P e = S (e_c[agg]) and P^T r = T^T (S^T r), so
prolongation and restriction are one SpMV with frozen coefficients on the level's own graph plus the aggregate
gather / member sum the pairwise build already has. Chebyshev bounds are frozen at setup (no max-reduction on device).
"""
import functools

import numpy as np
import scipy.sparse as sp

COARSE_DEG = 8

# Per-chip (SPMD) setup hooks, identity for a whole-mesh setup. lower/spmd.py sets them so each chip's process sets up
# its own chip's hierarchy (chip-local SA: nothing below the fine level couples two chips) and takes every choice that
# shapes the trace over the whole mesh: OWN(part) -> the rows this process owns (part: the rows' cores), GMAX / GSUM a
# max / sum over the chips, GUNION a sorted union of hashable items over the chips.
OWN = None
GMAX = lambda v: v
GSUM = lambda v: v
GUNION = lambda xs: sorted(set(xs))


def _own(part, n):
    return np.ones(n, bool) if OWN is None or part is None else np.asarray(OWN(np.asarray(part)), bool)


def ell(A):
    """CSR -> (d, off, nbr) with padding slots pointing at the row, coefficient 0."""
    A = A.tocsr()
    n = A.shape[0]
    B = A.tolil()
    B.setdiag(0)
    B = B.tocsr()
    B.eliminate_zeros()
    cnt = np.diff(B.indptr)
    S = max(int(cnt.max()), 1)
    nbr = np.tile(np.arange(n)[:, None], (1, S))
    off = np.zeros((n, S))
    rows = np.repeat(np.arange(n), cnt)
    slot = np.arange(B.nnz) - B.indptr[rows]
    nbr[rows, slot] = B.indices
    off[rows, slot] = B.data
    return A.diagonal().copy(), off, nbr


THETA = 0.0   # strength threshold: a_ij is strong if |a_ij| >= THETA sqrt(|a_ii a_jj|); 0 = every coupling strong


def aggregate(A, part=None, theta=None):
    """Standard aggregation on A's strong off-diagonal graph (theta 0: all couplings), edges across `part` removed."""
    A = A.tocsr()
    n = A.shape[0]
    ip, ix = A.indptr, A.indices
    rows = np.repeat(np.arange(n), np.diff(ip))
    keep = ix != rows
    theta = THETA if theta is None else theta
    if theta > 0:
        dg = np.abs(A.diagonal())
        keep &= np.abs(A.data) >= theta * np.sqrt(dg[rows] * dg[ix])
    if part is not None:
        keep &= part[ix] == np.repeat(part, np.diff(ip))
    G = sp.csr_matrix((np.ones(keep.sum()), (np.repeat(np.arange(n), np.diff(ip))[keep], ix[keep])), shape=(n, n))
    ip, ix = G.indptr, G.indices
    agg = np.full(n, -1, np.int64)
    na = 0
    for i in range(n):                                   # pass 1: free root with every neighbour free
        nb = ix[ip[i]:ip[i + 1]]
        if agg[i] < 0 and (agg[nb] < 0).all():
            agg[i] = na
            agg[nb] = na
            na += 1
    for i in np.nonzero(agg < 0)[0]:                     # pass 2: join an aggregated neighbour's aggregate
        nb = ix[ip[i]:ip[i + 1]]
        a = agg[nb]
        a = a[a >= 0]
        if len(a):
            agg[i] = a[0]
    for i in np.nonzero(agg < 0)[0]:                     # pass 3: the rest with their free neighbours
        if agg[i] >= 0:
            continue
        nb = ix[ip[i]:ip[i + 1]]
        agg[i] = na
        agg[nb[agg[nb] < 0]] = na
        na += 1
    return agg


def spectral_radius(M, its=30, seed=0):
    x = np.random.default_rng(seed).standard_normal(M.shape[0])
    lam = 0.0
    for _ in range(its):
        y = M @ x
        lam = np.linalg.norm(y) / np.linalg.norm(x)
        x = y / np.linalg.norm(y)
    return lam


def _keys(M):
    """Sorted (row * ncols + col) of a CSR pattern with sorted indices."""
    M.sort_indices()
    return np.repeat(np.arange(M.shape[0]), np.diff(M.indptr)).astype(np.int64) * M.shape[1] + M.indices


def layers(G, source, own=None):
    """A fixed sparse map G (lanes x inputs) as layers: layer t takes the t-th term of every lane that has one.
    source(cols) -> (input vector id, lane in it). Per layer: [(id, dst lanes, src lanes)] and its weight vector.
    own: the lanes this process sets up (SPMD); the layers and their parts are the union over the chips (GUNION)."""
    G = G.tocsr()
    nl = G.shape[0]
    if own is not None and not own.all():
        G = (sp.diags(own.astype(float)) @ G).tocsr()
        G.eliminate_zeros()
    G.sort_indices()                                     # (a product's indices come unsorted: the layers follow column order)
    cnt = np.diff(G.indptr)
    rank = np.arange(G.nnz) - np.repeat(G.indptr[:-1], cnt)
    lane = np.repeat(np.arange(nl), cnt)
    sid, pos = source(G.indices)
    o = np.lexsort((sid, rank))                          # by layer, then by input vector: every part a slice
    rank, sid, pos, lane, data = rank[o], sid[o], pos[o], lane[o], G.data[o]
    cut = np.r_[0, np.nonzero((np.diff(rank) != 0) | (np.diff(sid) != 0))[0] + 1, len(o)]
    got = {(int(rank[a]), int(sid[a])): (a, b) for a, b in zip(cut[:-1], cut[1:]) if b > a}
    out = []
    for r, k in GUNION(list(got)):
        while len(out) <= r:
            out.append(([], np.zeros(nl)))
        a, b = got.get((r, k), (0, 0))
        out[r][0].append((k, lane[a:b], pos[a:b]))
        out[r][1][lane[a:b]] = data[a:b]
    return out


def _traced(x):
    return hasattr(x, "tr") and hasattr(x, "space")


def gather(space, n, parts, dtype, tr=None):
    """parts: [(source vector, dst lanes, src lanes)]; unlisted lanes read 0. Traced: <= 8 sources per instruction
    (the DRAM reader's part cursors, vm_ncrisc.cpp)."""
    if tr is None:
        g = np.zeros(n, dtype)
        for src, dst, pos in parts:
            g[dst] = src[pos]
        return g
    from .trace import T
    out = None
    for c in range(0, len(parts), 8):
        g = T(tr, space, np.array(tr.gather(space, [(src.c[()], dst, pos) for src, dst, pos in parts[c:c + 8]]),
                                  object))
        out = g if out is None else out + g
    return out


def apply(lays, srcs, space, n, dtype, tr=None):
    """sum over layers of weight * gather: a fixed sparse map applied with gathers and constant multiplies."""
    out = None
    for parts, w in lays:
        g = gather(space, n, [(srcs[k], dst, pos) for k, dst, pos in parts], dtype, tr) * w.astype(dtype)
        out = g if out is None else out + g
    return out


class Level:
    """One SA level: fine ELL graph (nbr) -> coarse level, everything frozen at setup."""

    def __init__(self, A, nbr, part=None, smooth=True, mask=None, chip=None):
        """chip (chip-local SA, the rows' chips): A carries no coupling across chips and neither may the Galerkin
        pattern (mask has those slots off): the A P pattern comes from the live slots only, so no coarse lane couples
        two chips; the smoothing weight is 4/3 over the largest chip block's spectral radius (each block on its own,
        so a per-chip setup finds the same value)."""
        A = A.tocsr()
        local = chip is not None
        n = A.shape[0]
        d0 = A.diagonal()
        self.agg = agg = aggregate(A, part)
        nc = int(agg.max()) + 1
        self.n, self.nc = n, nc
        own = _own(part, n)
        self.own_c = own_c = np.zeros(nc, bool)          # coarse rows this process owns (an aggregate never spans cores)
        own_c[agg[own]] = True
        T = sp.csr_matrix((np.ones(n), (np.arange(n), agg)), shape=(n, nc))
        # members (nc, m) padded with the aggregate's first member and weight 0: restriction = (Mw * s[M]).sum(1)
        order = np.argsort(agg, kind="stable")
        cnt = np.bincount(agg, minlength=nc)
        start = np.r_[0, np.cumsum(cnt)[:-1]]
        pos = np.arange(n) - start[agg[order]]
        mw = GMAX(int(cnt[own_c].max(initial=1)))
        self.M = np.repeat(order[start][:, None], mw, axis=1)
        kp = pos < mw                                    # (an aggregate wider than mw is never this process's)
        self.M[agg[order][kp], pos[kp]] = order[kp]
        self.Mw = np.zeros(self.M.shape)
        self.Mw[agg[order][kp], pos[kp]] = 1.0
        DA = (sp.diags(1 / d0) @ A).tocsr()
        if not smooth:
            rho = None
        elif local:
            rho = GMAX(max(spectral_radius(DA[np.ix_(r, r)]) for r in (np.nonzero(chip == c)[0] for c in np.unique(chip[own]))))
        else:
            rho = spectral_radius(DA)
        self.w = w = 4.0 / 3.0 / rho if smooth else 0.0   # 0: plain aggregation, P = T
        P = (T - w * (DA @ T)).tocsr()
        P.sort_indices()
        self.P = P
        # frozen S on the fine ELL graph: S x = (1 - w) x + sum_s sof * x[nbr];  S^T r = (1 - w) r + sum_s stf * r[nbr]
        self.nbr = nbr
        rows = np.repeat(np.arange(n), nbr.shape[1])
        a1 = np.asarray(A[rows, nbr.ravel()]).ravel().reshape(nbr.shape)
        self.mask = mask = (nbr != np.arange(n)[:, None]) if mask is None else np.asarray(mask, bool)
        a1[~mask] = 0.0                                    # padding slots
        self.sof = -w * a1 / d0[:, None]
        self.stf = -w * a1 / d0[nbr]
        # Galerkin maps. Fine coefficient vector c = [d (n), off.ravel() (n*S)].
        S = nbr.shape[1]
        ci = np.r_[np.arange(n), rows]
        cj = np.r_[np.arange(n), nbr.ravel()]
        cid = np.r_[np.arange(n), n + np.arange(n * S)]
        live = np.r_[np.ones(n, bool), mask.ravel()]
        pi, pj = (ci[live], cj[live]) if local else (ci, cj)
        Apat = sp.csr_matrix((np.ones(len(pi)), (pi, pj)), shape=(n, n))
        Apat.data[:] = 1
        APpat = (Apat @ (P != 0).astype(float)).tocsr()
        APpat.data[:] = 1
        kAP = _keys(APpat)
        # G1: for each fine coefficient (i, j, cid) and each P_jJ: AP lane (i, J) += P_jJ * c[cid]
        ci, cj, cid = ci[live], cj[live], cid[live]
        cntj = np.diff(P.indptr)[cj]
        e_i = np.repeat(ci, cntj)
        e_c = np.repeat(cid, cntj)
        pos = np.repeat(P.indptr[cj], cntj) + (np.arange(cntj.sum()) - np.repeat(np.cumsum(cntj) - cntj, cntj))
        lane = np.searchsorted(kAP, e_i.astype(np.int64) * nc + P.indices[pos])
        self.G1 = sp.csr_matrix((P.data[pos], (lane, e_c)), shape=(len(kAP), n * (S + 1)))
        # G2: for each AP lane (i, J) and each P_iI: A_c lane (I, J) += P_iI * AP[i, J]
        Acpat = (P.T @ APpat).tocsr()
        Acpat.data[:] = 1
        kAc = _keys(Acpat)
        li = kAP // nc
        lJ = kAP % nc
        cnti = np.diff(P.indptr)[li]
        e_lane = np.repeat(np.arange(len(kAP)), cnti)
        pos = np.repeat(P.indptr[li], cnti) + (np.arange(cnti.sum()) - np.repeat(np.cumsum(cnti) - cnti, cnti))
        I = P.indices[pos]
        out = np.searchsorted(kAc, I.astype(np.int64) * nc + np.repeat(lJ, cnti))
        self.G2 = sp.csr_matrix((P.data[pos], (out, e_lane)), shape=(len(kAc), len(kAP)))
        # coarse ELL graph from the A_c pattern; lane -> (diag | slot)
        rI, cJ = kAc // nc, kAc % nc
        dg = rI == cJ
        self.diag_lane = np.zeros(nc, np.int64)
        self.diag_lane[rI[dg]] = np.nonzero(dg)[0]
        cnt = np.bincount(rI[~dg], minlength=nc)
        self.S = Sc = GMAX(max(int(cnt[own_c].max(initial=0)), 1))
        first = np.r_[0, np.cumsum(cnt)[:-1]]
        offl = np.nonzero(~dg)[0]
        slot = np.arange(len(offl)) - first[rI[offl]]
        offl, slot = offl[slot < Sc], slot[slot < Sc]    # (a row wider than Sc is never this process's)
        self.cnbr = np.tile(np.arange(nc)[:, None], (1, Sc))
        self.cnbr[rI[offl], slot] = cJ[offl]
        self.off_lane = np.full((nc, Sc), -1, np.int64)
        self.off_lane[rI[offl], slot] = offl
        # stacked layout (attach(stack=True)): slot k of row I at lane k * nc + I, so a coarse SpMV is one gather of
        # every neighbour value plus one block sum (trace.Tracer.bsum) instead of one gather per slot
        self.gspaces = None
        self.terms = (self.G1.nnz + self.G2.nnz) / n
        self.L1_src = lambda c: (np.where(c < n, 0, 1 + (c - n) % S), np.where(c < n, c, (c - n) // S))
        self.own_ap, self.own_ac = own[kAP // nc], own_c[rI]
        self.nAP, self.nAc = self.G1.shape[0], self.G2.shape[0]
        self.ap_row, self.ac_row = kAP // nc, rI
        self.slot_rows = [np.nonzero(self.off_lane[:, k] >= 0)[0] for k in range(Sc)]

    @functools.cached_property
    def L1(self):
        """whole-space layers of G1 (the unstacked path and the numpy reference; a stacked trace uses L1c)"""
        return layers(self.G1, self.L1_src, self.own_ap)

    @functools.cached_property
    def L2(self):
        return layers(self.G2, lambda c: (np.zeros(len(c), np.int64), c), self.own_ac)

    def galerkin(self, d, off, dtype=None):
        """Coarse (d, off) from the finer level's, by the two layered maps (works on numpy and traced arrays)."""
        tr = d.tr if _traced(d) else None
        dtype = dtype or d.dtype
        sp_ = (lambda name: getattr(self, name, None))
        if tr is not None and getattr(self, "L1c", None) is not None:   # chunked flat spaces (attach(stack=True))
            srcs = [d] + off if isinstance(off, list) else [d] + [off[:, s] for s in range(off.shape[1])]
            ap = [apply(lay, srcs, g, g.n, dtype, tr) for lay, g in zip(self.L1c, self.apc)]
            ac = [apply(lay, ap, g, g.n, dtype, tr) for lay, g in zip(self.L2c, self.acc)]
            dc = gather(self.space, self.nc, _parts(ac, self.acwhere, np.arange(self.nc), self.diag_lane, self.own_c), dtype, tr)
            return dc, [gather(g, g.n, _parts(ac, self.acwhere, dst, lane, self.own_c[dst % self.nc]), dtype, tr)
                        for g, (dst, _, lane) in zip(self.gspaces, self.gidx)]
        if tr is None:                                     # numpy: the maps as sparse products (no layers built)
            dt = dtype
            off = np.asarray(off)
            ap = self.G1 @ np.r_[d, off.ravel()].astype(dt)
            ap[~self.own_ap] = 0
            ac = self.G2 @ ap.astype(dt)
            ac[~self.own_ac] = 0
            ac = ac.astype(dt)
            cols = np.zeros((self.nc, self.S), dt)
            for k, r in enumerate(self.slot_rows):
                cols[r, k] = ac[self.off_lane[r, k]]
            return ac[self.diag_lane], cols
        if isinstance(off, list):                          # a stacked finer level: one source per chunk
            ap = apply(self.L1s, [d] + off, sp_("apspace"), self.nAP, dtype, tr)
        else:
            ap = apply(self.L1, [d] + [off[:, s] for s in range(off.shape[1])], sp_("apspace"), self.nAP, dtype, tr)
        ac = apply(self.L2, [ap], sp_("acspace"), self.nAc, dtype, tr)
        dc = gather(sp_("space"), self.nc, [(ac, np.arange(self.nc), self.diag_lane)], dtype, tr)
        if self.gspaces is not None and tr is not None:    # stacked: one gather of the coefficients per chunk
            return dc, [gather(g, g.n, [(ac, dst, lane)], dtype, tr) for g, (dst, _, lane) in zip(self.gspaces, self.gidx)]
        cols = [gather(sp_("space"), self.nc, [(ac, r, self.off_lane[r, k])], dtype, tr)
                for k, r in enumerate(self.slot_rows)]
        if tr is not None:
            from .trace import T
            return dc, T(tr, self.space, np.array([c.c[()] for c in cols], object))
        return dc, np.stack(cols, axis=1)

    def prolong(self, ec):
        dt = self.dtype(ec)
        e = ec[self.agg]
        if _traced(ec) and getattr(self, "fchunks", None):
            return dt(1 - self.w) * e + _ssum(e, [c[:3] for c in self.fchunks], e.space,
                                              [c[3].astype(dt) for c in self.fchunks])
        return dt(1 - self.w) * e + (self.sof.astype(dt) * e[self.nbr]).sum(axis=1)

    def restrict(self, r):
        dt = self.dtype(r)
        if _traced(r) and getattr(self, "fchunks", None):
            s = dt(1 - self.w) * r + _ssum(r, [c[:3] for c in self.fchunks], r.space,
                                           [c[4].astype(dt) for c in self.fchunks])
        else:
            s = dt(1 - self.w) * r + (self.stf.astype(dt) * r[self.nbr]).sum(axis=1)
        if _traced(r) and self.gspaces is not None:
            return _ssum(s, [c[:3] for c in self.mchunks], self.space, [m.astype(dt) for m in self.mw])
        return (self.Mw.astype(dt) * s[self.M]).sum(axis=1)

    @staticmethod
    def dtype(x):
        return x.tr.dtype.type if _traced(x) else x.dtype.type


def _chunks(name, base, leader, owner, I, k, pos):
    """Stacked spaces over `base` for entries (row I, slot k) reading lane pos: slot k at lane (k % C) * w + lane(I)
    of chunk k // C, w = base rows per core rounded up to 4 lanes (whole tiles above one tile), C slots per chunk so a
    chunk spans at most the fine level's T tiles (tabstream's map bound, and a DRAM-homed chunk's block shifts stay
    inside its T pages and the reader's int16 lane shift while T <= 32). -> [(Space, dst, pos, entry selection)]."""
    from .trace import Space
    own = _own(owner[leader], len(leader))
    rows = GMAX(int(np.bincount(owner[leader][own]).max(initial=0)))
    w = -(-rows // 4) * 4 if rows <= LANES else -(-rows // LANES) * LANES
    C = max(1, -(-GMAX(int(np.bincount(owner[_own(owner, len(owner))]).max(initial=0))) // LANES) * LANES // w)
    kmax = GMAX(int(k[own[I]].max(initial=0)))
    out = []
    for j, k0 in enumerate(range(0, kmax + 1, C)):
        sel = np.nonzero((k >= k0) & (k < k0 + C))[0]
        c = min(C, kmax + 1 - k0)
        out.append((Space(f"{name}{j}", base.n * c, cells=np.tile(leader, c), stack=(base, c, w)),
                    (k[sel] - k0) * base.n + I[sel], pos[sel], sel))
    return out, C


def _stack(tr, levels, l, leader, owner):
    """Stacked spaces of level l: its coarse SpMV (one chunk per C slots), its member sum, and level l + 1's first
    Galerkin map reading the SpMV chunks."""
    L = levels[l]
    sI, sk = np.nonzero(L.off_lane >= 0)
    L.gchunks, C = _chunks(f"amgs{l + 1}g", L.space, leader, owner, sI, sk, L.cnbr[sI, sk])
    L.gspaces = [g for g, _, _, _ in L.gchunks]
    L.gidx = [(dst, pos, L.off_lane[sI[sel], sk[sel]]) for _, dst, pos, sel in L.gchunks]
    L.g_entries = [(sI[sel], sk[sel]) for _, _, _, sel in L.gchunks]
    mI, mk = np.nonzero(L.Mw != 0)
    L.mchunks, _ = _chunks(f"amgs{l + 1}m", L.space, leader, owner, mI, mk, L.M[mI, mk])
    L.mw = [L.Mw[mI[sel], mk[sel]] for _, _, _, sel in L.mchunks]
    if l + 1 < len(levels):
        n, S, N = L.nc, L.S, levels[l + 1]
        assert np.array_equal(N.nbr, L.cnbr)     # level l + 1's smoother graph is level l's coarse graph
        N.L1s_src = lambda c: (np.where(c < n, 0, 1 + (c - n) % S // C), np.where(c < n, c, (c - n) % S % C * n + (c - n) // S))
        N.fchunks = [(g, dst, pos, N.sof[i, k], N.stf[i, k]) for g, (dst, pos, _), (i, k) in
                     zip(L.gspaces, L.gidx, L.g_entries)]


def _split(name, cells, owner, cap):
    """A flat space's lanes cut into chunks of at most `cap` lanes per core (by each lane's rank on its core):
    -> [Space], chunk id per lane, index in its chunk per lane."""
    from .trace import Space
    core = owner[cells]
    o = np.argsort(core, kind="stable")
    start = np.r_[0, np.cumsum(np.bincount(core, minlength=owner.max() + 1))[:-1]]
    rank = np.empty(len(cells), np.int64)
    rank[o] = np.arange(len(cells)) - start[core[o]]
    cid = rank // cap
    local = np.zeros(len(cells), np.int64)
    spaces = []
    for c in range(GMAX(int(cid[_own(core, len(cells))].max(initial=0))) + 1):
        lanes = np.nonzero(cid == c)[0]
        local[lanes] = np.arange(len(lanes))
        spaces.append(Space(f"{name}{c}", len(lanes), cells=cells[lanes]))
    return spaces, cid, local


def _flat(L, l, rowcell, leader, owner):
    """Galerkin flat spaces in chunks of at most the fine level's T tiles per core (tabstream's map bound, and the
    encoder's 16-bit table lanes): per chunk its rows of the two layered maps; A_c lanes read as (chunk, index)."""
    cap = -(-GMAX(int(np.bincount(owner[_own(owner, len(owner))]).max(initial=0))) // LANES) * LANES
    L.apc, apid, aploc = _split(f"amgs{l + 1}p", rowcell[L.ap_row], owner, cap)
    L.acc, acid, acloc = _split(f"amgs{l + 1}c", leader[L.ac_row], owner, cap)
    src = getattr(L, "L1s_src", L.L1_src)
    G1, G2 = L.G1.tocsr(), L.G2.tocsr()
    L.L1c = [layers(G1[r], src, L.own_ap[r]) for r in (np.nonzero(apid == c)[0] for c in range(len(L.apc)))]
    L.L2c = [layers(G2[r], lambda cc: (apid[cc], aploc[cc]), L.own_ac[r]) for r in (np.nonzero(acid == c)[0] for c in range(len(L.acc)))]
    L.acwhere = (acid, acloc)


def _parts(srcs, where, dst, lanes, own=None):
    """gather parts reading lanes of a chunked flat vector: one part per chunk touched (by any chip's own rows;
    own: per entry, whether this process sets up its destination)"""
    cid, loc = where
    if own is not None:
        dst, lanes = dst[own], lanes[own]
    c = cid[lanes]
    return [(srcs[k], dst[c == k], loc[lanes[c == k]]) for k in GUNION(c.tolist())]


def _ssum(x, chunks, base, weights):
    """sum over a stacked map: per chunk one gather of x and one block sum of weight * gathered (traced only)"""
    from .trace import T
    y = None
    for (g, dst, pos), wt in zip(chunks, weights):
        if isinstance(wt, np.ndarray):          # constant weights per entry -> a stacked constant (0 off the map)
            full = np.zeros(g.n, wt.dtype)
            full[dst] = wt
            wt = full
        xg = gather(g, g.n, [(x, dst, pos)], None, x.tr)
        t = T(x.tr, base, np.array(x.tr.bsum((xg * wt).c[()]), object))
        y = t if y is None else y + t
    return y


LANES = 1024


def attach(tr, levels, cells, stack=False, owner=None):
    """Make `levels` traceable on `tr`: per level a coarse Space (an aggregate on its leader's core, the leader being
    its first member's fine cell) and two flat spaces for the Galerkin lanes (AP lane (i, J) on row i's core, A_c
    lane (I, J) on I's core); every constant index array registered with the space its gather lands in."""
    from .trace import Space
    leader = np.arange(cells.n)
    fine = cells
    for l, L in enumerate(levels):
        rowcell = leader
        leader = leader[L.M[:, 0]]
        L.space = Space(f"amgs{l + 1}", L.nc, cells=leader.copy())
        L.apspace = Space(f"amgs{l + 1}p", L.nAP, cells=rowcell[L.ap_row])
        L.acspace = Space(f"amgs{l + 1}c", L.nAc, cells=leader[L.ac_row])
        L.fine_space = fine
        if stack:
            _flat(L, l, rowcell, leader, owner)
            _stack(tr, levels, l, leader, owner)
        for a in (L.M, L.cnbr):
            tr.index_space[id(a)] = L.space
        for a in (L.agg, L.nbr):
            tr.index_space[id(a)] = fine
        fine = L.space
    return levels


def gershgorin(d, off):
    return float(((np.abs(d) + np.abs(off).sum(axis=1)) / np.abs(d)).max(initial=0.0))


def chip_local(A, chip, nbr=None, mask=None):
    """A without its couplings across chips (chip[i]: row i's chip), no diagonal lumping (a chip's block keeps its
    Dirichlet-like edge, so it stays SPD); the ELL mask with those slots off."""
    A = A.tocoo()
    k = chip[A.row] == chip[A.col]
    A = sp.csr_matrix((A.data[k], (A.row[k], A.col[k])), shape=A.shape)
    if nbr is not None:
        mask = (nbr != np.arange(len(nbr))[:, None] if mask is None else np.asarray(mask, bool)) & (chip[nbr] == chip[:, None])
    return A, mask


def setup(A, part=None, max_coarse=500, smoothed=99, nbr=None, mask=None, chip_cores=None, depth=None, min_core=0):
    """Levels from the setup matrix A (CSR) and the frozen Chebyshev upper bounds per level (setup matrix's). The
    first `smoothed` levels get the smoothed P; below them plain aggregation (P = T) keeps coarse rows narrow.
    nbr, mask: the fine level's ELL layout (the program's; default A's own).
    chip_cores: cores per chip (part // chip_cores is a row's chip). The multigrid is chip-local: P and every coarse
    operator are built from A without its cross-chip couplings (A_c = P^T A_loc P), so the coarse levels are one
    independent hierarchy per chip (block-Jacobi over chips below the fine level, the coarsest solved per chip) and
    only the fine level, the PCG's own SpMV, exchanges values between chips. depth: at most this many coarse levels.
    min_core: no coarse level with fewer than this many rows per core (part's cores); such a level is all sync and halo
    (r13 2x4: 2.5 rows a core made the V-cycle 1.6x slower than stopping one level above it)."""
    local = chip_cores is not None
    if local:
        assert part is not None, "chip-local SA needs the core partition"
        A0, A = A, chip_local(A, np.asarray(part) // chip_cores)[0]
    d, off, nbr0 = ell(A0 if local else A)
    if nbr is None:
        nbr, mask = nbr0, None
    else:
        rows = np.repeat(np.arange(A.shape[0]), nbr.shape[1])
        off = np.where(mask, np.asarray((A0 if local else A).tocsr()[rows, nbr.ravel()]).reshape(nbr.shape), 0.0)
    assert OWN is None or local, "a per-chip SA setup is chip-local"
    own = _own(part, A.shape[0])
    levels, lmax = [], [GMAX(gershgorin(d[own], off[own]))]
    while GSUM(int(own.sum())) > max_coarse and (depth is None or len(levels) < depth):
        if local:
            mask = chip_local(A, np.asarray(part) // chip_cores, nbr, mask)[1]
        L = Level(A, nbr, part, len(levels) < smoothed, mask, np.asarray(part) // chip_cores if local else None)
        mask = None
        nc = GSUM(int(L.own_c.sum()))
        if nc > 0.7 * GSUM(int(own.sum())) or (min_core and nc < min_core * (GMAX(int(np.max(part))) + 1)):
            break
        levels.append(L)
        d, off = L.galerkin(d, off)
        nbr = L.cnbr
        own = L.own_c
        lmax.append(GMAX(gershgorin(d[own], off[own])))
        A = L.P.T @ A @ L.P
        if part is not None:
            cp = np.zeros(L.nc, np.int64)
            cp[L.agg] = part
            part = cp
    return levels, lmax


PRE, POST = 3, 3   # Chebyshev degrees of the pre and post smoothing on every level above the coarsest


class Ops:
    """Per-outer-iteration operators of every level from the fine (d, off, nbr); V-cycle with Chebyshev(PRE) pre and
    Chebyshev(POST) post smoothing, Chebyshev(COARSE_DEG) on the coarsest level."""

    def __init__(self, levels, lmax, d, off, nbr, dtype=np.float32, coarse=None, shape=None):
        """coarse: the coarse (d, off) per level, given instead of recomputed (a lagged Galerkin, krylov.amg_pcg).
        shape: the cycle's shape, dict(pre=, post=, coarse_deg=, gamma={level: coarse corrections}) (run_mesh
        --vpre/--vpost/--vcoarse/--vgamma); missing keys keep PRE / POST / COARSE_DEG and one correction per level
        (a V-cycle; gamma 2 at a level = a W at it). pre >= 1; post 0 skips the post smoothing (unsymmetric)."""
        shape = shape or {}
        self.pre, self.post = shape.get("pre", PRE), shape.get("post", POST)
        self.coarse_deg = shape.get("coarse_deg", COARSE_DEG)
        self.gamma = dict(shape.get("gamma") or {})
        assert self.pre >= 1 and self.post >= 0 and self.coarse_deg >= 1, shape
        self.levels, self.dtype = levels, dtype
        self.traced = _traced(d)
        self.A = [(self._c(d), self._c(off), nbr)]
        for l, L in enumerate(levels):
            d, off = L.galerkin(d, off) if coarse is None else coarse[l]
            self.A.append((self._c(d), off if isinstance(off, list) else self._c(off), L.cnbr))
        self.cheb = [((b + b / 30) / 2, (b - b / 30) / 2, self._c(1 / dd)) for b, (dd, _, _) in zip(lmax, self.A)]

    def _c(self, x):
        """fp rounding to the working type on numpy; traced values already are in it"""
        return x if self.traced else x.astype(self.dtype)

    def spmv(self, l, x):
        d, off, nbr = self.A[l]
        L = self.levels[l - 1] if l else None
        if self.traced and L is not None and L.gspaces is not None:
            return d * x + _ssum(x, [(g, dst, pos) for g, (dst, pos, _) in zip(L.gspaces, L.gidx)], L.space, off)
        return d * x + (off * x[nbr]).sum(axis=1)

    def smooth(self, l, b, x, deg):
        th, de, dinv = self.cheb[l]
        t = self.dtype
        r = b if x is None else self._c(b - self.spmv(l, x))
        dd = self._c(dinv * r / t(th))
        s1 = th / de
        rho = 1 / s1
        x = dd if x is None else self._c(x + dd)
        for _ in range(deg - 1):
            r = self._c(r - self.spmv(l, dd))
            rn = 1 / (2 * s1 - rho)
            dd = self._c(t(rn * rho) * dd + t(2 * rn / de) * dinv * r)
            x = self._c(x + dd)
            rho = rn
        return x

    def vcycle(self, r, l=0):
        if l == len(self.levels):
            return self.smooth(l, r, None, self.coarse_deg)
        L = self.levels[l]
        x = self.smooth(l, r, None, self.pre)
        for _ in range(self.gamma.get(l, 1)):
            rc = self._c(L.restrict(self._c(r - self.spmv(l, x))))
            x = self._c(x + self._c(L.prolong(self.vcycle(rc, l + 1))))
        return self.smooth(l, r, x, self.post) if self.post else x
