"""Host model of an aggregation-multigrid p solve on the device's partition: iterations and work on captured systems.

    python -m ttfluid.tools.mgmodel DIR --sys pf,it101 --cores 220 --chips 2 --cfg "agg=3 mode=post k=1" [--cfg ...]

DIR is a tools/pcapture.py output. The cells are split over cores as the device splits them (machine.rcb), and every
aggregate stays inside one core, so each core holds its own part of every level. A level is built by `agg` passes of
pairwise matching along the strongest couplings plus --extra passes that merge leftovers (aggregates of up to 2^agg cells,
piecewise-constant prolongation, Galerkin coarse operator), until a level has at most --coarsest rows. The preconditioner is one cycle from the fine
level, inside flexible PCG with OpenFOAM's normalised residual and the system's own tolerance and relTol.

A --cfg is a list of key=value:
  agg    pairwise passes per level, one number or a comma list per level (3 = aggregates of up to 8)
  mode   fine level: jac (no coarse level, the build's Jacobi PCG), add (smoother + coarse correction, both from the
         same residual), post (coarse correction, then smooth), sym (smooth, coarse correction, smooth)
  k, a   fine smoother: k Chebyshev steps on [lmax / a, lmax] of D^-1 A (k = 1 is damped Jacobi)
  cmode, ck, ca   the same for the coarse levels (default sym, 2, 4)
  gamma  cycles per coarse level visit (1 = V, 2 = W)
  kc, ac the coarsest level's Chebyshev solve (default 16, 30)
  depth  levels below the fine one (default: until --coarsest)
  w      weight of every coarse correction (piecewise-constant aggregation under-corrects; 1 = none)
  local  levels from this one down drop their couplings between cores, so their SpMVs need no halo exchange
         (1 = every coarse level is solved inside its core; default: none)
  fp32   1 = vectors and matrices in fp32

Per solve it prints the PCG iterations and the work: SpMVs, restrictions and prolongations per level, reductions.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

SMALL = 1e-20
T0 = time.time()


def log(*a):
    print(f"{time.time() - T0:8.1f} s", *a, flush=True)


# ---- aggregation ---------------------------------------------------------------------------------------------------

def pairwise(A, core, size, cap, q=4.0, rounds=40):
    """One matching pass inside each core. An edge is a candidate while both ends are free and the pair would hold at
    most cap cells. Every node takes its candidate of highest priority and mutual choices pair. The priority is the
    coupling in steps of 2^(1/q) with a hash of the edge inside a step: with the bare coupling, a mesh whose
    coefficients grow smoothly in one direction pairs one cell per round along each chain and the pass stalls.
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
        hit = np.flatnonzero(w == np.repeat(np.maximum.reduceat(w, start), np.diff(np.r_[start, len(i)])))
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


def galerkin(A, agg, nc):
    """P^T A P for the piecewise-constant P of agg."""
    n = A.shape[0]
    i = agg[np.repeat(np.arange(n), np.diff(A.indptr))]
    Ac = sp.csr_matrix((A.data, (i, agg[A.indices])), shape=(nc, nc))
    Ac.sum_duplicates()
    return Ac


class Level:
    def __init__(self, A, core):
        self.A, self.core, self.n = A, core, A.shape[0]
        d = A.diagonal()
        self.dinv = (1.0 / d).astype(A.dtype)
        self.lmax = float((np.abs(A).sum(axis=1).A1 / d).max())       # Gershgorin bound of D^-1 A: one reduction
        self.agg = None                                               # node -> node of the next level
        self._own = None

    @property
    def own(self):
        """The level without its couplings between cores: what a core can apply with no halo exchange."""
        if self._own is None:
            i = np.repeat(np.arange(self.n), np.diff(self.A.indptr))
            keep = self.core[i] == self.core[self.A.indices]
            self._own = sp.csr_matrix((self.A.data[keep], (i[keep], self.A.indices[keep])), shape=self.A.shape)
        return self._own


def hierarchy(A, core, passes, coarsest, cache=None, extra=2, q=4.0, minratio=2.0):
    """Levels from the fine one down. cache: an .npz with the aggregate of every node per level, written when absent."""
    aggs = None
    if cache is not None and cache.exists():
        with np.load(cache) as z:
            aggs = [z[f"agg{l}"] for l in range(len(z.files))]
    lev = [Level(A, core)]
    while lev[-1].n > coarsest and (aggs is None or len(lev) <= len(aggs)):
        L = lev[-1]
        if aggs is not None:
            agg = aggs[len(lev) - 1]
            first = np.zeros(int(agg.max()) + 1, np.int64)
            first[agg] = np.arange(L.n)
            Ac, cc = galerkin(L.A, agg, len(first)), L.core[first]
        else:
            agg, Ac, cc, size = np.arange(L.n), L.A, L.core, np.ones(L.n, np.int64)
            p = passes[min(len(lev) - 1, len(passes) - 1)]
            for _ in range(p + extra):                                 # the extra passes merge what was left over
                a, nc = pairwise(Ac, cc, size, 1 << p, q)
                if nc == len(a):
                    break
                first = np.zeros(nc, np.int64)
                first[a] = np.arange(len(a))
                Ac, cc, agg, size = galerkin(Ac, a, nc), cc[first], a[agg], np.bincount(a, size, nc).astype(np.int64)
        if Ac.shape[0] * minratio > L.n:                               # what is left pairs too slowly to be a level
            break
        L.agg = agg
        lev.append(Level(Ac, cc))
    if cache is not None and aggs is None:
        tmp = cache.with_name(f"{cache.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, **{f"agg{l}": L.agg.astype(np.int32) for l, L in enumerate(lev[:-1])})
        os.replace(tmp, cache)
    return lev


def stats(L, P, chips):
    """What one core holds of a level and what it needs from other cores (the halo of an SpMV)."""
    i = np.repeat(np.arange(L.n, dtype=np.int64), np.diff(L.A.indptr))
    j = L.A.indices.astype(np.int64)
    ci, cj = L.core[i].astype(np.int64), L.core[j].astype(np.int64)
    off = ci != cj
    rows, per = np.bincount(L.core, minlength=P), np.diff(L.A.indptr)
    halo = np.unique(ci[off] * L.n + j[off])                          # distinct (core, node of another core)
    hc = np.bincount(halo // L.n, minlength=P)
    xc = (ci[off] // (P // chips)) != (cj[off] // (P // chips))
    return dict(rows_core=[int(rows.min()), float(rows.mean()), int(rows.max())], nnz_row=[float(per.mean()), int(per.max())],
                nnz_core_max=int(np.bincount(ci, minlength=P).max()), off_core=float(off.mean()),
                halo_core=[float(hc.mean()), int(hc.max())], off_chip=float(xc.sum() / max(len(i), 1)))


# ---- cycle ---------------------------------------------------------------------------------------------------------

class Work:
    """Counts per level: SpMVs, restrictions, prolongations; reductions of the outer iteration."""

    def __init__(self, nlev):
        self.spmv, self.R, self.P = [0] * nlev, [0] * nlev, [0] * nlev
        self.red = 0


def cheb(L, r, k, a, W, l, A=None):
    """k Chebyshev steps for A z = r from z = 0 on [lmax / a, lmax] of D^-1 A: k - 1 SpMVs."""
    A = L.A if A is None else A
    theta, delta = 0.5 * L.lmax * (1 + 1 / a), 0.5 * L.lmax * (1 - 1 / a)
    sigma = theta / delta
    rho = 1 / sigma
    d = L.dinv * r / theta
    z = d.copy()
    rr = r
    for _ in range(k - 1):
        rr = rr - A @ d
        W.spmv[l] += 1
        rho1 = 1 / (2 * sigma - rho)
        d = rho1 * rho * d + (2 * rho1 / delta) * (L.dinv * rr)
        z += d
        rho = rho1
    return z


class Cycle:
    def __init__(self, lev, c, W):
        self.lev, self.c, self.W = lev, c, W

    def A(self, l):
        return self.lev[l].own if l >= self.c["local"] else self.lev[l].A

    def coarse(self, l, r):
        """The correction of level l - 1 from level l: gamma cycles."""
        W = self.W
        z = self(l, r)
        for _ in range(self.c["gamma"] - 1 if l < len(self.lev) - 1 else 0):
            W.spmv[l] += 1
            z = z + self(l, r - self.A(l) @ z)
        return z

    def __call__(self, l, r):
        lev, c, W = self.lev, self.c, self.W
        L, A = lev[l], self.A(l)
        if l == len(lev) - 1:
            return cheb(L, r, c["kc"], c["ac"], W, l, A)
        mode, k, a = (c["mode"], c["k"], c["a"]) if l == 0 else (c["cmode"], c["ck"], c["ca"])
        nc = lev[l + 1].n

        def corr(res):
            W.R[l] += 1
            W.P[l] += 1
            return c["w"] * self.coarse(l + 1, np.bincount(L.agg, res, nc).astype(res.dtype))[L.agg]
        if mode == "add":
            return cheb(L, r, k, a, W, l, A) + corr(r)
        if mode == "post":
            z = corr(r)
            W.spmv[l] += 1
            return z + cheb(L, r - A @ z, k, a, W, l, A)
        assert mode == "sym", mode
        z = cheb(L, r, k, a, W, l, A)
        W.spmv[l] += 1
        z = z + corr(r - A @ z)
        W.spmv[l] += 1
        return z + cheb(L, r - A @ z, k, a, W, l, A)


def fpcg(A, b, x0, M, ctl, W):
    """Flexible PCG (Polak-Ribiere beta) with OpenFOAM's normFactor and convergence test. Returns its, r0, final."""
    tol, relTol, maxit = float(ctl[0]), float(ctl[1]), int(ctl[2])
    x = x0.copy()
    Ax = A @ x
    t = np.asarray(A.sum(axis=1)).ravel() * x.mean()
    nf = np.abs(Ax - t).sum() + np.abs(b - t).sum() + SMALL
    r = b - Ax
    r0 = res = np.abs(r).sum() / nf
    it, p, rz_old, r_old = 0, None, None, None
    while it < maxit and not (res < tol or (relTol > 0 and res < relTol * r0)):
        z = M(r)
        rz = r @ z
        p = z if p is None else z + ((rz - z @ r_old) / rz_old) * p
        q = A @ p
        W.spmv[0] += 1
        W.red += 1                                    # one reduction event: r.z, z.r_old, p.q and the norm of r
        alpha = rz / (p @ q)
        x += alpha * p
        r_old, rz_old = r, rz
        r = r - alpha * q
        it += 1
        res = np.abs(r).sum() / nf
    true = np.abs(b - A @ x).sum() / nf
    return it, r0, res, true


# ---- driver --------------------------------------------------------------------------------------------------------

DEF = dict(agg="3", mode="post", k=1, a=4.0, cmode="sym", ck=2, ca=4.0, gamma=1, kc=16, ac=30.0, depth=None, local=99, w=1.0, fp32=0)


def parse_cfg(s):
    c = dict(DEF)
    for kv in s.split():
        k, v = kv.split("=")
        assert k in DEF, k
        c[k] = v if k in ("agg", "mode", "cmode") else float(v) if k in ("a", "ca", "ac", "w") else int(v)
    c["name"] = s
    return c


def partition(d, P, chips):
    f = d / f"part_{P}_{chips}.npy"
    if f.exists():
        return np.load(f)
    from ..resident.machine import rcb
    with np.load(d / "mesh.npz") as z:
        part = rcb(z["C"].astype(np.float64), P, chips=chips if chips > 1 else None, faces=(z["owner"], z["neighbour"]))
    np.save(f, part)
    return part


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--sys", required=True, help="comma list of captured system labels")
    ap.add_argument("--cores", type=int, default=220, help="cores over all chips")
    ap.add_argument("--chips", type=int, default=2)
    ap.add_argument("--coarsest", type=int, default=4000, help="stop coarsening at this many rows")
    ap.add_argument("--extra", type=int, default=2, help="matching passes per level after the first `agg`")
    ap.add_argument("--q", type=float, default=4.0, help="matching priority: coupling steps per factor of 2")
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--json", help="append one row per solve")
    a = ap.parse_args()
    d = Path(a.dir)
    cfgs = [parse_cfg(s) for s in a.cfg]
    core = partition(d, a.cores, a.chips)
    log(f"partition {a.cores} cores on {a.chips} chips, cells per core {np.bincount(core).min()}-{np.bincount(core).max()}")
    for label in a.sys.split(","):
        with np.load(d / f"sys_{label}.npz") as z:
            A = sp.csr_matrix((z["data"].astype(np.float64), z["indices"], z["indptr"]))
            b, x0, ctl = z["b"].astype(np.float64), z["x0"].astype(np.float64), z["ctl"]
        A.sort_indices()
        log(f"sys {label}: n {A.shape[0]} nnz {A.nnz} tol {ctl[0]} relTol {ctl[1]}")
        built = {}
        for c in cfgs:
            F = np.float32 if c["fp32"] else np.float64
            key = (c["agg"] if c["mode"] != "jac" else "jac", c["fp32"])
            if key not in built:
                Af = A.astype(F)
                if c["mode"] == "jac":
                    built[key] = [Level(Af, core)], None
                else:
                    t1 = time.time()
                    h = hierarchy(Af, core, [int(v) for v in c["agg"].split(",")], a.coarsest,
                                  d / f"agg_{label}_{c['agg'].replace(',', '-')}_{a.cores}_{a.chips}_{a.coarsest}_{a.extra}_{a.q:g}.npz",
                                  a.extra, a.q)
                    built[key] = h, [stats(L, a.cores, a.chips) for L in h]
                    log(f"  hierarchy agg={c['agg']}: rows {[L.n for L in h]} nnz {[L.A.nnz for L in h]} "
                        f"lmax {[round(L.lmax, 3) for L in h]} ({time.time() - t1:.0f} s)")
                    for l, st in enumerate(built[key][1]):
                        log(f"    level {l}: {st}")
            lev, st = built[key]
            if c["depth"] is not None:
                lev = lev[:c["depth"] + 1]
            W = Work(len(lev))
            M = (lambda r, L=lev[0], c=c, W=W: cheb(L, r, c["k"], c["a"], W, 0)) if c["mode"] == "jac" else \
                (lambda r, cy=Cycle(lev, c, W): cy(0, r))
            t1 = time.time()
            it, r0, res, true = fpcg(lev[0].A, b.astype(F), x0.astype(F), M, ctl, W)
            row = dict(sys=label, cfg=c["name"], its=it, r0=float(r0), final=float(res), true=float(true),
                       rows=[L.n for L in lev], nnz=[int(L.A.nnz) for L in lev], spmv=W.spmv, R=W.R, P=W.P, red=W.red,
                       fine_per_it=W.spmv[0] / max(it, 1),
                       coarse_nnz_eq=sum(s * L.A.nnz for s, L in zip(W.spmv[1:], lev[1:])) / lev[0].A.nnz,
                       lmax=[L.lmax for L in lev], levels=st[:len(lev)] if st else None, cores=a.cores, chips=a.chips,
                       wall=time.time() - t1)
            log(f"  {label} [{c['name']}] its {it} fine SpMV {W.spmv[0]} ({row['fine_per_it']:.2f}/it) coarse SpMV "
                f"{W.spmv[1:]} = {row['coarse_nnz_eq']:.2f} fine-eq by nnz, R {W.R[:-1]}, red {W.red}, "
                f"final {res / r0:.3e} of r0 (true {true / r0:.3e}), {row['wall']:.1f} s")
            if a.json:
                with open(a.json, "a") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
