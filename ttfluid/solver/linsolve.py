"""Krylov solvers for FvMatrix systems, with OpenFOAM's residual normalisation and convergence test.

Only parallel-friendly building blocks: SpMV, axpy, dot and a diagonal (Jacobi) preconditioner. These are the
operations the device implements; there is no sequential Gauss-Seidel or incomplete-Cholesky sweep here.
"""
from dataclasses import dataclass

import numpy as np

SMALL = 1e-20


@dataclass
class Controls:
    tolerance: float = 1e-6
    relTol: float = 0.0
    maxIter: int = 1000
    minIter: int = 0


@dataclass
class Perf:
    initial: float
    final: float
    iterations: int


class Operator:
    """y = A x for one component: interior LDU plus the boundary diagonal."""
    def __init__(self, M, diag):
        m = M.mesh
        self.o, self.n = m.owner[:m.nInternalFaces], m.neighbour
        self.d, self.up, self.lo = diag, M.upper, M.lower
        self.nCells = m.nCells

    def __call__(self, x):
        y = self.d * x
        y += np.bincount(self.o, self.up * x[self.n], self.nCells)
        y += np.bincount(self.n, self.lo * x[self.o], self.nCells)
        return y

    def sumA(self):
        return self.d + np.bincount(self.o, self.up, self.nCells) + np.bincount(self.n, self.lo, self.nCells)


def norm_factor(A, x, b, Ax):
    t = A.sumA() * x.mean()
    return np.abs(Ax - t).sum() + np.abs(b - t).sum() + SMALL


def _converged(r, r0, c):
    return r < c.tolerance or (c.relTol > 0 and r < c.relTol * r0)


def pcg(A, x, b, c, symmetric_check=True):
    """Jacobi-preconditioned conjugate gradients (OpenFOAM PCG with a diagonal preconditioner)."""
    Ax = A(x)
    nf = norm_factor(A, x, b, Ax)
    r = b - Ax
    r0 = np.abs(r).sum() / nf
    res, it = r0, 0
    if c.minIter <= 0 and _converged(res, r0, c):
        return Perf(r0, res, 0)
    rD = 1.0 / A.d
    rho_old, p = None, None
    while it < c.maxIter:
        w = rD * r
        rho = r @ w
        p = w if p is None else w + (rho / rho_old) * p
        q = A(p)
        alpha = rho / (p @ q)
        x += alpha * p
        r -= alpha * q
        rho_old = rho
        it += 1
        res = np.abs(r).sum() / nf
        if it >= c.minIter and _converged(res, r0, c):
            break
    return Perf(r0, res, it)


def pbicgstab(A, x, b, c):
    """Jacobi-preconditioned BiCGStab (OpenFOAM PBiCGStab structure, diagonal preconditioner)."""
    Ax = A(x)
    nf = norm_factor(A, x, b, Ax)
    r = b - Ax
    r0n = np.abs(r).sum() / nf
    res, it = r0n, 0
    if c.minIter <= 0 and _converged(res, r0n, c):
        return Perf(r0n, res, 0)
    rD = 1.0 / A.d
    r0 = r.copy()
    rho = alpha = omega = 1.0
    p = np.zeros_like(x)
    v = np.zeros_like(x)
    while it < c.maxIter:
        rho_old = rho
        rho = r0 @ r
        if abs(rho) < 1e-300:
            break
        if it == 0:
            p = r.copy()
        else:
            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
        y = rD * p
        v = A(y)
        alpha = rho / (r0 @ v)
        s = r - alpha * v
        it += 1
        res = np.abs(s).sum() / nf
        if it >= c.minIter and _converged(res, r0n, c):
            x += alpha * y
            r = s
            break
        z = rD * s
        t = A(z)
        tt = t @ t
        omega = (t @ s) / tt if tt > 0 else 0.0
        x += alpha * y + omega * z
        r = s - omega * t
        res = np.abs(r).sum() / nf
        if it >= c.minIter and _converged(res, r0n, c):
            break
    return Perf(r0n, res, it)


def solve(M, controls, method="auto"):
    """Solve FvMatrix M in place for every solved component (fvMatrix::solveSegregated). Returns max initial
    residual over components and the per-component performance."""
    psi = M.psi
    x = psi.internal
    perfs = []
    comps = range(1) if psi.width == 1 else [c for c in range(psi.width) if psi.mesh.solutionD[c]]
    for c in comps:
        diag, src = M.residual_parts(c)
        A = Operator(M, diag)
        xc = (x if psi.width == 1 else x[:, c]).copy()
        sym = np.array_equal(M.upper, M.lower)
        kind = method if method != "auto" else ("pcg" if sym else "pbicgstab")
        perf = (pcg if kind == "pcg" else pbicgstab)(A, xc, src, controls)
        if psi.width == 1:
            x[:] = xc
        else:
            x[:, c] = xc
        perfs.append(perf)
    psi.correct_bc()
    return max(p.initial for p in perfs), perfs


# =================================================================================================================
# Opt-in fast host path (SimpleFoam(host_solver="amg")): scipy CSR SpMV, pyamg smoothed-aggregation-preconditioned
# CG for symmetric systems, Jacobi BiCGStab on CSR otherwise. Same initial residual, normFactor and convergence test
# as pcg/pbicgstab above; only the iteration differs, so it is for tight solves where only the converged answer counts.
# =================================================================================================================

def to_csr(A):
    """scipy CSR of the operator exactly as A(x) applies it (an Operator or an ELL _EllOp)."""
    import scipy.sparse as sp
    n = A.d.shape[0]
    ar = np.arange(n)
    if hasattr(A, "off"):                      # ELL: y = d x + rowsum(off * x[nbr]); masked slots carry off = 0
        w = A.off.shape[1]
        rows = np.concatenate([ar, np.repeat(ar, w)])
        cols = np.concatenate([ar, np.asarray(A.nbr).ravel()])
        vals = np.concatenate([A.d, A.off.ravel()])
    else:                                      # LDU: y = d x + upper (owner <- neighbour) + lower (neighbour <- owner)
        rows = np.concatenate([ar, A.o, A.n])
        cols = np.concatenate([ar, A.n, A.o])
        vals = np.concatenate([A.d, A.up, A.lo])
    M = sp.csr_matrix((vals.astype(np.float64), (rows, cols)), shape=(n, n))
    M.sum_duplicates()
    M.eliminate_zeros()
    return M


def _host_log(kind, n, it, r0, res, dt):
    print(f"  host-{kind} n={n} it={it} init={r0:.6e} final={res:.6e} {dt:.1f} s", flush=True)


def amg_pcg(A, x, b, c, iter_scale=100):
    """CG preconditioned by a pyamg smoothed-aggregation V-cycle, on the (negated, SPD) CSR system."""
    import time
    import pyamg
    t0 = time.time()
    Ax = A(x)
    nf = norm_factor(A, x, b, Ax)
    r0 = np.abs(b - Ax).sum() / nf
    if c.minIter <= 0 and _converged(r0, r0, c):
        return Perf(r0, r0, 0)
    S = A.d.sum() < 0                          # laplacian(rAU, p): negative definite -> solve -A x = -b
    K = to_csr(A)
    if S:
        K = -K
    bb = -b if S else b
    ml = pyamg.smoothed_aggregation_solver(K, symmetry="hermitian", max_coarse=500)
    P = ml.aspreconditioner(cycle="V")
    tsetup = time.time() - t0
    r = bb - K @ x
    res, it, p, rho_old = r0, 0, None, None
    maxIter = c.maxIter * iter_scale
    while it < maxIter:
        z = P(r)
        rho = r @ z
        p = z if p is None else z + (rho / rho_old) * p
        q = K @ p
        alpha = rho / (p @ q)
        x += alpha * p
        r -= alpha * q
        rho_old = rho
        it += 1
        res = np.abs(r).sum() / nf
        if it >= c.minIter and _converged(res, r0, c):   # the recurrence residual, as OpenFOAM's PCG tests it
            break
    _host_log("amgpcg", len(x), it, r0, res, time.time() - t0)
    print(f"    true residual {np.abs(bb - K @ x).sum() / nf:.6e}", flush=True)
    print(f"    amg setup {tsetup:.1f} s, levels {len(ml.levels)}", flush=True)
    return Perf(r0, res, it)


def csr_pbicgstab(A, x, b, c, iter_scale=100):
    """pbicgstab above (Jacobi preconditioner) with its SpMV on scipy CSR."""
    import time
    t0 = time.time()
    Ax = A(x)
    nf = norm_factor(A, x, b, Ax)
    K = to_csr(A)
    r = b - K @ x
    r0n = np.abs(b - Ax).sum() / nf
    res, it = r0n, 0
    if c.minIter <= 0 and _converged(res, r0n, c):
        return Perf(r0n, res, 0)
    rD = 1.0 / A.d
    r0 = r.copy()
    rho = alpha = omega = 1.0
    p = np.zeros_like(x)
    v = np.zeros_like(x)
    maxIter = c.maxIter * iter_scale
    while it < maxIter:
        rho_old = rho
        rho = r0 @ r
        if abs(rho) < 1e-300:
            break
        if it == 0:
            p = r.copy()
        else:
            beta = (rho / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
        y = rD * p
        v = K @ y
        alpha = rho / (r0 @ v)
        s = r - alpha * v
        it += 1
        res = np.abs(s).sum() / nf
        if it >= c.minIter and _converged(res, r0n, c):
            x += alpha * y
            r = s
            break
        z = rD * s
        t = K @ z
        tt = t @ t
        omega = (t @ s) / tt if tt > 0 else 0.0
        x += alpha * y + omega * z
        r = s - omega * t
        res = np.abs(r).sum() / nf
        if it >= c.minIter and _converged(res, r0n, c):
            break
    _host_log("bicgstab", len(x), it, r0n, res, time.time() - t0)
    return Perf(r0n, res, it)


def host_krylov(symmetric):
    """ops.krylov replacement for SimpleFoam(host_solver="amg")."""
    return amg_pcg if symmetric else csr_pbicgstab
