"""linsolve.pcg / linsolve.pbicgstab as traced programs: the iteration loop and OpenFOAM's convergence test become
device control flow (labels, a jump back, conditional exits), so a whole linear solve runs without the host.

Same arithmetic as solver/ttfluid/solver/linsolve.py, one change of form: the first iteration is not peeled. PCG starts from
p = 0, rho_old = 1, so p = w + rho * 0 = w; BiCGStab from p = v = 0, alpha = omega = rho = 1, so p = r + beta * 0 = r.
Both are exact in floating point (x + 0 = x).
"""
import os

import numpy as np

from ttfluid.solver import linsolve
from ttfluid.solver.linsolve import Perf, SMALL

from .trace import SCALAR, T, _obj


class Var:
    """A loop-carried value: fixed registers, written by copies. `.t` refers to the registers, it is not a snapshot:
    read every old value a step needs before the step's `.set`."""

    def __init__(self, t):
        tr = t.tr
        self.tr, self.space = tr, t.space
        self.c = _obj(t.c.shape)
        for i in np.ndindex(*t.c.shape):
            r = tr.reg(tr.space[t.c[i]])
            tr.copy(r, t.c[i])
            self.c[i] = r

    @property
    def t(self):
        return T(self.tr, self.space, self.c.copy())

    def set(self, t):
        for i in np.ndindex(*self.c.shape):
            src = t.c[i] if isinstance(t, T) else t
            self.tr.copy(self.c[i], src)


def _scalar(tr, v):
    return T(tr, SCALAR, np.array(tr.imm(v), object))


def _vsmall(tr):
    """SolverPerformance::vsmall_ (the singularity test of PBiCGStab's rho): VSMALL of the program's precision. The
    double value 1e-300 is 0.0 as an fp32 constant, and `|rho| < 0` is never true: the test was dead on the device."""
    return linsolve.vsmall(tr.dtype)


def _converged(r, r0, c):
    """r < tolerance or (relTol > 0 and r < relTol r0), as a 0/1 scalar."""
    hit = r < c.tolerance
    if c.relTol > 0:
        hit = np.maximum(hit, r < c.relTol * r0)
    return hit


def _norm_factor(A, x, b, Ax):
    t = A.sumA() * x.mean()
    return np.abs(Ax - t).sum() + np.abs(b - t).sum() + SMALL


JOINT = False   # pcg, amg_pcg: the residual norm and the next iteration's rho leave in one cross-core reduction
CG1 = False     # pcg, amg_pcg as _pcg_cg1 (one reduction per iteration; needs Machine(joint_red=True) to pay)
BJOINT = False  # pbicgstab as _pbicgstab_joint (three cross-core reductions per iteration, not five)
USOLVE = "bicgstab"  # multi-component asymmetric U solve: bicgstab | bicgstab3 | bicgstab3s | jacobi3 | mr3
TSOLVE = "bicgstab"  # single-component asymmetric k/omega solve: bicgstab | jacobi | mr
UCAD = 1             # exit-test cadence for jacobi3/jacobi (sweeps per residual check; exit always tests true residual)
UOMEGA = 1.0         # Jacobi relaxation weight for jacobi3/jacobi (1.0 = undamped)
UBREAK = False       # broken-arm gate toggle: corrupts Uy slot-0 gather in off_matvec_multi (must FAIL parity/residual)


def pcg(A, x, b, c):
    if CG1:
        return _pcg_cg1(A, x, b, c)
    if JOINT:
        return _pcg_joint(A, x, b, c)
    tr = x.tr
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0 = np.abs(r).sum() / nf
    END, LOOP = tr.new_label(), tr.new_label()
    X, R = Var(x), Var(r)
    P, RHO_OLD, IT, RES = Var(np.zeros_like(x)), Var(_scalar(tr, 1.0)), Var(_scalar(tr, 0.0)), Var(r0)
    rD = 1.0 / A.d
    # rho = r.w for the next pass is taken right after the residual sum of the same r, so the two global sums can share
    # one reduction (a cross-chip all-gather on a mesh); same values, same order as taking it at the loop's top
    w = rD * r
    W, RHO = Var(w), Var(r @ w)
    if c.minIter <= 0:
        tr.jumpif(_converged(r0, r0, c).c[()], END)
    tr.trips[LOOP] = 300     # a pressure solve takes hundreds of passes, the others a handful
    tr.label(LOOP)
    tr.jumpif((IT.t >= float(c.maxIter)).c[()], END)
    r, w, rho = R.t, W.t, RHO.t
    p = w + (rho / RHO_OLD.t) * P.t
    q = A(p)
    alpha = rho / (p @ q)
    X.set(X.t + alpha * p)
    R.set(r - alpha * q)
    P.set(p)
    RHO_OLD.set(rho)
    it = IT.t + 1.0
    IT.set(it)
    rn = R.t
    res = np.abs(rn).sum() / nf
    RES.set(res)
    wn = rD * rn
    W.set(wn)
    RHO.set(rn @ wn)
    done = _converged(res, r0, c)
    if c.minIter > 0:
        done = done * (it >= float(c.minIter))
    tr.jumpif(done.c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    x[...] = X.t
    return Perf(r0, RES.t, IT.t)


def _pcg_cg1(A, x, b, c, M=None, maxit=None):
    """PCG (Jacobi unless M is given: amg_pcg passes its V-cycle) in the Chronopoulos-Gear form: one SpMV (w = A u, u = rD r) and ONE cross-core reduction per
    iteration, |r|, gamma = r.u and delta = w.u as three adjacent sums (Machine(joint_red=True) sends them together).
    s = A p is carried by the recurrence s = w + beta s. The same iterates as pcg in exact arithmetic, rounded
    differently; the residual it tests is the same |r| of the updated r, so iteration counts match up to rounding.
    One extra SpMV per solve: A u0 before the loop (the last iteration's w is computed and not used). The loop has one
    exit branch, converged or maxIter (maxIter is a Python constant, so a non-positive one skips the loop at trace time).
    With M = a V-cycle the extra application before the loop is one V-cycle, and so is the unused one at the exit."""
    tr = x.tr
    maxit = float(c.maxIter if maxit is None else maxit)
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0 = np.abs(r).sum() / nf
    END, LOOP = tr.new_label(), tr.new_label()
    if M is None:
        rD = 1.0 / A.d
        M = lambda v: rD * v
    u = M(r)
    w = A(u)
    gam, dlt = r @ u, w @ u
    X, R, U, W = Var(x), Var(r), Var(u), Var(w)
    P, S, IT, RES = Var(np.zeros_like(x)), Var(np.zeros_like(x)), Var(_scalar(tr, 0.0)), Var(r0)
    GAM, ALPHA, BETA = Var(gam), Var(gam / dlt), Var(_scalar(tr, 0.0))
    if c.minIter <= 0:
        tr.jumpif(_converged(r0, r0, c).c[()], END)
    if maxit <= 0:
        tr.jump(END)
    tr.trips[LOOP] = 300
    tr.label(LOOP)
    beta, alpha = BETA.t, ALPHA.t
    p = U.t + beta * P.t
    s = W.t + beta * S.t
    X.set(X.t + alpha * p)
    rn = R.t - alpha * s
    R.set(rn)
    P.set(p)
    S.set(s)
    it = IT.t + 1.0
    IT.set(it)
    un = M(rn)
    wn = A(un)
    a, g, d = np.abs(rn), rn * un, wn * un
    sa, sg, sd = a.sum(), g.sum(), d.sum()
    bn = sg / GAM.t
    ALPHA.set(sg / (sd - bn * sg / alpha))
    BETA.set(bn)
    GAM.set(sg)
    U.set(un)
    W.set(wn)
    res = sa / nf
    RES.set(res)
    done = _converged(res, r0, c)
    if c.minIter > 0:
        done = done * (it >= float(c.minIter))
    # the maxIter test rides on the exit test: one branch (one C -> B, N round) per iteration, not two
    tr.jumpif((done + (it >= maxit)).c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    x[...] = X.t
    return Perf(r0, RES.t, IT.t)


def _pcg_joint(A, x, b, c):
    """pcg with rho = r.(rD r) of iteration k + 1 computed at the end of iteration k, beside the residual norm, so the
    two sums are adjacent reduces the lowering sends as one (Machine(joint_red=True)). Same values in the same order:
    rho is the same sum of the same products, only earlier (the last one, past the exit, is never read)."""
    tr = x.tr
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0 = np.abs(r).sum() / nf
    END, LOOP = tr.new_label(), tr.new_label()
    X, R = Var(x), Var(r)
    P, RHO_OLD, IT, RES = Var(np.zeros_like(x)), Var(_scalar(tr, 1.0)), Var(_scalar(tr, 0.0)), Var(r0)
    if c.minIter <= 0:
        tr.jumpif(_converged(r0, r0, c).c[()], END)
    rD = 1.0 / A.d
    RHO = Var(r @ (rD * r))
    tr.trips[LOOP] = 300
    tr.label(LOOP)
    tr.jumpif((IT.t >= float(c.maxIter)).c[()], END)
    r = R.t
    w = rD * r
    rho = RHO.t
    p = w + (rho / RHO_OLD.t) * P.t
    q = A(p)
    alpha = rho / (p @ q)
    X.set(X.t + alpha * p)
    R.set(r - alpha * q)
    P.set(p)
    RHO_OLD.set(rho)
    it = IT.t + 1.0
    IT.set(it)
    rn = R.t
    a, g = np.abs(rn), rn * (rD * rn)
    sa, sg = a.sum(), g.sum()
    res = sa / nf
    RHO.set(sg)
    RES.set(res)
    done = _converged(res, r0, c)
    if c.minIter > 0:
        done = done * (it >= float(c.minIter))
    tr.jumpif(done.c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    x[...] = X.t
    return Perf(r0, RES.t, IT.t)


def amg_pcg(A, x, b, c, levels):
    """pcg with a multigrid V-cycle (ttfluid.resident.amg, aggregates fixed at setup) as the preconditioner instead of Jacobi.
    The coarse operators are recomputed from A by summation at the start of every solve."""
    from . import amg, sa
    tr = x.tr
    if getattr(levels, "lmax", None) is not None and getattr(levels, "lag", None):
        # lagged Galerkin: the coarse operators are recomputed on one solve in `k` and kept in loop-carried
        # registers in between; the fine operator (the PCG's A and the fine smoother) is always the current one
        k, CNT, store = levels.lag
        SKIP = tr.new_label()
        tr.jumpif((CNT.t > 0.5).c[()], SKIP)
        fresh = sa.Ops(levels, levels.lmax, A.d, A.off, A.nbr, tr.dtype.type, shape=levels.vshape)
        for (dv, ov), (d_, o_, _) in zip(store, fresh.A[1:]):
            dv.set(d_)
            for v, o in zip(ov, o_ if isinstance(o_, list) else [o_]):
                v.set(o)
        tr.label(SKIP)
        c1 = CNT.t + 1.0
        CNT.set(c1 * (c1 < float(k)))
        ops = sa.Ops(levels, levels.lmax, A.d, A.off, A.nbr, tr.dtype.type, shape=levels.vshape,
                     coarse=[(dv.t, [v.t for v in ov] if levels[0].gspaces else ov[0].t) for dv, ov in store])
    elif getattr(levels, "lmax", None) is not None:      # smoothed aggregation (program.SALevels)
        ops = sa.Ops(levels, levels.lmax, A.d, A.off, A.nbr, tr.dtype.type, shape=levels.vshape)
    else:
        ops = amg.Ops(levels, A.d, A.off, A.nbr, tr.dtype.type)
    maxit = min(c.maxIter, int(os.environ.get("CFD_AMG_MAXIT", 10 ** 9)))
    if CG1:
        return _pcg_cg1(A, x, b, c, ops.vcycle, maxit)
    if JOINT:
        return _amg_joint(A, x, b, c, ops.vcycle, maxit)
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0 = np.abs(r).sum() / nf
    END, LOOP = tr.new_label(), tr.new_label()
    X, R = Var(x), Var(r)
    P, RHO_OLD, IT, RES = Var(np.zeros_like(x)), Var(_scalar(tr, 1.0)), Var(_scalar(tr, 0.0)), Var(r0)
    if c.minIter <= 0:
        tr.jumpif(_converged(r0, r0, c).c[()], END)
    tr.trips[LOOP] = 10
    tr.label(LOOP)
    tr.jumpif((IT.t >= float(maxit)).c[()], END)
    r = R.t
    w = ops.vcycle(r)
    rho = r @ w
    p = w + (rho / RHO_OLD.t) * P.t
    q = A(p)
    alpha = rho / (p @ q)
    X.set(X.t + alpha * p)
    R.set(r - alpha * q)
    P.set(p)
    RHO_OLD.set(rho)
    it = IT.t + 1.0
    IT.set(it)
    res = np.abs(R.t).sum() / nf
    RES.set(res)
    done = _converged(res, r0, c)
    if c.minIter > 0:
        done = done * (it >= float(c.minIter))
    tr.jumpif(done.c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    x[...] = X.t
    return Perf(r0, RES.t, IT.t)


def _amg_joint(A, x, b, c, M, maxit):
    """amg_pcg with the V-cycle of iteration k + 1 applied to the updated residual before the convergence test, so |r|
    and the next rho = r.M(r) are adjacent sums the lowering sends as one reduction (Machine(joint_red=True)): two
    cross-core reductions per iteration, not three, and the maxIter test rides on the exit branch. Every value is the
    one amg_pcg computes, in the same order, and the solve exits at the same iteration; the cost is the exiting
    iteration's V-cycle, computed and not used."""
    tr = x.tr
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0 = np.abs(r).sum() / nf
    END, LOOP = tr.new_label(), tr.new_label()
    X, R = Var(x), Var(r)
    P, RHO_OLD, IT, RES = Var(np.zeros_like(x)), Var(_scalar(tr, 1.0)), Var(_scalar(tr, 0.0)), Var(r0)
    if c.minIter <= 0:
        tr.jumpif(_converged(r0, r0, c).c[()], END)
    if maxit <= 0:
        tr.jump(END)
    w = M(r)
    W, RHO = Var(w), Var(r @ w)
    tr.trips[LOOP] = 10
    tr.label(LOOP)
    rho = RHO.t
    p = W.t + (rho / RHO_OLD.t) * P.t
    q = A(p)
    alpha = rho / (p @ q)
    X.set(X.t + alpha * p)
    R.set(R.t - alpha * q)
    P.set(p)
    RHO_OLD.set(rho)
    it = IT.t + 1.0
    IT.set(it)
    rn = R.t
    wn = M(rn)
    a, g = np.abs(rn), rn * wn
    sa, sg = a.sum(), g.sum()
    RES.set(sa / nf)
    W.set(wn)
    RHO.set(sg)
    done = _converged(sa / nf, r0, c)
    if c.minIter > 0:
        done = done * (it >= float(c.minIter))
    tr.jumpif((done + (it >= float(maxit))).c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    x[...] = X.t
    return Perf(r0, RES.t, IT.t)


def pbicgstab(A, x, b, c):
    if BJOINT:
        return _pbicgstab_joint(A, x, b, c)
    tr = x.tr
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0n = np.abs(r).sum() / nf
    END, LOOP, EXIT_S = tr.new_label(), tr.new_label(), tr.new_label()
    X, R = Var(x), Var(r)
    zero = np.zeros_like(x)
    P, V = Var(zero), Var(zero)
    one = _scalar(tr, 1.0)
    RHO, ALPHA, OMEGA = Var(one), Var(one), Var(one)
    IT, RES = Var(_scalar(tr, 0.0)), Var(r0n)
    if c.minIter <= 0:
        tr.jumpif(_converged(r0n, r0n, c).c[()], END)
    rD = 1.0 / A.d
    r0 = r.copy()
    tr.trips[LOOP] = 5
    tr.label(LOOP)
    tr.jumpif((IT.t >= float(c.maxIter)).c[()], END)
    rho = r0 @ R.t
    beta = (rho / RHO.t) * (ALPHA.t / OMEGA.t)
    RHO.set(rho)
    tr.jumpif((np.abs(rho) < _vsmall(tr)).c[()], END)
    p = R.t + beta * (P.t - OMEGA.t * V.t)
    P.set(p)
    y = rD * p
    v = A(y)
    V.set(v)
    alpha = rho / (r0 @ v)
    ALPHA.set(alpha)
    s = R.t - alpha * v
    it = IT.t + 1.0
    IT.set(it)
    res = np.abs(s).sum() / nf
    RES.set(res)
    gate = (it >= float(c.minIter)) if c.minIter > 0 else None
    done = _converged(res, r0n, c)
    done = done if gate is None else done * gate
    tr.jumpif(done.c[()], EXIT_S)        # converged on s: x += alpha y, r = s, stop
    z = rD * s
    t = A(z)
    tt = t @ t
    omega = np.where(tt > 0, (t @ s) / tt, 0.0)
    OMEGA.set(omega)
    X.set(X.t + (alpha * y + omega * z))
    R.set(s - omega * t)
    res = np.abs(R.t).sum() / nf
    RES.set(res)
    done = _converged(res, r0n, c)
    done = done if gate is None else done * gate
    tr.jumpif(done.c[()], END)
    tr.jump(LOOP)
    tr.label(EXIT_S)
    X.set(X.t + alpha * y)
    R.set(s)
    tr.label(END)
    x[...] = X.t
    return Perf(r0n, RES.t, IT.t)


def _pbicgstab_joint(A, x, b, c):
    """pbicgstab with its sums regrouped into three adjacent sets the lowering sends as one reduction each
    (Machine(joint_red=True)): r0.v alone; |s|, t.t, t.s with t = A(rD s) computed before the test on s; |r| and the
    next iteration's rho = r0.r at the end of the iteration. Every sum is the same sum of the same products, so the
    iterates are the ones pbicgstab computes. The maxIter and rho == 0 tests at the loop head ride on the end-of-
    iteration branch (both leave X and R as they are). The one cost: when the test on s exits, its t was computed
    and not used (one SpMV more per solve that exits there)."""
    tr = x.tr
    Ax = A(x)
    nf = _norm_factor(A, x, b, Ax)
    r = b - Ax
    r0n = np.abs(r).sum() / nf
    END, LOOP, EXIT_S = tr.new_label(), tr.new_label(), tr.new_label()
    X, R = Var(x), Var(r)
    zero = np.zeros_like(x)
    P, V = Var(zero), Var(zero)
    one = _scalar(tr, 1.0)
    RHO_OLD, ALPHA, OMEGA = Var(one), Var(one), Var(one)
    IT, RES = Var(_scalar(tr, 0.0)), Var(r0n)
    if c.minIter <= 0:
        tr.jumpif(_converged(r0n, r0n, c).c[()], END)
    if c.maxIter <= 0:
        tr.jump(END)
    rD = 1.0 / A.d
    r0 = r.copy()
    RHO = Var(r0 @ r)
    tr.jumpif((np.abs(RHO.t) < _vsmall(tr)).c[()], END)
    tr.trips[LOOP] = 5
    tr.label(LOOP)
    rho = RHO.t
    beta = (rho / RHO_OLD.t) * (ALPHA.t / OMEGA.t)
    RHO_OLD.set(rho)
    p = R.t + beta * (P.t - OMEGA.t * V.t)
    P.set(p)
    y = rD * p
    v = A(y)
    V.set(v)
    alpha = rho / (r0 @ v)
    ALPHA.set(alpha)
    s = R.t - alpha * v
    it = IT.t + 1.0
    IT.set(it)
    z = rD * s
    t = A(z)
    a, q, w = np.abs(s), t * t, t * s
    sa, tt, ts = a.sum(), q.sum(), w.sum()
    res = sa / nf
    RES.set(res)
    gate = (it >= float(c.minIter)) if c.minIter > 0 else None
    done = _converged(res, r0n, c)
    done = done if gate is None else done * gate
    tr.jumpif(done.c[()], EXIT_S)        # converged on s: x += alpha y, r = s, stop
    omega = np.where(tt > 0, ts / tt, 0.0)
    OMEGA.set(omega)
    X.set(X.t + (alpha * y + omega * z))
    R.set(s - omega * t)
    rn = R.t
    a, g = np.abs(rn), r0 * rn
    sa, rho_n = a.sum(), g.sum()
    res = sa / nf
    RES.set(res)
    RHO.set(rho_n)
    done = _converged(res, r0n, c)
    done = done if gate is None else done * gate
    tr.jumpif((done + (it >= float(c.maxIter)) + (np.abs(rho_n) < _vsmall(tr))).c[()], END)
    tr.jump(LOOP)
    tr.label(EXIT_S)
    X.set(X.t + alpha * y)
    R.set(s)
    tr.label(END)
    x[...] = X.t
    return Perf(r0n, RES.t, IT.t)


def off_matvec_multi(A0, xs):
    """Shared off-diagonal SpMV [N @ x_k for x_k in xs] sharing A0.off and A0.nbr.
    Emits slot-interleaved gathers g_0, ..., g_{m-1} consecutively per ELL slot j so --g3 1 runs a single 3-wide
    dense gather across NCRISC/BRISC/UNPACK/MATH/PACK and --dedup 1 streams off[:, j] once for all components.
    Keeps live scratch in the fused segment at m + 4 tiles and tags slotfold[out] = j for --sliced."""
    m = len(xs)
    nbr = A0.nbr
    off = A0.off
    S = nbr.shape[1]
    if not isinstance(xs[0], T):
        outs = [None] * m
        for j in range(S):
            nb_j = nbr[:, j]
            aj = off[:, j]
            for k in range(m):
                xk = xs[0] if (UBREAK and j == 0 and k == 1 and m >= 2) else xs[k]
                term = aj * xk[nb_j]
                outs[k] = term if j == 0 else (outs[k] + term)
        return outs
    tr = xs[0].tr
    cells = xs[0].space
    dst = tr.arange(nbr.shape[0])
    outs = [None] * m
    for j in range(S):
        nb_j = nbr[:, j]
        gs = []
        for k in range(m):
            xk = xs[0] if (UBREAK and j == 0 and k == 1 and m >= 2) else xs[k]
            gs.append(T(tr, cells, np.array(tr.gather(cells, [(xk.c[()], dst, nb_j)]), object)))
        aj = off[:, j]
        for k in range(m):
            term = aj * gs[k]
            if j == 0:
                outs[k] = term
            else:
                outs[k] = outs[k] + term
                tr.slotfold[outs[k].c[()]] = j
    return outs


def _norm_factor_multi(As, xs, bs, Axs):
    """OpenFOAM normFactor for m components sharing off-diagonal A0.off: computes rowsum(off) once and groups the
    component sums adjacently so --jointred 1 sends them in two joint cross-core reductions. Bit-exact vs _norm_factor."""
    rs_off = As[0].rowsum(As[0].off)
    xsums = [x.sum() for x in xs]
    means = [s / float(x.size) for s, x in zip(xsums, xs)]
    ts = [(A.d + rs_off) * mu for A, mu in zip(As, means)]
    da = [np.abs(Ax - t) for Ax, t in zip(Axs, ts)]
    db = [np.abs(b - t) for b, t in zip(bs, ts)]
    sa = [u.sum() for u in da]
    sb = [u.sum() for u in db]
    return [a + b + SMALL for a, b in zip(sa, sb)]


def pbicgstab_multi(As, xs, bs, c, split_x=False):
    """Coupled multi-component BiCGStab sharing off-diagonal matrix reads, gather tables, and joint reductions.
    Each component keeps its own scalars, residual norm, and sticky freeze mask (exact np.where, never + 0*dx)
    so it stops at its own exact exit iteration. With split_x=False (bicgstab3), recomputes y_k = rD_k * P_k at
    the X update so y[0..m-1] die before z[0..m-1] are born (only m L1 gather colours needed) while staying 100%
    bitwise equal to sequential pbicgstab; split_x=True (bicgstab3s) updates X += alpha*y before z."""
    m = len(xs)
    tr = xs[0].tr
    vsm = _vsmall(tr)
    Nxs = off_matvec_multi(As[0], xs)
    Axs = [Nxs[k] + As[k].d * xs[k] for k in range(m)]
    nfs = _norm_factor_multi(As, xs, bs, Axs)
    rs = [bs[k] - Axs[k] for k in range(m)]
    r0s = [r.copy() for r in rs]
    abs_rs = [np.abs(r) for r in rs]
    r0rs = [r0 * r for r0, r in zip(r0s, rs)]
    sa0 = [a.sum() for a in abs_rs]
    rho0s = [g.sum() for g in r0rs]
    r0ns = [sa0[k] / nfs[k] for k in range(m)]

    END, LOOP = tr.new_label(), tr.new_label()
    X = [Var(x) for x in xs]
    R = [Var(r) for r in rs]
    P = [Var(0.0 * r) for r in rs]
    V = [Var(0.0 * r) for r in rs]
    RHO = [Var(rho0) for rho0 in rho0s]
    RHO_OLD = [Var(_scalar(tr, 1.0)) for _ in range(m)]
    ALPHA = [Var(_scalar(tr, 1.0)) for _ in range(m)]
    OMEGA = [Var(_scalar(tr, 1.0)) for _ in range(m)]
    IT = [Var(_scalar(tr, 0.0)) for _ in range(m)]
    RES = [Var(r0n) for r0n in r0ns]
    IT_LOOP = Var(_scalar(tr, 0.0))

    done0, all_done0 = [], None
    for k in range(m):
        dk = _converged(r0ns[k], r0ns[k], c) if c.minIter <= 0 else _scalar(tr, 0.0)
        dk = np.maximum(dk, np.abs(rho0s[k]) < vsm)
        done0.append(dk)
        all_done0 = dk if all_done0 is None else all_done0 * dk
    DONE = [Var(dk) for dk in done0]

    tr.jumpif(all_done0.c[()], END)
    if c.maxIter <= 0:
        tr.jump(END)
    rDs = [1.0 / A.d for A in As]
    tr.trips[LOOP] = 10
    tr.label(LOOP)

    act1s, rhos, ys = [], [], []
    for k in range(m):
        act_k = 1.0 - DONE[k].t
        act1s.append(act_k)
        rho_k = RHO[k].t
        rhos.append(rho_k)
        beta_raw = (rho_k / RHO_OLD[k].t) * (ALPHA[k].t / OMEGA[k].t)
        beta_k = np.where(act_k > 0.5, beta_raw, 0.0)
        RHO_OLD[k].set(np.where(act_k > 0.5, rho_k, RHO_OLD[k].t))
        p_k = R[k].t + beta_k * (P[k].t - OMEGA[k].t * V[k].t)
        P[k].set(p_k)
        ys.append(rDs[k] * p_k)

    Nys = off_matvec_multi(As[0], ys)
    vs = [Nys[k] + As[k].d * ys[k] for k in range(m)]
    for k in range(m):
        V[k].set(vs[k])
    r0v_prods = [r0s[k] * vs[k] for k in range(m)]
    r0vs = [g.sum() for g in r0v_prods]   # 1 joint reduction for all m components

    alphas, ss, zs, it_ks = [], [], [], []
    for k in range(m):
        act_k = act1s[k]
        rv_safe = np.where(act_k > 0.5, r0vs[k], 1.0)
        alpha_k = np.where(act_k > 0.5, rhos[k] / rv_safe, 0.0)
        ALPHA[k].set(np.where(act_k > 0.5, alpha_k, ALPHA[k].t))
        alphas.append(alpha_k)
        s_k = R[k].t - alpha_k * vs[k]
        ss.append(s_k)
        it_k = IT[k].t + act_k
        IT[k].set(it_k)
        it_ks.append(it_k)
        if split_x:
            X[k].set(np.where(act_k > 0.5, X[k].t + alpha_k * ys[k], X[k].t))
        zs.append(rDs[k] * s_k)

    Nzs = off_matvec_multi(As[0], zs)
    ts = [Nzs[k] + As[k].d * zs[k] for k in range(m)]
    abs_ss = [np.abs(s) for s in ss]
    tt_prods = [t * t for t in ts]
    ts_prods = [t * s for t, s in zip(ts, ss)]
    # 1 joint reduction per group of 3 (|s|, t.t, t.s)
    sa_s = [a.sum() for a in abs_ss]
    tts = [g.sum() for g in tt_prods]
    tss = [g.sum() for g in ts_prods]

    act2s, exit_ss, res_ss, r_news = [], [], [], []
    for k in range(m):
        act_k = act1s[k]
        res_s = sa_s[k] / nfs[k]
        res_ss.append(res_s)
        gate_k = (it_ks[k] >= float(c.minIter)) if c.minIter > 0 else None
        ds = _converged(res_s, r0ns[k], c)
        ds = ds if gate_k is None else ds * gate_k
        exit_s = act_k * ds
        act2 = act_k * (1.0 - ds)
        exit_ss.append(exit_s)
        act2s.append(act2)
        omega_raw = np.where(tts[k] > 0.0, tss[k] / tts[k], 0.0)
        OMEGA[k].set(np.where(act2 > 0.5, omega_raw, OMEGA[k].t))
        omega_eff = np.where(act2 > 0.5, omega_raw, 0.0)
        if split_x:
            X[k].set(np.where(act2 > 0.5, X[k].t + omega_eff * zs[k], X[k].t))
        else:
            y_k = rDs[k] * P[k].t   # recompute y_k (identical bits) so ys die before zs -> m L1 colours
            dx_k = np.where(exit_s > 0.5, alphas[k] * y_k, alphas[k] * y_k + omega_eff * zs[k])
            X[k].set(np.where(act_k > 0.5, X[k].t + dx_k, X[k].t))
        rn = np.where(act2 > 0.5, ss[k] - omega_eff * ts[k], ss[k])
        R[k].set(rn)
        r_news.append(rn)

    abs_rn = [np.abs(rn) for rn in r_news]
    rho_prods = [r0s[k] * r_news[k] for k in range(m)]
    sa_r = [a.sum() for a in abs_rn]
    rho_ns = [g.sum() for g in rho_prods]

    all_done = None
    for k in range(m):
        act2 = act2s[k]
        res_r = sa_r[k] / nfs[k]
        RES[k].set(np.where(act2 > 0.5, res_r, np.where(exit_ss[k] > 0.5, res_ss[k], RES[k].t)))
        gate_k = (it_ks[k] >= float(c.minIter)) if c.minIter > 0 else None
        dr = _converged(res_r, r0ns[k], c)
        dr = dr if gate_k is None else dr * gate_k
        d_rho = (np.abs(rho_ns[k]) < vsm)
        still_k = act2 * (1.0 - np.maximum(dr, d_rho))
        done_k = 1.0 - still_k
        DONE[k].set(done_k)
        RHO[k].set(np.where(still_k > 0.5, rho_ns[k], RHO[k].t))
        all_done = done_k if all_done is None else all_done * done_k

    it_l = IT_LOOP.t + 1.0
    IT_LOOP.set(it_l)
    tr.jumpif((all_done + (it_l >= float(c.maxIter))).c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    for k in range(m):
        xs[k][...] = X[k].t
    return [Perf(r0ns[k], RES[k].t, IT[k].t) for k in range(m)]


def smooth_jacobi_multi(As, xs, bs, c, omega=None, cad=None):
    """Coupled multi-component smoothSolver (Jacobi, omega=1.0 undamped default) sharing off-diagonal matrix reads,
    gather tables, and joint reductions. Works for both m=3 (U) and m=1 (k, omega).
    Only m loop-carried vectors (X[0..m-1]) stay live across the sweep, fitting 100% in L1 even at Small k2.
    In-place update fuses rk, |rk| partial sum, and X[k] update into one block with zero dxs/rks DRAM traffic;
    converged components freeze bit-exact via np.where. Exit always tests the exact true residual |b - A x|_1 / nf."""
    m = len(xs)
    tr = xs[0].tr
    w = float(UOMEGA if omega is None else omega)
    cad = max(1, int(UCAD if cad is None else cad))

    Nxs = off_matvec_multi(As[0], xs)
    Axs = [Nxs[k] + As[k].d * xs[k] for k in range(m)]
    nfs = _norm_factor_multi(As, xs, bs, Axs)
    rs = [bs[k] - Axs[k] for k in range(m)]
    abs_rs = [np.abs(r) for r in rs]
    sa0 = [a.sum() for a in abs_rs]
    r0ns = [sa0[k] / nfs[k] for k in range(m)]

    END, LOOP_END, LOOP = tr.new_label(), tr.new_label(), tr.new_label()
    IT = [Var(_scalar(tr, 0.0)) for _ in range(m)]
    RES = [Var(r0n) for r0n in r0ns]

    done0, all_done0 = [], None
    for k in range(m):
        dk = _converged(r0ns[k], r0ns[k], c) if c.minIter <= 0 else _scalar(tr, 0.0)
        done0.append(dk)
        all_done0 = dk if all_done0 is None else all_done0 * dk
    DONE = [Var(dk) for dk in done0]
    tr.jumpif(all_done0.c[()], END)
    if c.maxIter <= 0:
        tr.jump(END)

    # First sweep reuses rs already computed above (zero wasted SpMV; rs dies before LOOP).
    # X is born here (after all_done0 -> END) so it stays loop-local and never aliases the outer state
    # register across --span 3 colour-0 copies or the p solve: all m components get L1 colours 0..m-1.
    X = []
    for k in range(m):
        act0 = 1.0 - DONE[k].t
        step0 = rs[k] / As[k].d if w == 1.0 else (w * (rs[k] / As[k].d))
        X.append(Var(np.where(act0 > 0.5, xs[k] + step0, xs[k])))
        IT[k].set(IT[k].t + act0)
    IT_LOOP = Var(_scalar(tr, 1.0))

    tr.trips[LOOP] = 20
    tr.label(LOOP)
    Nxs = off_matvec_multi(As[0], [X[k].t for k in range(m)])
    # In-place update fused with residual computation: rk, |rk| partial sum, and X[k] update stay in SFPU
    # scratch (0 DRAM traffic for rk/dx); active components take one contraction step, converged ones stay frozen.
    abs_rs, act1s = [], []
    for k in range(m):
        act_k = 1.0 - DONE[k].t
        act1s.append(act_k)
        rk = (bs[k] - Nxs[k]) - As[k].d * X[k].t
        abs_rs.append(np.abs(rk))
        dxk = rk / As[k].d
        step_k = dxk if w == 1.0 else (w * dxk)
        X[k].set(np.where(act_k > 0.5, X[k].t + step_k, X[k].t))
        IT[k].set(IT[k].t + act_k)
    sums = [a.sum() for a in abs_rs]   # 1 joint reduction for all m true residuals

    acts, all_done = [], None
    for k in range(m):
        act_k = act1s[k]
        res_k = sums[k] / nfs[k]
        RES[k].set(np.where(act_k > 0.5, res_k, RES[k].t))
        gate_k = (IT[k].t >= float(c.minIter)) if c.minIter > 0 else None
        dk = _converged(res_k, r0ns[k], c)
        dk = dk if gate_k is None else dk * gate_k
        done_k = np.maximum(DONE[k].t, dk)
        DONE[k].set(done_k)
        acts.append(1.0 - done_k)
        all_done = done_k if all_done is None else all_done * done_k

    it_l = IT_LOOP.t + 1.0
    IT_LOOP.set(it_l)
    tr.jumpif((all_done + (it_l >= float(c.maxIter))).c[()], LOOP_END)

    for _ in range(cad - 1):
        step_ok = (IT_LOOP.t < float(c.maxIter))
        Nxs2 = off_matvec_multi(As[0], [X[k].t for k in range(m)])
        for k in range(m):
            a2 = acts[k] * step_ok
            if w == 1.0:
                x_new = (bs[k] - Nxs2[k]) / As[k].d
            else:
                rk2 = (bs[k] - Nxs2[k]) - As[k].d * X[k].t
                x_new = X[k].t + (w * rk2) / As[k].d
            X[k].set(np.where(a2 > 0.5, x_new, X[k].t))
            IT[k].set(IT[k].t + a2)
        IT_LOOP.set(IT_LOOP.t + step_ok)

    tr.jump(LOOP)
    tr.label(LOOP_END)
    for k in range(m):
        # all_done0 is 0 here; referencing xs[k] keeps X[k] from coalescing into the outer state register
        # so X[0..m-1] stay loop-local and can take L1 colour 0 alongside --span 3.
        xs[k][...] = np.where(all_done0 > 0.5, xs[k], X[k].t)
    tr.label(END)
    return [Perf(r0ns[k], RES[k].t, IT[k].t) for k in range(m)]


def mr_multi(As, xs, bs, c):
    """Coupled multi-component Minimum Residual (GMRES(1) + Jacobi), parameter-free 1-SpMV/it solver.
    Works for both m=3 (U) and m=1 (k, omega); z[0..m-1] are the sole SpMV inputs (m L1 colours)."""
    m = len(xs)
    tr = xs[0].tr
    Nxs = off_matvec_multi(As[0], xs)
    Axs = [Nxs[k] + As[k].d * xs[k] for k in range(m)]
    nfs = _norm_factor_multi(As, xs, bs, Axs)
    rs = [bs[k] - Axs[k] for k in range(m)]
    abs_rs = [np.abs(r) for r in rs]
    sa0 = [a.sum() for a in abs_rs]
    r0ns = [sa0[k] / nfs[k] for k in range(m)]

    END, LOOP = tr.new_label(), tr.new_label()
    X = [Var(x) for x in xs]
    R = [Var(r) for r in rs]
    IT = [Var(_scalar(tr, 0.0)) for _ in range(m)]
    RES = [Var(r0n) for r0n in r0ns]
    IT_LOOP = Var(_scalar(tr, 0.0))

    done0, all_done0 = [], None
    for k in range(m):
        dk = _converged(r0ns[k], r0ns[k], c) if c.minIter <= 0 else _scalar(tr, 0.0)
        done0.append(dk)
        all_done0 = dk if all_done0 is None else all_done0 * dk
    DONE = [Var(dk) for dk in done0]
    tr.jumpif(all_done0.c[()], END)
    if c.maxIter <= 0:
        tr.jump(END)

    tr.trips[LOOP] = 18
    tr.label(LOOP)
    zs = [R[k].t / As[k].d for k in range(m)]
    Nzs = off_matvec_multi(As[0], zs)
    qs = [Nzs[k] + As[k].d * zs[k] for k in range(m)]
    qq_prods = [q * q for q in qs]
    rq_prods = [R[k].t * qs[k] for k in range(m)]
    qqs = [g.sum() for g in qq_prods]
    rqs = [g.sum() for g in rq_prods]   # 1 joint reduction for all 2m dots

    act1s, r_news, it_ks = [], [], []
    for k in range(m):
        act_k = 1.0 - DONE[k].t
        act1s.append(act_k)
        alpha_k = np.where((act_k > 0.5) * (qqs[k] > 0.0), rqs[k] / qqs[k], 0.0)
        X[k].set(np.where(act_k > 0.5, X[k].t + alpha_k * zs[k], X[k].t))
        rn = np.where(act_k > 0.5, R[k].t - alpha_k * qs[k], R[k].t)
        R[k].set(rn)
        r_news.append(rn)
        it_k = IT[k].t + act_k
        IT[k].set(it_k)
        it_ks.append(it_k)

    abs_rn = [np.abs(rn) for rn in r_news]
    sa_r = [a.sum() for a in abs_rn]    # 1 joint reduction for all m residual norms

    all_done = None
    for k in range(m):
        act_k = act1s[k]
        res_k = sa_r[k] / nfs[k]
        RES[k].set(np.where(act_k > 0.5, res_k, RES[k].t))
        gate_k = (it_ks[k] >= float(c.minIter)) if c.minIter > 0 else None
        dk = _converged(res_k, r0ns[k], c)
        dk = dk if gate_k is None else dk * gate_k
        done_k = np.maximum(DONE[k].t, dk)
        DONE[k].set(done_k)
        all_done = done_k if all_done is None else all_done * done_k

    it_l = IT_LOOP.t + 1.0
    IT_LOOP.set(it_l)
    tr.jumpif((all_done + (it_l >= float(c.maxIter))).c[()], END)
    tr.jump(LOOP)
    tr.label(END)
    for k in range(m):
        xs[k][...] = X[k].t
    return [Perf(r0ns[k], RES[k].t, IT[k].t) for k in range(m)]
