"""linsolve.pcg / linsolve.pbicgstab as traced programs: the iteration loop and OpenFOAM's convergence test become
device control flow (labels, a jump back, conditional exits), so a whole linear solve runs without the host.

Same arithmetic as solver/ttfluid/solver/linsolve.py, one change of form: the first iteration is not peeled. PCG starts from
p = 0, rho_old = 1, so p = w + rho * 0 = w; BiCGStab from p = v = 0, alpha = omega = rho = 1, so p = r + beta * 0 = r.
Both are exact in floating point (x + 0 = x).
"""
import os

import numpy as np

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
    tr.jumpif((np.abs(rho) < 1e-300).c[()], END)
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
    tr.jumpif((np.abs(RHO.t) < 1e-300).c[()], END)
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
    tr.jumpif((done + (it >= float(c.maxIter)) + (np.abs(rho_n) < 1e-300)).c[()], END)
    tr.jump(LOOP)
    tr.label(EXIT_S)
    X.set(X.t + alpha * y)
    R.set(s)
    tr.label(END)
    x[...] = X.t
    return Perf(r0n, RES.t, IT.t)
