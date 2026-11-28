"""The device-side level tables (amg.build, amg.Ops) run the cycle the host model prices: on the first p system of
pitzDaily both need the same number of PCG iterations, and a capped coarse row width must change that number. The same
holds for a weighted coarse correction with two visits of the levels from 2 down, which must need fewer iterations."""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from ttfluid.resident import amg  # noqa: E402
from ttfluid.resident.machine import rcb  # noqa: E402
from ttfluid.solver import linsolve  # noqa: E402
from ttfluid.solver.simple import SimpleFoam  # noqa: E402
from ttfluid.tools import mgmodel as mm  # noqa: E402

CASE = Path(__file__).resolve().parents[1] / "cases" / "pitzDaily"
SHAPE = dict(fine="add", k=1, a=4.0, coarse="sym", ck=1, ca=4.0, kc=8, ac=30.0)   # V(1,1), 8 steps on the last level
CFG = "agg=3 mode=add k=1 cmode=sym ck=1 kc=8"
WSHAPE, WCFG = dict(SHAPE, w=1.25, g=2, gf=2), CFG + " w=1.25 gamma=2 gfrom=2"


class _Got(Exception):
    pass


def _first_p_system():
    s = SimpleFoam(str(CASE), backend="ell", dtype=np.float32, log=lambda *x: None)
    got = {}

    def krylov(symmetric):
        if not symmetric:
            return linsolve.pbicgstab

        def run(A, x, b, c):
            got.update(A=A, x=x.copy(), b=b.copy(), c=c)
            raise _Got
        return run
    s.ops.krylov = krylov
    with pytest.raises(_Got):
        s.iterate()
    return s, got


def test_ops_cycle_matches_model():
    s, got = _first_p_system()
    e, A, c = s.ops.e, got["A"], got["c"]
    nbr = np.asarray(A.nbr)
    w = np.where(e.internal, np.asarray(e.magSf, np.float64) * np.asarray(e.dc, np.float64), 0.0)
    part = rcb(np.asarray(s.mesh.C), 8)
    aggs = amg.matched(nbr, e.internal, w, part, max_coarse=20)
    sgn = -1.0 if np.asarray(A.d).sum() < 0 else 1.0
    d, off = (sgn * np.asarray(A.d)).astype(np.float64), (sgn * np.asarray(A.off)).astype(np.float64)
    K = (sgn * linsolve.to_csr(A)).astype(np.float64).tocsr()
    b, x0 = (sgn * got["b"]).astype(np.float64), got["x"].astype(np.float64)
    ctl = [c.tolerance, c.relTol, c.maxIter, c.minIter]

    def ops_its(cap, shape=SHAPE):
        levels = amg.build(nbr, np.asarray(e.internal), aggs, w, cap)
        ops = amg.Ops(levels, d, off, nbr, np.float64, shape)
        return mm.fpcg(K, b, x0, ops.vcycle, ctl, mm.Work(len(levels) + 1))[0]

    lev = [mm.Level(K, part)]
    for agg in aggs:
        L = lev[-1]
        nc = int(agg.max()) + 1
        first = np.zeros(nc, np.int64)
        first[agg] = np.arange(L.n)
        L.agg = agg
        lev.append(mm.Level(mm.galerkin(L.A, agg, nc), L.core[first]))

    def model_its(cfg):
        W = mm.Work(len(lev))
        return mm.fpcg(K, b, x0, lambda r, cy=mm.Cycle(lev, mm.parse_cfg(cfg), W): cy(0, r), ctl, W)[0]

    model, weighted = model_its(CFG), model_its(WCFG)
    jacobi = mm.fpcg(K, b, x0, lambda r: r / K.diagonal(), ctl, mm.Work(1))[0]

    assert len(aggs) >= 3, [int(a.max()) + 1 for a in aggs]
    assert ops_its(None) == model, (ops_its(None), model)
    assert model * 4 < jacobi, (model, jacobi)
    assert ops_its(6) != model, "a coarse row cap of 6 must change the count, or this check cannot fail"
    assert ops_its(None, WSHAPE) == weighted < model, (ops_its(None, WSHAPE), weighted, model)
    # SPD AMLI(2) polynomial W-cycle with separate fine weight wf needs fewer iterations than the V-cycle
    amli_shape = dict(SHAPE, fine="sym", wf=1.6, w=1.0, ktype="amli", kf=1, r1=1.15, r2=2.4)
    assert ops_its(None, amli_shape) < model
    # agg=4 coarsens more aggressively than agg=3 (fewer level-1 rows)
    aggs4 = amg.matched(nbr, e.internal, w, part, passes=4, max_coarse=20)
    assert int(aggs4[0].max()) + 1 < int(aggs[0].max()) + 1

