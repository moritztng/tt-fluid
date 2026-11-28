"""The device-side level tables (amg.build, amg.Ops) run the cycle the host model prices: on the first p system of
pitzDaily both need the same number of PCG iterations, and a capped coarse row width must change that number."""
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

    def ops_its(cap):
        levels = amg.build(nbr, np.asarray(e.internal), aggs, w, cap)
        ops = amg.Ops(levels, d, off, nbr, np.float64, SHAPE)
        return mm.fpcg(K, b, x0, ops.vcycle, ctl, mm.Work(len(levels) + 1))[0]

    lev = [mm.Level(K, part)]
    for agg in aggs:
        L = lev[-1]
        nc = int(agg.max()) + 1
        first = np.zeros(nc, np.int64)
        first[agg] = np.arange(L.n)
        L.agg = agg
        lev.append(mm.Level(mm.galerkin(L.A, agg, nc), L.core[first]))
    W = mm.Work(len(lev))
    model = mm.fpcg(K, b, x0, lambda r, cy=mm.Cycle(lev, mm.parse_cfg(CFG), W): cy(0, r), ctl, W)[0]
    jacobi = mm.fpcg(K, b, x0, lambda r: r / K.diagonal(), ctl, mm.Work(1))[0]

    assert len(aggs) >= 3, [int(a.max()) + 1 for a in aggs]
    assert ops_its(None) == model, (ops_its(None), model)
    assert model * 4 < jacobi, (model, jacobi)
    assert ops_its(6) != model, "a coarse row cap of 6 must change the count, or this check cannot fail"
