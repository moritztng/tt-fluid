"""--amg-depth 0 builds no coarse level, so the pressure solve is Jacobi PCG, never a lone Chebyshev smoother."""

import inspect

from ttfluid.resident import krylov, program  # noqa: E402


def test_no_amg_takes_jacobi_pcg(monkeypatch):
    calls = []
    monkeypatch.setattr(krylov, "pcg", lambda A, x, b, c: calls.append("pcg") or "perf")
    ops = object.__new__(program.TracedOps)
    ops.perfs = []
    ops.krylov(True)(None, None, None, None)
    assert calls == ["pcg"]


def test_depth0_attaches_no_levels():
    src = inspect.getsource(program.build)
    assert "amg_owner = None" in src.split("if amg_depth == 0:")[1].split("\n")[1]
