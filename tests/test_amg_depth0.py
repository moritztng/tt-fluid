"""--amg-depth 0 keeps the multigrid preconditioner: an empty level list must not fall back to Jacobi PCG."""

from ttfluid.resident import krylov, program  # noqa: E402


def test_empty_sa_levels_take_amg_pcg(monkeypatch):
    calls = []
    monkeypatch.setattr(krylov, "amg_pcg", lambda A, x, b, c, levels: calls.append(("amg", levels)) or "perf")
    monkeypatch.setattr(krylov, "pcg", lambda A, x, b, c: calls.append(("pcg", None)) or "perf")
    ops = object.__new__(program.TracedOps)
    ops.amg_levels, ops.perfs = program.SALevels(), []
    ops.krylov(True)(None, None, None, None)
    assert calls == [("amg", ops.amg_levels)]


def test_no_amg_takes_jacobi_pcg(monkeypatch):
    calls = []
    monkeypatch.setattr(krylov, "pcg", lambda A, x, b, c: calls.append("pcg") or "perf")
    ops = object.__new__(program.TracedOps)
    ops.perfs = []
    ops.krylov(True)(None, None, None, None)
    assert calls == ["pcg"]
