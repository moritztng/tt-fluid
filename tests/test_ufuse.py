"""--ufuse 1: the U components solved in one lockstep BiCGStab loop (resident.krylov.pbicgstab_fused) give bitwise the
emulator state of the one-after-another solve, with components leaving at different iterations. --ufuse 2 drops the
freeze of a component that has left: the same comparison must then fail (the check can fail)."""
import pickle
import sys
from pathlib import Path

import pytest

from ttfluid import cli


def _run(tmp_path, monkeypatch, name, extra):
    case = Path(__file__).resolve().parents[1] / "cases" / "pitzDaily"
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    ref = tmp_path / f"{name}.pkl"
    with pytest.raises(SystemExit):
        cli.main(["check", str(case), "-n", "3", "--chips", "2", "--out", str(tmp_path / "r.jsonl"),
                  "--", "--ref", str(ref), *extra])
    return pickle.load(open(ref, "rb"))


def _same(a, b):
    (ha, sa), (hb, sb) = a, b
    return ha == hb and sa.keys() == sb.keys() and all(
        sa[k].tobytes() == sb[k].tobytes() if hasattr(sa[k], "tobytes") else sa[k] == sb[k] for k in sa)


def test_fused_u_solve_is_bitwise(tmp_path, monkeypatch):
    base = _run(tmp_path, monkeypatch, "base", [])
    its = [(r["its0"], r["its1"]) for r in base[0]]
    assert any(a != b for a, b in its), "the components must leave at different iterations for the freeze to be tested"
    assert _same(base, _run(tmp_path, monkeypatch, "fused", ["--ufuse", "1"]))
    assert not _same(base, _run(tmp_path, monkeypatch, "nofreeze", ["--ufuse", "2"]))
