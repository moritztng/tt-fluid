"""The whole device pipeline on the CPU: trace, lower, encode and run pitzDaily on the emulator of the device program
(the exact reference a device run is checked against), for one Blackhole chip and for two."""
import pickle
import sys
from pathlib import Path

import pytest

from ttfluid import cli

# initial residuals of iteration 3 with the stock tutorial dictionaries. One and two chips partition the mesh
# differently, so their fp32 reductions run in a different order and the residuals differ by ~1 %.
LAST = {1: {"U": 0.2666557729244232, "p": 0.19618935883045197, "omega": 0.0917070209980011, "k": 0.17153163254261017},
        2: {"U": 0.26454228162765503, "p": 0.19852200150489807, "omega": 0.09140697866678238, "k": 0.17088072001934052}}


@pytest.mark.parametrize("chips", [1, 2])
def test_emulator_pitzdaily(tmp_path, monkeypatch, chips):
    case = Path(__file__).resolve().parents[1] / "cases" / "pitzDaily"
    monkeypatch.setattr(sys, "argv", sys.argv[:1])
    with pytest.raises(SystemExit):
        cli.main(["check", str(case), "-n", "3", "--chips", str(chips), "--out", str(tmp_path / "r.jsonl"),
                  "--", "--ref", str(tmp_path / "ref.pkl")])
    history, state = pickle.load(open(tmp_path / "ref.pkl", "rb"))
    assert len(history) == 3
    for k, v in LAST[chips].items():
        assert history[-1][k] == pytest.approx(v, rel=1e-5)
    assert all(state[n].dtype.kind == "f" for n in state if hasattr(state[n], "dtype"))
