"""The whole device pipeline on the CPU: trace, lower, encode and run pitzDaily on the emulator of the device program
(the exact reference a device run is checked against), for one Blackhole chip, two and a 2x2 block of four."""
import pickle
import sys
from pathlib import Path

import pytest

from ttfluid import cli

# initial residuals of iteration 3 with the stock tutorial dictionaries. Each chip count partitions the mesh
# differently, so their fp32 reductions run in a different order and the residuals differ by ~1 %.
LAST = {1: {"U": 0.2670932710170746, "p": 0.1979142725467682, "omega": 0.09137128293514252, "k": 0.1712660789489746},
        2: {"U": 0.2654341161251068, "p": 0.19957421720027924, "omega": 0.09150776267051697, "k": 0.17183400690555573},
        4: {"U": 0.26470062136650085, "p": 0.19779403507709503, "omega": 0.09139959514141083, "k": 0.1714317947626114}}


@pytest.mark.parametrize("chips", [1, 2, 4])
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
