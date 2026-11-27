"""The drag gate (bench/cdgrade.py) passes a correct run, fails a wrong one, and knows its own resolution."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import cdgrade  # noqa: E402

REF = cdgrade.series([str(ROOT / f) for f in cdgrade.REF])
H100 = ROOT / "results/nvidia/dgx-h100"
QB2 = ROOT / "results/tenstorrent/quietbox2-p300x4"


def test_correct_runs_pass():
    """Sixteen SPUMA runs of the same case: a 2 sigma band may fail one or two of them, not more."""
    grades = [cdgrade.grade(cdgrade.series([str(d / "coefficient.dat")]), REF) for d in sorted(H100.glob("A_*_g?"))]
    assert len(grades) == 16 and sum(g["grade"] == "FAIL" for g in grades) <= 2
    assert all(g["window"] == [201, 800] and round(g["Cd"]["band"], 1) == 17.8 for g in grades)


@pytest.mark.parametrize("error, grade", [(1.00, "PASS"), (1.01, "FAIL"), (0.99, "FAIL")])
def test_one_percent_error_fails(error, grade):
    """A 1 % error in the force (42 counts) fails on an 800-iteration run in both directions."""
    run = cdgrade.series([str(H100 / "A_DP_g6" / "coefficient.dat")])
    g = cdgrade.grade({i: (cd * error, cl) for i, (cd, cl) in run.items()}, REF)
    assert g["grade"] == grade and g["Cd"]["detects"] < 42


def test_band_tightens_to_the_bar_and_base_reports_the_floor():
    run = cdgrade.series([f"{QB2}/qb2k2_D1A.hist.json:0", f"{QB2}/qb2k2_D2A.hist.json:800"])
    g = cdgrade.grade(run, REF, base=run)
    assert g["window"] == [201, 1899] and g["Cd"]["band"] == cdgrade.BAND
    assert g["Cd"]["shift"] == 0 and 6.8 < g["Cd"]["floor"] < 6.9


def test_nan_and_gaps():
    run = cdgrade.series([str(H100 / "A_DP_g6" / "coefficient.dat")])
    assert cdgrade.grade({**run, 500: (float("nan"), 0.0)}, REF)["grade"] == "FAIL"
    with pytest.raises(SystemExit):
        cdgrade.grade({i: v for i, v in run.items() if i != 500}, REF)
