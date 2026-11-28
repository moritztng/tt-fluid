"""The drag gate (bench/cdgrade.py) passes a correct run, fails a wrong one, and knows its own resolution."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import cdgrade  # noqa: E402

REF = cdgrade.series(cdgrade.REF)
H100 = ROOT / "results/nvidia/dgx-h100"
QB2 = ROOT / "results/tenstorrent/quietbox2-p300x4"
GLX = ROOT / "results/tenstorrent/galaxy-blackhole"


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


def test_one_run_counts_for_600_iterations_however_long_and_base_reports_the_floor():
    """The mean of one run wanders over long windows, so 1,699 iterations get the band and the floor of 600."""
    run = cdgrade.series([f"{QB2}/qb2k2_D1A.hist.json:0", f"{QB2}/qb2k2_D2A.hist.json:800"])
    g = cdgrade.grade(run, REF, base=run)
    assert g["window"] == [201, 1899] and round(g["Cd"]["band"], 1) == 17.8
    assert g["Cd"]["shift"] == 0 and 11.5 < g["Cd"]["floor"] < 11.6


def test_nan_and_gaps():
    run = cdgrade.series([str(H100 / "A_DP_g6" / "coefficient.dat")])
    assert cdgrade.grade({**run, 500: (float("nan"), 0.0)}, REF)["grade"] == "FAIL"
    with pytest.raises(SystemExit):
        cdgrade.grade({i: v for i, v in run.items() if i != 500}, REF)


def test_state_file_reads_like_its_history(tmp_path):
    """run_mesh saves the history inside the final-state file; the gate reads it there."""
    np = pytest.importorskip("numpy")
    hist = QB2 / "qb2k2_D1A.hist.json"
    np.savez(tmp_path / "leg_state.npz", p=np.zeros(1, np.float32), hist=hist.read_text())
    assert cdgrade.series([f"{tmp_path}/leg_state.npz:300"]) == cdgrade.series([f"{hist}:300"])


def test_reordered_runs_track_and_other_changes_do_not():
    """Same binary on two GPUs differs only in reduction order; fp64 against mixed precision is a different solve."""
    a, b, dp = (cdgrade.series([str(H100 / d / "coefficient.dat")]) for d in ("A_SPDP_g1", "A_SPDP_g2", "A_DP_g2"))
    assert cdgrade.grade(a, REF, base=b)["reordered"] and not cdgrade.grade(a, REF, base=dp)["reordered"]
    wrong = {i: (cd * 1.001, cl) for i, (cd, cl) in b.items()}   # a 0.1 % force error is seen in the first iterations
    assert not cdgrade.grade(wrong, REF, base=b)["reordered"]
    assert "early" not in cdgrade.grade(a, REF, base={i: v for i, v in b.items() if i > cdgrade.TRACK_ITS})


@pytest.mark.parametrize("error, status", [(1.0, 0), (1.001, 1)])
def test_a_short_run_gets_the_reorder_check_alone(tmp_path, monkeypatch, error, status):
    """Six iterations decide whether a change is a reorder: no window, the exit status is the tracking check."""
    hist = json.loads((QB2 / "qb2k2_D1A.hist.json").read_text())[:6]
    short = tmp_path / "short.hist.json"
    short.write_text(json.dumps([{**r, "Cd_in": r["Cd_in"] * error} for r in hist]))
    monkeypatch.setattr(sys, "argv", ["cdgrade.py", "--run", f"{short}:0", "--base", f"{QB2}/qb2k2_D1A.hist.json:0"])
    with pytest.raises(SystemExit) as e:
        cdgrade.main()
    assert e.value.code == status


def test_runs_of_one_arithmetic_pool_into_one_grade(monkeypatch, capsys):
    """Eight SPUMA runs that differ only in reduction order grade as one 4,800-iteration window at the 12.5 bar."""
    spdp = [str(H100 / f"A_SPDP_g{g}" / "coefficient.dat") for g in range(8)]
    q = cdgrade.pooled([cdgrade.grade(cdgrade.series([f]), REF) for f in spdp])
    assert q["its"] == 4800 and q["Cd"]["band"] == cdgrade.BAND and q["grade"] == "PASS" and q["record"]
    assert abs(q["Cd"]["model"] - 20 / (48 - 7) ** 0.5) < 0.01 and q["Cd"]["sigma"] >= (q["Cd"]["model"] ** 2 + 3.5 ** 2) ** 0.5 - 0.05
    for error in (1.01, 0.99):   # the same eight runs with a 1 % force error must fail as a pool, in both directions
        wrong = [{i: (cd * error, cl) for i, (cd, cl) in cdgrade.series([f]).items()} for f in spdp]
        assert cdgrade.pooled([cdgrade.grade(w, REF) for w in wrong])["grade"] == "FAIL"
    two = [cdgrade.grade(cdgrade.series([f]), REF) for f in spdp[:2]]
    two[1]["Cd"]["diff"] = two[0]["Cd"]["diff"] + 40   # two runs that disagree: a wide band and a screen, not a record
    q = cdgrade.pooled(two)
    assert q["Cd"]["spread"] == pytest.approx(20) and q["Cd"]["band"] > 40 and not q["record"]
    monkeypatch.setattr(sys, "argv", ["cdgrade.py", "--run", spdp[0], "--pool", str(H100 / "A_DP_g0" / "coefficient.dat")])
    with pytest.raises(SystemExit) as e:   # fp64 is another arithmetic: it does not track, so it is not pooled
        cdgrade.main()
    assert "not pooled" in str(e.value.code)


def test_a_wrong_solve_fails():
    """SPUMA with the inlet 1 % and 0.3 % too fast (forces still normalised with the nominal speed): both must fail."""
    for f, counts in (("x1.010", 82.0), ("x1.003", 43.5)):
        g = cdgrade.grade(cdgrade.series([str(ROOT / "results/nvidia/h100-wrong-inlet" / f"{f}.dat")]), REF)
        assert g["grade"] == "FAIL" and round(g["Cd"]["diff"], 1) == counts and g["Cd"]["band"] < 18


def test_device_runs_from_starts_one_rounding_step_apart_pool_to_a_grade_of_record():
    """Seven 4-chip Galaxy runs whose start velocities differ by 1e-7: one arithmetic, pooled at the 12.5 bar."""
    runs = [cdgrade.series([f"{f}:0"]) for f in [GLX / "d_l0_b0.hist.json", *sorted(GLX.glob("glx1_*.hist.json"))]]
    early = [cdgrade.tracks(r, runs[0]) for r in runs[1:]]
    assert len(runs) == 7 and max(early) < 0.3   # they track through TRACK_ITS; by iteration 10 one is 18 counts away
    assert max(abs(r[i][0] - runs[0][i][0]) for r in runs[1:] for i in range(1, 11)) * 1e4 > 18
    q = cdgrade.pooled([cdgrade.grade(r, REF) for r in runs])
    assert q["record"] and q["grade"] == "PASS" and q["Cd"]["band"] == cdgrade.BAND and round(q["Cd"]["diff"], 1) == -3.3


def test_another_case_gets_means_and_no_grade(monkeypatch, capsys):
    """--means on the occDrivAer run: the window means against a chosen reference over the same iterations, no verdict."""
    occ = "occdrivaer-coarse"
    monkeypatch.setattr(sys, "argv", ["cdgrade.py", "--means", "--run", f"{ROOT}/results/tenstorrent/{occ}/dev.hist.json:0",
                                      "--ref", f"{ROOT}/results/openfoam/{occ}/serial.dat", "--from", "1001", "--to", "3999"])
    cdgrade.main()
    out = capsys.readouterr().out
    assert "diff +15.0 +- 4.6 counts" in out and "GRADE" not in out
