"""A run that diverges ends in the iteration it diverges in, and omega's floor follows the precision of the arithmetic.

Where these come from: Small on a 2x2 Blackhole block went NaN in SIMPLE iteration 337 of one start state and then ran
every solve of the 463 iterations left to its maxIter, 1,639 s at five times the healthy pace that looked like a hang.
A comparison with NaN is false, so no exit test of any loop was ever true again."""
import shutil
from pathlib import Path

import numpy as np

from ttfluid.resident.machine import Machine
from ttfluid.resident.memu import MachineEmu
from ttfluid.resident.program import build
from ttfluid.solver.simple import SimpleFoam

CASE = Path(__file__).resolve().parents[1] / "cases" / "pitzDaily"


def _history(n, poison):
    """pitzDaily on the emulator of the device program (16 cores, fp32) for at most n outer iterations."""
    s = SimpleFoam(str(CASE), backend="ell", dtype=np.float32, log=lambda *a: None)
    if poison:
        s.turb.k.internal[s.mesh.nCells // 2] = np.nan
    me = MachineEmu(Machine(build(s, n), s.mesh, 16, log=lambda *a: None))
    me.run()
    return me.history


def test_diverged_run_ends_in_that_iteration():
    h = _history(4, poison=True)
    assert not all(np.isfinite(v) for v in h[0].values())
    assert len(h) == 1, "a NaN iteration must be the last one, not %d more at every maxIter: %s" % (len(h) - 1, h[-1])


def test_finite_run_keeps_every_iteration():
    h = _history(3, poison=False)   # U and p start at residual exactly 1: the exit must not read that as diverged
    assert len(h) == 3 and all(np.isfinite(v) for r in h for v in r.values())


def test_omega_floor_follows_precision(tmp_path):
    """RASModel's kMin and omegaMin default to SMALL, which is 1e-15 in double and 1e-6 in single precision. The step
    divides by omega twice, so a cell left at a 1e-15 floor comes back near 1e18 and the next solve's dot products
    leave fp32; k is in no denominator and keeps 1e-15. The RAS dictionary overrides both, as in OpenFOAM."""
    for dtype, floor in ((np.float64, 1e-15), (np.float32, 1e-6)):
        t = SimpleFoam(str(CASE), backend="ell", dtype=dtype, log=lambda *a: None).turb
        assert (t.kMin, t.omegaMin) == (1e-15, floor)
    case = tmp_path / "pitzDaily"
    shutil.copytree(CASE, case)
    f = case / "constant" / "turbulenceProperties"
    f.write_text(f.read_text().replace("RASModel        kOmegaSST;",
                                       "RASModel        kOmegaSST;\n    omegaMin        1e-10;\n    kMin            kMin [0 2 -2 0 0] 1e-12;"))
    t = SimpleFoam(str(case), backend="ell", dtype=np.float32, log=lambda *a: None).turb
    assert (t.kMin, t.omegaMin) == (1e-12, 1e-10)
