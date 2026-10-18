import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ref():
    """OpenFOAM v2412 simpleFoam on the pitzDaily tutorial, every linear solve to 1e-13, iterations 1-3 (internal fields)."""
    return np.load(DATA / "pitzDaily_v2412.npz")


@pytest.fixture
def pitz(tmp_path):
    """A copy of cases/pitzDaily with the reference run's fvSolution (solves to 1e-13) and a 3-iteration controlDict."""
    case = tmp_path / "pitzDaily"
    shutil.copytree(ROOT / "cases" / "pitzDaily", case)
    shutil.copy(DATA / "fvSolution.tight", case / "system" / "fvSolution")
    shutil.copy(DATA / "controlDict.3its", case / "system" / "controlDict")
    return case
