"""The fp64 reference solver against OpenFOAM v2412 on pitzDaily: geometry, wall distance and SIMPLEC iterations."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from ttfluid.solver.mesh import Mesh
from ttfluid.solver.simple import SimpleFoam

DATA = Path(__file__).parent / "data"


def maxrel(x, ref):
    return np.abs(x - ref).max() / np.abs(ref).max()


def test_mesh_geometry(pitz, ref):
    m = Mesh(pitz)
    assert maxrel(m.C, ref["mesh/C"]) < 1e-12
    assert maxrel(m.V, ref["mesh/V"]) < 1e-12


@pytest.mark.parametrize("backend", ["face", "ell"])
def test_laminar_simplec(pitz, ref, backend):
    s = SimpleFoam(pitz, turbulence=False, backend=backend, log=lambda *a, **k: None)
    for it in (1, 2, 3):
        s.iterate()
        assert maxrel(s.U.internal, ref[f"lam/{it}/U"]) < 1e-9
        assert maxrel(s.p.internal, ref[f"lam/{it}/p"]) < 1e-9


def test_wall_distance(pitz, ref):
    s = SimpleFoam(pitz, log=lambda *a, **k: None)
    assert maxrel(s.turb.y, ref["mesh/wallDistance"]) < 1e-5


@pytest.mark.parametrize("backend", ["face", "ell"])
def test_sst_from_openfoam_state(pitz, ref, backend):
    """k-omega SST SIMPLEC iterations 2 and 3 started from OpenFOAM's own iteration-1 fields (tests/data/sst_t1)."""
    for f in (DATA / "sst_t1").iterdir():
        shutil.copy(f, pitz / "0" / f.name)
    s = SimpleFoam(pitz, backend=backend, log=lambda *a, **k: None)
    for it in (2, 3):
        s.iterate()
        for name, x in (("U", s.U.internal), ("p", s.p.internal), ("k", s.turb.k.internal),
                        ("omega", s.turb.omega.internal), ("nut", s.turb.nut.internal)):
            assert maxrel(x, ref[f"tight/{it}/{name}"]) < 1e-8, (it, name)
