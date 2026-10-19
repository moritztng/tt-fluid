"""The real HPC_motorbike Small and occDrivAer dictionaries parse as OpenFOAM v2412 reads them.

Truth (tests/data/case_dicts/<case>/), made with OpenFOAM v2412:
  expanded/<f>  `foamDictionary -expand` of each raw dict (includes, #includeEtc, $macros, #eval resolved)
  types.json    per-patch boundary type of U p k omega nut after GeometricBoundaryField::readField: the raw 0/ fields
                decomposed (decomposePar, 1 domain) on a 1-cell mesh carrying the case's real patch list and groups
raw/ holds the case dicts unmodified (polyMesh/boundary only, no mesh).
"""
import json
from pathlib import Path

import pytest

from ttfluid.solver import foamio  # noqa: E402
from ttfluid.solver.fields import _entry_key  # noqa: E402
from ttfluid.solver.mesh import Patch  # noqa: E402

DATA = Path(__file__).parent / "data" / "case_dicts"
CASES = {"small": "0.org", "occ": "0.orig"}
FIELDS = ("U", "p", "k", "omega", "nut")
# internalField as decomposePar wrote it (full precision; foamDictionary prints 6 digits): the #eval results
INTERNAL = {("occ", "k"): 0.01533527281494, ("occ", "omega"): 203.5205416714001, ("occ", "nut"): 7.535e-05}
SYSTEM = ("system/fvSolution", "system/fvSchemes", "system/controlDict", "constant/transportProperties",
          "constant/turbulenceProperties")


def _norm(v):
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    out = []
    for t in v:
        try:
            out.append(float(t))
        except ValueError:
            out.append(t.strip('"'))
    return out


def _same(a, b, path=""):
    if isinstance(a, dict) or isinstance(b, dict):
        assert isinstance(a, dict) and isinstance(b, dict), path
        assert list(a) == list(b) or set(a) == set(b), f"{path}: keys {sorted(set(a) ^ set(b))}"
        for k in a:
            _same(a[k], b[k], f"{path}/{k}")
        return
    assert len(a) == len(b), f"{path}: {a} vs {b}"
    for x, y in zip(a, b):
        if isinstance(x, float) and isinstance(y, float):
            assert x == pytest.approx(y, rel=1e-12, abs=0), f"{path}: {a} vs {b}"
        else:
            assert x == y, f"{path}: {a} vs {b}"


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("field", FIELDS)
def test_field_dict_matches_foamDictionary_expand(case, field):
    raw = foamio.read_dict(DATA / case / "raw" / CASES[case] / field, case=DATA / case / "raw")
    ref = foamio.read_dict(DATA / case / "expanded" / field)
    if (case, field) in INTERNAL:
        assert float(raw["internalField"][1]) == pytest.approx(INTERNAL[case, field], rel=1e-14)
        ref["internalField"] = raw["internalField"]
        for e in ref["boundaryField"].values():     # value $internalField, printed at 6 digits
            if e.get("value") == ["uniform", "%g" % INTERNAL[case, field]]:
                e["value"] = raw["internalField"]
    for k in ("dimensions", "internalField", "boundaryField"):
        _same(_norm(raw[k]), _norm(ref[k]), f"{case}/{field}/{k}")
    assert raw["boundaryField"].patterns == ref["boundaryField"].patterns


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", SYSTEM)
def test_case_dict_matches_foamDictionary_expand(case, name):
    raw = foamio.read_dict(DATA / case / "raw" / name, case=DATA / case / "raw")
    ref = foamio.read_dict(DATA / case / "expanded" / Path(name).name)
    for d in (raw, ref):
        d.pop("functions", None)      # function objects (#includeFunc) are not part of the solve
    _same(_norm(raw), _norm(ref), f"{case}/{name}")


@pytest.mark.parametrize("case", CASES)
def test_patch_bc_resolution_matches_openfoam(case):
    patches = [Patch(n, d["type"], d["startFace"], d["nFaces"], dict=d)
               for n, d in foamio.read_boundary(DATA / case / "raw/constant/polyMesh/boundary")]
    truth = json.loads((DATA / case / "types.json").read_text())
    n = 0
    for f in FIELDS:
        fld = foamio.read_field(DATA / case / "raw" / CASES[case] / f)
        for p in patches:
            if p.type == "empty":
                continue
            e = fld["boundary"][_entry_key(fld["boundary"], p)]
            assert e["type"] == truth[f][p.name], f"{case} {f} {p.name}: {e['type']} vs {truth[f][p.name]}"
            n += 1
    assert n == len(FIELDS) * len([p for p in patches if p.type != "empty"])


def test_verbatim_code_block():
    """#{ ... #} is one verbatim token (coded function objects), braces and ';' inside do not leak into the dict."""
    from ttfluid.solver.foamio import parse_dict_text
    d = parse_dict_text('f { a { type coded; codeData #{ int x; struct s { int y; }; #}; } b { type forces; n 2; } }')
    assert list(d["f"]) == ["a", "b"] and d["f"]["b"]["n"] == ["2"]
    assert d["f"]["a"]["codeData"] == ["#{ int x; struct s { int y; }; #}"]


def test_refuses_other_turbulence_models(tmp_path):
    """A case that asks for a model tt-fluid does not implement is refused, not silently run as k-omega SST."""
    import shutil

    from ttfluid.solver.simple import SimpleFoam
    case = tmp_path / "pitzDaily"
    shutil.copytree(Path(__file__).parents[1] / "cases" / "pitzDaily", case)
    tp = case / "constant" / "turbulenceProperties"
    tp.write_text(tp.read_text().replace("kOmegaSST", "kEpsilon"))
    with pytest.raises(ValueError, match="kEpsilon"):
        SimpleFoam(case)
