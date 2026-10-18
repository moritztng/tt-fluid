"""A run's starting state from an OpenFOAM time directory, in Image.state()'s layout, for run_mesh --state-in: a device
run restarted from fields (its own last state, or OpenFOAM's at iteration N) without lowering the case again.

    of_state.py IMAGES CASE TIMEDIR OUT.npz [--subs DIR] [--jobs J]

The solver is built exactly as the lowering builds it (SimpleFoam, ell, fp32), with every boundary entry from CASE/0 and
only the values from TIMEDIR (internalField and each patch's value; phi from TIMEDIR/phi when it is there, as
createPhi), so the patch merging and the BC types are the lowering's. The state is what program.build wraps: every
field's cells and patch values, and phi.

--subs: the case's prep.py sub-cases (a 65M case does not fit one SimpleFoam): one sub-solver per chip on its sub-mesh,
with the global ELL width, and each chip's owned elements taken into the global arrays (the SPMD lowering's rule:
owned values are exact, halo values are never read). Without it, one SimpleFoam on the whole case.
"""
import argparse
import copy
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from ttfluid.solver import ell as ell_mod  # noqa: E402
from ttfluid.solver import foamio, simple, turbulence  # noqa: E402

FIELDS = ("U", "p", "k", "omega", "nut", "phi")
T0 = time.time()


def log(*a):
    print(f"{time.time() - T0:7.1f} s", *a, flush=True)


def merged_fields(case, tdir):
    """read_field dicts: CASE/0's entries with TIMEDIR's values."""
    out = {}
    for f in FIELDS:
        c, t = case / "0" / f, tdir / f
        if not t.exists():
            assert f == "phi" or not c.exists(), (f, "missing in", tdir)
            continue
        dt = foamio.read_field(t)
        if not c.exists():
            out[f] = dt
            continue
        d = copy.copy(foamio.read_field(c))
        d["internal"] = dt["internal"]
        b = foamio.FoamDict()    # keeps the quoted (regex) keys: VolField.read takes them from .patterns ("side.*")
        b.patterns = list(getattr(d["boundary"], "patterns", []))
        for name, e in d["boundary"].items():
            e = dict(e)
            if "value" in dt["boundary"].get(name, {}):   # a coupled patch (cyclicAMI) is written without one
                e["value"] = dt["boundary"][name]["value"]
            b[name] = e
        d["boundary"] = b
        out[f] = d
    return out


def case_view(case, names, d):
    """CASE's system and constant with a 0/ naming exactly the fields read (SimpleFoam reads 0/phi only if it is
    there); their contents come from install(), never from these files."""
    (d / "0").mkdir(parents=True, exist_ok=True)
    for x in ("system", "constant"):
        if not (d / x).exists():
            (d / x).symlink_to((case / x).resolve())
    for f in FIELDS:
        (d / "0" / f).unlink(missing_ok=True)
        if f in names:
            (d / "0" / f).write_text("// of_state.py: read through install()\n")
    return d


def solver_state(case):
    s = simple.SimpleFoam(case, backend="ell", dtype=np.float32, log=lambda *x: None, ell_order="face")
    st = {}
    for f in s.fields():
        st[f.name] = np.asarray(f.internal, np.float32)
        for k, b in f.bc.items():
            st[f"{f.name}.{k}"] = np.asarray(b.value, np.float32)
    st["phi"] = np.asarray(s.phi, np.float32)
    return st


def install(fields):
    foamio.read_field = lambda p: copy.deepcopy(fields[Path(p).name])


G = {}   # the forked workers' shared inputs
ORIG = (simple.Mesh, turbulence.WallDistance, ell_mod.max_count)   # a worker runs many chips: unpatch before each load


def chip_state(k):
    import pickle
    from ttfluid.lower.prep import restrict_field
    simple.Mesh, turbulence.WallDistance, ell_mod.max_count = ORIG
    with open(G["subs"] / f"sub{k}.pkl", "rb") as f:
        sub = pickle.load(f)
    s = sub["mesh"]
    nI = G["nI"]
    install({f: restrict_field(d, sub["g"], s, nI) for f, d in G["fields"].items()})
    simple.Mesh = lambda c: s

    def wall_distance(mesh):
        wd = sub["wd"]
        wd.mesh = mesh
        return wd
    turbulence.WallDistance = wall_distance
    ell_mod.max_count = lambda counts: G["S"]
    st = solver_state(G["case"])
    own = sub["owner"] // sub["cores"] == k
    sg = sub["space_g"]
    out = {}
    for name, v in st.items():
        sp = "cells" if "." not in name else "patch:" + name.split(".", 1)[1]
        mine = own if sp == "cells" else own[s.patch[name.split(".", 1)[1]].faceCells]
        out[name] = (sg[sp][mine], v[mine])
    log("chip %d: %d owned cells" % (k, int(own.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("case")
    ap.add_argument("tdir")
    ap.add_argument("out")
    ap.add_argument("--subs")
    ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    from ttfluid.lower.image import Image
    img = Image.load(a.images)
    layout = {name: (shape, regs) for name, shape, regs in img.shared["state"]}
    n_of = img.shared["spaces"]
    case, tdir = Path(a.case), Path(a.tdir)
    fields = merged_fields(case, tdir)
    log("fields", list(fields), "from", tdir)
    view = case_view(case, list(fields), Path(a.out).with_suffix(".view"))
    if not a.subs:
        install(fields)
        st = solver_state(view)
    else:
        G.update(subs=Path(a.subs), case=view, fields=fields, S=len(layout["phi"][1]),
                 nI=len(fields["phi"]["internal"]) if "phi" in fields else None)
        st = {}
        for name, (shape, regs) in layout.items():
            st[name] = np.zeros((n_of[regs[0][0]],) + tuple(shape), np.float32)
        K = img.s["K"]
        with mp.get_context("fork").Pool(a.jobs) as pool:
            for part in pool.imap_unordered(chip_state, range(K)):
                for name, (ids, v) in part.items():
                    st[name][ids] = v
    miss = sorted(set(layout) - set(st))
    assert not miss, ("state names the solver did not produce", miss)
    for name, (shape, regs) in layout.items():
        assert st[name].shape == (n_of[regs[0][0]],) + tuple(shape), (name, st[name].shape, shape, n_of[regs[0][0]])
    np.savez(a.out, **{k: st[k] for k in layout})
    log("wrote", a.out, len(layout), "state arrays")


if __name__ == "__main__":
    main()
