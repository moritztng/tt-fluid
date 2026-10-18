"""One-time per case and partition: split a case into per-chip sub-cases for the SPMD lowering (spmd.py).

    prep.py CASE OUT --chips K [--cores 72] [--owner OWNER.npz] [--layers 2] [--chip-line]

Reads the mesh, the initial fields and the wall distance ONCE for the whole case (the expensive, mesh-only part of
SimpleFoam's setup; none of it depends on the tree or the lowering flags), takes the core of every cell from the same
partition run_mesh uses (rcb over cell centres, chip strips first with --chip-line, or an owner npz), and writes
OUT/sub<k>.pkl per chip k: the chip's cells plus `layers` face-neighbour layers, in global order, as a Mesh whose every
array is a restriction of the global one (nothing is recomputed, so every value is bitwise the global value), the
fields and wall distance restricted the same way, and the global core of every cell.
Faces between a kept cell and a dropped one are dropped: only halo cells lose faces, and nothing an owned cell
lowers to reads a halo cell's own face sums.
"""
import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np

from ttfluid.solver import foamio  # noqa: E402
from ttfluid.solver.fields import group_runs  # noqa: E402
from ttfluid.solver.mesh import Mesh  # noqa: E402

FIELDS = ("U", "p", "k", "omega", "nut", "phi")
FACE_ARRAYS = ("Sf", "Cf", "magSf", "weights", "delta", "deltaCoeffs", "nonOrthDeltaCoeffs", "nonOrthCorrectionVectors")
PATCH_ARRAYS = ("faceCells", "Sf", "magSf", "nf", "Cf", "delta", "deltaCoeffs")


def log(*a):
    print("%7.1f s" % (time.time() - T0), *a, flush=True)


def owners(m, P, K, chip_line, owner_npz):
    if owner_npz:
        return np.load(owner_npz)["parts"].astype(np.int64)
    from ttfluid.resident.machine import rcb
    return rcb(m.C, P, chips=K if chip_line else None)


def submesh(m, g, loc):
    """Mesh over global cells g (sorted); loc: global cell -> local index or -1."""
    nI = m.nInternalFaces
    o, n = loc[m.owner[:nI]], loc[m.neighbour]
    fi = np.nonzero((o >= 0) & (n >= 0))[0]
    s = Mesh.__new__(Mesh)
    s.case, s.merged = m.case, {}
    s.points = s.face_offsets = s.face_points = s.allCf = s.allSf = None
    s.nCells = len(g)
    s.C, s.V = m.C[g], m.V[g]
    s.solutionD = m.solutionD
    for k in FACE_ARRAYS:
        setattr(s, k, getattr(m, k)[fi])
    s.nInternalFaces = len(fi)
    owner, s.face_keep, s.patch_keep = [o[fi]], {"internal": fi}, {}
    s.patches, start = [], len(fi)
    for q in m.patches:
        for p in (q.parts or [q]):      # the sub-case is unmerged: SimpleFoam merges it again, the same runs
            keep = np.nonzero(loc[p.faceCells] >= 0)[0]
            sp = copy.copy(p)
            sp.start, sp.size, sp.parts = start, len(keep), None
            for k in PATCH_ARRAYS:
                setattr(sp, k, getattr(p, k)[keep])
            sp.faceCells = loc[p.faceCells[keep]]
            s.patches.append(sp)
            s.patch_keep[p.name] = keep
            owner.append(sp.faceCells)
            start += len(keep)
    s.patch = {p.name: p for p in s.patches}
    s.owner = np.concatenate(owner)
    s.neighbour = n[fi]
    s.nFaces = len(s.owner)
    return s


def restrict_field(d, g, s, nI_global):
    """A read_field dict on the sub-case: cell arrays by g, internal face arrays (phi) by the kept faces, patch
    value arrays by each patch's kept faces."""
    d = copy.copy(d)
    x = d["internal"]
    if np.ndim(x) and len(x) == nI_global and d.get("cls", "").startswith("surface"):
        d["internal"] = np.asarray(x)[s.face_keep["internal"]]
    elif np.ndim(x) and (d["width"] == 1 or np.ndim(x) == 2):
        d["internal"] = np.asarray(x)[g]
    b = copy.copy(d["boundary"])
    for name in list(b.keys()):
        e = dict(b[name])
        keep = s.patch_keep.get(name)
        for k, v in e.items():
            if keep is not None and isinstance(v, np.ndarray) and v.ndim and len(v) == s.patch_size_global[name]:
                e[k] = v[keep]
        b[name] = e
    d["boundary"] = b
    return d


def restrict_wd(wd, g, s, merged_keep):
    w = copy.copy(wd)
    w.mesh = None
    for k, v in vars(wd).items():
        if isinstance(v, np.ndarray) and v.ndim and len(v) == len(wd.mesh.C):
            setattr(w, k, v[g])
        elif isinstance(v, dict) and v and all(n in merged_keep for n in v):
            setattr(w, k, {n: (np.asarray(a)[merged_keep[n]] if np.ndim(a) else a) for n, a in v.items()})
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--chips", type=int, required=True)
    ap.add_argument("--cores", type=int, default=72)
    ap.add_argument("--owner", help="npz with key parts: global core of every cell (drivaer mesh_owner.py / part2d)")
    ap.add_argument("--chip-line", action="store_true", help="rcb in chip strips first (run_mesh with a 1 x K mesh or --strips)")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--sliced", action="store_true", help="lanes sorted widest cell first (lowering with --sliced)")
    ap.add_argument("--quarter", type=int, default=0, help="with --sliced: quarter-interleave each run of equal face count "
                    "in stretches of 4Q lanes (layout rcbq<Q>), so index neighbours are a 4-lane shift, not NCRISC copies")
    ap.add_argument("--turbulence", type=int, default=1)
    a = ap.parse_args()
    K, P = a.chips, a.chips * a.cores
    case, out = Path(a.case), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = Mesh(case)
    log("mesh", m.nCells, "cells", m.nFaces, "faces")
    raw = {f: foamio.read_field(case / "0" / f) for f in FIELDS if (case / "0" / f).exists()}
    log("fields", list(raw))
    patch_size = {p.name: p.size for p in m.patches}
    gm = copy.copy(m)                       # merged as SimpleFoam merges it, for the wall distance's patch keys
    gm.patches, gm.patch, gm.merged = list(m.patches), dict(m.patch), {}
    gm.merge_patches(group_runs(gm, [raw[f]["boundary"] for f in ("U", "p", "k", "omega", "nut") if f in raw]))
    wd = None
    if a.turbulence and "k" in raw:
        from ttfluid.solver.turbulence import WallDistance
        wd = WallDistance(gm)
        log("wall distance")
    own = owners(m, P, K, a.chip_line, a.owner)
    log("owners", np.bincount(own // a.cores, minlength=K).tolist())
    # every cell's lane on its core (Layout: rank among the core's cells in cell order), so a chip knows the lanes of
    # the halo cells it receives (Maps orders a halo by the sender's lanes)
    order = np.argsort(own, kind="stable")
    cnt = np.bincount(own, minlength=P)
    lane = np.zeros(m.nCells, np.int64)
    lane[order] = np.arange(m.nCells) - (np.cumsum(cnt) - cnt)[own[order]]
    if a.sliced:                            # run_mesh --sliced: widest cells first on each core
        from ttfluid.resident.machine import face_counts, sliced_lanes
        lane = sliced_lanes(own, face_counts(m), P, a.quarter)
    nI = m.nInternalFaces
    for k in range(K):
        inside = own // a.cores == k
        for _ in range(a.layers):
            grow = inside.copy()
            o, n = m.owner[:nI], m.neighbour
            grow[o[inside[n]]] = True
            grow[n[inside[o]]] = True
            inside = grow
        g = np.nonzero(inside)[0]
        loc = np.full(m.nCells, -1, np.int64)
        loc[g] = np.arange(len(g))
        s = submesh(m, g, loc)
        s.patch_size_global = patch_size
        merged_keep = {q.name: np.nonzero(loc[q.faceCells] >= 0)[0] for q in gm.patches}
        sub = dict(mesh=s, fields={f: restrict_field(d, g, s, nI) for f, d in raw.items()},
                   wd=restrict_wd(wd, g, s, merged_keep) if wd is not None else None,
                   g=g, owner=own[g], lane=lane[g], chip=k, K=K, cores=a.cores, nCells=m.nCells,
                   space_n={**{"cells": m.nCells}, **{"patch:" + q.name: q.size for q in gm.patches}},
                   space_g={**{"cells": g}, **{"patch:" + n: v for n, v in merged_keep.items()}})
        with open(out / f"sub{k}.pkl", "wb") as f:
            pickle.dump(sub, f, protocol=5)
        log("chip %d: %d owned + %d halo cells, %d faces" % (k, int(np.sum(own[g] // a.cores == k)),
                                                             int(np.sum(own[g] // a.cores != k)), s.nFaces))


T0 = time.time()
if __name__ == "__main__":
    main()
