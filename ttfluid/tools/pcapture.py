"""Capture a case's pressure systems as the device solver sees them, for host solver studies (tools/mgmodel.py).

    python -m ttfluid.tools.pcapture CASE OUT --state LABEL=STATE.npz [--state ...]

The solver is built as run_mesh builds it (SimpleFoam, ell, fp32, face order). Each STATE (a run_mesh _state.npz or a
lower/of_state.py start state: the fields after some SIMPLE iteration; "-" keeps the case's own 0/ fields) is put into
the solver, one SIMPLE iteration runs up to the p solve, and the system handed to the Krylov solver is written to
OUT/sys_LABEL.npz: the matrix in CSR with the sign that makes it positive definite, b with the same sign, x0 and the
case's solver controls. OUT/mesh.npz holds what a partition needs: cell centres, faces per cell and the internal faces.
--geo adds OUT/sys_geo.npz: the p Laplacian at uniform rAU (face weights |Sf| deltaCoeffs), the matrix a multigrid
setup has when the program is lowered, before any coefficient exists.
"""
import argparse
import time
from pathlib import Path

import numpy as np

from ..solver import linsolve
from ..solver.simple import SimpleFoam
from ..resident.machine import face_counts

T0 = time.time()


def log(*a):
    print(f"{time.time() - T0:8.1f} s", *a, flush=True)


class _Got(Exception):
    pass


def inject(s, st):
    """A state into the solver, in the solver's own dtypes."""
    for f in s.fields():
        f.internal[...] = st[f.name]
        for k, b in f.bc.items():
            b.value[...] = st[f"{f.name}.{k}"]
    s.phi = np.array(st["phi"], dtype=s.phi.dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case")
    ap.add_argument("out")
    ap.add_argument("--state", action="append", default=[], help="LABEL=STATE.npz, repeatable")
    ap.add_argument("--geo", action="store_true", help="also write sys_geo.npz, the Laplacian of the face weights")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = SimpleFoam(a.case, backend="ell", dtype=np.float32, log=lambda *x: None)
    m = s.mesh
    log("SimpleFoam", m.nCells, "cells")
    nI = m.nInternalFaces
    np.savez(out / "mesh.npz", C=m.C.astype(np.float32), counts=face_counts(m).astype(np.int16),
             owner=m.owner[:nI].astype(np.int32), neighbour=m.neighbour.astype(np.int32))
    if a.geo:
        import scipy.sparse as sp
        e, n, nbr = s.ops.e, m.nCells, np.asarray(s.ops.cnbr)
        w = np.where(e.internal, np.asarray(e.magSf, np.float64) * np.asarray(e.dc, np.float64), 0.0)
        k = np.asarray(e.internal, bool).ravel()
        i = np.repeat(np.arange(n), nbr.shape[1])[k]
        K = sp.csr_matrix((np.r_[w.sum(axis=1) + 1e-12 * w.max(), -w.ravel()[k]],
                           (np.r_[np.arange(n), i], np.r_[np.arange(n), nbr.ravel()[k]])), shape=(n, n))
        K.sum_duplicates()
        K.sort_indices()
        x = np.random.default_rng(0).standard_normal(n)
        np.savez(out / "sys_geo.npz", indptr=K.indptr.astype(np.int64), indices=K.indices.astype(np.int32),
                 data=K.data.astype(np.float32), b=(K @ x).astype(np.float32), x0=np.zeros(n, np.float32),
                 ctl=np.array([1e-7, 0.01, 1000, 0], np.float64))
        log(f"saved sys_geo: n {n} nnz {K.nnz}")
    cur = {}

    def krylov(symmetric):
        if not symmetric:
            return linsolve.pbicgstab

        def run(A, x, b, c):
            K = linsolve.to_csr(A)
            sgn = -1.0 if A.d.sum() < 0 else 1.0          # laplacian(rAU, p) is negative definite
            np.savez(out / f"sys_{cur['label']}.npz", indptr=K.indptr.astype(np.int64), indices=K.indices.astype(np.int32),
                     data=(sgn * K.data).astype(np.float32), b=(sgn * b).astype(np.float32), x0=x.astype(np.float32),
                     ctl=np.array([c.tolerance, c.relTol, c.maxIter, c.minIter], np.float64))
            log(f"saved sys_{cur['label']}: n {K.shape[0]} nnz {K.nnz} tol {c.tolerance} relTol {c.relTol}")
            raise _Got
        return run

    s.ops.krylov = krylov
    for label, path in (x.split("=", 1) for x in a.state):
        if path != "-":
            with np.load(path) as z:
                inject(s, z)
        cur["label"] = label
        try:
            s.iterate()
        except _Got:
            pass
    log("done")


if __name__ == "__main__":
    main()
