"""Finite-volume mesh: OpenFOAM polyMesh topology plus the geometry OpenFOAM derives from it.

Geometry follows OpenFOAM's primitiveMesh (face centres/areas by triangle fan around the point average, cell
centres/volumes by face pyramids around the face-centre average) and surfaceInterpolation (weights, deltaCoeffs,
non-orthogonal correction), so a solver built on it discretises exactly what simpleFoam discretises.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import foamio

VSMALL = 1e-300
ROOTVSMALL = 1e-150


@dataclass
class Patch:
    name: str
    type: str
    start: int
    size: int
    faceCells: np.ndarray = None
    Sf: np.ndarray = None
    magSf: np.ndarray = None
    nf: np.ndarray = None
    Cf: np.ndarray = None
    delta: np.ndarray = None       # nf (nf . (Cf - C_owner)), OpenFOAM fvPatch::delta
    deltaCoeffs: np.ndarray = None  # 1 / |delta|
    dict: dict = field(default_factory=dict)
    parts: list = None              # the patches a merged run was made of


def _dot(a, b):
    """a & b summed x, y, z in order (OpenFOAM's inner product; einsum may reorder)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _mag(a):
    return np.sqrt(_dot(a, a))


def face_geometry(points, offs, flat):
    """Face centres and area vectors, operation-for-operation as primitiveMeshTools::makeFaceCentresAndAreas
    (so the results are bitwise OpenFOAM's: summation order and constant placement matter for round-off)."""
    nf = len(offs) - 1
    sizes = np.diff(offs)
    Cf = np.empty((nf, 3))
    Sf = np.empty((nf, 3))
    for k in np.unique(sizes):
        idx = np.nonzero(sizes == k)[0]
        pts = points[flat[offs[idx][:, None] + np.arange(k)]]  # (m, k, 3)
        if k == 3:
            Cf[idx] = (1.0 / 3.0) * (pts[:, 0] + pts[:, 1] + pts[:, 2])
            Sf[idx] = 0.5 * np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
            continue
        fc = pts[:, 0].copy()
        for i in range(1, k):
            fc += pts[:, i]
        fc /= k
        sumN = np.zeros((len(idx), 3))
        sumA = np.zeros(len(idx))
        sumAc = np.zeros((len(idx), 3))
        for i in range(k):
            this, nxt = pts[:, i], pts[:, (i + 1) % k]
            c = this + nxt + fc
            n = np.cross(nxt - this, fc - this)
            a = _mag(n)
            sumN += n
            sumA += a
            sumAc += a[:, None] * c
        ok = sumA >= ROOTVSMALL
        Cf[idx] = np.where(ok[:, None], (1.0 / 3.0) * sumAc / np.where(ok, sumA, 1)[:, None], fc)
        Sf[idx] = np.where(ok[:, None], 0.5 * sumN, 0.0)
    return Cf, Sf


def cell_geometry(nCells, owner, neighbour, Cf, Sf):
    """Cell centres and volumes, as primitiveMeshTools::makeCellCentresAndVols (owner pass, then neighbour pass;
    np.add.at applies its updates sequentially in index order, like the C++ loops)."""
    nInt = len(neighbour)
    nFacesPerCell = np.bincount(owner, minlength=nCells) + np.bincount(neighbour, minlength=nCells)
    cEst = np.zeros((nCells, 3))
    np.add.at(cEst, owner, Cf)
    np.add.at(cEst, neighbour, Cf[:nInt])
    cEst /= nFacesPerCell[:, None]

    pyrO = _dot(Sf, Cf - cEst[owner])
    pyrN = _dot(Sf[:nInt], cEst[neighbour] - Cf[:nInt])
    pcO = (3.0 / 4.0) * Cf + (1.0 / 4.0) * cEst[owner]
    pcN = (3.0 / 4.0) * Cf[:nInt] + (1.0 / 4.0) * cEst[neighbour]
    C = np.zeros((nCells, 3))
    V = np.zeros(nCells)
    np.add.at(C, owner, pyrO[:, None] * pcO)
    np.add.at(V, owner, pyrO)
    np.add.at(C, neighbour, pyrN[:, None] * pcN)
    np.add.at(V, neighbour, pyrN)
    ok = np.abs(V) > VSMALL
    C = np.where(ok[:, None], C / np.where(ok, V, 1)[:, None], cEst)
    return C, V * (1.0 / 3.0)


class Mesh:
    def __init__(self, case):
        pm = Path(case) / "constant" / "polyMesh"
        self.points = foamio.read_points(pm / "points")
        offs, flat = foamio.read_faces(pm / "faces")
        self.owner = foamio.read_labels(pm / "owner")
        self.neighbour = foamio.read_labels(pm / "neighbour")
        self.nInternalFaces = nInt = len(self.neighbour)
        self.nCells = int(max(self.owner.max(), self.neighbour.max() if nInt else -1)) + 1
        self.nFaces = len(self.owner)

        Cf, Sf = face_geometry(self.points, offs, flat)
        # full polyMesh face data (all faces, empty patches included) for mesh-wave style algorithms
        self.case = str(case)
        self.face_offsets, self.face_points = offs, flat
        self.allCf, self.allSf = Cf, Sf
        self.C, self.V = cell_geometry(self.nCells, self.owner, self.neighbour, Cf, Sf)

        # internal faces
        o, n = self.owner[:nInt], self.neighbour
        self.Sf, self.Cf = Sf[:nInt], Cf[:nInt]
        # surfaceInterpolation / basicFvGeometryScheme, same operation order
        self.magSf = _mag(self.Sf)
        nHat = self.Sf / self.magSf[:, None]
        dOwn = np.abs(_dot(self.Sf, self.Cf - self.C[o]))
        dNei = np.abs(_dot(self.Sf, self.C[n] - self.Cf))
        self.weights = dNei / (dOwn + dNei)
        delta = self.C[n] - self.C[o]
        self.delta = delta
        self.deltaCoeffs = 1.0 / _mag(delta)
        nd = _dot(nHat, delta)
        self.nonOrthDeltaCoeffs = 1.0 / np.maximum(nd, 0.05 * _mag(delta))
        self.nonOrthCorrectionVectors = nHat - delta * self.nonOrthDeltaCoeffs[:, None]

        # boundary patches (empty patches carry no faces in the finite-volume system)
        self.patches = []
        for name, d in foamio.read_boundary(pm / "boundary"):
            p = Patch(name, d["type"], d["startFace"], d["nFaces"], dict=d)
            s = slice(p.start, p.start + p.size)
            if p.type == "empty":
                p.size = 0
                s = slice(p.start, p.start)
            p.faceCells = self.owner[s]
            p.Sf, p.Cf = Sf[s], Cf[s]
            p.magSf = _mag(p.Sf)
            p.nf = p.Sf / np.maximum(p.magSf, VSMALL)[:, None]
            d = p.Cf - self.C[p.faceCells]
            p.delta = p.nf * _dot(p.nf, d)[:, None]
            p.deltaCoeffs = 1.0 / np.maximum(_mag(p.delta), VSMALL)
            self.patches.append(p)
        self.patch = {p.name: p for p in self.patches}
        self.merged = {}

        # solution directions: a direction is solved unless every face normal in it belongs to an empty patch
        emptyDirs = np.zeros(3)
        for name, d in foamio.read_boundary(pm / "boundary"):
            if d["type"] == "empty":
                s = slice(d["startFace"], d["startFace"] + d["nFaces"])
                emptyDirs += np.abs(Sf[s]).sum(0)
        self.solutionD = ~(emptyDirs > 0.5 * np.abs(emptyDirs).max()) if emptyDirs.any() else np.ones(3, bool)

    def merge_patches(self, runs):
        """Replace each run of consecutive patches (names, new name) by one patch over their faces. The face order
        is unchanged, so every face sum is the same; a device program then carries one boundary vector for the run
        instead of one per patch (motorBike: 67 surface patches with the same conditions)."""
        for names, new in runs:
            ps = [self.patch[n] for n in names]
            assert all(a.start + a.size == b.start for a, b in zip(ps, ps[1:])), names
            cat = lambda k: np.concatenate([getattr(q, k) for q in ps])
            q = Patch(new, ps[0].type, ps[0].start, sum(x.size for x in ps), dict=ps[0].dict, parts=ps)
            for k in ("faceCells", "Sf", "magSf", "nf", "Cf", "delta", "deltaCoeffs"):
                setattr(q, k, cat(k))
            i = self.patches.index(ps[0])
            self.patches[i:i + len(ps)] = [q]
            self.merged[new] = [(x.name, x.size) for x in ps]
        self.patch = {p.name: p for p in self.patches}
