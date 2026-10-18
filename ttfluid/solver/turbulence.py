"""Incompressible RAS k-omega SST as OpenFOAM v2406 runs it in simpleFoam (kOmegaSSTBase.C, default coefficients),
with the wall functions of the pitzDaily case and OpenFOAM's wall distances.

Sources followed (OpenFOAM v2406):
  TurbulenceModels/turbulenceModels/Base/kOmegaSST/kOmegaSSTBase.{C,H}
  .../derivedFvPatchFields/wallFunctions/omegaWallFunctions/omegaWallFunction   (binomial blending, n = 2)
  .../derivedFvPatchFields/wallFunctions/nutWallFunctions/nutkWallFunction       (stepwise blending)
  .../derivedFvPatchFields/wallFunctions/kqRWallFunctions/kqRWallFunction        (zeroGradient)
  finiteVolume/cfdTools/general/bound/bound.C, fvc::average
  finiteVolume/fvMesh/wallDist (meshWave patchDistMethod, nearWallDist), meshTools/cellDist (patchWave,
  cellDistFuncs, wallPoint), meshTools/algorithms/MeshWave/FaceCellWave.C, face::nearestPointClassify,
  triangle::nearestPointClassify.

Field conventions: VolField internal arrays plus per-patch BC objects (fields.py); a "boundary field" of a derived
quantity is a dict {patch name: values} over the non-empty patches, evaluated pointwise from the operands' patch
values exactly as GeometricField algebra does.
"""
import math
from pathlib import Path

import numpy as np

from .ops import FaceOps
from .fields import VolField, FixedValue

SMALL = 1e-15
GREAT = 1e15
ROOTVSMALL = 1e-150


def _dot(a, b):
    """a & b for vectors, x + y + z in order (as OpenFOAM's inner product)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _vdot(a, T):
    """a & T for a vector and a tensor (sum over the first tensor index, in order)."""
    return a[:, 0, None] * T[:, 0] + a[:, 1, None] * T[:, 1] + a[:, 2, None] * T[:, 2]


def _entry_scalar(entry, key, default):
    v = entry.get(key)
    return default if v is None else float(v[0] if isinstance(v, list) else v)


# ---- boundary conditions owned by the turbulence model ------------------------------------------------------------

class OmegaWallFunction(FixedValue):
    """omegaWallFunction: fixedValue whose value (and the wall-cell omega, and G) the model sets in updateCoeffs.
    On read the patch value is the cell value (constructor calls extrapolateInternal())."""

    def __init__(self, field, patch, entry):
        super().__init__(field, patch, entry)
        self.value = field.internal[patch.faceCells].copy()


class NutkWallFunction(FixedValue):
    """nutkWallFunction: fixedValue; values computed by KOmegaSST.nutk_wall()."""


class NutUSpaldingWallFunction(FixedValue):
    """nutUSpaldingWallFunction: fixedValue; values computed by KOmegaSST.nutU_spalding() from the previous values."""


REGISTRY = {"omegaWallFunction": OmegaWallFunction, "nutkWallFunction": NutkWallFunction,
            "nutUSpaldingWallFunction": NutUSpaldingWallFunction}


# ---- geometry helpers (exact transcriptions) ------------------------------------------------------------------------

def _tdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _mag(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _tri_nearest_dist(a, b, c, p):
    """triangle::nearestPointClassify distance (Ericson, Real-time collision detection)."""
    ab = _sub(b, a); ac = _sub(c, a); ap = _sub(p, a)
    d1 = _tdot(ab, ap); d2 = _tdot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return _mag(_sub(a, p))
    bp = _sub(p, b)
    d3 = _tdot(ab, bp); d4 = _tdot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return _mag(_sub(b, p))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        if (d1 - d3) < ROOTVSMALL:
            return _mag(_sub(a, p))
        v = d1 / (d1 - d3)
        q = (a[0] + v * ab[0], a[1] + v * ab[1], a[2] + v * ab[2])
        return _mag(_sub(q, p))
    cp = _sub(p, c)
    d5 = _tdot(ab, cp); d6 = _tdot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return _mag(_sub(c, p))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        if (d2 - d6) < ROOTVSMALL:
            return _mag(_sub(a, p))
        w = d2 / (d2 - d6)
        q = (a[0] + w * ac[0], a[1] + w * ac[1], a[2] + w * ac[2])
        return _mag(_sub(q, p))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        if ((d4 - d3) + (d5 - d6)) < ROOTVSMALL:
            return _mag(_sub(b, p))
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = (b[0] + w * (c[0] - b[0]), b[1] + w * (c[1] - b[1]), b[2] + w * (c[2] - b[2]))
        return _mag(_sub(q, p))
    if (va + vb + vc) < ROOTVSMALL:
        q = tuple((1.0 / 3.0) * (a[i] + b[i] + c[i]) for i in range(3))
        return _mag(_sub(q, p))
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    q = (a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w)
    return _mag(_sub(q, p))


def _face_centre(pts):
    """face::centre (area-weighted triangle-fan centre around the point average)."""
    n = len(pts)
    if n == 3:
        return tuple((1.0 / 3.0) * (pts[0][i] + pts[1][i] + pts[2][i]) for i in range(3))
    cp = [0.0, 0.0, 0.0]
    for q in pts:
        cp[0] += q[0]; cp[1] += q[1]; cp[2] += q[2]
    cp = (cp[0] / n, cp[1] / n, cp[2] / n)
    sumA = 0.0
    sumAc = [0.0, 0.0, 0.0]
    for i in range(n):
        p0, p1 = pts[i], pts[(i + 1) % n]
        ttc = (p0[0] + p1[0] + cp[0], p0[1] + p1[1] + cp[1], p0[2] + p1[2] + cp[2])
        u = _sub(p0, cp); v = _sub(p1, cp)
        x = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        ta = _mag(x)
        sumA += ta
        sumAc[0] += ta * ttc[0]; sumAc[1] += ta * ttc[1]; sumAc[2] += ta * ttc[2]
    if sumA > 1e-300:
        return (sumAc[0] / (3.0 * sumA), sumAc[1] / (3.0 * sumA), sumAc[2] / (3.0 * sumA))
    return cp


def _face_nearest_dist(pts, p):
    """face::nearestPointClassify distance."""
    if len(pts) == 3:
        return _tri_nearest_dist(pts[0], pts[1], pts[2], p)
    ctr = _face_centre(pts)
    best = GREAT
    n = len(pts)
    for i in range(n):
        d = _tri_nearest_dist(pts[i], pts[(i + 1) % n], ctr, p)
        if abs(d) < abs(best):
            best = d
    return best


def _vmag(a):
    return np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vtri_nearest_dist(a, b, c, p):
    """_tri_nearest_dist over arrays: a, b, c, p are 3-tuples of 1-d arrays. Each lane takes the first branch the
    scalar code would take and evaluates that branch's expression with the same operations in the same order, so
    every lane is bit-identical to the scalar transcription."""
    N = len(a[0])
    out = np.empty(N)
    with np.errstate(all="ignore"):
        ab = _sub(b, a); ac = _sub(c, a); ap = _sub(p, a)
        d1 = _tdot(ab, ap); d2 = _tdot(ac, ap)
        bp = _sub(p, b)
        d3 = _tdot(ab, bp); d4 = _tdot(ac, bp)
        cp = _sub(p, c)
        d5 = _tdot(ab, cp); d6 = _tdot(ac, cp)
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        e43 = d4 - d3; e56 = d5 - d6
        s = va + vb + vc
        conds = [(d1 <= 0.0) & (d2 <= 0.0), (d3 >= 0.0) & (d4 <= d3), (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
                 (d6 >= 0.0) & (d5 <= d6), (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
                 (va <= 0.0) & (e43 >= 0.0) & (e56 >= 0.0), s < ROOTVSMALL]
        br = np.full(N, 7, np.int8)
        for k in range(6, -1, -1):
            br[conds[k]] = k
        g = lambda X, i: tuple(x[i] for x in X)
        mg = lambda q, P: _vmag(_sub(q, P))
        for k in range(8):
            i = np.nonzero(br == k)[0]
            if not len(i):
                continue
            A, P = g(a, i), g(p, i)
            if k == 0:
                out[i] = mg(A, P)
            elif k == 1:
                out[i] = mg(g(b, i), P)
            elif k == 2:
                D1, D3 = d1[i], d3[i]
                v = D1 / (D1 - D3); AB = g(ab, i)
                r = mg((A[0] + v * AB[0], A[1] + v * AB[1], A[2] + v * AB[2]), P)
                out[i] = np.where((D1 - D3) < ROOTVSMALL, mg(A, P), r)
            elif k == 3:
                out[i] = mg(g(c, i), P)
            elif k == 4:
                D2, D6 = d2[i], d6[i]
                w = D2 / (D2 - D6); AC = g(ac, i)
                r = mg((A[0] + w * AC[0], A[1] + w * AC[1], A[2] + w * AC[2]), P)
                out[i] = np.where((D2 - D6) < ROOTVSMALL, mg(A, P), r)
            elif k == 5:
                E43, E56 = e43[i], e56[i]
                B, Cc = g(b, i), g(c, i)
                w = E43 / (E43 + E56)
                r = mg((B[0] + w * (Cc[0] - B[0]), B[1] + w * (Cc[1] - B[1]), B[2] + w * (Cc[2] - B[2])), P)
                out[i] = np.where((E43 + E56) < ROOTVSMALL, mg(B, P), r)
            elif k == 6:
                B, Cc = g(b, i), g(c, i)
                out[i] = mg(tuple((1.0 / 3.0) * (A[j] + B[j] + Cc[j]) for j in range(3)), P)
            else:
                denom = 1.0 / s[i]
                v = vb[i] * denom
                w = vc[i] * denom
                AB, AC = g(ab, i), g(ac, i)
                out[i] = mg((A[0] + AB[0] * v + AC[0] * w, A[1] + AB[1] * v + AC[1] * w,
                             A[2] + AB[2] * v + AC[2] * w), P)
    return out


def _vface_centre(P):
    """_face_centre for M faces of n points each, P (M, n, 3); same accumulation order."""
    M, n, _ = P.shape
    if n == 3:
        return (1.0 / 3.0) * (P[:, 0] + P[:, 1] + P[:, 2])
    cp = np.zeros((M, 3))
    for k in range(n):
        cp = cp + P[:, k]
    cp = cp / n
    sumA = np.zeros(M)
    sumAc = np.zeros((M, 3))
    for i in range(n):
        p0, p1 = P[:, i], P[:, (i + 1) % n]
        ttc = p0 + p1 + cp
        u = p0 - cp; v = p1 - cp
        x = (u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1], u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2],
             u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
        ta = _vmag(x)
        sumA = sumA + ta
        sumAc = sumAc + ta[:, None] * ttc
    ok = sumA > 1e-300
    with np.errstate(all="ignore"):
        c = sumAc / (3.0 * sumA)[:, None]
    return np.where(ok[:, None], c, cp)


def _csr_expand(off, rows):
    """Concatenated index ranges off[r]:off[r+1] for r in rows, plus the row position each entry came from."""
    lens = off[rows + 1] - off[rows]
    pos = np.repeat(np.arange(len(rows)), lens)
    start = np.cumsum(lens) - lens
    return np.repeat(off[rows], lens) + (np.arange(int(lens.sum())) - start[pos]), pos


class WallDistance:
    """OpenFOAM wall distances for the wall-type patches.

    y      : cell values of wallDist::y() with the meshWave method and correctWalls = true (kOmegaSST's y_)
    y_b    : its patch values: SMALL on the wall patches (fixedValue: sqrt(0) + SMALL from patchWave), the cell
             value elsewhere (zeroGradient, re-evaluated by y.correctBoundaryConditions())
    near   : {wall patch: nearWallDist values}, turbulenceModel::y() used by the wall functions

    Vectorised with numpy; every value is bit-identical to the scalar (loop) transcription it replaced: the mesh
    wave replays FaceCellWave's visiting order exactly (see _mesh_wave), the nearest-point distances run the same
    floating-point operations lane by lane, and the min-reductions are order-independent (NaN skipped, as `<`).
    """

    CHUNK = 1 << 20

    def __init__(self, mesh, tol=0.01):
        self.mesh = mesh
        wall = [p for p in mesh.patches if p.type == "wall"]
        pts = mesh.points
        offs = np.asarray(mesh.face_offsets, dtype=np.int64)
        flat = np.asarray(mesh.face_points, dtype=np.int64)
        nC = mesh.nCells

        # OpenFOAM 2412 (cellDistFuncs::useCombinedWallPatch = true): all wall patches form one patch, so a wall
        # face's point neighbours (getPointNeighbours: itself, then every wall face sharing a point) cross patch
        # boundaries. Where two wall patches meet (motorBike's parts) this changes the wall cells' distances.
        wf = np.concatenate([np.arange(p.start, p.start + p.size, dtype=np.int64) for p in wall]) if wall \
            else np.zeros(0, np.int64)
        wc = np.concatenate([np.asarray(p.faceCells, dtype=np.int64) for p in wall]) if wall \
            else np.zeros(0, np.int64)
        nW = len(wf)
        self._wf = wf
        ent, wpos = _csr_expand(offs, wf)          # wall face point entries, in wall-face order
        wpt = flat[ent]
        self._wn = (offs[wf + 1] - offs[wf])
        self._woff = np.concatenate([[0], np.cumsum(self._wn)])
        self._wpt = wpt
        # point -> wall faces (local wall index), CSR
        o = np.argsort(wpt, kind="stable")
        self._pf_pt = wpt[o]
        self._pf_w = wpos[o]
        upts, pstart = np.unique(self._pf_pt, return_index=True)
        self._pf_off = np.concatenate([pstart, [len(o)]]).astype(np.int64)
        pidx = np.full(len(pts), -1, np.int64)
        pidx[upts] = np.arange(len(upts))          # wall point -> its pf row
        # face centres of the wall faces, by point count
        self._wctr = np.zeros((nW, 3))
        for n in np.unique(self._wn):
            sel = np.nonzero(self._wn == n)[0]
            P = pts[wpt[self._woff[sel][:, None] + np.arange(n)]]
            self._wctr[sel] = _vface_centre(P)

        # point neighbours of each wall face: (w, g) pairs, g over the wall faces of any point of w
        if nW:
            ix, pos = _csr_expand(self._pf_off, pidx[wpt])
            pr = np.unique(wpos[pos] * nW + self._pf_w[ix])
            nb_w, nb_g = pr // nW, pr % nW
        else:
            nb_w = nb_g = np.zeros(0, np.int64)

        # ---- patchWave: FaceCellWave<wallPoint> from the wall face centres --------------------------------------
        y = self._mesh_wave(wf, tol)
        # correctBoundaryCells: cells with a face on a wall get the true distance to the nearest point-neighbour
        # wall face of that face (later faces of the same cell overwrite earlier ones, as in OpenFOAM) ...
        near_w = self._smallest(wc[nb_w], nb_g, nb_w, nW)
        if nW:
            _, ri = np.unique(wc[::-1], return_index=True)
            last = nW - 1 - ri
            y[wc[last]] = near_w[last]
        done = np.zeros(nC, bool)
        done[wc] = True
        # ... then every other cell sharing a point with the wall, once, from the first wall point it is met at
        # (points in order of first use along the wall faces), against that point's wall faces only
        if nW:
            iswp = np.zeros(len(pts), bool)
            iswp[wpt] = True
            useq, first = np.unique(wpt, return_index=True)
            prank = np.full(len(pts), np.iinfo(np.int64).max, np.int64)
            prank[useq] = first                    # rank = first position along the wall faces
            e = np.nonzero(iswp[flat])[0]
            fe = np.searchsorted(offs, e, side="right") - 1
            rk = prank[flat[e]]
            nI = mesh.nInternalFaces
            cr = np.full(nC, np.iinfo(np.int64).max, np.int64)
            np.minimum.at(cr, np.asarray(mesh.owner, np.int64)[fe], rk)
            inn = fe < nI
            np.minimum.at(cr, np.asarray(mesh.neighbour, np.int64)[fe[inn]], rk[inn])
            cells = np.nonzero((cr != np.iinfo(np.int64).max) & ~done)[0]
            v = wpt[cr[cells]]
            ix, pos = _csr_expand(self._pf_off, pidx[v])
            y[cells] = self._smallest(cells[pos], self._pf_w[ix], pos, len(cells))
        self.y = y
        self.y_b = {}
        for p in mesh.patches:
            if p.type == "empty":
                continue
            self.y_b[p.name] = np.full(p.size, SMALL) if p.type == "wall" else y[p.faceCells].copy()

        # ---- nearWallDist (turbulenceModel::y()), same combined wall patch ----------------------------------------
        self.near = {}
        for p in mesh.patches:
            if p.type == "empty":
                continue
            if p.type == "wall":
                s = sum(q.size for q in wall[:wall.index(p)])
                self.near[p.name] = near_w[s:s + p.size].copy()
            else:
                self.near[p.name] = np.zeros(p.size)

    def _face_dist(self, c, g):
        """_face_nearest_dist(wall face g, cell centre of c) for pair arrays."""
        C = self.mesh.C
        pts = self.mesh.points
        out = np.empty(len(c))
        n_of = self._wn[g]
        for n in np.unique(n_of):
            sel = np.nonzero(n_of == n)[0]
            for s0 in range(0, len(sel), self.CHUNK):
                s = sel[s0:s0 + self.CHUNK]
                gg = g[s]
                P = pts[self._wpt[self._woff[gg][:, None] + np.arange(n)]]
                pc = C[c[s]]
                pt = (pc[:, 0], pc[:, 1], pc[:, 2])
                col = lambda X: (X[:, 0], X[:, 1], X[:, 2])
                if n == 3:
                    out[s] = _vtri_nearest_dist(col(P[:, 0]), col(P[:, 1]), col(P[:, 2]), pt)
                    continue
                ctr = col(self._wctr[gg])
                best = np.full(len(s), GREAT)
                for i in range(n):
                    d = _vtri_nearest_dist(col(P[:, i]), col(P[:, (i + 1) % n]), ctr, pt)
                    best = np.where(np.abs(d) < np.abs(best), d, best)
                out[s] = best
        return out

    def _smallest(self, c, g, grp, ngrp):
        """smallest(): min over each group of the pair distances, starting from GREAT; `h < d` skips NaN (fmin)."""
        res = np.full(ngrp, GREAT)
        if len(c) == 0:
            return res
        d = self._face_dist(c, g)
        o = np.argsort(grp, kind="stable")
        gs = grp[o]
        st = np.nonzero(np.r_[True, gs[1:] != gs[:-1]])[0]
        res[gs[st]] = np.fmin(GREAT, np.fmin.reduceat(d[o], st))
        return res

    def _mesh_wave(self, wf, tol):
        """FaceCellWave<wallPoint> (patchWave), vectorised but replaying the scalar visiting order exactly.

        Per half-sweep the sources (changed faces, then changed cells) are fixed; each target sees its candidate
        origins in loop order (changed-list position, then owner/neighbour or the cell's face order). Targets are
        independent, so candidate k of every target is applied in round k. The next changed list is ordered by
        each target's first successful update in the loop, as the scalar code appends it."""
        m = self.mesh
        nC, nF, nI = m.nCells, m.nFaces, m.nInternalFaces
        own = np.asarray(m.owner, np.int64)
        nei = np.asarray(m.neighbour, np.int64)
        Cf = m.allCf
        Cc = m.C
        # primitiveMesh::calcCells: owned faces in face order, then neighbour faces in face order
        cc = np.concatenate([own, nei])
        cfi = np.concatenate([np.arange(nF, dtype=np.int64), np.arange(nI, dtype=np.int64)])
        o = np.argsort(cc, kind="stable")
        cf_idx = cfi[o]
        cf_off = np.zeros(nC + 1, np.int64)
        np.cumsum(np.bincount(cc, minlength=nC), out=cf_off[1:])
        del cc, cfi, o

        # wallPoint equality compares origin coordinates: give wall faces with identical centres one id
        cid = np.full(nF, -1, np.int64)
        if len(wf):
            _, fi, inv = np.unique(Cf[wf], axis=0, return_index=True, return_inverse=True)
            cid[wf] = wf[fi[inv.ravel()]]
        fo = np.full(nF, -1, np.int64); fd = np.full(nF, -GREAT)   # face origin (canonical wall face) / distSqr
        co = np.full(nC, -1, np.int64); cd = np.full(nC, -GREAT)
        markF = np.full(nF, -1, np.int64); markC = np.full(nC, -1, np.int64)
        Ox, Oy, Oz = (np.ascontiguousarray(Cf[:, i]) for i in range(3))

        def sweep(tgt, org, X, to, td, mark):
            """Apply candidate origins org to targets tgt (in event order); returns targets changed, in order of
            their first change."""
            ev = np.arange(len(tgt))
            od = np.argsort(tgt, kind="stable")
            ts = tgt[od]
            st = np.r_[True, ts[1:] != ts[:-1]]
            if st.all():
                rounds = [od]
            else:
                gstart = np.maximum.accumulate(np.where(st, np.arange(len(ts)), 0))
                rank = np.arange(len(ts)) - gstart
                orr = np.argsort(rank, kind="stable")
                lanes_all = od[orr]
                cnt = np.cumsum(np.bincount(rank))
                rounds = np.split(lanes_all, cnt[:-1])
            chg = []
            for L in rounds:
                t = tgt[L]; w = org[L]
                nz = to[t] != w
                t, w, L = t[nz], w[nz], L[nz]
                dx = X[0][t] - Ox[w]; dy = X[1][t] - Oy[w]; dz = X[2][t] - Oz[w]
                dist2 = dx * dx + dy * dy + dz * dz
                cdv = td[t]
                with np.errstate(all="ignore"):
                    diff = cdv - dist2
                    keep = (diff < 0) | (diff < SMALL) | ((cdv > SMALL) & (diff / cdv < tol))
                take = ~(cdv > -SMALL) | ~keep
                t = t[take]
                to[t] = w[take]
                td[t] = dist2[take]
                new = mark[t] < 0          # targets are distinct within a round; the earliest round is the first
                t = t[new]
                mark[t] = ev[L[take][new]]
                chg.append(t)
            chg = np.concatenate(chg)
            out = chg[np.argsort(mark[chg])]
            mark[chg] = -1
            return out

        Xc = tuple(np.ascontiguousarray(Cc[:, i]) for i in range(3))
        Xf = (Ox, Oy, Oz)
        fo[wf] = cid[wf]; fd[wf] = 0.0
        changedFaces = wf.copy()
        maxIter = nC + 1
        it = 0
        while it < maxIter:
            # faceToCell
            F = changedFaces
            nb = np.where(F < nI, nei[np.minimum(F, max(nI - 1, 0))] if nI else -1, -1)
            tgt = np.stack([own[F], nb], axis=1).ravel()
            src = np.repeat(F, 2)
            ok = tgt >= 0
            tgt, src = tgt[ok], src[ok]
            changedCells = sweep(tgt, fo[src], Xc, co, cd, markC)
            if not len(changedCells):
                break
            # cellToFace
            ix, pos = _csr_expand(cf_off, changedCells)
            changedFaces = sweep(cf_idx[ix], co[changedCells][pos], Xf, fo, fd, markF)
            it += 1
            if not len(changedFaces):
                break
        valid = cd > -SMALL
        return np.where(valid, np.sqrt(np.where(valid, cd, 0.0)), cd)


# ---- small field algebra helpers ---------------------------------------------------------------------------------

def _symm_magsqr2(g):
    """2*magSqr(symm(T)) for T (..., 3, 3)."""
    s = 0.5 * (g + np.swapaxes(g, -1, -2))
    return 2 * (s * s).sum(axis=(-1, -2))


def _gbynu0(g):
    """T && devTwoSymm(T)."""
    tr = (2.0 / 3.0) * (g[..., 0, 0] + g[..., 1, 1] + g[..., 2, 2])
    d = g + np.swapaxes(g, -1, -2)
    for i in range(3):
        d[..., i, i] = 2 * g[..., i, i] - tr
    return (g * d).sum(axis=(-1, -2))


# ---- the model --------------------------------------------------------------------------------------------------

class KOmegaSST:
    alphaK1, alphaK2 = 0.85, 1.0
    alphaOmega1, alphaOmega2 = 0.5, 0.856
    gamma1, gamma2 = 5.0 / 9.0, 0.44
    beta1, beta2 = 0.075, 0.0828
    betaStar = 0.09
    a1, b1, c1 = 0.31, 1.0, 10.0
    kMin = omegaMin = SMALL
    # wall-function coefficients (wallFunctionCoefficients defaults)
    Cmu, kappa, E = 0.09, 0.41, 9.8
    wf_beta1 = 0.075   # omegaWallFunction beta1
    wf_n = 2.0         # omegaWallFunction binomial exponent

    def __init__(self, mesh, U, nu, case_dir, time="0", relax=None, wall_distance=None, ops=None, div=None):
        self.mesh, self.U, self.nu = mesh, U, float(nu)
        # (bounded, scheme) for div(phi,k) and div(phi,omega); pitzDaily's bounded Gauss limitedLinear 1 by default
        self.div = div or {n: (True, ("limitedLinear", 1.0)) for n in ("k", "omega")}
        self.ops = ops or FaceOps(mesh)
        d = Path(case_dir) / str(time)
        self.k = VolField.read(mesh, d / "k", registry=REGISTRY)
        self.omega = VolField.read(mesh, d / "omega", registry=REGISTRY)
        self.nut = VolField.read(mesh, d / "nut", registry=REGISTRY)
        self.k.name, self.omega.name, self.nut.name = "k", "omega", "nut"
        self.relax = relax or {"k": 0.9, "omega": 0.9}
        self.wd = wall_distance or WallDistance(mesh)
        self.y, self.y_b = self.wd.y, self.wd.y_b

        self.yPlusLam = 11.0
        for _ in range(10):
            self.yPlusLam = math.log(max(self.E * self.yPlusLam, 1.0)) / self.kappa
        self.Cmu25 = self.Cmu ** 0.25

        # omegaWallFunction corner weights: 1 / (number of omegaWallFunction faces on the face cell)
        self.owf = [n for n, b in self.omega.bc.items() if isinstance(b, OmegaWallFunction)]
        cnt = np.zeros(mesh.nCells)
        for n in self.owf:
            np.add.at(cnt, mesh.patch[n].faceCells, 1.0)
        self.cornerWeights = {n: 1.0 / cnt[mesh.patch[n].faceCells] for n in self.owf}

        # constructor: bound(k), bound(omega); simpleFoam then calls turbulence->validate() = correctNut()
        self.bound(self.k, self.kMin)
        self.bound(self.omega, self.omegaMin)
        self.correctNut()

    # ---- pieces -------------------------------------------------------------------------------------------------
    def _bvals(self, f):
        return {n: b.value for n, b in f.bc.items()}

    def F1(self, CD, CD_b):
        def f(k, om, y, cd):
            cdp = np.maximum(cd, 1.0e-10)
            arg1 = np.minimum(np.minimum(np.maximum((1.0 / self.betaStar) * np.sqrt(k) / (om * y),
                                                    500.0 * self.nu / (y * y * om)),
                                         (4 * self.alphaOmega2) * k / (cdp * (y * y))), 10.0)
            return np.tanh(arg1 ** 4)
        kb, ob = self._bvals(self.k), self._bvals(self.omega)
        return (f(self.k.internal, self.omega.internal, self.y, CD),
                {n: f(kb[n], ob[n], self.y_b[n], CD_b[n]) for n in kb})

    def F2(self):
        def f(k, om, y):
            arg2 = np.minimum(np.maximum((2.0 / self.betaStar) * np.sqrt(k) / (om * y),
                                         500.0 * self.nu / (y * y * om)), 100.0)
            return np.tanh(arg2 * arg2)
        kb, ob = self._bvals(self.k), self._bvals(self.omega)
        return f(self.k.internal, self.omega.internal, self.y), {n: f(kb[n], ob[n], self.y_b[n]) for n in kb}

    def nutk_wall(self, name):
        """nutkWallFunction::calcNut (stepwise)."""
        p = self.mesh.patch[name]
        y = self.wd.near[name]
        k = self.k.internal[p.faceCells]
        yPlus = self.Cmu25 * y * np.sqrt(k) / self.nu
        nutLog = self.nu * yPlus * self.kappa / np.log(np.maximum(self.E * yPlus, 1 + 1e-4))
        nutw = np.where(yPlus > self.yPlusLam, nutLog, self.nu)
        return nutw - self.nu

    def nutU_spalding(self, name, nutw):
        """nutUSpaldingWallFunction::calcNut: Newton on Spalding's law for uTau from the current nutw (maxIter steps,
        a face stopping once |du/u| <= tolerance: a frozen lane, so it traces). With a tolerance other than the
        default 0.01, a face whose one-step error is already under it keeps its nutw."""
        b = self.nut.bc[name]
        maxIter = int(_entry_scalar(b.entry, "maxIter", 10))
        tolerance = _entry_scalar(b.entry, "tolerance", 0.01)
        Uw = self.U.bc[name]
        g = Uw.snGrad()
        magGradU = np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2 + g[:, 2] ** 2)
        d = self.U.internal[self.mesh.patch[name].faceCells] - Uw.value
        magUp = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
        tiny = ROOTVSMALL if magGradU.dtype == np.float64 else 1e-18
        uTau, _ = self._spalding_uTau(name, nutw, magGradU, magUp, maxIter, tolerance, tiny)
        out = np.maximum(0.0, uTau * uTau / (magGradU + tiny) - self.nu)
        if tolerance != 0.01:
            _, err = self._spalding_uTau(name, nutw, magGradU, magUp, 1, tolerance, tiny)
            out = np.where(err < tolerance, nutw, out)
        return out

    def _spalding_uTau(self, name, nutw, magGradU, magUp, maxIter, tolerance, tiny):
        """calcUTau: (uTau, err of the last step taken; 0 on faces that never started)."""
        y, nuw, kappa, E = self.wd.near[name], self.nu, self.kappa, self.E
        ut = np.sqrt((nutw + nuw) * magGradU)
        live = started = ut > tiny
        err = np.zeros_like(ut)
        for _ in range(maxIter):
            u = np.where(live, ut, 1.0)
            kUu = np.minimum(kappa * magUp / u, 50.0)
            fkUu = np.exp(kUu) - 1.0 - kUu * (1.0 + 0.5 * kUu)
            f = -u * y / nuw + magUp / u + (fkUu - kUu ** 3 / 6.0) / E
            df = y / nuw + magUp / (u * u) + kUu * fkUu / (E * u)
            new = u + f / df
            err = np.where(live, np.abs((u - new) / u), err)
            ut = np.where(live, new, ut)
            live = live & (ut > tiny) & (err > tolerance)
        return np.where(started, np.maximum(ut, 0.0), 0.0), err

    def correctNut(self, S2=None, S2_b=None):
        if S2 is None:
            g, gb = self.ops.grad(self.U)
            S2, S2_b = _symm_magsqr2(g), {n: _symm_magsqr2(v) for n, v in gb.items()}
        F2, F2_b = self.F2()
        kb, ob = self._bvals(self.k), self._bvals(self.omega)
        self.nut.internal[:] = self.a1 * self.k.internal / np.maximum(self.a1 * self.omega.internal,
                                                                     self.b1 * F2 * np.sqrt(S2))
        for n, b in self.nut.bc.items():
            if isinstance(b, NutkWallFunction):
                b.value = self.nutk_wall(n)
            elif isinstance(b, NutUSpaldingWallFunction):
                b.value = self.nutU_spalding(n, b.value)
            elif isinstance(b, FixedValue):
                pass  # fixedValue patches ignore assignment
            else:
                b.value = self.a1 * kb[n] / np.maximum(self.a1 * ob[n], self.b1 * F2_b[n] * np.sqrt(S2_b[n]))

    def omega_wall_update(self, G):
        """omegaWallFunction::updateCoeffs on all omegaWallFunction patches: sets wall-cell G and omega and the
        patch values (binomial blending n = 2, G from the log-law)."""
        m = self.mesh
        k = self.k.internal
        G0 = np.zeros_like(k, dtype=float)
        om0 = np.zeros_like(k, dtype=float)
        for n in self.owf:
            p = m.patch[n]
            fc = p.faceCells
            w = self.cornerWeights[n]
            y = self.wd.near[n]
            nutw = self.nut.bc[n].value
            nuw = self.nu
            sk = np.sqrt(k[fc])
            omegaVis = w * 6.0 * nuw / (self.wf_beta1 * (y * y))
            omegaLog = w * sk / (self.Cmu25 * self.kappa * y)
            np.add.at(om0, fc, np.power(np.power(omegaVis, self.wf_n) + np.power(omegaLog, self.wf_n),
                                        1.0 / self.wf_n))
            magGradUw = np.linalg.norm(self.U.bc[n].snGrad(), axis=1)
            np.add.at(G0, fc, w * (nutw + nuw) * magGradUw * self.Cmu25 * sk / (self.kappa * y))
        for n in self.owf:
            fc = m.patch[n].faceCells
            self.omega.bc[n].value = om0[fc].copy()
        for n in self.owf:
            fc = m.patch[n].faceCells
            G[fc] = G0[fc]
            self.omega.internal[fc] = om0[fc]

    def bound(self, f, lb):
        """bound(): cells below lb get max(average of max(f, lb) over the faces, lb) where f <= 0."""
        vals = [f.internal] + [b.value for b in f.bc.values()]
        if min(v.min() for v in vals if v.size) >= lb:
            return
        self._bound_apply(f, lb)

    def _bound_apply(self, f, lb):
        """bound() without its early return; a no-op when nothing is below lb (what a traced program runs)."""
        x = f.internal
        xm = np.maximum(x, lb)
        av = self.ops.face_average(self.ops.interp(xm, {n: np.maximum(b.value, lb) for n, b in f.bc.items()}))
        f.internal[:] = np.maximum(np.maximum(x, av * (-x >= 0)), lb)
        for b in f.bc.values():
            if not isinstance(b, FixedValue):
                b.value = np.maximum(b.value, lb)

    def nuEff(self):
        """nut + nu as a face value: linear interpolation of the cell field, patch values on patches."""
        return self.ops.interp(self.nut.internal + self.nu, {n: b.value + self.nu for n, b in self.nut.bc.items()})

    # ---- kOmegaSSTBase::correct ------------------------------------------------------------------------------------
    def correct(self, phi, solver):
        m = self.mesh
        k, omega, nut = self.k, self.omega, self.nut
        nu = self.nu

        ops = self.ops
        divU = ops.surface_integrate(phi)
        gU, gUb = ops.grad(self.U)
        S2, S2_b = _symm_magsqr2(gU), {n: _symm_magsqr2(v) for n, v in gUb.items()}
        GbyNu0 = _gbynu0(gU)
        G = nut.internal * GbyNu0

        self.omega_wall_update(G)

        gk, gkb = ops.grad(k)
        go, gob = ops.grad(omega)
        ob = self._bvals(omega)
        CD = (2 * self.alphaOmega2) * _dot(gk, go) / omega.internal
        CD_b = {n: (2 * self.alphaOmega2) * _dot(gkb[n], gob[n]) / ob[n] for n in gkb}
        F1, F1_b = self.F1(CD, CD_b)
        F23, _ = self.F2()

        # ---- omega equation
        gamma = F1 * (self.gamma1 - self.gamma2) + self.gamma2
        beta = F1 * (self.beta1 - self.beta2) + self.beta2
        om = omega.internal
        GbyNu = np.minimum(GbyNu0, (self.c1 / self.a1) * self.betaStar * om
                           * np.maximum(self.a1 * om, self.b1 * F23 * np.sqrt(S2)))
        aO = F1 * (self.alphaOmega1 - self.alphaOmega2) + self.alphaOmega2
        aO_b = {n: v * (self.alphaOmega1 - self.alphaOmega2) + self.alphaOmega2 for n, v in F1_b.items()}
        D = aO * nut.internal + nu
        D_b = {n: aO_b[n] * nut.bc[n].value + nu for n in aO_b}

        eq = ops.div(phi, omega, self.div["omega"][1], bounded=self.div["omega"][0])
        if hasattr(ops, "sub_laplacian"):
            ops.sub_laplacian(eq, ops.interp(D, D_b), omega)                 # eq -= laplacian, never whole
        else:
            eq -= ops.laplacian(ops.interp(D, D_b), omega)
        eq.source += ops.V * (gamma * GbyNu)
        eq += ops.SuSp((2.0 / 3.0) * gamma * divU, omega)
        eq += ops.Sp(beta * om, omega)
        eq += ops.SuSp((F1 - 1.0) * CD / om, omega)
        eq.relax(self.relax["omega"])
        for n in self.owf:   # boundaryManipulate: omegaWallFunction::manipulateMatrix
            fc = m.patch[n].faceCells
            eq.setValues(fc, omega.internal[fc].copy())
        self.res_omega = solver(eq, "omega")
        omega.correct_bc()
        self.bound(omega, self.omegaMin)

        # ---- k equation
        aK = F1 * (self.alphaK1 - self.alphaK2) + self.alphaK2
        aK_b = {n: v * (self.alphaK1 - self.alphaK2) + self.alphaK2 for n, v in F1_b.items()}
        D = aK * nut.internal + nu
        D_b = {n: aK_b[n] * nut.bc[n].value + nu for n in aK_b}
        om = omega.internal
        eq = ops.div(phi, k, self.div["k"][1], bounded=self.div["k"][0])
        if hasattr(ops, "sub_laplacian"):
            ops.sub_laplacian(eq, ops.interp(D, D_b), k)                 # eq -= laplacian, never whole
        else:
            eq -= ops.laplacian(ops.interp(D, D_b), k)
        eq.source += ops.V * np.minimum(G, (self.c1 * self.betaStar) * k.internal * om)
        eq += ops.SuSp((2.0 / 3.0) * divU, k)
        eq += ops.Sp(self.betaStar * om, k)
        eq.relax(self.relax["k"])
        self.res_k = solver(eq, "k")
        k.correct_bc()
        self.bound(k, self.kMin)

        self.correctNut(S2, S2_b)
        return self.res_omega, self.res_k
