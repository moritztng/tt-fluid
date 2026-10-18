"""Cell-centric (ELL) form of the mesh: the layout the device solver runs on.

Every finite-volume face loop in SIMPLE and k-omega SST is a sum over a cell's faces of something computed from the
cell's own value and the value on the other side of the face. Store, for each cell, one slot per face: the index of the
value on the other side, plus that face's geometry as seen from this cell (outward Sf, own-side weight, deltaCoeffs).
A face loop then becomes three steps with no scatter and no atomics:

    gather   x_ext[nbr]          (the only irregular access; on device it reads the core's own SRAM or its halo)
    per-slot arithmetic          (elementwise, SFPU)
    row sum over slots           (local reduction)

Interior faces appear in both adjacent rows (each computed once per side, a 2x redundancy on cheap arithmetic that
buys the removal of every scatter). Boundary faces are slots whose "other side" is a ghost entry holding the patch
face value, so x_ext = [cell values ; boundary face values] and boundary conditions only write ghost entries and
per-slot coefficients.
"""
import numpy as np


def max_count(counts):
    """Faces of the widest cell (the ELL width); the SPMD lowering (lower/spmd.py) takes it over the whole mesh."""
    return counts.max()


class Ell:
    def __init__(self, mesh, dtype=np.float64, order="face"):
        """order: slot order within a row. "face" = face index (OpenFOAM's order); "dir" = by the direction of the
        outward face normal (-x +x -y +y -z +z), so on a hex lattice slot k points the same way in every cell and a
        gather over slot k is one uniform shift (resident/ttfluid/resident/machine.py turns it into an offset load)."""
        m = mesh
        self.dtype = dtype
        nC, nInt = m.nCells, m.nInternalFaces
        o, n = m.owner[:nInt], m.neighbour
        # boundary faces in patch order become ghost entries nC .. nC + nB - 1
        bcells, bslices, off = [], {}, nC
        for p in m.patches:
            bslices[p.name] = slice(off, off + p.size)
            bcells.append(p.faceCells)
            off += p.size
        self.nGhost = off - nC
        self.ghost = bslices
        bcells = np.concatenate(bcells) if bcells else np.zeros(0, np.int64)

        # slot lists: (row cell, other entry, face id, side) with side +1 = owner side, -1 = neighbour side,
        # face id >= 0 for internal faces, -1 - k for ghost k
        rows = np.concatenate([o, n, bcells])
        other = np.concatenate([n, o, nC + np.arange(self.nGhost)])
        face = np.concatenate([np.arange(nInt), np.arange(nInt), -1 - np.arange(self.nGhost)])
        side = np.concatenate([np.ones(nInt), -np.ones(nInt), np.ones(self.nGhost)])
        fkey = np.where(face >= 0, face, nInt - 1 - face)   # face order within a row, ghosts after interior faces
        if order == "dir":
            gS = np.concatenate([p.Sf for p in m.patches]) if self.nGhost else np.zeros((0, 3))
            out = np.concatenate([m.Sf[:nInt], -m.Sf[:nInt], gS])
            d = np.argmax(np.abs(out), axis=1)
            fkey = (2 * d + (out[np.arange(len(d)), d] > 0)) * (nInt + self.nGhost + 1) + fkey
        order = np.lexsort((fkey, rows))
        rows, other, face, side = rows[order], other[order], face[order], side[order]
        counts = np.bincount(rows, minlength=nC)
        self.S = S = int(max_count(counts))
        start = np.zeros(nC + 1, np.int64)
        np.cumsum(counts, out=start[1:])
        slot = np.arange(len(rows)) - start[rows]
        # index arrays in int32 while every index fits (memory: (nC, S) of them, S the widest row)
        it = np.int32 if 2 * nInt + nC + self.nGhost < 2 ** 31 else np.int64
        self.nbr = np.tile(np.arange(nC, dtype=it)[:, None], (1, S))   # padding points at the cell itself, coefficient 0
        self.mask = np.zeros((nC, S), bool)
        self.face = np.full((nC, S), np.iinfo(it).min, it)
        self.side = np.zeros((nC, S), np.int8)
        self.nbr[rows, slot] = other
        self.mask[rows, slot] = True
        self.face[rows, slot] = face
        self.side[rows, slot] = side
        del rows, other, face, side, fkey, order, slot
        self.boundary = self.mask & (self.face < 0)
        self.internal = self.mask & (self.face >= 0)
        self.padding = float((~self.mask).sum()) / self.mask.size

        # per-slot geometry seen from the row cell; the ones used once or twice an iteration (w, corr, dself, dother,
        # dCC, dlim) are properties built on access from the mesh, the same expressions, so the same bits
        self.mesh, self.nCells = m, nC
        self._gS, self._gCf = (np.concatenate([getattr(p, k) for p in m.patches]) if self.nGhost else np.zeros((0, 3))
                               for k in ("Sf", "Cf"))
        fi = self.face[self.internal]
        sgn = self.side[self.internal]
        gk = -1 - self.face[self.boundary]
        Sf = np.zeros((nC, S, 3))
        Sf[self.internal] = m.Sf[fi] * sgn[:, None]
        Sf[self.boundary] = self._gS[gk]
        self.Sf = self._f(Sf)
        del Sf
        magSf = np.zeros((nC, S))
        magSf[self.internal] = m.magSf[fi]
        magSf[self.boundary] = np.linalg.norm(self._gS[gk], axis=1)
        self.magSf = self._f(magSf)
        dc = np.zeros((nC, S))         # nonOrthDeltaCoeffs (internal) / patch deltaCoeffs (boundary)
        dc[self.internal] = m.nonOrthDeltaCoeffs[fi]
        dc[self.boundary] = (np.concatenate([p.deltaCoeffs for p in m.patches]) if self.nGhost else np.zeros(0))[gk]
        self.dc = self._f(dc)
        wown = np.zeros((nC, S))
        wown[self.internal] = m.weights[fi]
        self.wown = self._f(wown)
        del magSf, dc, wown, fi, sgn, gk
        self.ghost_nf = self._f(np.concatenate([p.nf for p in m.patches]) if self.nGhost else np.zeros((0, 3)))
        self.V = self._f(m.V)

    def _f(self, a):
        return np.ascontiguousarray(a, dtype=self.dtype)

    def _vslots(self, r, where, vals):
        out = np.zeros(where.shape + (3,))
        out[where] = vals
        return self._f(out)

    def _rowcells(self, r, where):
        return np.nonzero(where)[0] + r.start

    @property
    def w(self):
        """Weight of the row cell's own value in the face interpolate."""
        I = self.internal
        fi, sgn = self.face[I], self.side[I]
        out = np.zeros(self.mask.shape)
        out[I] = np.where(sgn > 0, self.mesh.weights[fi], 1.0 - self.mesh.weights[fi])
        return self._f(out)

    # per-slot vectors of rows r (a slice), built from the mesh on access; the properties are all rows
    def slot_corr(self, r):
        """nonOrthCorrectionVectors, oriented from the row cell."""
        I = self.internal[r]
        return self._vslots(r, I, self.mesh.nonOrthCorrectionVectors[self.face[r][I]] * self.side[r][I][:, None])

    def slot_dself(self, r):
        """linearUpwind displacement from the face centre to the row cell's centre."""
        I, m = self.internal[r], self.mesh
        return self._vslots(r, I, m.Cf[self.face[r][I]] - m.C[self._rowcells(r, I)])

    def slot_dother(self, r):
        """... and to the other side's cell centre."""
        I, m = self.internal[r], self.mesh
        return self._vslots(r, I, m.Cf[self.face[r][I]] - m.C[self.nbr[r][I]])

    def slot_dlim(self, r):
        """cellLimited extrapolation: face centre minus the row cell's centre, boundary faces too."""
        d = self.slot_dself(r)
        B = self.boundary[r]
        d[B] = self._gCf[-1 - self.face[r][B]] - self.mesh.C[self._rowcells(r, B)]
        return d

    def slot_dCC(self, r):
        I, m = self.internal[r], self.mesh
        return self._vslots(r, I, m.C[self.nbr[r][I]] - m.C[self._rowcells(r, I)])

    corr = property(lambda self: self.slot_corr(slice(0, self.nCells)))
    dself = property(lambda self: self.slot_dself(slice(0, self.nCells)))
    dother = property(lambda self: self.slot_dother(slice(0, self.nCells)))
    dlim = property(lambda self: self.slot_dlim(slice(0, self.nCells)))
    dCC = property(lambda self: self.slot_dCC(slice(0, self.nCells)))

    # ---- the three primitives ------------------------------------------------------------------------------------
    def extend(self, x, ghosts):
        """x_ext = [cells ; ghost values] where ghosts maps patch name -> patch face values."""
        g = np.zeros((self.nGhost,) + x.shape[1:], dtype=x.dtype)
        for name, s in self.ghost.items():
            if name in ghosts:
                g[s.start - self.nCells:s.stop - self.nCells] = ghosts[name]
        return np.concatenate([x, g])

    def gather(self, x_ext):
        return x_ext[self.nbr]

    @staticmethod
    def rowsum(slots):
        return slots.sum(axis=1)

    # ---- operators built from them --------------------------------------------------------------------------------
    def face_values(self, x, x_ext):
        """Linear interpolate on every slot: w x_self + (1-w) x_other; boundary slots carry the ghost value."""
        xo = self.gather(x_ext)
        ws = self.w.reshape(self.w.shape + (1,) * (x.ndim - 1))
        return ws * x[:, None] + (1 - ws) * xo

    def grad(self, x, x_ext):
        """Gauss linear gradient, (n,3) for scalars and (n,3,3) with g_ij = d x_j / d x_i for vectors."""
        xf = self.face_values(x, x_ext)
        if x.ndim == 1:
            return self.rowsum(self.Sf * xf[..., None]) / self.V[:, None]
        return self.rowsum(self.Sf[..., :, None] * xf[..., None, :]) / self.V[:, None, None]
