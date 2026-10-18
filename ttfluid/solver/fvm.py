"""Implicit finite-volume operators and the LDU matrix (OpenFOAM fvm + fvMatrix).

A matrix is diag (nCells), lower/upper (nInternalFaces, row neighbour / row owner), source (nCells[, 3]) and per-patch
internalCoeffs/boundaryCoeffs, with OpenFOAM's sign convention: the system solved is A psi = source.
"""
import numpy as np

from . import fvc

SMALL = 1e-15


def _dot(a, b):
    """a & b for vectors, x + y + z in order (as OpenFOAM's inner product)."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _vdot(a, T):
    """a & T for a vector and a tensor (sum over the first tensor index, in order)."""
    return a[:, 0, None] * T[:, 0] + a[:, 1, None] * T[:, 1] + a[:, 2, None] * T[:, 2]


class FvMatrix:
    def __init__(self, psi):
        m = psi.mesh
        self.psi, self.mesh = psi, m
        self.diag = np.zeros(m.nCells)
        self.lower = np.zeros(m.nInternalFaces)
        self.upper = np.zeros(m.nInternalFaces)
        self.source = np.zeros_like(psi.internal)
        shp = lambda p: (p.size,) if psi.width == 1 else (p.size, psi.width)
        self.internalCoeffs = {k: np.zeros(shp(b.patch)) for k, b in psi.bc.items()}
        self.boundaryCoeffs = {k: np.zeros(shp(b.patch)) for k, b in psi.bc.items()}
        self.faceFluxCorrection = None   # explicit face flux added by corrected schemes (laplacian, linearUpwind)

    # ---- algebra -------------------------------------------------------------------------------------------
    def __iadd__(self, o):
        if isinstance(o, FvMatrix):
            self.diag += o.diag; self.lower += o.lower; self.upper += o.upper; self.source += o.source
            for k in self.internalCoeffs:
                self.internalCoeffs[k] += o.internalCoeffs[k]
                self.boundaryCoeffs[k] += o.boundaryCoeffs[k]
        else:  # explicit field su (per unit volume) on the left-hand side
            self.source -= _vol(self.mesh, o)
        return self

    def __isub__(self, o):
        if isinstance(o, FvMatrix):
            self.diag -= o.diag; self.lower -= o.lower; self.upper -= o.upper; self.source -= o.source
            for k in self.internalCoeffs:
                self.internalCoeffs[k] -= o.internalCoeffs[k]
                self.boundaryCoeffs[k] -= o.boundaryCoeffs[k]
        else:
            self.source += _vol(self.mesh, o)
        return self

    def negSumDiag(self):
        o, n = self.mesh.owner[:self.mesh.nInternalFaces], self.mesh.neighbour
        np.subtract.at(self.diag, o, self.lower)
        np.subtract.at(self.diag, n, self.upper)

    # ---- lduMatrix pieces ------------------------------------------------------------------------------------
    def amul(self, x, diag=None):
        """A x for the interior + diag part (no boundary coefficients)."""
        o, n = self.mesh.owner[:self.mesh.nInternalFaces], self.mesh.neighbour
        d = self.diag if diag is None else diag
        y = d * x
        np.add.at(y, o, self.upper * x[n])
        np.add.at(y, n, self.lower * x[o])
        return y

    def H_ldu(self, x):
        """-(off-diagonal) x."""
        o, n = self.mesh.owner[:self.mesh.nInternalFaces], self.mesh.neighbour
        shp = (-1,) + (1,) * (x.ndim - 1)
        y = np.zeros_like(x)
        np.subtract.at(y, o, self.upper.reshape(shp) * x[n])
        np.subtract.at(y, n, self.lower.reshape(shp) * x[o])
        return y

    def sumMagOffDiag(self):
        o, n = self.mesh.owner[:self.mesh.nInternalFaces], self.mesh.neighbour
        s = np.zeros(self.mesh.nCells)
        np.add.at(s, o, np.abs(self.upper))
        np.add.at(s, n, np.abs(self.lower))
        return s

    def _cmpt(self, a, c):
        return a if self.psi.width == 1 else a[:, c]

    def boundaryDiag(self, c):
        d = np.zeros(self.mesh.nCells)
        for k, b in self.psi.bc.items():
            np.add.at(d, b.patch.faceCells, self._cmpt(self.internalCoeffs[k], c))
        return d

    def cmptAvBoundaryDiag(self):
        d = np.zeros(self.mesh.nCells)
        for k, b in self.psi.bc.items():
            ic = self.internalCoeffs[k]
            np.add.at(d, b.patch.faceCells, ic if ic.ndim == 1 else ic.mean(1))
        return d

    def boundarySource(self):
        s = np.zeros_like(self.source)
        for k, b in self.psi.bc.items():
            np.add.at(s, b.patch.faceCells, self.boundaryCoeffs[k])
        return s

    # ---- fvMatrix operations ---------------------------------------------------------------------------------
    def relax(self, alpha):
        if alpha <= 0:
            return
        D0 = self.diag.copy()
        D = self.diag
        sumOff = self.sumMagOffDiag()
        for k, b in self.psi.bc.items():
            ic = self.internalCoeffs[k]
            np.add.at(D, b.patch.faceCells, np.abs(ic) if ic.ndim == 1 else np.abs(ic).max(1))
        D[:] = np.maximum(np.abs(D), sumOff)
        D /= alpha
        for k, b in self.psi.bc.items():
            ic = self.internalCoeffs[k]
            np.subtract.at(D, b.patch.faceCells, ic if ic.ndim == 1 else ic.min(1))
        x = self.psi.internal
        self.source += ((D - D0)[:, None] * x) if x.ndim > 1 else (D - D0) * x

    def A(self):
        return (self.diag + self.cmptAvBoundaryDiag()) / self.mesh.V

    def H(self):
        x = self.psi.internal
        if x.ndim == 1:
            h = (self.cmptAvBoundaryDiag() - self.boundaryDiag(0)) * x
        else:
            av = self.cmptAvBoundaryDiag()
            h = np.stack([(av - self.boundaryDiag(c)) * x[:, c] for c in range(x.shape[1])], 1)
        h += self.H_ldu(x) + self.source
        h += self.boundarySource()
        h /= self.mesh.V.reshape((-1,) + (1,) * (x.ndim - 1))
        if x.ndim > 1:
            h[:, ~self.mesh.solutionD] = 0.0
        return h

    def H1(self):
        o, n = self.mesh.owner[:self.mesh.nInternalFaces], self.mesh.neighbour
        h = np.zeros(self.mesh.nCells)
        np.subtract.at(h, n, self.lower)
        np.subtract.at(h, o, self.upper)
        return h / self.mesh.V

    def residual_parts(self, c):
        """(diag with boundary, source with boundary) for component c, as fvMatrix::solveSegregated builds them."""
        diag = self.diag + self.boundaryDiag(c)
        src = self.source + self.boundarySource()
        return diag, self._cmpt(src, c)

    def flux(self):
        """Face flux of the solved matrix: upper*psi_N - lower*psi_P on internal faces, plus patch fluxes."""
        o, n = self.mesh.owner[:self.mesh.nInternalFaces], self.mesh.neighbour
        x = self.psi.internal
        shp = (-1,) + (1,) * (x.ndim - 1)
        f = self.upper.reshape(shp) * x[n] - self.lower.reshape(shp) * x[o]
        if self.faceFluxCorrection is not None:
            f = f + self.faceFluxCorrection
        fb = {k: self.internalCoeffs[k] * x[b.patch.faceCells] - self.boundaryCoeffs[k]
              for k, b in self.psi.bc.items()}
        return f, fb

    def setValues(self, cells, values):
        """Fix psi in cells (fvMatrix::setValues): eliminate them from the system."""
        m = self.mesh
        x = self.psi.internal
        x[cells] = values
        nInt = m.nInternalFaces
        o, n = m.owner[:nInt], m.neighbour
        isfixed = np.zeros(m.nCells, bool)
        isfixed[cells] = True
        D = self.diag
        S = self.source
        # faces where the owner is fixed: move the neighbour's coupling to its source
        fi = np.nonzero(isfixed[o])[0]
        np.subtract.at(S, n[fi], self.lower[fi] * x[o[fi]])
        fj = np.nonzero(isfixed[n])[0]
        np.subtract.at(S, o[fj], self.upper[fj] * x[n[fj]])
        f = np.nonzero(isfixed[o] | isfixed[n])[0]
        self.upper[f] = 0.0
        self.lower[f] = 0.0
        S[cells] = D[cells] * values
        for k, b in self.psi.bc.items():
            fc = isfixed[b.patch.faceCells]
            self.internalCoeffs[k][fc] = 0.0
            self.boundaryCoeffs[k][fc] = 0.0


def _vol(mesh, su):
    su = np.asarray(su)
    return su * mesh.V.reshape((-1,) + (1,) * (su.ndim - 1))


# ---- implicit operators ----------------------------------------------------------------------------------------

def Sp(coeff, psi):
    """Implicit source coeff*psi (coeff per unit volume)."""
    M = FvMatrix(psi)
    M.diag += psi.mesh.V * coeff
    return M


def SuSp(coeff, psi):
    """Implicit where coeff > 0, explicit where coeff < 0 (fvm::SuSp)."""
    M = FvMatrix(psi)
    V = psi.mesh.V
    M.diag += V * np.maximum(coeff, 0.0)
    M.source -= V * np.minimum(coeff, 0.0) * psi.internal
    return M


def laplacian(gamma_f, gamma_b, psi, correct=True):
    """Gauss linear corrected laplacian(gamma, psi); gamma_f on internal faces, gamma_b per patch."""
    mesh = psi.mesh
    M = FvMatrix(psi)
    gms = gamma_f * mesh.magSf
    M.upper = mesh.nonOrthDeltaCoeffs * gms
    M.lower = M.upper.copy()
    M.negSumDiag()
    for k, b in psi.bc.items():
        pg = gamma_b[k] * b.patch.magSf
        pg = pg if psi.width == 1 else pg[:, None]
        M.internalCoeffs[k] = pg * b.gradientInternalCoeffs()
        M.boundaryCoeffs[k] = -pg * b.gradientBoundaryCoeffs()
    if correct:
        corr = nonorth_correction(psi)
        cf = (gms[:, None] * corr) if corr.ndim > 1 else gms * corr
        M.faceFluxCorrection = cf
        M.source -= fvc.scatter_faces(mesh, cf)
    return M


def nonorth_correction(psi, g=None):
    """nonOrthCorrectionVectors . interpolate(grad(psi)) on internal faces (correctedSnGrad::correction)."""
    mesh = psi.mesh
    if g is None:
        g, _ = fvc.grad(psi)
    gf = fvc.interpolate(mesh, g)
    if psi.width == 1:
        return _dot(mesh.nonOrthCorrectionVectors, gf)
    return _vdot(mesh.nonOrthCorrectionVectors, gf)


def upwind_weights(phi):
    return (phi >= 0).astype(float)


def limited_linear_weights(psi, phi, k=1.0):
    """limitedLinear(k) TVD weights (NVDTVD r, limiter max(min(2r/k, 1), 0))."""
    mesh = psi.mesh
    o, n = mesh.owner[:mesh.nInternalFaces], mesh.neighbour
    x = psi.internal
    g, _ = fvc.grad(psi)
    d = mesh.C[n] - mesh.C[o]
    gradf = x[n] - x[o]
    gradcf = np.where(phi > 0, _dot(d, g[o]), _dot(d, g[n]))
    big = np.abs(gradcf) >= 1000 * np.abs(gradf)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Foam::sign(s) = (s >= 0) ? 1 : -1 (sign(0) = +1, unlike np.sign): matters wherever gradf == 0,
        # e.g. everywhere in a uniform field, where OpenFOAM's limiter is 1 (linear), not 0 (upwind)
        fsign = lambda a: np.where(a >= 0, 1.0, -1.0)
        r = np.where(big, 2 * 1000 * fsign(gradcf) * fsign(gradf) - 1, 2 * (gradcf / gradf) - 1)
    lim = np.maximum(np.minimum(2.0 / max(k, SMALL) * r, 1.0), 0.0)
    return lim * mesh.weights + (1 - lim) * upwind_weights(phi)


def div(phi, phib, psi, scheme, bounded=True):
    """Gauss convection fvm::div(phi, psi). scheme: 'upwind' | 'linearUpwind' | ('limitedLinear', k) | 'linear'."""
    mesh = psi.mesh
    M = FvMatrix(psi)
    corr = None
    if scheme == "upwind":
        w = upwind_weights(phi)
    elif scheme == "linear":
        w = mesh.weights
    elif scheme == "linearUpwind":
        w = upwind_weights(phi)
        g, _ = fvc.grad(psi)
        o, n = mesh.owner[:mesh.nInternalFaces], mesh.neighbour
        dO = mesh.Cf - mesh.C[o]
        dN = mesh.Cf - mesh.C[n]
        if psi.width == 1:
            corr = np.where(phi > 0, _dot(dO, g[o]), _dot(dN, g[n]))
        else:
            corr = np.where((phi > 0)[:, None], _vdot(dO, g[o]), _vdot(dN, g[n]))
    elif isinstance(scheme, tuple) and scheme[0] == "limitedLinear":
        w = limited_linear_weights(psi, phi, scheme[1])
    else:
        raise ValueError(scheme)
    M.lower = -w * phi
    M.upper = M.lower + phi
    M.negSumDiag()
    for k, b in psi.bc.items():
        pf = phib[k] if psi.width == 1 else phib[k][:, None]
        M.internalCoeffs[k] = pf * b.valueInternalCoeffs(1.0)
        M.boundaryCoeffs[k] = -pf * b.valueBoundaryCoeffs(1.0)
    if corr is not None:
        ff = phi * corr if corr.ndim == 1 else phi[:, None] * corr
        M.source -= fvc.scatter_faces(mesh, ff)
    if bounded:
        M.diag -= fvc.scatter_faces(mesh, phi, phib)
    return M
