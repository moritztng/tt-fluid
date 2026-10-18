"""Finite-volume operator backends. The SIMPLE and k-omega SST algorithms are written once against this interface.

FaceOps   face-by-face loops with scatter, exactly as OpenFOAM's fvc/fvm (the oracle, checked against OpenFOAM).
EllOps    the device form: every face loop is gather -> per-slot arithmetic -> row sum on the cell-centric ELL
          layout (ell.Ell), Krylov solves on ELL SpMV. Runs in fp64 or fp32.

Both take and return cell arrays and per-patch dicts. A face quantity (flux, interpolated coefficient) is an opaque
"surface" value: a Surf (internal faces + patch dict) for FaceOps, a per-slot array for EllOps. Surfaces support
+, -, * among themselves and with scalars, which is all the algorithms do with them.
"""
import functools

import numpy as np

from . import fvc, fvm, linsolve
from .ell import Ell


# =================================================================================================================
# face backend
# =================================================================================================================

class Surf:
    """Face field: internal-face values and a patch dict."""
    __array_priority__ = 100

    def __init__(self, i, b):
        self.i, self.b = i, b

    def _op(self, o, f):
        if isinstance(o, Surf):
            return Surf(f(self.i, o.i), {k: f(v, o.b[k]) for k, v in self.b.items()})
        return Surf(f(self.i, o), {k: f(v, o) for k, v in self.b.items()})

    def __add__(self, o): return self._op(o, lambda a, b: a + b)
    def __sub__(self, o): return self._op(o, lambda a, b: a - b)
    def __mul__(self, o): return self._op(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._op(o, lambda a, b: b * a)
    def __radd__(self, o): return self._op(o, lambda a, b: b + a)
    def __neg__(self): return Surf(-self.i, {k: -v for k, v in self.b.items()})


class FaceOps:
    name = "face"

    def __init__(self, mesh, dtype=np.float64):
        if dtype != np.float64:
            raise ValueError("the face backend is the fp64 oracle")
        self.mesh, self.dtype = mesh, dtype
        self.V = mesh.V
        self.magSf = Surf(mesh.magSf, {p.name: p.magSf for p in mesh.patches if p.type != "empty"})

    def cells_on(self, x):
        """Patch dict of the face-cell values of x (extrapolated patch values)."""
        return {p.name: x[p.faceCells] for p in self.mesh.patches if p.type != "empty"}

    def grad(self, field):
        return fvc.grad(field)

    def interp(self, x, xb=None):
        return Surf(fvc.interpolate(self.mesh, x), self.cells_on(x) if xb is None else xb)

    def flux(self, v, vb):
        return Surf(*fvc.flux(v, vb, self.mesh))

    def surface(self, i, b):
        """A face field read from disk (internal faces in face order, {patch: values})."""
        return Surf(np.asarray(i, float), {k: np.asarray(v, float) for k, v in b.items()})

    def flux_tensor(self, T, Tb):
        """Sf & interpolate(T) for a tensor field (Gauss linear div of a tensor, before the surface integral)."""
        m = self.mesh
        return Surf(fvm._vdot(m.Sf, fvc.interpolate(m, T)), {k: fvm._vdot(m.patch[k].Sf, v) for k, v in Tb.items()})

    def snGrad(self, field, g):
        """corrected snGrad: orthogonal part + non-orthogonal correction from the cell gradient g."""
        m = self.mesh
        x = field.internal
        o, n = m.owner[:m.nInternalFaces], m.neighbour
        i = m.nonOrthDeltaCoeffs * (x[n] - x[o]) + fvm.nonorth_correction(field, g)
        return Surf(i, {k: b.snGrad() for k, b in field.bc.items()})

    def surface_integrate(self, s):
        return fvc.surface_integrate(self.mesh, s.i, s.b)

    def face_average(self, s):
        """Area-weighted average over each cell's faces of a face scalar (bound())."""
        m = self.mesh
        o, n = m.owner[:m.nInternalFaces], m.neighbour
        num = np.zeros(m.nCells)
        den = np.zeros(m.nCells)
        sf = s.i * m.magSf
        np.add.at(num, o, sf); np.add.at(num, n, sf)
        np.add.at(den, o, m.magSf); np.add.at(den, n, m.magSf)
        for k, v in s.b.items():
            p = m.patch[k]
            np.add.at(num, p.faceCells, p.magSf * v)
            np.add.at(den, p.faceCells, p.magSf)
        return num / den

    def div(self, phi, field, scheme, bounded=True):
        return FaceMatrix(fvm.div(phi.i, phi.b, field, scheme, bounded=bounded))

    def laplacian(self, gamma, field):
        return FaceMatrix(fvm.laplacian(gamma.i, gamma.b, field))

    def Sp(self, coeff, field):
        return FaceMatrix(fvm.Sp(coeff, field))

    def SuSp(self, coeff, field):
        return FaceMatrix(fvm.SuSp(coeff, field))


class FaceMatrix:
    """fvm.FvMatrix behind the backend-neutral matrix interface."""

    def __init__(self, M):
        self.M = M

    @property
    def source(self):
        return self.M.source

    @source.setter
    def source(self, v):
        self.M.source = v

    def __iadd__(self, o):
        self.M += o.M
        return self

    def __isub__(self, o):
        self.M -= o.M
        return self

    def relax(self, alpha):
        self.M.relax(alpha)

    def A(self):
        return self.M.A()

    def H(self):
        return self.M.H()

    def H1(self):
        return self.M.H1()

    def flux(self):
        return Surf(*self.M.flux())

    def setValues(self, cells, values):
        self.M.setValues(cells, values)

    def solve(self, controls, symmetric=False):
        r, _ = linsolve.solve(self.M, controls, "pcg" if symmetric else "pbicgstab")
        return r


# =================================================================================================================
# ELL (device) backend
# =================================================================================================================

def _b(a, x):
    """Broadcast a per-slot / per-cell array against a field-shaped x."""
    return a.reshape(a.shape + (1,) * (x.ndim - a.ndim))


class EllOps:
    name = "ell"

    def __init__(self, mesh, dtype=np.float64, order="face", chunk=1 << 16):
        self.mesh, self.dtype = mesh, dtype
        self.chunk = chunk        # rows per block of the host (numpy) face loops: bounds their slot temporaries
        self.e = e = Ell(mesh, dtype, order)
        self.V = e.V
        self.magSf = e.magSf
        self.gc = np.concatenate([p.faceCells for p in mesh.patches]) if e.nGhost else np.zeros(0, np.int64)
        self.gk = np.where(e.boundary, -1 - e.face, 0)
        # interior-only gather index; ghost slots (masked out) read the cell itself, so a partition sees no halo
        # traffic for them (clamping them to the last cell made every core gather from that cell's core)
        self.cnbr = np.where(e.nbr < e.nCells, e.nbr, np.arange(e.nCells, dtype=e.nbr.dtype)[:, None])
        self.device = None        # a device.DeviceSpmv: the pressure solve's SpMV runs on it, and with
        self.device_all = False   # device_all every linear solve's SpMV (U, k, omega BiCGStab too)
        self.grad_limit = set()   # fields whose grad is cellLimited Gauss linear 1 (fvSchemes gradSchemes)

    @staticmethod
    def krylov(symmetric):
        return linsolve.pcg if symmetric else linsolve.pbicgstab

    # ---- ghost plumbing -------------------------------------------------------------------------------------------
    def ghosts(self, d, x):
        """Ghost-ordered array from a patch dict (patches missing from d, e.g. empty ones, have no ghosts)."""
        e = self.e
        g = np.zeros((e.nGhost,) + x.shape[1:], self.dtype)
        for k, s in e.ghost.items():
            if k in d:
                g[s.start - e.nCells:s.stop - e.nCells] = d[k]
        return g

    def bslots(self, d, like):
        """Per-slot array (zero off the boundary slots) from a patch dict."""
        return self._bslots_rows(self.ghosts(d, like), slice(None))

    def _bslots_rows(self, gh, r):
        """bslots of rows r from a ghost array, in place (the off-boundary slots set to 0, as np.where gives)."""
        g = gh[self.gk[r]]
        g[~self.e.boundary[r]] = 0
        return g

    def rowmax(self, s, init):
        out = init
        for j in range(s.shape[1]):
            out = np.maximum(out, s[:, j])
        return out

    def rowmin(self, s, init):
        out = init
        for j in range(s.shape[1]):
            out = np.minimum(out, s[:, j])
        return out

    def rowsum(self, s):
        """Sum over slots in slot order (fixed, deterministic: what a core does)."""
        out = s[:, 0].copy()
        for j in range(1, s.shape[1]):
            out += s[:, j]
        return out

    def rowdot(self, a, b):
        """rowsum(a * b), slot by slot: a core then folds each slot's product as it forms it (same order, same sum)."""
        return self.rowsum(a * b)

    def cells_on(self, x):
        return {p.name: x[p.faceCells] for p in self.mesh.patches if p.type != "empty"}

    # ---- face values ------------------------------------------------------------------------------------------------
    def interp(self, x, xb=None):
        """Linear interpolate per slot as OpenFOAM orders it: w_own (x_own - x_nei) + x_nei, the same operands from
        both sides of a face; boundary slots carry the patch value (default: the cell value)."""
        e = self.e
        if not hasattr(x, "space"):     # a traced array passes through
            return self._interp_host(np.asarray(x, self.dtype), xb, slice(0, self.e.nCells))
        xo = x[self.cnbr]
        xs = np.broadcast_to(x[:, None], xo.shape)
        own = _b(e.side > 0, xo)
        xP, xN = np.where(own, xs, xo), np.where(own, xo, xs)
        f = _b(e.wown, xo) * (xP - xN) + xN
        xbd = self.cells_on(x) if xb is None else xb
        bv = self.bslots(xbd, x)
        f = np.where(_b(e.boundary, f), bv, f)
        return np.where(_b(e.mask, f), f, 0)

    def _rows(self):
        n, c = self.e.nCells, max(int(self.chunk), 1)
        return [slice(a, min(a + c, n)) for a in range(0, n, c)]

    def _interp_host(self, x, xb, r, g=None):
        """interp of rows r on numpy arrays, in place: the same operations in the same order, two slot arrays live,
        not six. g: the ghost array, when the caller already built it."""
        e = self.e
        cn = self.cnbr[r]
        xs = np.broadcast_to(x[r, None], cn.shape + x.shape[1:])
        own = _b(e.side[r] > 0, xs)
        f = x[cn]                                  # x_P - x_N, then w_own (x_P - x_N) + x_N
        np.copyto(f, xs, where=own)
        xN = x[cn]
        np.copyto(xN, xs, where=~own)
        f -= xN
        f *= _b(e.wown[r], f)
        f += xN
        del xN
        if g is None:
            g = self.ghosts(self.cells_on(x) if xb is None else xb, x)
        bd = e.boundary[r]
        f[bd] = g[self.gk[r][bd]]
        f[~e.mask[r]] = 0
        return f

    def _host_ghosts(self, x, xb):
        return self.ghosts(self.cells_on(x) if xb is None else xb, x)

    def flux(self, v, vb):
        return _dot(self.e.Sf, self.interp(v, vb))

    def surface(self, i, b):
        """A face field read from disk as per-slot values, signed outward from each row cell."""
        e, m = self.e, self.mesh
        out = np.zeros(e.mask.shape, self.dtype)
        out[e.internal] = np.asarray(i, float)[e.face[e.internal]] * e.side[e.internal]
        g = np.zeros(e.nGhost)
        for name, sl in e.ghost.items():
            if name in b:
                g[sl.start - e.nCells:sl.stop - e.nCells] = b[name]
        out[e.boundary] = g[-1 - e.face[e.boundary]]
        return out

    def flux_tensor(self, T, Tb):
        if hasattr(T, "space"):
            return _vdot(self.e.Sf, self.interp(T, Tb))
        T = np.asarray(T, self.dtype)
        g = self._host_ghosts(T, Tb)
        out = np.empty((self.e.nCells, self.e.S) + T.shape[2:], self.dtype)
        for r in self._rows():
            out[r] = _vdot(self.e.Sf[r], self._interp_host(T, Tb, r, g))
        return out

    def grad(self, field):
        """Gauss linear gradient and its patch values (normal part replaced by the patch snGrad, as gaussGrad)."""
        e = self.e
        x = field.internal
        xb = {k: b.value for k, b in field.bc.items()}
        if hasattr(x, "space"):
            xf = self.interp(x, xb)
            if x.ndim == 1:
                g = self.rowsum(e.Sf * xf[..., None]) / e.V[:, None]
            else:
                g = self.rowsum(e.Sf[..., :, None] * xf[..., None, :]) / e.V[:, None, None]
        else:
            x = np.asarray(x, self.dtype)
            gh = self._host_ghosts(x, xb)
            g = np.empty((e.nCells, 3) + x.shape[1:], self.dtype)
            for r in self._rows():
                xf = self._interp_host(x, xb, r, gh)
                if x.ndim == 1:
                    g[r] = self.rowsum(e.Sf[r] * xf[..., None]) / e.V[r, None]
                else:
                    g[r] = self.rowsum(e.Sf[r][..., :, None] * xf[..., None, :]) / e.V[r, None, None]
            del xf
        if getattr(field, "name", None) in self.grad_limit:
            g = self._cell_limited(field, g)
        gb = {}
        for k, b in field.bc.items():
            p = b.patch
            gi = g[p.faceCells]
            nf = p.nf.astype(self.dtype)
            sn = b.snGrad()
            if x.ndim == 1:
                gb[k] = gi + nf * (sn - _dot(nf, gi))[:, None]
            else:
                gb[k] = gi + nf[:, :, None] * (sn - _vdot(nf, gi))[:, None, :]
            gb[k] = gb[k].astype(self.dtype)
        return g, gb

    def _cell_limited(self, field, g):
        """cellLimited<minmod> Gauss linear 1 (cellLimitedGrad::calcGrad): per cell and component, the extrapolation to
        every face centre may not leave [min, max] of the cell, its face neighbours and its patch values."""
        x = field.internal
        bvd = {k: b.value for k, b in field.bc.items()}
        if hasattr(x, "space") or hasattr(g, "space"):
            return self._cell_limited_traced(x, g, bvd)
        gh = self.ghosts(bvd, x)
        out = np.empty(g.shape, g.dtype)
        for r in self._rows():
            out[r] = self._cell_limited_rows(x, g, self._bslots_rows(gh, r), r)
        return out

    def _cell_limited_traced(self, x, g, bvd):
        e = self.e
        xo = x[self.cnbr]
        xs = np.broadcast_to(x[:, None], xo.shape)
        bv = self.bslots(bvd, x)
        nb = np.where(_b(e.internal, xo), xo, np.where(_b(e.boundary, bv), bv, xs))
        maxD = self.rowmax(nb, x) - x
        minD = self.rowmin(nb, x) - x
        if x.ndim == 1:
            ext = _dot(e.dlim, g[:, None, :])
        else:
            ext = _vdot(e.dlim, g[:, None])
        small = 1e-15
        pos, neg = ext > small, ext < -small
        r = np.where(pos, maxD[:, None] / np.where(pos, ext, 1.0),
                     np.where(neg, minD[:, None] / np.where(neg, ext, 1.0), 1.0))
        r = np.where(_b(e.mask, r), np.minimum(r, 1.0), 1.0)
        lim = self.rowmin(r, np.ones_like(x))
        return g * lim[:, None] if x.ndim == 1 else g * lim[:, None, :]

    def _cell_limited_rows(self, x, g, bv, r):
        """_cell_limited on host rows r: the same operations in the same order, one block of slot temporaries."""
        e = self.e
        cn = self.cnbr[r]
        dlim = e.slot_dlim(r)
        xr, gr = x[r], g[r]
        xo = x[cn]
        xs = np.broadcast_to(xr[:, None], xo.shape)
        nb = np.where(_b(e.internal[r], xo), xo, np.where(_b(e.boundary[r], bv), bv, xs))
        del xo
        maxD = self.rowmax(nb, xr) - xr
        minD = self.rowmin(nb, xr) - xr
        del nb
        if x.ndim == 1:
            ext = _dot(dlim, gr[:, None, :])
        else:
            ext = _vdot(dlim, gr[:, None])
        del dlim
        small = 1e-15
        pos, neg = ext > small, ext < -small
        q = np.where(pos, maxD[:, None] / np.where(pos, ext, 1.0),
                     np.where(neg, minD[:, None] / np.where(neg, ext, 1.0), 1.0))
        del pos, neg, ext
        q = np.where(_b(e.mask[r], q), np.minimum(q, 1.0), 1.0)
        lim = self.rowmin(q, np.ones_like(xr))
        return gr * lim[:, None] if x.ndim == 1 else gr * lim[:, None, :]

    def nonorth_corr(self, g):
        e = self.e
        if hasattr(g, "space"):
            gf = self.interp(g)
            c = _dot(e.corr, gf) if g.ndim == 2 else _vdot(e.corr, gf)
            return np.where(_b(e.internal, c), c, 0)
        g = np.asarray(g, self.dtype)
        gh = self._host_ghosts(g, None)
        out = np.empty((e.nCells, e.S) + g.shape[2:], self.dtype)
        for r in self._rows():
            gf = self._interp_host(g, None, r, gh)
            cr = e.slot_corr(r)
            c = _dot(cr, gf) if g.ndim == 2 else _vdot(cr, gf)
            out[r] = np.where(_b(e.internal[r], c), c, 0)
        return out

    def snGrad(self, field, g):
        e = self.e
        x = field.internal
        bvd = {k: b.snGrad() for k, b in field.bc.items()}
        if hasattr(x, "space") or hasattr(g, "space"):
            xo = x[self.cnbr]
            i = e.dc * (xo - x[:, None]) + self.nonorth_corr(g)
            bv = self.bslots(bvd, x)
            return np.where(e.boundary, bv, np.where(e.internal, i, 0))
        g = np.asarray(g, self.dtype)
        gg = self._host_ghosts(g, None)
        gb = self.ghosts(bvd, x)
        out = np.empty((e.nCells, e.S), self.dtype)
        for r in self._rows():
            gf = self._interp_host(g, None, r, gg)
            cr = e.slot_corr(r)
            c = _dot(cr, gf) if g.ndim == 2 else _vdot(cr, gf)
            del gf, cr
            c = np.where(_b(e.internal[r], c), c, 0)
            i = e.dc[r] * (x[self.cnbr[r]] - x[r, None]) + c
            del c
            out[r] = np.where(e.boundary[r], self._bslots_rows(gb, r), np.where(e.internal[r], i, 0))
        return out

    def surface_integrate(self, s):
        return self.rowsum(s) / _b(self.e.V, s[:, 0])

    def face_average(self, s):
        e = self.e
        return self.rowsum(s * e.magSf) / self.rowsum(e.magSf)

    # ---- matrices ------------------------------------------------------------------------------------------------------
    def _scale_bc(self, s, ic, bc):
        """(s ic, -s bc) per slot; in place on numpy operands (s ic == ic s and -(s bc) == (-s) bc, bit for bit)."""
        if hasattr(s, "space") or not isinstance(s, np.ndarray) or ic.dtype != np.result_type(s, ic):
            return _b(s, ic) * ic, -_b(s, bc) * bc
        ic *= _b(s, ic)
        bc *= _b(s, bc)
        np.negative(bc, out=bc)
        return ic, bc

    def _bc_coeffs(self, field, kind):
        ic, bc = {}, {}
        for k, b in field.bc.items():
            if kind == "value":
                ic[k], bc[k] = b.valueInternalCoeffs(1.0), b.valueBoundaryCoeffs(1.0)
            else:
                ic[k], bc[k] = b.gradientInternalCoeffs(), b.gradientBoundaryCoeffs()
        return self.bslots(ic, field.internal), self.bslots(bc, field.internal)

    def laplacian(self, gamma, field):
        e = self.e
        M = EllMatrix(self, field)
        gms = gamma * e.magSf
        M.off = np.where(e.internal, e.dc * gms, 0)
        M.diag = -self.rowsum(M.off)
        ic, bc = self._bc_coeffs(field, "gradient")
        M.bdiag, M.bsrc = self._scale_bc(gms, ic, bc)
        g, _ = self.grad(field)
        corr = self.nonorth_corr(g)
        if type(corr) is np.ndarray and type(gms) is np.ndarray and corr.dtype == np.result_type(gms, corr):
            corr *= _b(gms, corr)                    # gms corr == corr gms, in place on the host
            fc = corr
        else:
            fc = _b(gms, corr) * corr
        del corr
        M.fluxCorr = fc
        M.source = M.source - self.rowsum(fc)
        return M

    def sub_laplacian(self, M, gamma, field):
        """M -= laplacian(gamma, field), each coefficient array subtracted as soon as it is built, so the laplacian's
        matrix never exists whole (host numpy only; otherwise the plain merge). Same values, same operations."""
        e = self.e
        x = field.internal
        if (type(x) is not np.ndarray or type(gamma) is not np.ndarray
                or not all(_inplace(getattr(M, k), getattr(M, k)) for k in ("diag", "off", "bdiag", "bsrc", "source"))):
            M -= self.laplacian(gamma, field)
            return M
        gms = gamma * e.magSf
        off = np.where(e.internal, e.dc * gms, 0)
        M._sub("diag", -self.rowsum(off))
        M._sub("off", off)
        del off
        ic, bc = self._bc_coeffs(field, "gradient")
        bdiag, bsrc = self._scale_bc(gms, ic, bc)
        del ic, bc
        M._sub("bdiag", bdiag)
        del bdiag
        M._sub("bsrc", bsrc)
        del bsrc
        g, _ = self.grad(field)
        corr = self.nonorth_corr(g)
        del g
        if corr.dtype == np.result_type(gms, corr):
            corr *= _b(gms, corr)
            fc = corr
        else:
            fc = _b(gms, corr) * corr
        del corr
        M._sub("source", np.zeros(M.source.shape, self.dtype) - self.rowsum(fc))
        if M.fluxCorr is None:
            M.fluxCorr = -1 * fc
        else:
            M.fluxCorr = M.fluxCorr + -1 * fc
        return M

    def div(self, phi, field, scheme, bounded=True):
        """Gauss convection with OpenFOAM's weights; upwind decisions taken on the owner-oriented flux."""
        e = self.e
        x = field.internal
        M = EllMatrix(self, field)
        own = e.side > 0
        phiO = np.where(own, phi, -phi)              # owner-oriented flux of each slot's face
        corr = None
        if scheme in ("upwind", "linearUpwind", "linearUpwindV"):
            wO = (phiO >= 0).astype(self.dtype)
        elif scheme == "linear":
            wO = e.wown
        elif isinstance(scheme, tuple) and scheme[0] == "limitedLinear":
            wO = self._limited_linear_wown(field, phiO, scheme[1])
        else:
            raise ValueError(scheme)
        wS = np.where(own, wO, 1 - wO)               # weight of this row's own value
        del wO
        M.off = np.where(e.internal, phi * (1 - wS), 0)
        M.diag = self.rowsum(np.where(e.internal, wS * phi, 0))
        del wS
        if scheme not in ("linearUpwind", "linearUpwindV"):
            del phiO, own
        ic, bc = self._bc_coeffs(field, "value")
        pf = np.where(e.boundary, phi, 0)
        M.bdiag, M.bsrc = self._scale_bc(pf, ic, bc)
        del pf
        if scheme in ("linearUpwind", "linearUpwindV"):
            g, _ = self.grad(field)
            up_self = np.where(own, phiO > 0, phiO <= 0)
            if hasattr(x, "space") or hasattr(g, "space") or hasattr(phi, "space"):
                ff = self._lu_flux(x, g, phi, phiO, own, up_self, scheme, None, e.dother, e.dself)
                M.source = M.source - self.rowsum(ff)
            else:
                rs = np.empty(M.source.shape, self.dtype)
                for r in self._rows():
                    ff = self._lu_flux(x, g, phi[r], phiO[r], own[r], up_self[r], scheme, r, e.slot_dother(r),
                                       e.slot_dself(r))
                    rs[r] = self.rowsum(ff)
                del ff
                M.source = M.source - rs
        if bounded:
            M.diag = M.diag - self.rowsum(phi)
        return M

    def _lu_flux(self, x, g, phi, phiO, own, up_self, scheme, r, dother, dself):
        """phi x linearUpwind(V) correction on the slots of rows r (all rows on the traced path)."""
        e = self.e
        cn = self.cnbr if r is None else self.cnbr[r]   # the traced path needs the array itself (index space by id)
        r = slice(None) if r is None else r
        go = g[cn]
        if hasattr(go, "space"):
            gup = np.where(_b(up_self, go), np.broadcast_to(g[:, None], go.shape), go)
            d = np.where(up_self[..., None], dself, dother)
        else:
            gup = go
            np.copyto(gup, np.broadcast_to(g[r, None], gup.shape), where=_b(up_self, gup))
            d = dother
            np.copyto(d, dself, where=up_self[..., None])
        del go
        corr = _vdot(d, gup) if x.ndim > 1 else _dot(d, gup)
        del gup, d
        if scheme == "linearUpwindV" and x.ndim > 1:
            # linearUpwindV::correction: the correction may not pass the linear-interpolate's own difference
            xo = x[cn]
            xs = np.broadcast_to(x[r, None], xo.shape)
            xP, xN = np.where(_b(own, xo), xs, xo), np.where(_b(own, xo), xo, xs)
            wl = _b(e.wown[r], xo)
            maxCorr = np.where(_b(phiO > 0, xo), (1.0 - wl) * (xN - xP), wl * (xP - xN))
            sfCorrs = _dot(corr, corr)
            maxCorrs = _dot(corr, maxCorr)
            vsmall = 1e-300 if np.dtype(self.dtype) == np.float64 else 1e-37
            f = np.where(sfCorrs > 0, np.where(maxCorrs < 0, 0.0,
                         np.where(sfCorrs > maxCorrs, maxCorrs / (sfCorrs + vsmall), 1.0)), 1.0)
            corr = corr * f[..., None]
        ff = _b(phi, corr) * corr
        return np.where(_b(e.internal[r], ff), ff, 0)

    def _limited_linear_wown(self, field, phiO, k):
        """limitedLinear(k) owner weight per slot (NVDTVD r; OpenFOAM's sign(0) = +1)."""
        e = self.e
        x = field.internal
        g, _ = self.grad(field)
        xo = x[self.cnbr]
        own = e.side > 0
        xP, xN = np.where(own, x[:, None], xo), np.where(own, xo, x[:, None])
        gs, go = np.broadcast_to(g[:, None], (e.nCells, e.S, 3)), g[self.cnbr]
        gP, gN = np.where(own[..., None], gs, go), np.where(own[..., None], go, gs)
        d = np.where(own[..., None], e.dCC, -e.dCC)  # C_nei - C_own
        gradf = xN - xP
        gradcf = np.where(phiO > 0, _dot(d, gP), _dot(d, gN))
        sgn = lambda a: np.where(a >= 0, 1.0, -1.0)
        big = np.abs(gradcf) >= 1000 * np.abs(gradf)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(big, 2 * 1000 * sgn(gradcf) * sgn(gradf) - 1, 2 * (gradcf / gradf) - 1)
        lim = np.maximum(np.minimum((2.0 / max(k, 1e-15)) * r, 1.0), 0.0)
        return (lim * e.wown + (1 - lim) * (phiO >= 0)).astype(self.dtype)

    def Sp(self, coeff, field):
        M = EllMatrix(self, field)
        M.diag = M.diag + self.V * coeff
        return M

    def SuSp(self, coeff, field):
        M = EllMatrix(self, field)
        M.diag = M.diag + self.V * np.maximum(coeff, 0)
        M.source = M.source - self.V * np.minimum(coeff, 0) * field.internal
        return M


def _inplace(a, b):
    """a may take a op= b in place: both plain host arrays of one shape and dtype, a owning its memory."""
    return (type(a) is np.ndarray and type(b) is np.ndarray and a.shape == b.shape and a.dtype == b.dtype
            and a.flags.writeable and a.base is None)


class EllMatrix:
    """A x = source on the ELL layout: diag + off (interior slots); boundary slots split as OpenFOAM's
    internalCoeffs (bdiag, per component) and boundaryCoeffs (bsrc)."""

    def __init__(self, ops, field):
        e = ops.e
        self.ops, self.e, self.psi = ops, e, field
        dt, w = ops.dtype, field.width
        cw = () if w == 1 else (w,)
        self.diag = np.zeros(e.nCells, dt)
        self.off = np.zeros((e.nCells, e.S), dt)
        self.bdiag = np.zeros((e.nCells, e.S) + cw, dt)
        self.bsrc = np.zeros((e.nCells, e.S) + cw, dt)
        self.source = np.zeros((e.nCells,) + cw, dt)
        self.fluxCorr = None

    def _sub(self, k, b):
        a = getattr(self, k)
        if _inplace(a, b):
            np.subtract(a, b, out=a)
        else:
            setattr(self, k, a + -1 * b)

    def _merge(self, o, sgn):
        for k in ("diag", "off", "bdiag", "bsrc", "source"):
            a, b = getattr(self, k), getattr(o, k)
            if _inplace(a, b) and not np.shares_memory(a, b):
                (np.add if sgn > 0 else np.subtract)(a, b, out=a)   # a + 1*b == a + b, a + -1*b == a - b
            else:
                setattr(self, k, a + sgn * b)
        if o.fluxCorr is not None:
            self.fluxCorr = sgn * o.fluxCorr if self.fluxCorr is None else self.fluxCorr + sgn * o.fluxCorr
        return self

    def __iadd__(self, o):
        return self._merge(o, 1)

    def __isub__(self, o):
        return self._merge(o, -1)

    def _rs(self, a):
        return self.ops.rowsum(a)

    def offx(self, x):
        if type(x) is not np.ndarray or type(self.off) is not np.ndarray:
            xo = x[self.ops.cnbr]
            return self._rs(_b(self.off, xo) * xo)
        out = np.empty(x.shape, np.result_type(self.off, x))
        for r in self.ops._rows():
            xo = x[self.ops.cnbr[r]]
            out[r] = self._rs(_b(self.off[r], xo) * xo)
        return out

    def bdiag_c(self, c):
        return self._rs(self.bdiag if self.psi.width == 1 else self.bdiag[..., c])

    def bdiag_av(self):
        return self._rs(self.bdiag if self.psi.width == 1 else self.bdiag.mean(-1))

    def relax(self, alpha):
        if alpha <= 0:
            return
        x = self.psi.internal
        D0 = self.diag.copy()
        bd = self.bdiag
        bmax = np.abs(bd) if bd.ndim == 2 else np.abs(bd).max(-1)
        bmin = bd if bd.ndim == 2 else bd.min(-1)
        D = self.diag + self._rs(bmax)
        D = np.maximum(np.abs(D), self._rs(np.abs(self.off))) / alpha
        D = D - self._rs(bmin)
        self.diag = D
        self.source = self.source + _b(D - D0, x) * x

    def A(self):
        return (self.diag + self.bdiag_av()) / self.e.V

    def H(self):
        x = self.psi.internal
        av = self.bdiag_av()
        if x.ndim == 1:
            h = (av - self.bdiag_c(0)) * x
        else:
            h = np.stack([(av - self.bdiag_c(c)) * x[:, c] for c in range(x.shape[1])], 1)
        h = h - self.offx(x) + self.source + self._rs(self.bsrc)
        h = h / _b(self.e.V, h)
        if x.ndim > 1:
            h[:, ~self.ops.mesh.solutionD] = 0
        return h

    def H1(self):
        return -self._rs(self.off) / self.e.V

    def flux(self):
        """Outward face flux per slot (fvMatrix::flux): off (x_other - x_self) on interior slots (a symmetric matrix
        gives the owner's upper x_N - lower x_P from either side), internalCoeffs x - boundaryCoeffs on patches."""
        e = self.e
        x = self.psi.internal
        f = np.where(e.internal, self.off * (x[self.ops.cnbr] - x[:, None]), 0)
        f = f + np.where(e.boundary, self.bdiag * x[:, None] - self.bsrc, 0)
        if self.fluxCorr is not None:
            f = f + self.fluxCorr
        return f

    def setValues(self, cells, values):
        e = self.e
        x = self.psi.internal
        x[cells] = values
        fixed = np.zeros(e.nCells, bool)
        fixed[cells] = True
        nb_fixed = e.internal & fixed[self.ops.cnbr]
        self.source = self.source - self._rs(np.where(nb_fixed, self.off * x[self.ops.cnbr], 0))
        self.off = np.where(nb_fixed | fixed[:, None], 0, self.off)
        self.bdiag[cells] = 0
        self.bsrc[cells] = 0
        self.source[cells] = self.diag[cells] * values

    def solve(self, controls, symmetric=False):
        psi = self.psi
        x = psi.internal
        bsrc = self.source + self._rs(self.bsrc)
        comps = [None] if psi.width == 1 else [c for c in range(psi.width) if self.ops.mesh.solutionD[c]]
        perfs = []
        for c in comps:
            diag = self.diag + self.bdiag_c(0 if c is None else c)
            b = bsrc if c is None else bsrc[:, c]
            A = _EllOp(diag, self.off, self.ops.cnbr, self.ops.rowsum, self.ops.rowdot)
            if self.ops.device is not None and (symmetric or self.ops.device_all):   # SpMV on the chip
                A = _DeviceOp(A, self.ops.device)
            xc = (x if c is None else x[:, c]).copy()
            perf = self.ops.krylov(symmetric)(A, xc, b, controls)
            if c is None:
                x[:] = xc
            else:
                x[:, c] = xc
            perfs.append(perf)
        psi.correct_bc()
        return functools.reduce(np.maximum, [p.initial for p in perfs])


class _DeviceOp:
    """The ELL SpMV run by a device.DeviceSpmv; sumA (normFactor only, once per solve) stays on the host."""

    def __init__(self, A, dev):
        self.A, self.dev, self.d = A, dev, A.d
        dev.set_matrix(A.d, A.off)

    def __call__(self, x):
        return self.dev(x).astype(x.dtype)

    def sumA(self):
        return self.A.sumA()


class _EllOp:
    """y = diag x + rowsum(off * x[nbr]): the ELL SpMV."""

    def __init__(self, d, off, nbr, rowsum, rowdot=None):
        self.d, self.off, self.nbr, self.rowsum = d, off, nbr, rowsum
        self.rowdot = rowdot or (lambda a, b: rowsum(a * b))

    def __call__(self, x):
        # off-diagonal sum first: d*x then forms in the run that consumes it instead of being stored (same sum)
        return self.rowdot(self.off, x[self.nbr]) + self.d * x

    def sumA(self):
        return self.d + self.rowsum(self.off)


def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _vdot(a, T):
    """a & T: sum over T's first vector index."""
    return a[..., 0, None] * T[..., 0, :] + a[..., 1, None] * T[..., 1, :] + a[..., 2, None] * T[..., 2, :]
