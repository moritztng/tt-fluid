"""Explicit finite-volume operators (OpenFOAM fvc). Every face loop here is one of three primitives:
gather cell values to faces, combine per face, scatter-add face values to cells."""
import numpy as np


def scatter_faces(mesh, internal_face_vals, patch_face_vals=None):
    """Sum of face values into cells: +owner, -neighbour (the divergence pattern). Shapes (nInt, ...).

    Accumulated in OpenFOAM's order (face by face, owner then neighbour; then patches in order), so sums of
    near-cancelling terms round exactly as in fvc::surfaceIntegrate / gaussGrad::gradf. np.add.at applies its
    updates sequentially in index order."""
    nInt = mesh.nInternalFaces
    o, n = mesh.owner[:nInt], mesh.neighbour
    v = np.asarray(internal_face_vals)
    out = np.zeros((mesh.nCells,) + v.shape[1:])
    idx = np.empty(2 * nInt, dtype=o.dtype)
    idx[0::2], idx[1::2] = o, n
    vals = np.empty((2 * nInt,) + v.shape[1:])
    vals[0::2], vals[1::2] = v, -v
    np.add.at(out, idx, vals)
    if patch_face_vals:
        for p in mesh.patches:
            pv = patch_face_vals.get(p.name)
            if pv is not None and p.size:
                np.add.at(out, p.faceCells, pv)
    return out


def interpolate(mesh, x, w=None):
    """Linear (or given-weight) interpolation of cell values x to internal faces."""
    w = mesh.weights if w is None else w
    o, n = mesh.owner[:mesh.nInternalFaces], mesh.neighbour
    ws = w.reshape((-1,) + (1,) * (x.ndim - 1))
    # surfaceInterpolationScheme::interpolate: lambda*(xP - xN) + xN (exact for a uniform field)
    return ws * (x[o] - x[n]) + x[n]


def boundary_values(field):
    return {name: b.value for name, b in field.bc.items()}


def surface_integrate(mesh, phi, phi_b):
    """sum over cell faces of an outward flux, divided by V (fvc::surfaceIntegrate)."""
    out = scatter_faces(mesh, phi, phi_b)
    return out / mesh.V.reshape((-1,) + (1,) * (out.ndim - 1))


def grad(field, cell_vals=None, bvals=None):
    """Gauss linear gradient. Returns (cell gradient (n,3[,3]), patch gradient {patch: ...}).

    For a vector field the gradient is the tensor grad(U)_ij = d U_j / d x_i, shape (n, 3, 3)."""
    mesh = field.mesh
    x = field.internal if cell_vals is None else cell_vals
    bv = boundary_values(field) if bvals is None else bvals
    xf = interpolate(mesh, x)
    if x.ndim == 1:
        flux = mesh.Sf * xf[:, None]
        pflux = {k: mesh.patch[k].Sf * v[:, None] for k, v in bv.items()}
    else:
        flux = mesh.Sf[:, :, None] * xf[:, None, :]
        pflux = {k: mesh.patch[k].Sf[:, :, None] * v[:, None, :] for k, v in bv.items()}
    g = surface_integrate(mesh, flux, pflux)
    # boundary gradient: cell value with the normal component replaced by the patch snGrad (gaussGrad)
    gb = {}
    for name, b in field.bc.items():
        p = b.patch
        gi = g[p.faceCells]
        sn = b.snGrad()
        nf = p.nf
        if x.ndim == 1:
            ndg = nf[:, 0] * gi[:, 0] + nf[:, 1] * gi[:, 1] + nf[:, 2] * gi[:, 2]
            gb[name] = gi + nf * (sn - ndg)[:, None]
        else:
            ndg = nf[:, 0, None] * gi[:, 0] + nf[:, 1, None] * gi[:, 1] + nf[:, 2, None] * gi[:, 2]
            gb[name] = gi + nf[:, :, None] * (sn - ndg)[:, None, :]
    return g, gb


def flux(field_internal, bvals, mesh):
    """Interpolated vector field dotted with Sf: the face volume flux (fvc::flux)."""
    phi = np.einsum("ij,ij->i", mesh.Sf, interpolate(mesh, field_internal))
    phib = {k: np.einsum("ij,ij->i", mesh.patch[k].Sf, v) for k, v in bvals.items()}
    return phi, phib


def div_flux(mesh, phi, phib):
    return surface_integrate(mesh, phi, phib)


def snGrad_internal(mesh, x):
    """Uncorrected-by-scheme snGrad on internal faces with the 'corrected' scheme: orthogonal part + correction."""
    o, n = mesh.owner[:mesh.nInternalFaces], mesh.neighbour
    return mesh.nonOrthDeltaCoeffs * (x[n] - x[o])
