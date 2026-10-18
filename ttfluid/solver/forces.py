"""Wall forces and force coefficients, as OpenFOAM v2412's `forces` / `forceCoeffs` function objects compute them.

Sources (src/functionObjects/forces): forces.C calcForcesMoments / devRhoReff / setCoordinateSystem,
forceCoeffs.C calcForceCoeffs / calcMomentCoeffs / selectCoeffs, forceCoeffs.H coeffDesc::value; the gradient is
fvc::grad(U) with the case's grad(U) scheme (gaussGrad, cellLimitedGrad), coordinate axes from
coordinateRotations::axes (E3_E1 order).

Incompressible case, rho = rhoInf:
  fP = rhoInf * Sf * (p_b - pRef/rhoInf)                         p kinematic, p_b the patch value
  fV = Sf & (-rhoInf * (nut_b + nu) * devTwoSymm(gradU_b))        nut_b the wall-function nut patch value
  gradU_b = G_P + n (snGrad(U) - n & G_P)                         G_P the (limited) cell gradient, gaussGrad BC
  M = (Cf - CofR) ^ f
Coefficients are the local-frame components (e1 = dragDir made orthogonal to e3, e3 = liftDir, e2 = e3 ^ e1) over
0.5 rhoInf magUInf^2 Aref (moments: also over lRef). pitchAxis is read by nothing in v2412.
"""
import re

import numpy as np

from . import fvc

SMALL = 1e-15


# ---- gradient -------------------------------------------------------------------------------------------------

def cell_limited(field, g, k=1.0):
    """cellLimitedGrad<Type, minmod>::calcGrad limiter applied to the Gauss gradient g (face loop, as OpenFOAM)."""
    m = field.mesh
    x = field.internal
    nInt = m.nInternalFaces
    o, n = m.owner[:nInt], m.neighbour
    vec = x.ndim > 1
    mx, mn = x.copy(), x.copy()
    np.maximum.at(mx, o, x[n]); np.minimum.at(mn, o, x[n])
    np.maximum.at(mx, n, x[o]); np.minimum.at(mn, n, x[o])
    for name, b in field.bc.items():
        fc = b.patch.faceCells
        np.maximum.at(mx, fc, b.value); np.minimum.at(mn, fc, b.value)
    mx = mx - x
    mn = mn - x
    if k < 1.0:
        mm = (1.0 / k - 1.0) * (mx - mn)
        mx, mn = mx + mm, mn - mm
    lim = np.ones_like(x)

    def limit_face(cells, d):
        ext = np.einsum("fi,fi...->f...", d, g[cells])
        mxc, mnc = mx[cells], mn[cells]
        r = np.where(ext > SMALL, mxc / np.where(ext > SMALL, ext, 1.0),
                     np.where(ext < -SMALL, mnc / np.where(ext < -SMALL, ext, 1.0), np.inf))
        np.minimum.at(lim, cells, r)

    limit_face(o, m.Cf - m.C[o])
    limit_face(n, m.Cf - m.C[n])
    for name, b in field.bc.items():
        p = b.patch
        limit_face(p.faceCells, p.Cf - m.C[p.faceCells])
    lim = np.minimum(lim, 1.0)
    return g * lim[:, None] if not vec else g * lim[:, None, :]


def grad_with_bc(field, limited_k=None):
    """fvc::grad(field): Gauss linear (optionally cellLimited k) cell gradient plus gaussGrad patch values."""
    g, _ = fvc.grad(field)
    if limited_k is not None and limited_k >= SMALL:
        g = cell_limited(field, g, limited_k)
    gb = {}
    for name, b in field.bc.items():
        p = b.patch
        gi = g[p.faceCells]
        nf = p.nf
        ndg = np.einsum("fi,fi...->f...", nf, gi)
        gb[name] = gi + np.einsum("fi,f...->fi...", nf, b.snGrad() - ndg)
    return g, gb


def dev_two_symm(T):
    """devTwoSymm: 2 symm(T) - (2/3) tr(T) I, T shape (n, 3, 3)."""
    s = T + np.swapaxes(T, 1, 2)
    tr = np.trace(T, axis1=1, axis2=2)
    return s - (2.0 / 3.0) * tr[:, None, None] * np.eye(3)


# ---- setup ----------------------------------------------------------------------------------------------------

def _selected(p, specs):
    groups = p.dict.get("inGroups", [])
    groups = groups if isinstance(groups, list) else [groups]
    groups = [g for g in groups if isinstance(g, str) and not g.isdigit()]
    return any(re.fullmatch(s.strip('"'), w) for s in specs for w in [p.name] + groups)


def select_patches(mesh, specs):
    """polyBoundaryMesh::patchSet(wordRes): names, regexes ("motorBike.*") and group names, in patch order."""
    return [p.name for p in mesh.patches if _selected(p, specs)]


def axes(dragDir, liftDir):
    """coordSystem::cartesian(origin, e3=liftDir, e1=dragDir): rotation tensor with the local axes as columns."""
    e3 = np.asarray(liftDir, float); e3 = e3 / np.linalg.norm(e3)
    e1 = np.asarray(dragDir, float); e1 = e1 - (e1 @ e3) * e3; e1 = e1 / np.linalg.norm(e1)
    return np.stack([e1, np.cross(e3, e1), e3], axis=1)


# ---- forces ---------------------------------------------------------------------------------------------------

def forces(mesh, p, U, nut_b, nu, patches, CofR, rhoInf=1.0, pRef=0.0, grad_limit=None, faces=None):
    """Pressure / viscous force and moment sums over `patches` (global frame), as forces::calcForcesMoments.

    p, U: VolFields (p kinematic); nut_b: {patch: nut patch values}; nu: laminar viscosity; grad_limit: the k of
    'cellLimited Gauss linear k' for grad(U) (None for plain Gauss linear); faces: {patch: face indices} to sum
    over part of a patch (a merged run of which only some parts are force patches)."""
    _, gUb = grad_with_bc(U, grad_limit)
    origin = np.asarray(CofR, float)
    pRef_k = pRef / rhoInf
    Fp, Fv, Mp, Mv = (np.zeros(3) for _ in range(4))
    for name in patches:
        pa = mesh.patch[name]
        f = slice(None) if faces is None or name not in faces else faces[name]
        Sf = pa.Sf[f]
        Md = pa.Cf[f] - origin
        fP = rhoInf * Sf * (p.bc[name].value[f] - pRef_k)[:, None]
        nuEff = nut_b[name][f] + nu
        R = (-rhoInf * nuEff)[:, None, None] * dev_two_symm(gUb[name][f])
        fV = np.einsum("fi,fij->fj", Sf, R)
        Fp += fP.sum(0); Fv += fV.sum(0)
        Mp += np.cross(Md, fP).sum(0); Mv += np.cross(Md, fV).sum(0)
    return dict(Fp=Fp, Fv=Fv, Mp=Mp, Mv=Mv)


COEFF_ORDER = ["Cd", "Cd(f)", "Cd(r)", "Cl", "Cl(f)", "Cl(r)", "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs(f)", "Cs(r)"]


def force_coeffs(F, magUInf, lRef, Aref, dragDir=(1, 0, 0), liftDir=(0, 0, 1), rhoInf=1.0):
    """forceCoeffs: {name: total} in coefficient.dat column order, plus pressure/viscous parts under 'parts'."""
    pDyn = 0.5 * rhoInf * magUInf ** 2
    fs = 1.0 / (Aref * pDyn + SMALL)
    ms = 1.0 / (Aref * pDyn * lRef + SMALL)
    R = axes(dragDir, liftDir)
    Cf = {k: fs * (R.T @ F[k]) for k in ("Fp", "Fv")}   # localVector = R^T v
    Cm = {k: ms * (R.T @ F[k]) for k in ("Mp", "Mv")}
    f = Cf["Fp"] + Cf["Fv"]
    m = Cm["Mp"] + Cm["Mv"]
    out = {"Cd": f[0], "Cs": f[1], "Cl": f[2], "CmRoll": m[0], "CmPitch": m[1], "CmYaw": m[2]}
    # front/rear split (forceCoeffs.H coeffDesc::value): 0.5 C{d,s,l} +/- Cm{Roll,Yaw,Pitch}
    for c, mc in (("Cd", "CmRoll"), ("Cs", "CmYaw"), ("Cl", "CmPitch")):
        out[c + "(f)"] = 0.5 * out[c] + out[mc]
        out[c + "(r)"] = 0.5 * out[c] - out[mc]
    res = {k: float(out[k]) for k in COEFF_ORDER}
    res["parts"] = dict(Cf_p=Cf["Fp"], Cf_v=Cf["Fv"], Cm_p=Cm["Mp"], Cm_v=Cm["Mv"])
    return res


# ---- in the solver loop ---------------------------------------------------------------------------------------

def _num(v):
    v = [x for x in (v if isinstance(v, list) else [v]) if x not in ("(", ")")]
    return [float(x) for x in v]


def coeff_hook(solver, func="forceCoeffsAll"):
    """SimpleFoam.force_hook for the controlDict function `func` (or "CONTROLDICT:FUNC"): Cd and Cl of the state an outer iteration starts
    from, i.e. OpenFOAM's coefficient.dat row of the iteration before, from what iterate has at its start: p patch
    values, the gaussGrad patch values of fvc::grad(U) (the case's grad(U) scheme, as forces uses) and nuEff.

    Per face the coefficient is linear in those: rhoInf [(p_b - pRef/rhoInf) (Sf.e) - nuEff M:gradU_b] with the
    constant M = Sf e^T + e Sf^T - 2/3 (Sf.e) I (Sf & devTwoSymm(G) & e = M:G), so a traced run (resident program)
    carries it as a few multiply-adds per car face and one sum per patch and coefficient."""
    if ":" in func:   # "CONTROLDICT:FUNC": the function from another controlDict (a device case copy without functions)
        from .simple import _dict
        path, func = func.rsplit(":", 1)
        fd = _dict(path)["functions"][func]
    else:
        fd = solver.case.control["functions"][func]
    f1 = lambda k, d: _num(fd.get(k, d))[-1]
    rho, pref = f1("rhoInf", 1.0), f1("pRef", 0.0) / f1("rhoInf", 1.0)
    scale = rho / (0.5 * rho * f1("magUInf", None) ** 2 * f1("Aref", None) + SMALL)
    R = axes(_num(fd["dragDir"]), _num(fd["liftDir"]))
    mesh, dt = solver.mesh, solver.ops.dtype
    specs = [s for s in fd["patches"] if s not in ("(", ")")]
    w = {}
    for q in mesh.patches:   # a merged run (SimpleFoam merges patches with equal BCs): weights 0 off its force parts
        on = np.concatenate([np.full(x.size, _selected(x, specs)) for x in (q.parts or [q])])
        if not on.any():
            continue
        Sf = q.Sf * on[:, None]
        for c, e in (("Cd_in", R[:, 0]), ("Cl_in", R[:, 2])):
            a = Sf @ e
            M = Sf[:, :, None] * e[None, None, :]
            M = M + np.swapaxes(M, 1, 2) - (2.0 / 3.0) * a[:, None, None] * np.eye(3)
            w[q.name, c] = (a.astype(dt), M.astype(dt))
    pats = sorted({k for k, _ in w}, key=[q.name for q in mesh.patches].index)

    def hook(p, gUb, nub):
        out = {}
        for c in ("Cd_in", "Cl_in"):
            tot = None
            for name in pats:
                a, M = w[name, c]
                g = gUb[name]
                v = None
                for i in range(3):
                    for j in range(3):
                        t = M[:, i, j] * g[:, i, j]
                        v = t if v is None else v + t
                s = ((p.bc[name].value - pref) * a - nub[name] * v).sum()
                tot = s if tot is None else tot + s
            out[c] = tot * scale
        return out
    return hook
