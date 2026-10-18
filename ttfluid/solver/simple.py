"""Steady incompressible SIMPLE/SIMPLEC (OpenFOAM simpleFoam), reading and writing an OpenFOAM case."""
import re
import time
from pathlib import Path

import numpy as np

from . import fvc, linsolve
from .fields import InletOutlet, VolField
from .foamio import read_dict
from .mesh import Mesh
from .ops import EllOps, FaceOps


# ---- case dictionaries -----------------------------------------------------------------------------------------

def _dict(path):
    return read_dict(path)


def _lookup(d, name):
    """Exact key first, then OpenFOAM regex keys ("(U|k)"), last match wins as in OpenFOAM."""
    if name in d:
        return d[name]
    hit = None
    for k, v in d.items():
        if re.fullmatch(k.strip('"'), name):
            hit = v
    return hit


def _scalar(v, default=None):
    if v is None:
        return default
    return float(v[0] if isinstance(v, list) else v)


def _scheme(tokens, all_divs):
    """'bounded Gauss linearUpwind grad(U)' -> (bounded, scheme)."""
    t = [x for x in tokens]
    if t and t[0].startswith("$"):
        t = all_divs[t[0][1:]]
    bounded = t[0] == "bounded"
    t = t[1:] if bounded else t
    assert t[0] == "Gauss", t
    name = t[1]
    if name == "limitedLinear":
        return bounded, ("limitedLinear", float(t[2]))
    return bounded, name


class Case:
    def __init__(self, case):
        self.path = Path(case)
        self.solution = _dict(self.path / "system/fvSolution")
        self.schemes = _dict(self.path / "system/fvSchemes")
        self.control = _dict(self.path / "system/controlDict")
        self.transport = _dict(self.path / "constant/transportProperties")

    def solver_controls(self, name):
        d = _lookup(self.solution["solvers"], name)
        return linsolve.Controls(tolerance=_scalar(d.get("tolerance"), 1e-6), relTol=_scalar(d.get("relTol"), 0.0),
                                 maxIter=int(_scalar(d.get("maxIter"), 1000)),
                                 minIter=int(_scalar(d.get("minIter"), 0)))

    def relax(self, name):
        eq = self.solution.get("relaxationFactors", {}).get("equations", {})
        return _scalar(_lookup(eq, name), 0.0)

    def div_scheme(self, key):
        divs = self.schemes["divSchemes"]
        return _scheme(divs[key], divs)

    def velocity_damping(self):
        """(UMax, C) of each velocityDampingConstraint in fvOptions/fvConstraints (selectionMode all only)."""
        out = []
        for f in ("system/fvOptions", "constant/fvOptions", "system/fvConstraints", "constant/fvConstraints"):
            if not (self.path / f).exists():
                continue
            for name, d in _dict(self.path / f).items():
                if not isinstance(d, dict) or d.get("type", [None])[0] != "velocityDampingConstraint":
                    continue
                c = d.get(f"{d['type'][0]}Coeffs", d)
                if c.get("selectionMode", d.get("selectionMode", ["all"]))[0] != "all":
                    raise NotImplementedError(f"{name}: velocityDampingConstraint on a cell subset")
                out.append((_scalar(c["UMax"]), _scalar(c.get("C"), 1.0)))
        return out

    def residual_control(self):
        rc = self.solution["SIMPLE"].get("residualControl", {})
        return {k.strip('"'): _scalar(v) for k, v in rc.items()}


# ---- the solver ------------------------------------------------------------------------------------------------

class SimpleFoam:
    """simpleFoam on a chosen operator backend: backend='face' (OpenFOAM's face loops, fp64 oracle) or 'ell' (the
    device form, fp64 or fp32)."""

    def __init__(self, case, turbulence=True, backend="face", dtype=np.float64, log=print, ell_order="face",
                 merge_patches=True, chunk=1 << 16, host_solver=None):
        self.case = Case(case)
        self.mesh = m = Mesh(case)
        if merge_patches:
            from . import foamio
            from .fields import group_runs
            m.merge_patches(group_runs(m, [foamio.read_field(self.case.path / "0" / f)["boundary"]
                                           for f in ("U", "p", "k", "omega", "nut") if (self.case.path / "0" / f).exists()]))
        self.log = log
        self.matrix_hook = None
        self.ops = EllOps(m, dtype, ell_order, chunk=chunk) if backend == "ell" else FaceOps(m, dtype)
        if host_solver is not None:      # opt-in fast host linear solves (pyamg CG for p, CSR BiCGStab otherwise)
            assert host_solver == "amg" and backend == "ell", (host_solver, backend)
            self.ops.krylov = linsolve.host_krylov
        for key, v in self.case.schemes.get("gradSchemes", {}).items():
            t = v if isinstance(v, list) else [v]
            if t and t[0] == "cellLimited":
                assert t[1:3] == ["Gauss", "linear"] and float(t[3]) == 1.0, (key, t)
                assert key.startswith("grad(") and backend == "ell", (key, backend)
                self.ops.grad_limit.add(key[5:-1])
            elif key != "default":
                assert t[:2] == ["Gauss", "linear"], (key, t)
        self.U = VolField.read(m, self.case.path / "0/U")
        self.p = VolField.read(m, self.case.path / "0/p")
        self.U.name, self.p.name = "U", "p"
        nu = self.case.transport["nu"]
        self.nu = float(nu[-1]) if isinstance(nu, list) else float(nu)
        self.turb = None
        if turbulence:
            from .turbulence import KOmegaSST
            self.turb = KOmegaSST(m, self.U, self.nu, self.case.path, ops=self.ops,
                                  relax={n: self.case.relax(n) for n in ("k", "omega")},
                                  div={n: self.case.div_scheme(f"div(phi,{n})") for n in ("k", "omega")})
        for f in self.fields():
            f.internal = f.internal.astype(dtype)
            f.correct_bc()
        if self.turb is not None and dtype != np.float64:
            t = self.turb
            t.y = t.y.astype(dtype)
            t.y_b = {k: v.astype(dtype) for k, v in t.y_b.items()}
        # createPhi reads 0/phi when it exists (potentialFoam -writephi in motorBike's Allrun), else Sf & U_f
        phi0 = self.case.path / "0/phi"
        if phi0.exists():
            from . import foamio
            d = foamio.read_field(phi0)
            pv = {k: e["value"] for k, e in d["boundary"].items() if "value" in e}
            for new, parts in m.merged.items():
                pv[new] = np.concatenate([np.broadcast_to(np.asarray(pv.pop(n), float), (k,)) for n, k in parts])
            self.phi = self.ops.surface(d["internal"], pv)
        else:
            self.phi = self.ops.flux(self.U.internal, fvc.boundary_values(self.U))
        self.consistent = self.case.solution["SIMPLE"].get("consistent", ["no"])[0] in ("yes", "true", "on")
        # velocityDampingConstraint: diag += C V^(2/3) max(|U| - UMax, 0), after relax (UEqn.H fvOptions.constrain)
        self.damping = [(UMax, (C * np.cbrt(m.V) ** 2).astype(dtype)) for UMax, C in self.case.velocity_damping()]
        self.time = 0

    def fields(self):
        t = self.turb
        return [self.U, self.p] + ([t.k, t.omega, t.nut] if t is not None else [])

    def _nuEff(self):
        """(face value, cell values, patch values) of the effective viscosity."""
        if self.turb is not None:
            nut = self.turb.nut
            cells, patches = nut.internal + self.nu, {k: b.value + self.nu for k, b in nut.bc.items()}
        else:
            cells = np.full(self.mesh.nCells, self.nu, self.ops.dtype)
            patches = {k: np.full(b.patch.size, self.nu) for k, b in self.U.bc.items()}
        return self.ops.interp(cells, patches), cells, patches

    def check_outflow(self):
        """inletOutlet is modelled as zeroGradient (fields.InletOutlet): hold that every face of such a patch flows
        out. Host path only; a traced flux is not a number yet."""
        phi, e = self.phi, getattr(self.ops, "e", None)
        if e is None or not isinstance(phi, np.ndarray):
            return
        g = np.empty(e.nGhost)
        g[-1 - e.face[e.boundary]] = phi[e.boundary]
        for f in (self.U, getattr(self.turb, "k", None), getattr(self.turb, "omega", None)):
            for name, b in (f.bc.items() if f is not None else ()):
                if isinstance(b, InletOutlet):
                    sl = e.ghost[name]
                    worst = g[sl.start - e.nCells:sl.stop - e.nCells].min()
                    if worst < 0:
                        raise NotImplementedError(f"inflow {worst:.3g} on inletOutlet patch {name} of {f.name}")

    def iterate(self):
        ops, m, U, p = self.ops, self.mesh, self.U, self.p
        V = ops.V[:, None]
        res = {}
        self._phase("mom")
        # --- momentum predictor (UEqn.H): div(phi,U) + turbulence->divDevSigma(U), relaxed, == -grad(p)
        bounded, scheme = self.case.div_scheme("div(phi,U)")
        nuf, nuc, nub = self._nuEff()
        gU, gUb = ops.grad(U)
        if self.force_hook is not None:
            res.update(self.force_hook(p, gUb, nub))
        UEqn = ops.div(self.phi, U, scheme, bounded=bounded)
        if hasattr(ops, "sub_laplacian"):
            ops.sub_laplacian(UEqn, nuf, U)                                 # UEqn -= laplacian, never whole
        else:
            UEqn -= ops.laplacian(nuf, U)
        T = nuc[:, None, None] * _dev2T(gU)
        Tb = {k: nub[k][:, None, None] * _dev2T(v) for k, v in gUb.items()}
        UEqn.source += V * ops.surface_integrate(ops.flux_tensor(T, Tb))   # - fvc::div(nuEff dev2(T(grad U)))
        UEqn.relax(self.case.relax("U"))
        for UMax, cV in self.damping:
            M = getattr(UEqn, "M", UEqn)
            magU = np.sqrt(U.internal[:, 0] ** 2 + U.internal[:, 1] ** 2 + U.internal[:, 2] ** 2)
            M.diag = M.diag + cV * np.maximum(magU - UMax, 0.0)
        gp, _ = ops.grad(p)
        UEqn.source -= V * gp
        self._phase("U_solve")
        res["U"] = UEqn.solve(self.case.solver_controls("U"))
        self._phase("p_asm")
        UEqn.source += V * gp

        # --- pressure correction (pEqn.H), SIMPLEC when consistent
        rAU = 1.0 / UEqn.A()
        HbyA = rAU[:, None] * UEqn.H()
        # constrainHbyA: U's value on the non-assignable patches (fixedValue, noSlip, slip), else extrapolated
        HbyAb = {k: HbyA[b.patch.faceCells] if b.assignable else b.value.copy() for k, b in U.bc.items()}
        phiHbyA = ops.flux(HbyA, HbyAb)
        rAtU = rAU
        if self.consistent:
            rAtU = 1.0 / (1.0 / rAU - UEqn.H1())
            gp, _ = ops.grad(p)
            phiHbyA = phiHbyA + ops.interp(rAtU - rAU) * ops.snGrad(p, gp) * ops.magSf
            HbyA = HbyA - (rAU - rAtU)[:, None] * gp
        if not any(b.fixes_value for b in p.bc.values()):
            raise NotImplementedError("pressure reference cell")
        pEqn = ops.laplacian(ops.interp(rAtU), p)
        pEqn.source += V[:, 0] * ops.surface_integrate(phiHbyA)          # == fvc::div(phiHbyA)
        if self.matrix_hook is not None:      # e.g. drivaer/grade_pmatrix.py: the p system as handed to the solver
            self.matrix_hook("p", pEqn)
        self._phase("p_solve")
        res["p"] = pEqn.solve(self.case.solver_controls("p"), symmetric=True)
        self._phase("flux")
        self.phi = phiHbyA - pEqn.flux()
        self.check_outflow()
        gp, _ = ops.grad(p)
        U.internal[:] = HbyA - rAtU[:, None] * gp
        U.internal[:, ~m.solutionD] = 0.0
        U.correct_bc()

        # --- turbulence
        if self.turb is not None:
            self._phase("turb_asm")
            res["omega"], res["k"] = self.turb.correct(self.phi, self._solve_turb)
        self.time += 1
        return res

    def _solve_turb(self, M, name):
        self._phase(name + "_solve")
        r = M.solve(self.case.solver_controls(name))
        self._phase("turb_asm")
        return r

    force_hook = None   # forces.coeff_hook: Cd/Cl of the incoming state in res (run_mesh --forces)
    phase_hook = None   # resident/ttfluid/resident/program.py build(phase=True): a marker at each SIMPLE phase boundary

    def _phase(self, name):
        if self.phase_hook is not None:
            self.phase_hook(name)

    def converged(self, res):
        rc = self.case.residual_control()
        for name, r in res.items():
            crit = _lookup(rc, name)
            if crit is not None and r > crit:
                return False
        return bool(rc)

    def run(self, max_iter=None, write=True, out=None):
        end = int(_scalar(self.case.control["endTime"])) if max_iter is None else max_iter
        t0 = time.perf_counter()
        history = []
        for _ in range(end):
            res = self.iterate()
            history.append(res)
            self.log(f"Time = {self.time} " + " ".join(f"{k}={v:.3e}" for k, v in res.items()))
            if self.converged(res):
                self.log(f"SIMPLE solution converged in {self.time} iterations")
                break
        wall = time.perf_counter() - t0
        if write:
            self.write(out or self.case.path)
        return history, wall

    def write(self, case):
        t = self.time
        self.U.write(case, t, "U")
        self.p.write(case, t, "p")
        if self.turb is not None:
            for f, n in ((self.turb.k, "k"), (self.turb.omega, "omega"), (self.turb.nut, "nut")):
                f.write(case, t, n)


def _dev2T(g):
    """dev2(T(g)) = T(g) - (2/3) tr(g) I."""
    T = np.swapaxes(g, -1, -2)
    tr = T[..., 0, 0] + T[..., 1, 1] + T[..., 2, 2]
    return T - (2.0 / 3.0) * tr[..., None, None] * np.eye(3, dtype=g.dtype)
