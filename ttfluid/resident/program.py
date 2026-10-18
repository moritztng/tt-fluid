"""The resident program of a case: the solver's state as device registers, one traced SIMPLE iteration, and the outer
loop with simpleFoam's residualControl test, so the device runs the case to convergence on its own.

    prog = build(SimpleFoam(case, backend="ell", dtype=np.float32), max_outer=2000)

Construction (mesh, wall distance, initial fields, the first flux) stays on the host: it runs once.
"""
import numpy as np

from ttfluid.solver.ops import EllOps

from . import krylov
from .trace import SCALAR, Space, T, Tracer, _obj


def host_any(a, cells=None):
    """a.any() for a host array over the cells (or over the cells `cells`): the SPMD lowering (lower/spmd.py)
    replaces it to decide over the whole mesh."""
    return bool(np.any(a))


class TracedOps(EllOps):
    """EllOps over traced arrays: same constants, traced Krylov solves, boundary slots as gathers."""

    def __init__(self, ops, tr, pspace):
        self.__dict__.update(ops.__dict__)
        self.tr, self.pspace = tr, pspace
        self.perfs, self._bsl = [], {}     # _bsl: each patch's (cells, slots, positions) for bslots

    # slot folds: the k-th step of a fold over ELL slots is tagged slotfold[result] = k, so the Machine can run slot k
    # only over the cells that have a k-th face (a lane prefix of each core under --sliced), the rest being padding,
    # which every fold here already treats as neutral (0 to a sum, the init or the cell's own value to a min / max)
    def _fold(self, f, out, s, j0):
        for j in range(j0, s.shape[1]):
            out = f(out, s[:, j])
            if isinstance(out, T) and out.space is self.pspace["__cells__"]:
                for r in out.c.ravel():
                    self.tr.slotfold[r] = j
        return out

    def rowsum(self, s):
        return self._fold(np.add, s[:, 0].copy(), s, 1) if isinstance(s, T) else super().rowsum(s)

    def rowdot(self, a, b):
        # each slot's product folded as it forms, so no slot-wide product vector is stored and re-read
        if not isinstance(b, T):
            return super().rowdot(a, b)
        out = a[:, 0] * b[:, 0]
        for j in range(1, b.shape[1]):
            out = out + a[:, j] * b[:, j]
            if out.space is self.pspace["__cells__"]:
                for r in out.c.ravel():
                    self.tr.slotfold[r] = j
        return out

    def rowmax(self, s, init):
        return self._fold(np.maximum, init, s, 0) if isinstance(s, T) else super().rowmax(s, init)

    def rowmin(self, s, init):
        return self._fold(np.minimum, init, s, 0) if isinstance(s, T) else super().rowmin(s, init)

    def krylov(self, symmetric):
        f = krylov.pcg if symmetric else krylov.pbicgstab
        if symmetric and getattr(self, "amg_levels", None) is not None:   # an empty SALevels (--amg-depth 0) is falsy
            levels = self.amg_levels
            f = lambda A, x, b, c: krylov.amg_pcg(A, x, b, c, levels)

        def run(A, x, b, c):
            perf = f(A, x, b, c)
            self.perfs.append(("pcg" if symmetric else "pbicgstab", perf))
            return perf
        return run

    def bslots(self, d, like):
        """Per-slot array, patch values on each patch's boundary slots, 0 elsewhere."""
        e, tr = self.e, self.tr
        nC = e.nCells
        tail = tuple(like.shape[1:])
        host = np.zeros((nC, e.S) + tail, tr.dtype)
        parts = {}
        for k, s in e.ghost.items():
            if k not in d:
                continue
            v = d[k]
            if k not in self._bsl:
                sel = e.boundary & (self.gk >= s.start - nC) & (self.gk < s.stop - nC)
                cells, slots = np.nonzero(sel)
                self._bsl[k] = cells, slots, self.gk[cells, slots] - (s.start - nC)
            cells, slots, pos = self._bsl[k]
            if not isinstance(v, T):
                host[cells, slots] = np.broadcast_to(np.asarray(v, tr.dtype), (s.stop - s.start,) + tail)[pos]
                continue
            vc = np.broadcast_to(v.c, tail)
            for j in range(e.S):
                on = slots == j
                if not host_any(on, cells):
                    continue
                for ci in np.ndindex(*tail):
                    src = vc[ci]
                    if tr.space[src] is SCALAR:   # a patch-uniform traced value: give it the patch's space
                        src = tr.ew("add", _zeros(tr, self.pspace[k]), src)
                    parts.setdefault((j,) + ci, []).append((src, cells[on], pos[on]))
        if not parts:
            return host
        cells_sp = self.pspace["__cells__"]
        out = _obj((e.S,) + tail)
        for idx in np.ndindex(*out.shape):
            g = tr.gather(cells_sp, parts[idx]) if idx in parts else None
            h = host[(slice(None),) + idx]
            hc = tr.const(h, cells_sp) if host_any(h) else None
            out[idx] = g if hc is None else (hc if g is None else tr.ew("add", g, hc))
            if out[idx] is None:
                out[idx] = tr.imm(0.0)
        return T(tr, cells_sp, out)


def _zeros(tr, space):
    from .trace import _zero_vec
    return _zero_vec(tr, space)


def _zfill(tr, space, shape=()):
    c = np.empty(shape, object)
    for i in np.ndindex(*shape):
        c[i] = _zeros(tr, space)
    return c


# SIMPLE phases of one outer iteration, in order (ttfluid.solver.simple iterate: a marker label opens each; build(phase=True)).
# "tail" is the state writeback, the outputs and the convergence test after iterate. Index = the device phase id.
PHASES = ("mom", "U_solve", "p_asm", "p_solve", "flux", "turb_asm", "omega_solve", "k_solve", "tail")


class Program:
    def __init__(self, tr, cells, pspace, state, outs, perf_names):
        self.code, self.space, self.dtype = tr.code, tr.space, tr.dtype
        self.src = tr.src               # trace.SRC: register -> solver source line (census only)
        self.slotfold = tr.slotfold     # register -> ELL slot k of a fold step (TracedOps._fold, machine.slot_extents)
        self.cells, self.pspace = cells, pspace
        self.state = state            # [(name, object array of registers)]
        self.outs = outs              # names of the per-iteration scalars, in 'out' order
        self.perf_names = perf_names
        self.trips = tr.trips
        self.phase_at = getattr(tr, "phase_at", {})   # build(phase=True): label -> PHASES index (encode.py C_PH / B_PH)


def amg_setup(ops, owner, log=print):
    """Multigrid levels for the pressure PCG: pairwise aggregates on the face graph weighted by |Sf| deltaCoeffs
    (the p Laplacian at uniform rAU), never across `owner` cores."""
    from . import amg
    e = ops.e
    w = np.where(e.internal, np.asarray(e.magSf, np.float64) * np.asarray(e.dc, np.float64), 0.0)
    aggs = amg.pairwise(ops.cnbr, e.internal, w, owner)
    levels = amg.build(ops.cnbr, e.internal, aggs, w)
    log(f"amg: levels {[e.nCells] + [L.nc for L in levels]}, coarse slots {[L.S for L in levels]}")
    return levels


def sa_setup(ops, owner, smoothed=99, log=print, chip_cores=None, depth=None, min_core=0):
    """Smoothed-aggregation levels (ttfluid.resident.sa) built from the geometric p Laplacian (face weights |Sf| deltaCoeffs, the
    pairwise setup's), on the program's own fine ELL layout, aggregates never across `owner` cores. chip_cores (a mesh):
    the coarse levels are chip-local (sa.setup)."""
    import scipy.sparse as sp
    from . import sa
    e = ops.e
    n = len(owner)
    w = np.where(e.internal, np.asarray(e.magSf, np.float64) * np.asarray(e.dc, np.float64), 0.0)
    rows = np.repeat(np.arange(n), ops.cnbr.shape[1])
    m = np.asarray(e.internal, bool).ravel()
    A = sp.csr_matrix((np.r_[w.sum(axis=1) + 1e-12 * w.max(), -w.ravel()[m]],
                       (np.r_[np.arange(n), rows[m]], np.r_[np.arange(n), ops.cnbr.ravel()[m]])), shape=(n, n))
    levels, lmax = sa.setup(A, owner, smoothed=smoothed, nbr=ops.cnbr, mask=e.internal, chip_cores=chip_cores, depth=depth, min_core=min_core)
    log(f"amg sa: levels {[n] + [L.nc for L in levels]}, coarse slots {[L.S for L in levels]}, lmax "
        f"{[round(x, 3) for x in lmax]}")
    return levels, lmax


class SALevels(list):
    """ttfluid.resident.sa levels plus their frozen Chebyshev bounds (krylov.amg_pcg picks sa.Ops for these) and cycle shape."""
    lmax = None
    vshape = None


def build(solver, max_outer=2000, amg_owner=None, amg_kind="pairwise", amg_smoothed=99, amg_stack=False, amg_lag=1, amg_lag_warm=0,
          amg_chip_cores=None, amg_depth=None, amg_min_core=0, phase=False, amg_vshape=None):
    tr = Tracer(solver.ops.dtype)
    if phase:   # a label (a block boundary every pass keeps) opens each phase; the encoder puts a marker on it
        tr.phase_at = {}

        def mark(name):
            L = tr.new_label()
            tr.label(L)
            tr.phase_at[L] = PHASES.index(name)
        solver.phase_hook = mark
    m = solver.mesh
    ops = solver.ops
    assert isinstance(ops, EllOps)
    cells = Space("cells", m.nCells)
    pspace = {p.name: Space("patch:" + p.name, p.size, p.faceCells) for p in m.patches if p.type != "empty"}
    for p in m.patches:
        if p.name in pspace:
            tr.index_space[id(p.faceCells)] = pspace[p.name]
    tr.index_space[id(ops.cnbr)] = cells
    tops = TracedOps(ops, tr, dict(pspace, __cells__=cells))
    if amg_owner is not None and amg_kind == "sa":
        from . import sa
        levels, lmax = sa_setup(ops, np.asarray(amg_owner), amg_smoothed, chip_cores=amg_chip_cores, depth=amg_depth, min_core=amg_min_core)
        tops.amg_levels = SALevels(sa.attach(tr, levels, cells, stack=amg_stack, owner=np.asarray(amg_owner)))
        tops.amg_levels.lmax = lmax
        tops.amg_levels.vshape = amg_vshape   # sa.Ops cycle shape (None: 3/3/8 V)
        for L in tops.amg_levels:          # Layout places every space the Program lists
            for sp_ in ([L.space] + L.apc + L.acc + L.gspaces + [c[0] for c in L.mchunks] if L.gspaces else [L.space, L.apspace, L.acspace]):
                pspace[sp_.name] = sp_
    elif amg_owner is not None:
        assert amg_chip_cores is None, "chip-local multigrid: --amg-kind sa only"
        from . import amg
        tops.amg_levels = amg.attach(tr, amg_setup(ops, np.asarray(amg_owner)), cells)
        for L in tops.amg_levels:          # Layout places every space the Program lists
            pspace[L.space.name] = L.space
            pspace[L.fspace.name] = L.fspace

    # ---- the state: every field's cells and patch values, and the flux
    state = []

    def wrap(name, arr, space):
        arr = np.asarray(arr, tr.dtype)
        lead = arr.shape[0]
        assert lead == space.n, (name, arr.shape, space)
        c = _obj(arr.shape[1:])
        for i in np.ndindex(*arr.shape[1:]):
            r = tr.reg(space)
            tr.code.append(("state", r, np.ascontiguousarray(arr[(slice(None),) + i])))
            c[i] = r
        state.append((name, c))
        return T(tr, space, c)

    holders = []   # (name, get, set)
    for f in solver.fields():
        holders.append((f.name, (lambda f=f: f.internal), (lambda v, f=f: setattr(f, "internal", v)), cells))
        for k, b in f.bc.items():
            holders.append((f"{f.name}.{k}", (lambda b=b: b.value), (lambda v, b=b: setattr(b, "value", v)),
                            pspace[k]))
    holders.append(("phi", lambda: solver.phi, lambda v: setattr(solver, "phi", v), cells))
    for name, get, put, sp in holders:
        put(wrap(name, get(), sp))

    solver.ops = tops
    if solver.turb is not None:
        solver.turb.ops = tops
        solver.turb.bound = solver.turb._bound_apply

    OUTER, END = tr.new_label(), tr.new_label()
    one = T(tr, SCALAR, np.array(tr.imm(1.0), object))
    IT = krylov.Var(one * 0.0)
    if amg_lag > 1 and amg_owner is not None and amg_kind == "sa":
        # lagged Galerkin (krylov.amg_pcg): solve counter and the coarse operators' registers, zero until the first
        # solve computes them
        z = lambda sp, shape=(): T(tr, sp, _zfill(tr, sp, shape))
        store = []
        for L in tops.amg_levels:
            offs = [z(g) for g in L.gspaces] if L.gspaces else [z(L.space, (L.S,))]
            store.append((krylov.Var(z(L.space)), [krylov.Var(o) for o in offs]))
        # the counter starts at -warm: the first warm + 1 solves all recompute (the transient), then one in amg_lag
        tops.amg_levels.lag = (amg_lag, krylov.Var(one * float(-amg_lag_warm)), store)
    tr.label(OUTER)
    res = solver.iterate()
    if phase:
        solver.phase_hook("tail")
        solver.phase_hook = None

    # ---- write the new state back into the state registers (through temporaries where a new value is itself a
    # state register about to be overwritten)
    state_regs = {r for _, c in state for r in c.ravel()}
    moves = []
    for (name, get, put, sp), (_, c) in zip(holders, state):
        new = get()
        nc = np.broadcast_to(new.c, c.shape) if isinstance(new, T) else None
        for i in np.ndindex(*c.shape):
            src = nc[i] if nc is not None else tr.const(np.broadcast_to(np.asarray(new, tr.dtype),
                                                                        (sp.n,) + c.shape)[(slice(None),) + i], sp)
            if tr.space[src] is SCALAR and sp is not SCALAR:
                src = tr.ew("add", _zeros(tr, sp), src)
            if src != c[i]:
                moves.append((c[i], src))
    tmp = {}
    for dst, src in moves:
        if src in state_regs:
            t = tr.reg(tr.space[src])
            tr.copy(t, src)
            tmp[src] = t
    for dst, src in moves:
        tr.copy(dst, tmp.get(src, src))

    outs = []
    for k, v in res.items():
        tr.out(k, v.c[()])
        outs.append(k)
    for i, (kind, perf) in enumerate(tops.perfs):
        tr.out(f"its{i}", perf.iterations.c[()])
        outs.append(f"its{i}")
    it = IT.t + 1.0
    IT.set(it)
    tr.out("outer", IT.c[()])
    outs.append("outer")
    rc = solver.case.residual_control()
    from ttfluid.solver.simple import _lookup
    conv = None
    for k, v in res.items():
        crit = _lookup(rc, k)
        if crit is not None:
            ok = v <= crit
            conv = ok if conv is None else conv * ok
    if conv is not None:
        tr.jumpif(conv.c[()], END)
    tr.jumpif((IT.t < float(max_outer)).c[()], OUTER)
    tr.label(END)
    tr.code.append(("keep", sorted(state_regs)))     # the state is the program's result: live at the end
    return Program(tr, cells, pspace, state, outs, [k for k, _ in tops.perfs])


def apply_state(solver, state, iterations):
    """Load a program's state (name -> array) into a host solver, e.g. to write the fields in OpenFOAM format."""
    for f in solver.fields():
        f.internal = np.asarray(state[f.name], np.float64)
        for k, b in f.bc.items():
            b.value = np.asarray(state[f"{f.name}.{k}"], np.float64)
    solver.phi = state["phi"]
    solver.time = iterations
