"""SPMD lowering: every chip's process traces its own sub-case (prep.py) and every value-dependent choice the trace
makes is taken over the whole mesh through a coordinator, so each chip ends up with the global program restricted to
its cells.

    spmd.py SUBDIR CASE [--check] [-n 100] [--machine JSON --images DIR --mesh R,C]
    spmd.py SUBDIR -- <run_mesh.py arguments incl. --images DIR>     every chip's process runs run_mesh.py itself
    spmd.py SUBDIR --replay B -- <the same>          chips run alone, B at a time, passes to a fixed point (main_replay)

Stage 1 (this file so far): SimpleFoam + build per chip. The workers run in lockstep; each collective carries a call
counter and a kind, and the coordinator stops on the first call where the chips disagree (a divergence, not a hang).
Collective choices:
  Tracer.const   fold to an immediate iff every chip's owned values are one value (the global rule on the whole
                 vector), dedup to the register whose owned values are equal on every chip (the global key)
  host_any       bslots' boundary-slot and host-constant presence, over owned cells
  max_count      the ELL width over owned cells
  _space_by_len  the gather's space: the one candidate every chip agrees on
Values on halo cells (a chip's cells it does not own) are never read: the device reads them from their owner.
--check also traces the whole case in this process and compares the programs: same instructions and registers, and
every array (state, const, gather) equal on each chip's owned elements, in global numbering.
"""
import argparse
import copy
import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from ttfluid.solver import ell as ell_mod  # noqa: E402
from ttfluid.solver import foamio, simple, turbulence  # noqa: E402
from ttfluid.resident import program, trace  # noqa: E402
from ttfluid.resident.trace import SCALAR, Tracer  # noqa: E402


def log(*a):
    print("%7.1f s" % (time.time() - T0), *a, flush=True)


# ---- worker side ---------------------------------------------------------------------------------------------------
def sizes_of(g, names):
    """Bytes reachable from each of g[names] (numpy buffers + Python object sizes), shared objects counted once."""
    import gc
    seen, out = set(), {}
    for name in names:
        tot, arr, stack = 0, 0, [g[name]]
        while stack:
            o = stack.pop()
            if id(o) in seen or isinstance(o, (type, type(sys))) or callable(o) and not hasattr(o, "__dict__"):
                continue
            seen.add(id(o))
            if isinstance(o, np.ndarray):
                if o.base is None:
                    arr += o.nbytes
                else:
                    stack.append(o.base)
                continue
            tot += sys.getsizeof(o)
            stack.extend(gc.get_referents(o))
        out[name] = (round(arr / 1e9, 3), round(tot / 1e9, 3))
    return out


def rss_kb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4


class Coll:
    def __init__(self, conn):
        self.conn, self.n, self.prof, self.log = conn, 0, [], []

    def __call__(self, kind, x):
        self.n += 1
        rss = rss_kb()
        self.prof.append((kind, rss, time.time()))     # this process's RSS when it reaches each collective
        site = call_site()
        if kind != "done":
            self.log.append((kind, pickle.dumps(x, protocol=5), site))
        self.conn.send((self.n, kind, x, site, rss))
        r = self.conn.recv()
        if isinstance(r, str) and r == STOP:
            raise Stopped()
        return r


STOP = "__spmd_stop__"


class Stopped(BaseException):
    """Pass 0 of --replay stopped the lockstep chips at the RAM budget."""


class TraceOnly(BaseException):
    """SPMD_TRACE_ONLY=1: a replay chip that guessed inside its trace stops at the trace end."""


def call_site():
    f, site = sys._getframe(2), []
    while f is not None and len(site) < 3:
        if "spmd.py" not in f.f_code.co_filename or not site:
            site.append("%s:%d" % (Path(f.f_code.co_filename).stem, f.f_lineno))
        f = f.f_back
    return " < ".join(site)


class Replay:
    """A chip run alone (--replay): answer each collective from the previous pass's answers (a list of (kind, pickled
    answer)) while its kind matches, else guess with this chip's contribution alone; log every contribution."""
    def __init__(self, path):
        self.ans = pickle.load(open(path, "rb")) if path and Path(path).exists() else []
        self.n, self.log, self.guessed, self.prof = 0, [], 0, []

    def __call__(self, kind, x):
        c = self.n
        self.n += 1
        self.prof.append((kind, rss_kb(), time.time()))
        if kind == "done":
            return None
        site = call_site()
        self.log.append((kind, pickle.dumps(x, protocol=5), site))
        if c < len(self.ans) and self.ans[c][0] == kind and self.ans[c][2] == site:
            return pickle.loads(self.ans[c][1])
        self.guessed += 1
        return reduce_loose(kind, [x])[0]


def spmd_tracer(C, own_of, first_of):
    class SpmdTracer(Tracer):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._okey, self._creg = {}, {}

        def const(self, values, space):
            v = np.ascontiguousarray(values, self.dtype).ravel()
            assert v.size == space.n, (v.size, space)
            o = v if space is SCALAR else v[own_of(space)]
            uni = bool(o.size) and bool(np.all(o == o[0]))
            key = (space.name, hashlib.blake2b(o.tobytes(), digest_size=16).digest())
            # registers named by the call that made them (the same on every chip in lockstep), so an answer stays
            # right in a replay pass whose earlier folds numbered the registers differently. The contribution is every
            # earlier call with these owned values, whatever its answer: the chips' intersection is the call's global
            # class, whose first call made the register, so the answer depends on the data alone, not on earlier answers
            got = C("const", (space.name, int(o.size), uni, o[:1].tobytes(), first_of(space), self._okey.get(key, [])))
            self._okey.setdefault(key, []).append(C.n)
            if got[0] == "imm":
                return self.imm(np.frombuffer(got[1], self.dtype)[0])
            r = self._creg.get(got[1])
            if r is not None and self.space[r].name == space.name:
                return r
            if r is not None:   # a replay answer made under earlier guesses names another space's register: a guess
                C.guessed += 1
            r = self.reg(space)
            self.code.append(("const", r, v))
            self._creg[C.n] = r
            return r
    return SpmdTracer


LOOSE = False     # replay passes before the fixed point: contributions made under guessed answers need not agree


def r_const(xs):
    names = {x[0] for x in xs}
    assert len(names) == 1, names
    full = [x for x in xs if x[1]]
    if full and all(x[2] for x in full):
        first = min(full, key=lambda x: x[4])        # the chip owning the space's first element holds v[0]
        v0 = np.frombuffer(first[3], np.float32)[0]
        if all(np.frombuffer(x[3], np.float32)[0] == v0 for x in full) and not np.signbit(v0):
            return ("imm", first[3])
    inter = set(xs[0][5])
    for x in xs[1:]:
        inter &= set(x[5])
    return ("reg", min(inter) if inter else None)


def r_space(xs):
    inter = set(xs[0])
    for x in xs[1:]:
        inter &= set(x)
    assert len(inter) == 1 or LOOSE and inter, ("gather space", inter, xs)
    return min(inter)


def r_inter(xs):
    inter = set(xs[0])
    for x in xs[1:]:
        inter &= set(x)
    assert len(inter) <= 1 or LOOSE, inter
    return min(inter) if inter else None


def r_split(xs):
    """Machine._split_gathers' choice for one map from every chip's (zero count, per-part {offset: count})."""
    cover, best = sum(x[0] for x in xs), "zero"
    for k in range(len(xs[0][1])):
        if xs[0][1][k] is None:
            continue
        tot = {}
        for x in xs:
            for v, c in x[1][k].items():
                tot[v] = tot.get(v, 0) + c
        if tot:
            cmax = max(tot.values())
            if cmax > cover:
                best, cover = (k, min(v for v, c in tot.items() if c == cmax)), cmax
    return None if cover < xs[0][2] else best


REDUCE = {"const": r_const, "any": any, "max": max, "space": r_space, "eq": lambda xs: all(x == xs[0] for x in xs),
          "inter": r_inter, "first": lambda xs: min(set(xs[0]).intersection(*xs[1:]), default=None), "split": r_split, "union": lambda xs: sorted(set().union(*xs)),
          "cat": lambda xs: [y for x in xs for y in x]}


def spmd_machine(C, sub, own_of, prog, mesh, mkw):
    """Machine on the chip's sub-program, P = K * cores with every element's true core."""
    from ttfluid.resident.machine import Machine
    assert mkw.get("layout", "rcb") == "rcb", "prep lanes are rcb ranks (lattice: prep must save lattice_lanes)"
    install_machine(C, sub, own_of, {})
    return Machine(prog, mesh, sub["K"] * sub["cores"], log=lambda *x: None, **mkw)


def install_machine(C, sub, own_of, state):
    """Machine patches: each element-dependent choice over the whole mesh, tables for the chip's own cores only,
    cross-chip halo sends exchanged. state["maps"]: the last finished Maps."""
    from ttfluid.resident import machine as mach
    from ttfluid.resident.machine import LANES, Layout, Maps, Machine
    P0, k = sub["cores"], sub["chip"]
    lo, hi = k * P0, (k + 1) * P0
    g = sub["g"]
    mach.rcb = lambda C_, P_, chips=None: sub["owner"]
    mach.gmax = lambda x: C("max", x)
    lay0, fin0, nh0 = Layout.__init__, Maps.finish, Maps.needs_halo

    def lay_init(self, prog, cell_owner, P, cell_lane=None, **kw):
        lay0(self, prog, cell_owner, P, cell_lane, **kw)
        self.lane[prog.cells.name] = np.asarray(sub["lane"], np.int64)   # halo cells too: their lane on their owner
        for cnt in self.count.values():       # other chips' cores: their counts here are partial
            cnt[:lo] = 0
            cnt[hi:] = 0
        spaces = [prog.cells] + list(prog.pspace.values())
        amg_vec = kw.get("amg_vec", False)
        self.wide = {sp.name for sp in spaces if sp is prog.cells or (amg_vec and sp.name.startswith("amg"))
                     or C("any", bool(self.count[sp.name].max() > LANES))}
        self.T = C("max", -(-max(int(self.count[nm].max()) for nm in self.wide) // LANES))
        # multigrid (and with ptiles patch) spaces' own tile counts (Layout.Ts), over the mesh
        ownt = lambda nm: nm.startswith("amg") or (kw.get("ptiles", False) and nm != prog.cells.name)
        self.Ts = {nm: max(1, -(-C("max", int(self.count[nm].max())) // LANES)) if ownt(nm) else self.T
                   for nm in sorted(self.wide)}

    def map_of(self, dst_space, parts, lanes=None):
        o = own_of(dst_space)
        h = hashlib.blake2b(dst_space.name.encode() + (b"|%d" % lanes if lanes is not None else b""), digest_size=16)
        if lanes is not None:       # --sliced: a narrow register's lanes only (Maps.map_of)
            ln = self.lay.lane[dst_space.name]
            parts = [(sr, np.asarray(d)[ln[np.asarray(d)] < lanes], np.asarray(p)[ln[np.asarray(d)] < lanes]) for sr, d, p in parts]
        # the gathers reuse a few index arrays thousands of times: digest each (space, d, p) once; the memo holds the
        # arrays, so their ids cannot be recycled while it lives
        memo = self.__dict__.setdefault("_pmemo", {})
        for sr, d, p in parts:
            mk = (dst_space.name, sr, id(d), id(p))
            if mk not in memo:
                d_, p_ = np.asarray(d, np.int64), np.asarray(p, np.int64)
                sel = o[d_]
                memo[mk] = (d, p, self.prog.space[sr].name.encode() + b"|%d|" % int(sel.sum())
                            + hashlib.blake2b(d_[sel].tobytes() + p_[sel].tobytes(), digest_size=32).digest())
            h.update(memo[mk][2])
        okey = self.__dict__.setdefault("_okey", {})
        key = h.digest()
        got = C("first", okey.get(key, []))        # maps named by their call, as Tracer.const's registers
        okey.setdefault(key, []).append(C.n)
        cmap = self.__dict__.setdefault("_cmap", {})
        if got in cmap:
            return cmap[got]
        cmap[C.n] = len(self.maps)
        self.maps.append((dst_space, [(self.prog.space[sr], np.asarray(d), np.asarray(p)) for sr, d, p in parts]))
        self.lanes.append(lanes)
        return len(self.maps) - 1

    def finish(self, *a, **kw):
        for i, (dsp, parts) in enumerate(self.maps):     # only the chip's own destinations
            o = own_of(dsp)
            self.maps[i] = (dsp, [(ssp, d[o[d]], p[o[d]]) for ssp, d, p in parts])
        self.cores = (lo, hi)       # tables of the chip's own cores only
        fin0(self, *a, **kw)
        state["maps"] = self
        if len(self.plans) == 1:
            self.hmax = C("max", int(self.hmax))
            self.plans[0]["hmax"] = self.hmax_q[0] = self.hmax
        else:                           # multigrid spaces: every plan's slot size over the whole mesh
            for q, pl in enumerate(self.plans):
                pl["hmax"] = self.hmax_q[q] = C("max", int(pl["hmax"]))
            self.hmax = max(self.hmax_q)
        # cross-chip halo: receivers here name the cells (global ids) they need from each remote sender
        cells = self.prog.cells.name
        own = self.lay.owner[cells]
        req = []
        for i, h in enumerate(self.halo_cells):
            if not lo <= i < hi or not len(h):
                continue
            for j in np.unique(own[h]):
                if not lo <= j < hi:
                    sel = np.nonzero(own[h] == j)[0]
                    req.append((int(j), i, int(self.pos[i][sel[0]]), g[h[sel]]))
        allreq = C("cat", req)
        for key in [x for x in self.send if not lo <= x[0] < hi]:
            del self.send[key]          # the remote sender's chip fills these
        lane_c = self.lay.lane[cells]
        for j, i, off, gids in allreq:     # (the lanes are global now; the exchange gives the sender the offsets)
            if lo <= j < hi:
                self.send[(j, i)] = lane_c[np.searchsorted(g, gids)]
                self.recv_off[(j, i)] = off
        self.cross = {(j, i): (off, len(gids)) for j, i, off, gids in allreq}
        self.senders, self.receivers = mach.defaultdict(list), mach.defaultdict(list)
        for (j, i) in sorted(set(self.send) | set(self.cross)):
            self.senders[i].append(j)
            self.receivers[j].append(i)
        # Encoded reads the halo lists per plan: plan 0 is the cells' (send / recv_off were updated in place)
        self.plans[0]["senders"], self.plans[0]["receivers"] = self.senders, self.receivers

    Layout.__init__ = lay_init
    Maps.map_of = map_of
    Maps.finish = finish
    Maps.needs_halo = lambda self, mid: set(C("union", sorted(nh0(self, mid))))

    def split_gathers(self, min_per_core=24):
        cells = self.prog.cells.name
        self.base = {}
        for mid, t in enumerate(self.maps.tab):
            dsp, parts = self.maps.maps[mid]
            stats = []
            d = t["idx"] - t["lane"]
            for kk, (ssp, _, _) in enumerate(parts):
                if ssp.name != cells or dsp.name != cells:
                    stats.append(None)
                    continue
                sel = (t["part"] == kk) & (t["kind"] == 0)
                vals, cnt = np.unique(d[sel & (d % 4 == 0)], return_counts=True)
                stats.append(dict(zip(vals.tolist(), cnt.tolist())))
            best = C("split", (len(t["zc"]), stats, min_per_core * self.P))
            if best is None:
                continue
            cover = None if best == "zero" else (t["part"] == best[0]) & (t["kind"] == 0) & (d == best[1])
            self.base[mid] = None if best == "zero" else tuple(best)
            self.maps.tab[mid] = Maps.drop(t, cover)
        out = []
        for ins in self.mir:
            if ins[0] == "gather" and ins[2] in self.base:
                b = self.base[ins[2]]
                u = [("ld", self.nslot - 1, "__anchor__"), ("fill", 0, 0.0)] if b is None else \
                    [("lds", 0, ins[3][b[0]][0], b[1])]
                out.append(("blk", u, [(0, ins[1])]))
                if C("any", bool(len(self.maps.tab[ins[2]]["core"]) + len(self.maps.tab[ins[2]]["zc"]))):
                    out.append(ins)
                continue
            out.append(ins)
        self.mir = out
    Machine._split_gathers = split_gathers

    def b_gathers(self):
        out, nb = [], 0
        for x in self.mir:
            if x[0] == "gather" and len(x) == 4 and self.phys[x[1]] < mach.DBASE and \
                    all(self.phys[q] < mach.DBASE and h is None for q, h in x[3]) and \
                    not C("any", bool((self.maps.tab[x[2]]["kind"] == 1).any())):
                x = x + (3,)
                nb += 1
            out.append(x)
        self._log("bgather: %d gathers on BRISC" % nb)
        self.mir = out
    Machine._b_gathers = b_gathers


def spmd_image(C, sub, own_of, prog, m, out, plan_kw, page, flags):
    """Encoded for the chip's own cores, the shared halo plan from every chip's cross pairs, the chip's image written
    to out/chip<k> (chip 0 also writes out/shared.npz). Returns the chip's digest."""
    from ttfluid.lower.image import Image
    from ttfluid.resident import chipplan
    from ttfluid.resident.encode import Encoded
    P0, K, k = sub["cores"], sub["K"], sub["chip"]
    lo, hi = k * P0, (k + 1) * P0
    sg = sub["space_g"]
    E = Encoded(m, page=bool(page), pdev=P0, tab_stream=False, cores=np.arange(lo, hi))
    L0 = C("max", int(E.tab.shape[1]))
    mp = m.maps
    runs = dict(mp.cross)
    runs.update({key: (mp.recv_off[key], len(mp.send[key])) for key in mp.send if key in mp.recv_off})
    R, Cm = plan_kw["mesh"]
    pl = chipplan.plan(mp.cross, K, P0, R, Cm, gw=plan_kw["gw"], nsh=plan_kw["nsh"], nhalo=m.nhalo, hmax=mp.hmax,
                       runs=runs)
    tab = E.tab.copy()          # rewritten in place, row by row (list rows of numpy scalars were ~40 B a word)
    chipplan.rewrite(tab, E.tab_halo, range(lo, hi), pl)
    cells_per_chip = C("cat", [int(np.sum(m.lay.owner[prog.cells.name][own_of(prog.cells)] // P0 == k))])
    gtab = pl["gtab"]
    info = dict(K=K, mesh=[R, Cm], place={x: pl["cells"][pl["place"][x]] for x in range(K)}, T=m.T, hmax=mp.hmax,
                nhalo=m.nhalo, cross_pairs={f"{x}->{y}": len(v) for (x, y), v in pl["pairs"].items()},
                gateways={f"{x}->{y}" + (f".{q}" if plan_kw["gw"] > 1 else ""): g for (x, y, q), g in pl["gw"].items()},
                shadow_kb=pl["NSH"] * pl["SHS"] * 4 / 1024,
                gw_load={g: (v["gP"], v["gR"], len(v["rec"])) for g, v in gtab.items()}, cells_per_chip=cells_per_chip)
    spaces = {sp.name: sp for sp in [prog.cells] + list(prog.pspace.values())}
    const_state = {}
    for name, cr in prog.state:                  # state registers that stayed constants: assembled over the chips
        for i, r in enumerate(cr.ravel()):
            r2 = m.ren(r)
            if r2 in m.phys:
                continue
            sp = m.space[r2]
            v = np.asarray(m.prog_pre[r2], np.float32)
            if sp is SCALAR:
                const_state[f"{name}/{i}"] = v
                continue
            o = own_of(sp)
            full = np.zeros(sub["space_n"][sp.name], np.float32)
            for gids, vals in C("cat", [(sg[sp.name][o], v[o])]):
                full[gids] = vals
            const_state[f"{name}/{i}"] = full
    shared = Image.shared_of(m, E, gtab, info, K, P0, flags, L0, space_n=sub["space_n"], const_state=const_state)

    def members(name):
        o = own_of(spaces[name])
        return sg[name][o], m.lay.owner[name][o] - lo, m.lay.lane[name][o]
    img = Image(shared, {k: Image.chip_of(E, tab, slice(0, P0), L0, shared, members)})
    img.write_chip(out, k)
    if k == 0:
        img.write_shared(out)
    return img.digest(k)


def machine_summary(m, lo, hi):
    tabs = [{key: v for key, v in t.items()} for t in m.maps.tab]
    for t in tabs:
        for c, keys in (("core", ("core", "lane", "kind", "part", "idx")), ("zc", ("zc", "zl"))):
            sel = (t[c] >= lo) & (t[c] < hi)
            for key in keys:
                t[key] = t[key][sel]
    sends = {key: np.asarray(v) for key, v in m.maps.send.items() if lo <= key[0] < hi}
    return dict(mir=pickle.dumps(m.mir), phys=m.phys, T=m.T, ntiles=m.ntiles, hmax=m.maps.hmax, nhalo=m.nhalo,
                tabs=tabs, sends=sends, recv_off={key: v for key, v in m.maps.recv_off.items() if lo <= key[1] < hi})


def install_trace(C, sub):
    """The per-chip trace: the sub-case in place of the case's mesh, fields and wall distance, and every
    value-dependent trace choice through the coordinator. Returns own_of(space) -> owned mask."""
    own_cell = sub["owner"] // sub["cores"] == sub["chip"]
    sg = sub["space_g"]
    owns, firsts = {}, {}

    def own_of(sp):
        if sp.cells is None:
            return own_cell
        if sp.name not in owns:
            owns[sp.name] = own_cell[sp.cells]
        return owns[sp.name]

    def first_of(sp):
        if sp is SCALAR:
            return 0
        if sp.name not in firsts:
            if sp.name in sg:
                g = sg[sp.name][own_of(sp)]
                firsts[sp.name] = int(g[0]) if len(g) else 1 << 62
            else:                       # a multigrid space (ttfluid.resident.sa): by its elements' cells
                g = sub["g"][sp.cells[own_of(sp)]]
                firsts[sp.name] = int(g.min()) if len(g) else 1 << 62
        return firsts[sp.name]

    simple.Mesh = lambda c: sub["mesh"]
    foamio.read_field = lambda p: copy.deepcopy(sub["fields"][Path(p).name])

    def wall_distance(mesh):
        wd = sub["wd"]
        wd.mesh = mesh
        return wd
    turbulence.WallDistance = wall_distance
    program.Tracer = spmd_tracer(C, own_of, first_of)
    from ttfluid.resident import sa           # chip-local smoothed aggregation: this chip's hierarchy, shapes over the mesh
    sa.OWN = lambda part: part // sub["cores"] == sub["chip"]
    sa.GMAX = lambda v: C("max", v)
    sa.GSUM = lambda v: sum(C("cat", [v]))
    sa.GUNION = lambda xs: [tuple(x) if isinstance(x, list) else x for x in C("union", sorted(set(xs)))]
    program.host_any = lambda a, cells=None: C("any", bool(np.any(a[own_cell[cells]] if cells is not None
                                                                  else a[own_cell])))
    ell_mod.max_count = lambda counts: C("max", int(counts[own_cell].max(initial=0)))

    def space_by_len(tr, k):
        hits = {s.name: s for s in set(tr.space) if s.n == k and s is not SCALAR}
        return hits[C("space", sorted(hits))]
    trace._space_by_len = space_by_len
    trace.host_max = lambda k: C("max", int(k))
    mean0 = trace.T.mean

    def mean(self, axis=None):     # over the whole mesh: the global element count
        if axis is None and self.space is not SCALAR:
            return self._reduce("sum") / float(sub["space_n"][self.space.name] * int(np.prod(self.c.shape)))
        return mean0(self, axis)
    trace.T.mean = mean
    return own_of


def worker(sub_path, case, n, conn, out, mkw=None, images=None):
    C = Coll(conn)
    try:
        with open(sub_path, "rb") as f:
            sub = pickle.load(f)
        sg = sub["space_g"]
        own_of = install_trace(C, sub)
        s = simple.SimpleFoam(case, backend="ell", dtype=np.float32, log=lambda *x: None, ell_order="face")
        prog = program.build(s, n)
        if mkw is not None:
            m = spmd_machine(C, sub, own_of, prog, s.mesh, mkw)
            P0 = sub["cores"]
            mres = machine_summary(m, sub["chip"] * P0, (sub["chip"] + 1) * P0)
            if images is not None:
                mres["digest"] = spmd_image(C, sub, own_of, prog, m, images["dir"], images["plan"], images["page"],
                                            images["flags"])
        res = dict(code=prog.code, space=[(sp.name, sp.n) for sp in prog.space], chip=sub["chip"],
                   own={name: own_of(sp) for name, sp in [("cells", prog.cells)] + [(sp.name, sp) for sp in prog.pspace.values()]},
                   space_g=sg, outs=prog.outs)
        if mkw is not None:
            res["machine"] = mres
        with open(out, "wb") as f:
            pickle.dump(res, f, protocol=5)
        C("done", None)
    except BaseException as e:
        import traceback
        traceback.print_exc()
        conn.send((-1, "error", repr(e)))
        raise


class _CkPickler:
    """Trace checkpoints keep the trace module's SCALAR space by reference (code tests `space is SCALAR`)."""
    @staticmethod
    def dump(obj, path):
        import cloudpickle

        class P(cloudpickle.Pickler):
            def persistent_id(self, o):
                return "SCALAR" if o is SCALAR else None
        tmp = Path(str(path) + ".tmp")
        with open(tmp, "wb") as f:
            P(f, protocol=5).dump(obj)
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        class U(pickle.Unpickler):
            def persistent_load(self, pid):
                assert pid == "SCALAR", pid
                return SCALAR
        with open(path, "rb") as f:
            return U(f).load()


def install_checkpoint(C, ckpt, state):
    """--replay trace checkpoint. ckpt = (path, "save"): after build() pickle this chip's traced program, its mesh and
    the collective log so far; (path, "load"): skip SimpleFoam and build, take them from the pickle (the coordinator
    asks for this only when every answer the trace used is the answer it gets now, so the trace is the same)."""
    if ckpt is None:
        return
    path, mode, ipath, imode = ckpt
    if mode == "save":
        # the same for the case setup (SimpleFoam's constructor: host numerics on the initial fields, about half of a
        # chip's trace time at 65M): (ipath, "save") pickles the constructed case, (ipath, "load") takes it
        if imode == "load":
            di = _CkPickler.load(ipath)
            C.n, C.log, C.guessed = di["n"], list(di["log"]), di["guessed"]
            state["ci_n"] = di["n"]
            simple.SimpleFoam = lambda *a, **kw: di["s"]
        else:
            sf0 = simple.SimpleFoam

            def sf(*a, **kw):
                s_ = sf0(*a, **kw)
                _CkPickler.dump(dict(s=s_, n=C.n, log=list(C.log), guessed=getattr(C, "guessed", 0)), ipath)
                Path(str(ipath) + ".n").write_text(str(C.n))
                state["ci_n"] = C.n
                return s_
            simple.SimpleFoam = sf
        b0 = program.build

        def build(s, *a, **kw):
            prog = b0(s, *a, **kw)
            _CkPickler.dump(dict(prog=prog, mesh=s.mesh, n=C.n, log=list(C.log), guessed=getattr(C, "guessed", 0)), path)
            Path(str(path) + ".n").write_text(str(C.n))
            state["ck_n"] = C.n
            if os.environ.get("SPMD_TRACE_ONLY") == "1" and C.guessed:
                raise TraceOnly     # this pass cannot be the fixed point and the trace reruns: Machine would be waste
            return prog
        program.build = build
        return
    d = _CkPickler.load(path)
    C.n, C.log, C.guessed = d["n"], list(d["log"]), d["guessed"]
    state["ck_n"] = d["n"]
    stub = type("TracedCase", (), {})()
    stub.mesh = d["mesh"]
    simple.SimpleFoam = lambda *a, **kw: stub
    program.build = lambda s, *a, **kw: d["prog"]


def run_mesh_worker(sub_path, argv, conn, out, replay=None, ckpt=None):
    """--run-mesh: this chip's process runs run_mesh.py itself with the same arguments (so every lowering flag means what
    it means there) and the per-chip patches: trace and Machine as worker(), Encoded for the chip's cores with the
    table width taken over all chips, the halo plan from every chip's cross pairs, and --images writing this chip."""
    import runpy
    import resource
    C = Coll(conn) if replay is None else Replay(replay)
    k = None
    try:
        with open(sub_path, "rb") as f:
            sub = pickle.load(f)
        P0, k = sub["cores"], sub["chip"]
        lo, hi = k * P0, (k + 1) * P0
        own_cell = sub["owner"] // P0 == k
        sg = sub["space_g"]
        own_of = install_trace(C, sub)
        state = {}
        install_machine(C, sub, own_of, state)
        from ttfluid.resident import chipplan, encode
        E0 = encode.Encoded

        class ChipEncoded(E0):
            def _tables(self):
                super()._tables()       # the cursor tables at the mesh's width: tabstream's L1 rows end in their zeros
                W = C("max", int(self.tab.shape[1]))
                self.tab = np.pad(self.tab, ((0, 0), (0, W - self.tab.shape[1])))

            def __init__(self, m, *a, **kw):
                super().__init__(m, *a, cores=np.arange(lo, hi), **kw)
                L0 = C("max", int(self.tab.shape[1]))
                self.tab = np.pad(self.tab, ((0, 0), (0, L0 - self.tab.shape[1])))
                if self.tab_stream:      # the DRAM part too: padded to the widest core of the mesh
                    B0 = C("max", int(self.tab_blob.shape[1]))
                    self.tab_blob = np.pad(self.tab_blob, ((0, 0), (0, B0 - self.tab_blob.shape[1])))
        encode.Encoded = ChipEncoded
        plan0, rw0 = chipplan.plan, chipplan.rewrite

        def plan(cross, K, P, R, Cm, **kw):
            mp = state["maps"]
            runs = dict(mp.cross)
            runs.update({key: (mp.recv_off[key], len(mp.send[key])) for key in mp.send if key in mp.recv_off})
            kw["runs"] = runs
            return plan0(mp.cross, K, P, R, Cm, **kw)
        chipplan.plan = plan
        chipplan.rewrite = lambda tab, tab_halo, cores, pl, *a: rw0(tab, tab_halo, range(lo, hi), pl, *a)
        from ttfluid.lower import image

        def from_lowering(cls, m, E, tab, gtab, info, K, P, flags):
            if k == 0 and os.environ.get("SPMD_SIZES"):
                f = sys._getframe(1)
                log("held (GB, first holder wins):", sizes_of(f.f_globals, ["E", "tab", "m", "prog", "s"]))
            prog = m.prog
            L0 = max(len(t) for t in tab)
            info = dict(info, cells_per_chip=C("cat", [int(own_cell.sum())]))
            spaces = {sp.name: sp for sp in [prog.cells] + list(prog.pspace.values())}
            const_state = {}
            for name, cr in prog.state:          # state registers that stayed constants: assembled over the chips
                for i, r in enumerate(cr.ravel()):
                    r2 = m.ren(r)
                    if r2 in m.phys:
                        continue
                    sp = m.space[r2]
                    v = np.asarray(m.prog_pre[r2], np.float32)
                    if sp is SCALAR:
                        const_state[f"{name}/{i}"] = v
                        continue
                    o = own_of(sp)
                    full = np.zeros(sub["space_n"][sp.name], np.float32)
                    for gids, vals in C("cat", [(sg[sp.name][o], v[o])]):
                        full[gids] = vals
                    const_state[f"{name}/{i}"] = full
            shared = cls.shared_of(m, E, gtab, info, K, P, flags, L0, space_n=sub["space_n"], const_state=const_state)

            def members(name):
                o = own_of(spaces[name])
                return sg[name][o], m.lay.owner[name][o] - lo, m.lay.lane[name][o]
            img = cls(shared, {k: cls.chip_of(E, tab, slice(0, P), L0, shared, members)})
            dig = img.digest(k)
            state["digest"] = dig
            img.write = lambda d: (img.write_chip(d, k), k == 0 and img.write_shared(d))
            img.digest = lambda q: dig if q == k else None
            return img
        image.Image.from_lowering = classmethod(from_lowering)
        install_checkpoint(C, ckpt, state)
        argv = list(argv)
        if "--owner" in argv:       # the partition came with the sub-case (prep.py --owner); run_mesh's would be global
            del argv[argv.index("--owner"):argv.index("--owner") + 2]
        sys.argv = ["run_mesh.py"] + argv
        try:
            runpy.run_path(str(Path(__file__).resolve().parents[1] / "device" / "run_mesh.py"), run_name="__main__")
        except SystemExit as e:
            assert not e.code, ("run_mesh exited", e.code)
        res = dict(chip=k, digest=state.get("digest"), cells=int(len(sub["g"])), prof=C.prof + [("end", rss_kb(), time.time())],
                   maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if replay is not None:
            res.update(log=C.log, guessed=C.guessed, error=None, ck_n=state.get("ck_n"), ci_n=state.get("ci_n"))
        with open(out, "wb") as f:
            pickle.dump(res, f, protocol=5)
        C("done", None)
    except Stopped:
        os._exit(0)
    except TraceOnly:
        with open(out, "wb") as f:
            pickle.dump(dict(chip=k, digest=None, log=C.log, guessed=C.guessed, error=None, ck_n=state.get("ck_n"),
                             ci_n=state.get("ci_n"),
                             maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), f, protocol=5)
    except BaseException as e:
        import traceback
        traceback.print_exc()
        if replay is None:
            conn.send((-1, "error", repr(e)))
            raise
        with open(out, "wb") as f:      # a guessed answer can break a later step: the log so far still counts
            pickle.dump(dict(chip=k, digest=None, log=C.log, guessed=C.guessed, error=repr(e),
                             maxrss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), f, protocol=5)


# ---- coordinator ---------------------------------------------------------------------------------------------------
def coordinate(conns):
    ncall = 0
    while True:
        msgs = [c.recv() for c in conns]
        bad = [(k, m) for k, m in enumerate(msgs) if m[1] == "error"]
        if bad:
            raise RuntimeError(f"worker error: {bad}")
        if len({(m[0], m[1]) for m in msgs}) != 1:
            raise RuntimeError(f"divergence at call {ncall}: {[(m[0], m[1]) for m in msgs]}")
        ncall += 1
        if msgs[0][1] == "done":
            for c in conns:
                c.send(None)
            return ncall
        r = REDUCE[msgs[0][1]]([m[2] for m in msgs])
        for c in conns:
            c.send(r)


def coordinate_log(conns, budget_kb):
    """Pass 0 of --replay: lockstep, logging every answer (kind, pickled answer, call site), until the chips' summed RSS
    passes budget_kb; then every chip is told to stop. The logged answers are the lockstep ones."""
    log_, ncall = [], 0
    while True:
        msgs = [c.recv() for c in conns]
        bad = [(k, m) for k, m in enumerate(msgs) if m[1] == "error"]
        if bad:
            raise RuntimeError(f"worker error: {bad}")
        if len({(m[0], m[1]) for m in msgs}) != 1:
            raise RuntimeError(f"divergence at call {ncall}: {[(m[0], m[1]) for m in msgs]}")
        ncall += 1
        if msgs[0][1] == "done":
            for c in conns:
                c.send(None)
            return log_, False
        if sum(m[4] for m in msgs) > budget_kb:
            for c in conns:
                c.send(STOP)
            return log_, True
        r = REDUCE[msgs[0][1]]([m[2] for m in msgs])
        log_.append((msgs[0][1], pickle.dumps(r, protocol=5), msgs[0][3]))
        for c in conns:
            c.send(r)


# ---- the gate against the global trace ------------------------------------------------------------------------------
def same(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[x], b[x]) for x in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    return a == b


def gate_machine(prog, mesh, results, mkw):
    from ttfluid.resident.machine import Machine
    K, P0 = len(results), mkw["chip_cores"]
    m = Machine(prog, mesh, K * P0, log=lambda *x: None, **mkw)
    bad = []
    for res in results:
        k = res["chip"]
        ref, got = machine_summary(m, k * P0, (k + 1) * P0), res["machine"]
        diff = [key for key in ref if not same(ref[key], got[key])]
        if diff:
            bad.append((k, "machine", diff))
    return bad, m


def gate(case, n, results, mkw=None):
    s = simple.SimpleFoam(case, backend="ell", dtype=np.float32, log=lambda *x: None, ell_order="face")
    prog = program.build(s, n)
    gspace = [(sp.name, sp.n) for sp in prog.space]
    bad = []
    for res in results:
        k = res["chip"]
        if len(res["code"]) != len(prog.code):
            bad.append((k, "code length", len(res["code"]), len(prog.code)))
        if [x[0] for x in res["space"]] != [x[0] for x in gspace]:
            i = next((i for i, (x, y) in enumerate(zip(res["space"], gspace)) if x[0] != y[0]), None)
            bad.append((k, "register spaces differ from reg", i))
        own, sg = res["own"], res["space_g"]
        nm = lambda r: gspace[r][0]
        for i, (a, b) in enumerate(zip(res["code"], prog.code)):
            if a[0] != b[0]:
                bad.append((k, i, "op", str(a)[:150], str(b)[:150], [str(x)[:100] for x in prog.code[i - 3:i]]))
                break
            if a[0] in ("state", "const"):
                if a[1] != b[1]:
                    bad.append((k, i, a[0], "reg", a[1], b[1]))
                    break
                name = nm(a[1])
                if name == "scalar":
                    ok = np.array_equal(a[2], b[2])
                else:
                    o = own[name]
                    ok = np.array_equal(np.asarray(a[2])[o].view(np.uint32), np.asarray(b[2])[sg[name][o]].view(np.uint32))
                if not ok:
                    bad.append((k, i, a[0], a[1], name))
                    break
            elif a[0] == "gather":
                if a[1] != b[1] or [p[0] for p in a[2]] != [p[0] for p in b[2]]:
                    bad.append((k, i, "gather regs"))
                    break
                dn = nm(a[1])
                o = own[dn]
                gN = dict(gspace)[dn]
                gw, gsrc = np.full(gN, -1), np.zeros(gN, np.int64)
                for j, (src, d, p) in enumerate(b[2]):
                    gw[np.asarray(d, np.int64)] = j
                    gsrc[np.asarray(d, np.int64)] = np.asarray(p)
                lw, lsrc = np.full(len(o), -1), np.zeros(len(o), np.int64)
                for j, (src, d, p) in enumerate(a[2]):
                    lw[np.asarray(d, np.int64)] = j
                    lsrc[np.asarray(d, np.int64)] = sg[nm(src)][np.asarray(p, np.int64)] if len(np.asarray(p)) else 0
                go = sg[dn][o]
                if not (np.array_equal(lw[o], gw[go]) and np.array_equal(np.where(lw[o] >= 0, lsrc[o], 0),
                                                                          np.where(gw[go] >= 0, gsrc[go], 0))):
                    bad.append((k, i, "gather", a[1], dn))
                    break
            elif a != b:
                bad.append((k, i, "instr", str(a)[:120], str(b)[:120]))
                break
    if mkw is not None and not bad:
        bad += gate_machine(prog, s.mesh, results, mkw)[0]
    return bad, len(prog.code)


def main_run_mesh(sub_dir, argv):
    """spmd.py SUBDIR -- <run_mesh.py arguments, with --images DIR>: every chip's process runs run_mesh.py."""
    import json
    subs = sorted(Path(sub_dir).glob("sub*.pkl"), key=lambda p: int(p.stem[3:]))
    ctx = mp.get_context("fork")
    conns, procs, outs = [], [], []
    for sp in subs:
        here, there = ctx.Pipe()
        out = sp.with_name("res" + sp.stem[3:] + ".pkl")
        p = ctx.Process(target=run_mesh_worker, args=(sp, argv, there, out))
        p.start()
        there.close()
        conns.append(here)
        procs.append(p)
        outs.append(out)
    ncall = coordinate(conns)
    for p in procs:
        p.join()
    dig, rss, cells, profs = {}, [], 0, []
    for o in outs:
        r = pickle.load(open(o, "rb"))
        dig[str(r["chip"])] = r["digest"]
        rss.append(r["maxrss_kb"] / 1e6)
        cells += r["cells"]
        profs.append(r["prof"])
    log("worker peak RSS GB: max %.2f, sum %.2f (%d B per sub-case cell)" % (max(rss), sum(rss), sum(rss) * 1e9 / cells))
    # lockstep: call c happens on every chip at once, so the summed RSS at call c is the host's footprint then
    tot = [sum(p[c][1] for p in profs) for c in range(len(profs[0]))]
    last = None
    for c, (kind, _, t) in enumerate(profs[0]):
        if kind != last or c == len(tot) - 1 or tot[c] == max(tot):
            log("  rss call %5d %-7s t %7.1f s  sum %6.2f GB  max %5.2f GB" % (c, kind, t - profs[0][0][2], tot[c] / 1e6,
                                                                       max(p[c][1] for p in profs) / 1e6))
        last = kind
    d = argv[argv.index("--images") + 1]
    Path(d, "digests.json").write_text(json.dumps(dig, indent=1))
    log("lowered %d chips, %d collective calls; images %s, digests %s" % (len(subs), ncall, d, json.dumps(dig)))
    return 0


def reduce_loose(kind, xs):
    global LOOSE
    try:
        return REDUCE[kind](xs), True
    except Exception:
        LOOSE = True
        try:
            return REDUCE[kind](xs), False
        finally:
            LOOSE = False


def main_replay(sub_dir, argv, B, passes=8):
    """spmd.py SUBDIR --replay B -- <run_mesh.py arguments>: the chips run alone, B at a time, each pass answering the
    collectives from the previous pass's contributions (the first pass guesses from each chip's own). Exact at the fixed
    point: if in a pass no chip guessed, every chip made the same calls, and reducing this pass's contributions gives
    back the answers they used, then by induction over the calls each chip ran exactly as in lockstep (call 1's
    contributions depend on the data alone; given equal answers up to call c, call c + 1's contributions are equal).
    Host memory: B chips' footprint, not K's."""
    import json
    subs = sorted(Path(sub_dir).glob("sub*.pkl"), key=lambda p: int(p.stem[3:]))
    work = Path(sub_dir, "replay")
    work.mkdir(exist_ok=True)
    K = len(subs)
    aps = [work / ("answers%d.pkl" % i) for i in range(K)]
    hints = [[] for _ in range(K)]                # per chip: the answers its next pass uses
    # trace checkpoints (SPMD_CKPT=0 turns them off): per chip (calls the trace made, the answers it used for them)
    use_ck = os.environ.get("SPMD_CKPT", "1") != "0"
    cks = [work / ("trace%d.pkl" % i) for i in range(K)]
    cis = [work / ("init%d.pkl" % i) for i in range(K)]
    ck, ci = [None] * K, [None] * K
    # SPMD_RESUME=1: continue a killed run at the pass it was in: that pass's answers files are the hints, chips whose
    # result is newer than their answers file finished it, and a checkpoint stays valid with the answers it used (.ans)
    resume = os.environ.get("SPMD_RESUME") == "1" and all(a_.exists() for a_ in aps)
    ans = lambda c_: Path(str(c_) + ".ans")

    def note_ck(c_, v):
        if v is None:
            ans(c_).unlink(missing_ok=True)
        else:
            ans(c_).write_bytes(pickle.dumps(v, protocol=5))
        return v
    if resume:
        hints = [pickle.load(open(a_, "rb")) for a_ in aps]
        ck = [pickle.loads(ans(c_).read_bytes()) if ans(c_).exists() and c_.exists() else None for c_ in cks]
        ci = [pickle.loads(ans(c_).read_bytes()) if ans(c_).exists() and c_.exists() else None for c_ in cis]
        log("resume: hints from the answers files (%d..%d calls), %d traces and %d cases checkpointed" % (
            min(map(len, hints)), max(map(len, hints)), sum(x is not None for x in ck), sum(x is not None for x in ci)))
    else:
        for c_ in cks + cis:
            for x in (c_, Path(str(c_) + ".n"), ans(c_)):
                x.unlink(missing_ok=True)
    ctx = mp.get_context("fork")
    p0 = 0 if resume else float(os.environ.get("SPMD_P0_GB", 0))
    if p0:      # pass 0: lockstep while the K chips fit in p0 GB; its answers are exact
        t0 = time.time()
        conns, procs = [], []
        for i, sp in enumerate(subs):
            here, there = ctx.Pipe()
            p = ctx.Process(target=run_mesh_worker, args=(sp, argv, there, work / ("p0_%d.pkl" % i), None,
                                                          (cks[i], "save", cis[i], "save") if use_ck else None))
            p.start()
            there.close()
            conns.append(here)
            procs.append(p)
        lg, stopped = coordinate_log(conns, p0 * 1e6)
        for p in procs:
            p.join()
        hints = [list(lg) for _ in range(K)]
        for i in range(K):      # a chip that finished its trace under exact answers
            nf = Path(str(cks[i]) + ".n")
            if nf.exists() and int(nf.read_text()) <= len(lg):
                ck[i] = note_ck(cks[i], (int(nf.read_text()), list(lg[:int(nf.read_text())])))
            nf = Path(str(cis[i]) + ".n")
            if nf.exists() and int(nf.read_text()) <= len(lg):
                ci[i] = note_ck(cis[i], (int(nf.read_text()), list(lg[:int(nf.read_text())])))
        log("pass 0 (lockstep under %.0f GB): %.1f s, %d exact answers, %s" % (p0, time.time() - t0, len(lg),
                                                                            "stopped at the budget" if stopped else "ran to the end"))
    for npass in range(1, passes + 1):
        outs = [work / ("res%d.pkl" % i) for i in range(K)]
        kept = set()
        if resume and npass == 1:   # results are a function of the answers: keep those made under this answers file
            kept = {i for i in range(K) if outs[i].exists() and outs[i].stat().st_mtime > aps[i].stat().st_mtime}
            log("resume: chips %s kept from the interrupted pass" % sorted(kept))
        else:
            for i in range(K):
                with open(aps[i], "wb") as f:
                    pickle.dump(hints[i], f, protocol=5)
        t0 = time.time()
        mode = [None] * K
        for i in range(K):
            if use_ck:
                valid = lambda c_: c_ is not None and hints[i][:c_[0]] == c_[1]
                mode[i] = (cks[i], "load" if valid(ck[i]) else "save", cis[i], "load" if valid(ci[i]) else "save")
        todo, live = [(i, subs[i], outs[i], aps[i]) for i in range(K) if i not in kept], []
        while todo or live:
            while todo and len(live) < B:
                i, sp, out, a_ = todo.pop(0)
                out.unlink(missing_ok=True)
                for c_, m_ in ((cks[i], mode[i][1]), (cis[i], mode[i][3])) if mode[i] else ():
                    if m_ == "save":    # a kill mid-save must not leave a stale answer record
                        ans(c_).unlink(missing_ok=True)
                p = ctx.Process(target=run_mesh_worker, args=(sp, argv, None, out, a_, mode[i]))
                p.start()
                live.append(p)
            live[0].join(0.5)
            live = [p for p in live if p.is_alive()]
        res = [pickle.load(open(o, "rb")) for o in outs]
        logs = [r["log"] for r in res]
        for i, r in enumerate(res):     # a trace (case) saved this pass: valid while its calls keep the answers it used
            if mode[i] and mode[i][1] == "save" and r.get("ck_n") is not None:
                ck[i] = note_ck(cks[i], (r["ck_n"], list(hints[i][:r["ck_n"]])) if len(hints[i]) >= r["ck_n"] else None)
            if mode[i] and mode[i][3] == "save" and r.get("ci_n") is not None:
                ci[i] = note_ck(cis[i], (r["ci_n"], list(hints[i][:r["ci_n"]])) if len(hints[i]) >= r["ci_n"] else None)
        # the agreed prefix: every chip made the same call; its strict reduce is the lockstep answer if the inputs were
        new, strict = [], True
        for c in range(min(len(x) for x in logs)):
            kinds = {(x[c][0], x[c][2]) for x in logs}
            if len(kinds) != 1:
                break
            kind = kinds.pop()[0]
            try:
                a, ok = reduce_loose(kind, [pickle.loads(x[c][1]) for x in logs])
            except Exception:
                break
            strict &= ok
            new.append((kind, pickle.dumps(a, protocol=5), logs[0][c][2]))
        errs = [(r["chip"], r["error"]) for r in res if r["error"]]
        fixed = (strict and not errs and all(r["guessed"] == 0 for r in res) and all(len(x) == len(new) for x in logs)
                 and all(h == new for h in hints))
        diff = min(next((c for c in range(min(len(new), len(h))) if new[c] != h[c]), min(len(new), len(h))) for h in hints)
        c = len(new)
        log("  first answer change at %d: %s; calls disagree at %d: %s" % (
            diff, logs[0][diff][2] if diff < len(logs[0]) else "-", c,
            sorted({(x[c][0], x[c][2]) if len(x) > c else ("end", "") for x in logs})[:4]))
        log("pass %d: %.1f s, worker peak RSS max %.2f GB (x %d at once), calls %d..%d, agreed prefix %d, first answer "
            "change %d, guessed %s, errors %s, traces loaded %d, cases loaded %d, %s" % (npass, time.time() - t0, max(r["maxrss_kb"] for r in res) / 1e6, B,
                                                      min(map(len, logs)), max(map(len, logs)), len(new), diff,
                                                      [r["guessed"] for r in res], errs[:2], sum(1 for x in mode if x and x[1] == "load"),
                                                      sum(1 for x in mode if x and x[1] == "save" and x[3] == "load"), "FIXED POINT" if fixed else "again"))
        if fixed:
            dig = {str(r["chip"]): r["digest"] for r in res}
            d = argv[argv.index("--images") + 1]
            Path(d, "digests.json").write_text(json.dumps(dig, indent=1))
            log("lowered %d chips in %d passes; images %s, digests %s" % (K, npass, d, json.dumps(dig)))
            return 0
        # next hints, per chip: the agreed prefix, then call c reduced over the chips whose call c has this chip's kind
        # and call site (chips on another code path make other calls); nothing past its own log (stale hints misfire)
        nh = []
        for i in range(K):
            h = list(new)
            for c in range(len(new), len(logs[i])):
                kind, site = logs[i][c][0], logs[i][c][2]
                xs = [pickle.loads(x[c][1]) for x in logs if len(x) > c and x[c][0] == kind and x[c][2] == site]
                try:
                    h.append((kind, pickle.dumps(reduce_loose(kind, xs)[0], protocol=5), site))
                except Exception:       # contributions made on different paths need not even reduce
                    break
            nh.append(h)
        assert nh != hints, "replay made no progress"
        hints = nh
    raise RuntimeError("no fixed point in %d passes" % passes)


def main():
    if "--" in sys.argv:
        i = sys.argv.index("--")
        if "--replay" in sys.argv[2:i]:
            return main_replay(sys.argv[1], sys.argv[i + 1:], int(sys.argv[sys.argv.index("--replay") + 1]),
                               int(os.environ.get("SPMD_PASSES", 8)))
        return main_run_mesh(sys.argv[1], sys.argv[i + 1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("sub")
    ap.add_argument("case")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--machine", help="JSON Machine kwargs (chip_cores required): also lower each chip's Machine")
    ap.add_argument("--images", help="write the chips' device images here (needs --machine)")
    ap.add_argument("--mesh", default="1,2", help="the chips' mesh R,C (run_mesh --mesh)")
    ap.add_argument("--gw", type=int, default=1)
    ap.add_argument("--nsh", type=int, default=2)
    ap.add_argument("--page", type=int, default=0)
    a = ap.parse_args()
    import json
    mkw = json.loads(a.machine) if a.machine else None
    images = None
    if a.images:
        assert mkw is not None, "--images needs --machine"
        images = dict(dir=a.images, page=a.page, plan=dict(mesh=[int(x) for x in a.mesh.split(",")], gw=a.gw, nsh=a.nsh),
                      flags=dict(spmd=True, case=str(a.case), n=a.n, machine=mkw, mesh=a.mesh, gw=a.gw, nsh=a.nsh, page=a.page))
    subs = sorted(Path(a.sub).glob("sub*.pkl"), key=lambda p: int(p.stem[3:]))
    ctx = mp.get_context("fork")
    conns, procs, outs = [], [], []
    for sp in subs:
        here, there = ctx.Pipe()
        out = sp.with_name("prog" + sp.stem[3:] + ".pkl")
        p = ctx.Process(target=worker, args=(sp, a.case, a.n, there, out, mkw, images))
        p.start()
        there.close()
        conns.append(here)
        procs.append(p)
        outs.append(out)
    ncall = coordinate(conns)
    for p in procs:
        p.join()
    log("traced %d chips, %d collective calls" % (len(subs), ncall))
    if images is not None:
        dig = {}
        for o in outs:
            r = pickle.load(open(o, "rb"))
            dig[r["chip"]] = r["machine"]["digest"]
        Path(a.images, "digests.json").write_text(json.dumps(dig, indent=1))
        log("images %s, digests %s" % (a.images, json.dumps(dig)))
    if a.check:
        results = [pickle.load(open(o, "rb")) for o in outs]
        bad, ncode = gate(a.case, a.n, results, mkw)
        log("gate vs global trace (%d instructions): %s" % (ncode, "PASS" if not bad else bad[:10]))
        return 0 if not bad else 1


T0 = time.time()
if __name__ == "__main__":
    sys.exit(main())
