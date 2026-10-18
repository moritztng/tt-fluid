"""Executor of a lowered Machine program, all cores at once, in program order. Storage: one (P, size * 1024) array per
physical slot (size T for a cell vector, 1 otherwise); a block that touches a cell vector runs once per tile.

It checks the lowering (partition, lanes, gather maps, halo plan, Dest blocks, tile allocation) against the IR
Emulator; the device kernels run the same MIR. Device semantics it reproduces: comparisons as sub + sign test,
pow by its lowering, a reduction as per-core partials folded in a fixed order, padding lanes as garbage (NaN).
"""
import re
import os
from collections import Counter, defaultdict

import numpy as np

from .machine import LANES, DBASE, KBASE, bf16_rne, widen_bf16
from .trace import SCALAR

UN = {"neg": np.negative, "abs": np.abs, "sqrt": np.sqrt, "tanh": np.tanh, "log": np.log, "exp": np.exp,
      "gtz": lambda x: (x > 0), "gez": lambda x: (x >= 0), "ltz": lambda x: (x < 0), "lez": lambda x: (x <= 0),
      "eqz": lambda x: (x == 0), "nez": lambda x: (x != 0)}
BIN = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "max": np.maximum,
       "min": np.minimum}


def _put(tile, lay, sp, values):
    """values of space sp into a (P, n * LANES) tile, lanes past n tiles dropped (a --sliced register's tail)."""
    own, ln = lay.owner[sp.name], lay.lane[sp.name]
    k = ln < tile.shape[1]
    tile[own[k], ln[k]] = values[k] if values.ndim else values


class _Owner(dict):
    """owner map that also marks the tiles an overlaid register (virtual base -> device base) shares with one-tile
    slots, over the register's own width (an offset's registers differ in width: a coarse level fills its front)."""

    def __init__(self, ovdev, wid, rwid):
        super().__init__()
        self.base, self.rwid = ovdev, rwid
        self.at = defaultdict(set)
        for v, o in ovdev.items():
            for t in range(o, o + wid[v]):
                self.at[t].add(v)

    def _covers(self, v, t):
        r = self.get(v)
        return r is not None and not isinstance(r, tuple) and self.base[v] <= t < self.base[v] + self.rwid(r)

    def __setitem__(self, k, r):
        if k in self.base:
            for t in range(self.base[k], self.base[k] + self.rwid(r)):
                for v in self.at[t] - {k}:
                    if self._covers(v, t):
                        super().__setitem__(v, ("overlaid", r))
                super().__setitem__(t, ("overlaid", r))
        else:
            for v in self.at.get(k, ()):
                if self._covers(v, k):
                    super().__setitem__(v, ("one-tile", r))
        super().__setitem__(k, r)


class MachineEmu:
    def __init__(self, m, dtype=None):
        self.m = m
        self._bsc_seen = set()
        self.F = np.dtype(dtype or m.prog.dtype).type
        P = m.P
        T = m.T
        self.A = {b: np.full((P, LANES), np.nan, self.F) for b in range(m.nsmall)}
        self.A.update({b: np.full((P, n * LANES), np.nan, self.F) for b, n in m.vslots})
        # wplace 2: an overlaid L1 colour lives in its own array at a virtual base; owner tracks the device tiles it
        # shares with one-tile registers, so a read of a value the other one overwrote fails the allocation check
        self.phys = {**m.phys, **getattr(m, "ovl", {})}
        self.A.update({v: np.full((P, T * LANES), np.nan, self.F) for v in set(getattr(m, "ovdev", {}))})
        self.A.update({b: np.full((P, n * LANES), np.nan, self.F) for b, n in m.dslots})
        self._ends = {b + a.shape[1] // LANES: b for b, a in self.A.items()}   # the slot that ends where b starts
        self.H = np.full((max(m.nhalo, 1), P, m.maps.hmax), np.nan, self.F)
        for r, v in m.pre.items():
            t = self.phys.get(r)
            if t is None:
                continue
            if v is None:
                self.A[t][...] = 0
            else:
                sp = m.space[r]
                tile = np.full((P, m.size(r) * LANES), np.nan, self.F)
                _put(tile, m.lay, sp, np.asarray(v, self.F))
                if r in m.masks.values():
                    tile[np.isnan(tile)] = 0
                if t >= m.nsmall and tile.shape[1] < self.A[t].shape[1]:   # a multigrid level in a T-tile slot
                    tile = np.concatenate([tile, np.full((P, self.A[t].shape[1] - tile.shape[1]), np.nan, self.F)], 1)
                self.A[t] = tile
        self.ctile = {}
        for r in m.streamed:
            sp = m.space[r]
            tile = np.full((P, m.size(r) * LANES), np.nan, self.F)
            _put(tile, m.lay, sp, np.asarray(m.pre[r], self.F))
            self.ctile[r] = tile
        for r, b in m.alias.items():     # DRAM-homed fetched constants read in place (bf16 shadow when narrowed)
            c = m.fetched_from[r]
            if c in getattr(m, "pre_h", {}) and b not in self.A:
                sp = m.space[c]
                tile = np.full((P, m.size(c) * LANES), np.nan, self.F)
                _put(tile, m.lay, sp, widen_bf16(m.pre_h[c]))
                self.A[b] = tile
            self.A.setdefault(b, self.ctile[c].copy())
        self._page = {}
        self.lds_adjacent = False   # offset loads read past the tile as NaN (strict) or into the next tile (device)
        self.howner = {}    # halo buffer slot -> halo whose values it holds
        self._ovt = {r: tuple(range(m.phys[r], m.phys[r] + min(m.size(r), T))) for r in getattr(m, "ovl", {})}
        self.owner = _Owner(getattr(m, "ovdev", {}), getattr(m, "ovwid", {}), lambda r: min(m.size(r), T))     # physical tile -> register whose value it holds (checking the allocation)
        for r, t in self.phys.items():
            if r in m.pre:
                self.owner[t] = r
        self.check = True
        self.labels = {ins[1]: i for i, ins in enumerate(m.mir) if ins[0] == "label"}
        self.history, self.counts = [], Counter()
        self.pcc = Counter()     # executions per MIR index (region census)
        self.optiles = Counter()  # executed tile ops: "op" compute uops, "ld" loads (a --sliced A/B weighted by trips)
        # the same per SIMPLE phase (program.PHASES, from the phase markers of build(phase=True)), plus "ldr" DRAM page
        # loads, "st" stores into DRAM homes, "blk" block runs, "gather" gathers and "red" reductions
        self.phase_at, self.phase, self.phtiles = getattr(m.prog, "phase_at", {}), "top", defaultdict(Counter)
        self.branches = []      # outcome of every jumpif executed, in order (replayed by devsim)
        self.record = None
        self.record_from = 0
        self.record_from = 0
        self._prep()

    def _prep(self):
        m = self.m
        self.gt = []
        for t in m.maps.tab:
            self.gt.append(t)
        # halo: per (sender j, receiver i): receiver positions and sender lanes
        cat = lambda x: np.concatenate(x) if x else np.zeros(0, np.int64)
        self.hplan = []    # per halo plan: receiver core, receiver position, sender core, sender lane
        for pl in m.maps.plans:
            rc, rp, sc, sl = [], [], [], []
            for (j, i), lanes in pl["send"].items():
                off = pl["recv_off"][(j, i)]
                rc.append(np.full(len(lanes), i)); rp.append(off + np.arange(len(lanes)))
                sc.append(np.full(len(lanes), j)); sl.append(lanes)
            self.hplan.append((cat(rc), cat(rp), cat(sc), cat(sl)))

    def run(self, max_outer=None, log=None):
        m, A, H, F = self.m, self.A, self.H, self.F
        mir, phys, hphys = m.mir, self.phys, m.hphys
        pc, row, nout = 0, {}, 0
        with np.errstate(all="ignore"):
            while pc < len(mir):
                ins = mir[pc]
                pc += 1
                op = ins[0]
                self.counts[op] += 1
                self.pcc[pc - 1] += 1
                if op == "label" and ins[1] in self.phase_at:
                    from .program import PHASES
                    self.phase = PHASES[self.phase_at[ins[1]]]
                pt = self.phtiles[self.phase]
                if op in ("gather", "red", "red2", "sync", "halo"):
                    pt[op] += 1
                if op in ("blk", "seg"):
                    # a fused segment runs all its blocks tile by tile; ownership is checked in program order
                    blks = [ins] if op == "blk" else [x for x in ins[1] if x[0] == "blk"]
                    vec = False
                    for x in blks:
                        regs = [u[2] for u in x[1] if u[0] in ("ld", "lds", "rvf")] + [r for _, r in x[2]]
                        vec |= any(phys[r] >= m.nsmall for r in regs if r != "__anchor__")
                        for u in x[1]:
                            if u[0] in ("ld", "lds", "rvf"):
                                self._chk(u[2], pc)
                            elif u[0] == "ldr":
                                vec = True
                                for y in ([u[2]] if u[2] is not None else []) + [y for y, _ in u[5]]:
                                    self._chk(y, pc)
                        for _, r in x[2]:
                            self.owner[phys[r]] = r
                    self._bt = m.tiles(blks)
                    nt = self._bt if vec else 1
                    if self.check and vec and self._ovt:
                        self._ovl_loop(blks, pc)
                    for x in blks:
                        for u in x[1]:
                            self.optiles["ld" if u[0] in ("ld", "lds", "rvf", "ldr") else "op"] += nt
                            pt["ld" if u[0] in ("ld", "lds", "rvf", "ldr") else "op"] += nt
                            if u[0] == "ldr":   # by source: a constant, a gather page the reader builds, a vector home
                                pt["ldr"] += nt
                                pt["ldr:" + ("gather" if u[2] is None else "const" if u[2] in m.const_regs else "vec")] += nt
                        pt["st"] += nt * sum(1 for _, r in x[2] if r != "__anchor__" and phys[r] >= DBASE)
                    pt["blk"] += len(blks)
                    for t in range(self._bt if vec else 1):
                        for x in blks:
                            self._block(x, t)
                    self._page.clear()      # ldr pages are rebuilt at t == 0 of each run; kept, they grew to GBs
                    for x in blks:
                        for _, r in x[2]:
                            if self.record is not None and len(self.history) >= self.record_from and \
                                    r not in self.record and isinstance(r, int) and r < len(m.prog.space):
                                self.record[r] = self._glob(r, A[phys[r]])
                elif op == "gather":
                    t = self.gt[ins[2]]
                    dst = phys[ins[1]]
                    for x, _ in ins[3]:
                        self._chk(x, pc)
                    out = A[dst].copy()
                    mode = ins[4] if len(ins) > 4 else 0     # gsplit: 1 local + zeros, 2 halo values only
                    for k, (s, h) in enumerate(ins[3]):
                        sel = t["part"] == k
                        loc = sel & (t["kind"] == 0)
                        if mode != 2:
                            out[t["core"][loc], t["lane"][loc]] = A[phys[s]][t["core"][loc], t["idx"][loc]]
                        rem = sel & (t["kind"] == 1)
                        if rem.any() and mode not in (1, 4):   # bloc: 4 = 1 on BRISC
                            self._hchk(h, pc)
                            out[t["core"][rem], t["lane"][rem]] = H[hphys[("H", h)]][t["core"][rem], t["idx"][rem]]
                    if mode != 2:
                        out[t["zc"], t["zl"]] = 0
                    A[dst] = out
                    self.owner[dst] = ins[1]
                    if self.record is not None and len(self.history) >= self.record_from and ins[1] not in self.record:
                        self.record[ins[1]] = self._glob(ins[1], out)
                    self.counts["g:values"] += (len(t["core"]) + len(t["zc"])) / m.P
                    pt["gval"] += (len(t["core"]) + len(t["zc"])) / m.P   # values written per core: NCRISC scalar copies
                    lv = re.match(r"amgs(\d+)", m.space[ins[1]].name) if isinstance(ins[1], int) else None
                    pt["gval:L%s" % (lv.group(1) if lv else 0)] += (len(t["core"]) + len(t["zc"])) / m.P
                    pt["gzero:L%s" % (lv.group(1) if lv else 0)] += len(t["zc"]) / m.P
                    pt["ghalo:L%s" % (lv.group(1) if lv else 0)] += int((t["kind"] == 1).sum()) / m.P
                elif op == "bscal":     # lane 0 per core in fp32 (BRISC soft-float: IEEE, a mad rounds twice)
                    D = {}
                    for u in ins[1]:
                        k = u[0]
                        if k == "ld":
                            self._chk(u[2], pc)
                            D[u[1]] = A[phys[u[2]]][:, 0].copy()
                        elif k == "fill":
                            D[u[1]] = np.full(m.P, u[2], F)
                        elif k == "cpd":
                            D[u[2]] = D[u[1]].copy()
                        elif k == "un":
                            D[u[2]] = UN[u[1]](D[u[2]]).astype(F)
                        elif u[1] == "where":
                            D[u[2]] = np.where(D[u[3]] != 0, D[u[4]], D[u[5]]).astype(F)
                        elif u[1] in ("mad", "msb"):
                            pr = (D[u[3]] * D[u[4]]).astype(F)
                            D[u[2]] = (D[u[5]] + pr if u[1] == "mad" else D[u[5]] - pr).astype(F)
                        else:
                            D[u[2]] = BIN[u[1]](D[u[3]], D[u[4]]).astype(F)
                    if os.environ.get("BSCAL_CHECK"):   # the same uops on whole tiles: lanes that differ from lane 0
                        W = {}
                        for u in ins[1]:
                            k = u[0]
                            if k == "ld":
                                W[u[1]] = A[phys[u[2]]].copy()
                            elif k == "fill":
                                W[u[1]] = np.full((m.P, LANES), u[2], F)
                            elif k == "cpd":
                                W[u[2]] = W[u[1]].copy()
                            elif k == "un":
                                W[u[2]] = UN[u[1]](W[u[2]]).astype(F)
                            elif u[1] == "where":
                                W[u[2]] = np.where(W[u[3]] != 0, W[u[4]], W[u[5]]).astype(F)
                            elif u[1] in ("mad", "msb"):
                                pr = (W[u[3]] * W[u[4]]).astype(F)
                                W[u[2]] = (W[u[5]] + pr if u[1] == "mad" else W[u[5]] - pr).astype(F)
                            else:
                                W[u[2]] = BIN[u[1]](W[u[3]], W[u[4]]).astype(F)
                        with np.errstate(all="ignore"):
                            for s, r in ins[2]:
                                w = W[s]
                                nu = int(np.sum(~np.isclose(w, w[:, :1], rtol=1e-5, equal_nan=True)))
                                if nu and pc not in self._bsc_seen:
                                    self._bsc_seen.add(pc)
                                    print("BSCAL_CHECK pc %d r %d: %d lanes differ from lane 0; loads %s" % (
                                        pc, r, nu, [(u[2], int(np.sum(~np.isclose(A[phys[u[2]]], A[phys[u[2]]][:, :1],
                                        rtol=1e-5, equal_nan=True)))) for u in ins[1] if u[0] == "ld"]), flush=True)
                    for s, r in ins[2]:
                        y = np.full((m.P, LANES), np.nan, F)   # only lane 0 is written: a full-tile reader sees NaN
                        y[:, 0] = D[s]
                        A[phys[r]] = y
                        self.owner[phys[r]] = r
                        if self.record is not None and len(self.history) >= self.record_from and \
                                r not in self.record and isinstance(r, int) and r < len(m.prog.space):
                            self.record[r] = self._glob(r, y)
                elif op == "fetch":
                    c = m.fetched_from[ins[1]]
                    t = phys[ins[1]]
                    if A[t].shape == self.ctile[c].shape:
                        A[t] = self.ctile[c].copy()
                    else:                       # a multigrid level's constant in a T-tile slot
                        A[t][...] = np.nan
                        A[t][:, :self.ctile[c].shape[1]] = self.ctile[c]
                    self.owner[phys[ins[1]]] = ins[1]
                elif op == "halo":
                    self._chk(ins[2], pc)
                    hs = hphys[("H", ins[1])]
                    self.howner[hs] = ins[1]
                    rc, rp, sc, sl = self.hplan[m.halo_plan[hs]]
                    H[hs][rc, rp] = A[phys[ins[2]]][sc, sl]
                elif op in ("red", "red2"):
                    scope, G = (ins[3] if op == "red" and len(ins) > 3 else 0), m.maps.group
                    for fr, pr in zip(ins[1::2], ins[2::2]) if op == "red2" else [(ins[1], ins[2])]:
                        self._chk(pr, pc)
                        f = A[phys[fr]]
                        x = A[phys[pr]][:, 0]
                        if scope == 1:     # each chip's own partials
                            f[:, :G] = np.repeat(x.reshape(-1, G), G, axis=0)
                        elif scope == 2:   # one value per chip: its core 0's
                            f[:, :m.P // G] = x[::G][None, :]
                        else:
                            f[:, :m.P] = x[None, :]
                elif op == "out":
                    row[ins[1]] = float(A[phys[ins[2]]][0, 0])
                    if ins[1] == "outer":
                        self.history.append(row)
                        if log:
                            log(row)
                        row = {}
                        nout += 1
                        if max_outer and nout >= max_outer:
                            return self.history
                elif op == "jumpif":
                    v = A[phys[ins[1]]][:, 0]
                    assert np.all((v != 0) == (v[0] != 0)), "cores disagree on a branch"
                    self.branches.append(bool(v[0] != 0))
                    if v[0] != 0:
                        pc = self.labels[ins[2]]
                elif op == "jump":
                    pc = self.labels[ins[1]]
        return self.history

    def _tile(self, r, t):
        """View of tile t of register r (a one-tile register: its only tile)."""
        b = self.phys[r]
        a = self.A[b]
        return a[:, t * LANES:(t + 1) * LANES] if b >= self.m.nsmall else a

    def _shifted(self, r, t, d):
        """Tile t of r read at lane offset d: lanes outside the vector are NaN, or (lds_adjacent, as the device) the
        neighbouring slots' lanes."""
        m, F = self.m, self.F
        b = self.phys[r]
        x = self.A[b]
        if self.lds_adjacent:
            prev = self.A.get(self._ends.get(b)) if b > 0 else None
            nxt = self.A.get(b + x.shape[1] // LANES)
            parts = ([prev[:, -LANES:]] if prev is not None else []) + [x] + ([nxt[:, :LANES]] if nxt is not None else [])
            x, base = np.concatenate(parts, 1), (LANES if prev is not None else 0)
        else:
            base = 0
        o = base + t * LANES + d
        out = np.full((m.P, LANES), np.nan, F)
        a0, a1 = max(o, 0), min(o + LANES, x.shape[1])
        if a1 > a0:
            out[:, a0 - o:a1 - o] = x[:, a0:a1]
        return out

    def _ldr_vector(self, u):
        """All tiles of an ldr page: the source read at the shift (or zeros), then map u[4]'s entries patched in."""
        m, F = self.m, self.F
        _, _, src, d, mid, parts = u
        n = self._bt     # an ldr feeds a vector (a DRAM-homed source or destination): its block's tile count
        if src is not None:
            out = np.concatenate([self._shifted(src, t, d) for t in range(n)], 1)
        else:
            out = np.zeros((m.P, n * LANES), F)
        if mid is not None:
            self._patch(out, mid, parts)
        return out

    def _patch(self, out, mid, parts):
        m, A, H = self.m, self.A, self.H
        t = self.gt[mid]
        for k, (s, h) in enumerate(parts):
            sel = t["part"] == k
            loc = sel & (t["kind"] == 0)
            out[t["core"][loc], t["lane"][loc]] = A[self.phys[s]][t["core"][loc], t["idx"][loc]]
            rem = sel & (t["kind"] == 1)
            if rem.any():
                self._hchk(h, None)
                out[t["core"][rem], t["lane"][rem]] = H[m.hphys[("H", h)]][t["core"][rem], t["idx"][rem]]
        out[t["zc"], t["zl"]] = 0
        self.counts["g:values"] += (len(t["core"]) + len(t["zc"])) / m.P

    def _block(self, ins, t):
        m, F = self.m, self.F
        D = [None] * m.nslot
        for u in ins[1]:
            k = u[0]
            if t == 0:
                self.counts["u:" + (k if k != "op" and k != "un" else u[1])] += 1
            if k == "ld":
                D[u[1]] = self._tile(u[2], t).copy()
            elif k == "ldr":
                if t == 0:
                    self._page[id(u)] = self._ldr_vector(u)
                D[u[1]] = self._page[id(u)][:, t * LANES:(t + 1) * LANES].copy()
            elif k == "lds":
                D[u[1]] = self._shifted(u[2], t, u[3])
            elif k == "fill":
                D[u[1]] = np.full((m.P, LANES), u[2], F)
            elif k == "rvf":
                D[u[1]] = np.repeat(self._tile(u[2], 0)[:, :1], LANES, 1)
            elif k == "cpd":
                D[u[2]] = D[u[1]].copy()
            elif k == "op":
                if u[1] == "where":
                    D[u[2]] = np.where(D[u[3]] != 0, D[u[4]], D[u[5]]).astype(F)
                elif u[1] in ("mad", "msb"):   # one rounding, as SFPMAD
                    pr = D[u[3]].astype(np.float64) * D[u[4]]
                    D[u[2]] = (D[u[5]] + pr if u[1] == "mad" else D[u[5]] - pr).astype(F)
                else:
                    D[u[2]] = BIN[u[1]](D[u[3]], D[u[4]]).astype(F)
            elif k == "un":
                D[u[2]] = UN[u[1]](D[u[2]]).astype(F)
            elif k == "fold":
                x = D[u[1]]
                sm = x.reshape(m.P, 32, 32).sum(1, dtype=F).sum(1, dtype=F)   # columns, then rows
                y = np.full((m.P, LANES), np.nan, F)
                y[:, 0] = sm
                D[u[1]] = y
            else:
                raise NotImplementedError(k)
        for s_, r in ins[2]:
            self._tile(r, t)[...] = bf16_rne(D[s_]) if r in m.narrow_home else D[s_]

    def _ovl_loop(self, blks, pc):
        """wplace 2: a tile loop runs pass t over tile t of every vector, so an overlaid register and a one-tile value
        on one of its device tiles must not both appear in one loop with either written (memu keeps the overlay in
        its own array and would not see the clobber)."""
        m, use, dfn = self.m, set(), set()
        for x in blks:
            for u in x[1]:
                if u[0] in ("ld", "lds", "rvf"):
                    use.add(u[2])
                elif u[0] == "ldr":
                    use.update(([u[2]] if u[2] is not None else []) + [y for y, _ in u[5]])
            dfn.update(r for _, r in x[2])
        tiles = lambda r: self._ovt[r] if r in self._ovt else (m.phys[r],) if m.phys.get(r, DBASE) < m.nsmall else ()
        ov = [r for r in use | dfn if r in self._ovt]
        for r in ov:
            for y in use | dfn:
                if y != r and y != "__anchor__" and not isinstance(y, tuple) and (r in dfn or y in dfn) and \
                        set(tiles(r)) & set(tiles(y)):
                    raise AssertionError(f"MIR {pc - 1}: overlaid {r} and {y} share device tiles in one tile loop")

    def _hchk(self, h, pc):
        if self.check:
            hs = self.m.hphys[("H", h)]
            assert self.howner.get(hs) == h, f"MIR {pc and pc - 1}: halo slot {hs} holds halo {self.howner.get(hs)}, read as {h}"

    def _chk(self, r, pc):
        if self.check and r != "__anchor__":
            t = self.phys[r]
            if t >= KBASE:      # a constant read in place (several fetch registers alias it)
                return
            assert self.owner.get(t) == r, f"MIR {pc - 1}: tile {t} holds {self.owner.get(t)}, read as {r}"

    def _glob(self, r, tile):
        sp = self.m.space[r]
        if sp is SCALAR:
            return tile[0, 0].copy()
        return self.m.lay.collect(sp, tile)

    def state(self):
        m = self.m
        out = {}
        for name, c in m.prog.state:
            arrs = []
            for r in c.ravel():
                r2 = m.ren(r)
                sp = m.space[r2]
                if r2 in self.phys:
                    arrs.append(m.lay.collect(sp, self.A[self.phys[r2]]))
                else:   # never touched by the program: its initial value
                    arrs.append(np.asarray(m.prog_pre[r2], self.F))
            out[name] = np.stack(arrs, -1).reshape(arrs[0].shape + c.shape) if c.ndim else arrs[0]
        return out
