"""Lower a traced program to the device machine.

P cores; core i owns a set of cells (and the patch faces of those cells). Every vector is one 1024-lane fp32 tile per
core: lane l of a cell vector on core i is the l-th cell core i owns; lanes past the core's count are padding and
may hold anything. A scalar is a tile with the value in every lane. Three agents per core walk the same control flow:

  C  compute (the three TRISCs): fused elementwise blocks over Dest slots, reduction partials, scalar math;
  B  BRISC: halo pushes to the neighbour cores, the cross-core sum of reduction partials, outputs to the host;
  N  NCRISC: constant-map gathers (from own lanes or a halo buffer) and halo credits.

The lowered program (MIR) is a list of
    ('blk', uops, stores)       C: uops over Dest slots, then pack `stores` [(slot, reg)]
          uops: ('ld', s, reg) ('fill', s, value) ('rvf', s, reg) ('op', name, out, *ins) ('un', name, s)
                ('cpd', src, dst) ('fold', s)
    ('gather', dst, map_id, [(src, halo)])  N
    ('halo', h, x)              B pushes this core's lanes of x that other cores read into their halo slot h
    ('red', F, part)            B: F[0:P] = lane 0 of every core's `part`; F[P:] = 0 (F is a fixed zeroed tile)
    ('out', name, reg)          B (core 0)
    ('label', L) ('jump', L) ('jumpif', reg, L)       all agents
    ('sync', X, Y)              X signals, Y waits; X, Y in C B N ('C' waits on UNPACK, signals on PACK)
    ('tok',)                    C: PACK -> UNPACK order (a packed tile is read back)
Registers are IR register numbers plus temporaries; after allocation they are arena tile numbers (`phys`).
"""
import bisect
import os
from collections import Counter, defaultdict

import numpy as np

from .analysis import PURE, live_outs, liveness
from .trace import SCALAR, Space

SCRATCH = Space("scratch", 1024)   # one-tile temporaries of a fused run

NSLOT = 4      # fp32 Dest tiles per acquire (half sync)
LANES = 1024
DBASE = 1 << 20   # phys of DRAM-homed vectors: DBASE + the slot's tile offset (Machine._place)


def bf16_rne(x):
    """fp32 -> the nearest bf16 (ties to even), as fp32; denormals flush to 0 as the unpacker does."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.uint64)
    w = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)
    w[np.abs(w) < np.finfo(np.float32).tiny] = 0
    return w


def bf16_bits(x):
    """bf16_rne(x) as its upper 16 bits (uint16): half the bytes, widen_bf16 gives the fp32 values back exactly."""
    return (bf16_rne(x).view(np.uint32) >> 16).astype(np.uint16)


def widen_bf16(h):
    return (np.asarray(h, np.uint16).astype(np.uint32) << 16).view(np.float32)


NH = 32           # halo slots per core (hcnt[NH], the sync region and the halo directory on device)
KBASE = 1 << 24   # phys of DRAM-homed fetched constants, read in place: KBASE + the constant's tile offset (coff)
BINARY = ("add", "sub", "mul", "div", "max", "min")
CMP = {"gt": "gtz", "ge": "gez", "lt": "ltz", "le": "lez", "eq": "eqz", "ne": "nez"}
UNARY = ("neg", "abs", "sqrt", "tanh", "log", "exp")


# =================================================================================================================
# partition
# =================================================================================================================
def rcb(xyz, P, chips=None):
    """Recursive coordinate bisection of points into P parts of near-equal size. chips = K > 1: chip first, for a
    line of K chips: K equal strips along the longest axis (strip k on chip k, neighbours one hop apart), then
    bisection of each strip into its P / K cores."""
    if chips and chips > 1:
        d = int(np.argmax(xyz.max(0) - xyz.min(0)))
        order = np.argsort(xyz[:, d], kind="stable")
        parts = np.zeros(len(xyz), np.int64)
        cut = [int(round(len(xyz) * k / chips)) for k in range(chips + 1)]
        for k in range(chips):
            sel = order[cut[k]:cut[k + 1]]
            parts[sel] = rcb(xyz[sel], P // chips) + k * (P // chips)
        return parts
    parts = np.zeros(len(xyz), np.int64)

    def rec(idx, p0, k):
        if k == 1:
            parts[idx] = p0
            return
        k1 = k // 2
        pts = xyz[idx]
        d = int(np.argmax(pts.max(0) - pts.min(0)))
        order = idx[np.argsort(pts[:, d], kind="stable")]
        m = int(round(len(idx) * k1 / k))
        rec(order[:m], p0, k1)
        rec(order[m:], p0 + k1, k - k1)
    rec(np.arange(len(xyz)), 0, P)
    return parts


def _quarter(m, r, Q):
    """Position of entry m of a run of r in index order once each 4Q-long stretch is split into 4 interleaved quarters
    (4 * (j % q) + j // q, j = m - stretch start), so index neighbours sit 4 lanes apart. A permutation of the run: the
    last stretch interleaves its first 4 * (len // 4) entries and keeps the rest in place."""
    base = m - m % (4 * Q)
    q = np.minimum(4 * Q, r - base) // 4
    j = m - base
    return np.where(j < 4 * q, base + 4 * (j % np.maximum(q, 1)) + j // np.maximum(q, 1), m)


def quarter_lanes(owner, P, Q):
    """Lanes of a partition in index order, each run of 4Q lanes split into 4 interleaved quarters (lane 4 * (m % Q) + m
    // Q), so index neighbours (x +- 1 in a block-structured mesh) sit 4 lanes apart: a lane shift, not a residue entry,
    except at the 3 quarter seams per run. A core's last, shorter run uses its own quarter, so at most 3 lanes idle."""
    lane = np.zeros(len(owner), np.int64)
    for i in range(P):
        idx = np.nonzero(owner == i)[0]
        m = np.arange(len(idx))
        base = m - m % (4 * Q)
        r = np.minimum(4 * Q, len(idx) - base)
        q = -(-r // 4)
        lane[idx] = base + 4 * ((m - base) % q) + (m - base) // q
    return lane


class Layout:
    """Owner core and lane of every element of every space."""

    def __init__(self, prog, cell_owner, P, cell_lane=None, amg_vec=False, ptiles=False):
        self.P = P
        self.owner, self.lane, self.count = {}, {}, {}
        spaces = [prog.cells] + list(prog.pspace.values())
        for sp in spaces:
            own = cell_owner if sp is prog.cells else cell_owner[sp.cells]
            # lane = rank of the element among its core's elements, in element order
            order = np.argsort(own, kind="stable")
            cnt = np.bincount(own, minlength=P).astype(np.int64)
            lane = np.zeros(sp.n, np.int64)
            lane[order] = np.arange(sp.n) - (np.cumsum(cnt) - cnt)[own[order]]
            if sp is prog.cells and cell_lane is not None:
                lane = np.asarray(cell_lane, np.int64)
                cnt = np.array([lane[own == i].max(initial=-1) + 1 for i in range(P)])
            if getattr(sp, "stack", None) is not None:   # block k at lanes k w + (lane in base), sa._stack
                base, S, w = sp.stack
                own = self.owner[base.name][np.arange(sp.n) % base.n]
                lane = (np.arange(sp.n) // base.n) * w + self.lane[base.name][np.arange(sp.n) % base.n]
                cnt = np.full(P, S * w, np.int64)
            self.owner[sp.name], self.lane[sp.name], self.count[sp.name] = own, lane, cnt
        # vector spaces: the cells, and any patch space with more than a tile's elements on one core (motorBike's
        # surface); each takes T tiles per core, T from the widest
        # amg_vec: every multigrid space is a vector register too (one tile or more), so --l1 can home it in DRAM
        self.wide = {sp.name for sp in spaces if sp is prog.cells or self.count[sp.name].max() > LANES
                     or (amg_vec and sp.name.startswith("amg"))}
        self.T = -(-max(int(self.count[n].max()) for n in self.wide) // LANES)
        # multigrid spaces (amg<l>, amg<l>f) take their own tile count, and with ptiles the wide patch spaces too (a
        # body surface is a few tiles a core, not the cells' T); otherwise the widest one's, so a program lowers as before
        ownt = lambda n: n.startswith("amg") or (ptiles and n != prog.cells.name)
        self.Ts = {n: max(1, -(-int(self.count[n].max()) // LANES)) if ownt(n) else self.T for n in self.wide}

    def scatter(self, space, values, T=1):
        """Global vector -> (P, T * LANES): T tiles per core."""
        out = np.zeros((self.P, T * LANES), values.dtype)
        ln = self.lane[space.name]
        if ln.size and ln.max() >= T * LANES:      # a --sliced register: its first T tiles only
            k = ln < T * LANES
            out[self.owner[space.name][k], ln[k]] = values[k]
            return out
        out[self.owner[space.name], ln] = values
        return out

    def collect(self, space, tiles):
        return tiles[self.owner[space.name], self.lane[space.name]]


# =================================================================================================================
# IR clean-up: constant folding, dead code, copy coalescing
# =================================================================================================================
def _ews_reads(run):
    return {a for x in run for a in (x[3:] if x[0] == "ew" else x[2:3])}


def _uses_defs(ins):
    op = ins[0]
    if op == "ew":
        return list(ins[3:]), [ins[1]]
    if op == "gather":
        return [s for s, _, _ in ins[2]], [ins[1]]
    if op == "reduce":
        return [ins[3]], [ins[1]]
    if op in ("copy", "bsum"):
        return [ins[2]], [ins[1]]
    if op in ("const", "state", "imm"):
        return [], [ins[1]]
    if op == "jumpif":
        return [ins[1]], []
    if op == "out":
        return [ins[2]], []
    if op == "keep":
        return list(ins[1]), []
    return [], []


EWF = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "max": np.maximum,
       "min": np.minimum, "pow": np.power, "neg": np.negative, "abs": np.abs, "sqrt": np.sqrt, "tanh": np.tanh,
       "log": np.log, "exp": np.exp, "gt": np.greater, "ge": np.greater_equal, "lt": np.less,
       "le": np.less_equal, "eq": np.equal, "ne": np.not_equal, "where": lambda c, a, b: np.where(c != 0, a, b)}


def fold_constants(code, space, dtype):
    """ew over immediates only -> immediate (registers keep their numbers: the ew becomes an imm)."""
    F = np.dtype(dtype).type
    imm = {}
    out = []
    defs = Counter(d for ins in code for d in _uses_defs(ins)[1])
    for ins in code:
        if ins[0] == "imm":
            imm[ins[1]] = F(ins[2])
        elif ins[0] == "ew" and defs[ins[1]] == 1 and all(a in imm for a in ins[3:]):
            with np.errstate(all="ignore"):
                v = F(EWF[ins[2]](*[imm[a] for a in ins[3:]]))
            imm[ins[1]] = v
            ins = ("imm", ins[1], float(v))
        out.append(ins)
    return out


def _aid(a):
    """An index array's identity by its memory (base pointer, shape, strides, dtype): the trace's index arrays are
    constant views, so equal identity is equal content; equal content elsewhere is only a missed match."""
    a = np.asarray(a)
    return (a.__array_interface__["data"][0], a.shape, a.strides, a.dtype.str)


def cse(code):
    """Value numbering: an ew / gather / bsum / reduce that repeats an earlier one (same op on the same registers) is
    dropped and its uses read the earlier result. Same operations on the same values, so bit-exact; what goes is the
    trace's recomputation (grad U built three times in the momentum phase, interp of one field per caller, ...).
    Only single-definition registers take part as results. A copy into a register forgets every entry that reads it;
    a jump target forgets everything (a value from before a loop head or a merge may not hold on every path)."""
    defs = Counter(d for ins in code for d in _uses_defs(ins)[1])
    targets = {ins[-1] for ins in code if ins[0] in ("jump", "jumpif")}
    memo, readers, ren, out = {}, defaultdict(list), {}, []
    rn = lambda r: ren.get(r, r)
    for ins in code:
        op = ins[0]
        if op == "label" and ins[1] in targets:
            memo.clear()
            readers.clear()
        if op == "ew":
            ins = ins[:3] + tuple(rn(a) for a in ins[3:])
            args = ins[3:]
            key = ("ew", ins[2]) + (tuple(sorted(args)) if ins[2] in ("add", "mul") else args)
        elif op == "gather":
            ins = ("gather", ins[1], [(rn(s), c, p) for s, c, p in ins[2]])
            args = tuple(s for s, _, _ in ins[2])
            key = ("gather",) + tuple((s, _aid(c), _aid(p)) for s, c, p in ins[2])
        elif op == "bsum":
            ins, args = ("bsum", ins[1], rn(ins[2])), (rn(ins[2]),)
            key = ("bsum",) + args
        elif op == "reduce":
            ins, args = ("reduce", ins[1], ins[2], rn(ins[3])), (rn(ins[3]),)
            key = ("reduce", ins[2]) + args
        else:
            if op == "copy":
                ins = ("copy", ins[1], rn(ins[2]))
                for k in readers.pop(ins[1], ()):
                    memo.pop(k, None)
                for k in [k for k, v in memo.items() if v == ins[1]]:
                    del memo[k]
            elif op == "jumpif":
                ins = ("jumpif", rn(ins[1]), ins[2])
            elif op == "out":
                ins = ("out", ins[1], rn(ins[2]))
            elif op == "keep":
                ins = ("keep", sorted({rn(r) for r in ins[1]}))
            out.append(ins)
            continue
        r = ins[1]
        if defs[r] == 1 and key in memo:
            ren[r] = memo[key]
            continue
        if defs[r] == 1:
            memo[key] = r
            for a in set(args):
                if defs[a] > 1:
                    readers[a].append(key)
        out.append(ins)
    return out, ren


def dce(code):
    keep = [True] * len(code)
    for i, lo in live_outs(code, strong=True):
        keep[i] = code[i][0] not in PURE or code[i][1] in lo
    return [ins for ins, k in zip(code, keep) if k]


def coalesce(code, space):
    """Merge the two sides of every copy whose live ranges do not overlap; drop the copies that become self-copies.
    Immediates are never merged (they have no tile); two preloaded registers (const/state) are never merged."""
    pre = {ins[1] for ins in code if ins[0] in ("const", "state")}
    imm = {ins[1] for ins in code if ins[0] == "imm"}
    # interference among copy operands only: every query and every merge is between two of them
    cop = {r for ins in code if ins[0] == "copy" for r in ins[1:]}
    adj = defaultdict(set)
    for i, lo in live_outs(code):
        ins = code[i]
        u, d = _uses_defs(ins)
        for x in cop.intersection(d):
            for y in cop & lo:
                if y != x and not (ins[0] == "copy" and y == ins[2]):
                    adj[x].add(y)
                    adj[y].add(x)
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x
    members = defaultdict(lambda: None)
    for ins in code:
        if ins[0] != "copy":
            continue
        a, b = find(ins[1]), find(ins[2])
        if a == b or a in imm or b in imm or (a in pre and b in pre) or space[a] is not space[b]:
            continue
        if b in adj[a]:
            continue
        if b in pre:          # keep the preloaded register as the representative
            a, b = b, a
        parent[b] = a
        for y in adj.pop(b, ()):
            adj[y].discard(b)
            if y != a:
                adj[y].add(a)
                adj[a].add(y)
    ren = lambda r: find(r)

    def rn(ins):
        op = ins[0]
        if op == "ew":
            return ("ew", ren(ins[1]), ins[2]) + tuple(ren(a) for a in ins[3:])
        if op == "gather":
            return ("gather", ren(ins[1]), [(ren(s), d, p) for s, d, p in ins[2]])
        if op == "reduce":
            return ("reduce", ren(ins[1]), ins[2], ren(ins[3]))
        if op in ("copy", "bsum"):
            return (op, ren(ins[1]), ren(ins[2]))
        if op == "jumpif":
            return ("jumpif", ren(ins[1]), ins[2])
        if op == "out":
            return ("out", ins[1], ren(ins[2]))
        if op in ("const", "state"):
            return (op, ren(ins[1]), ins[2])
        if op == "keep":
            return ("keep", sorted({ren(r) for r in ins[1]}))
        return ins
    out = [rn(ins) for ins in code]
    out = [ins for ins in out if not (ins[0] == "copy" and ins[1] == ins[2])]
    return out, ren


# =================================================================================================================
# gather maps and the halo plan
# =================================================================================================================
HRUN = 8   # hrun: shortest halo run moved by one NoC copy
ROWS = ("core", "lane", "kind", "part", "idx")   # a map table's written rows; zc/zl its unwritten lanes


class Maps:
    """Distinct gather maps, lowered per core, and the halo plan they imply.

    Per map, flat arrays over every (core, dst lane) the gather writes: kind (0 local, 1 halo, 3 zero), part, src
    lane or halo index. The halo of core i lists the remote cells any map reads on i, ordered by (owner, cell)."""

    def __init__(self, prog, lay):
        self.prog, self.lay = prog, lay
        self.key2id, self.maps, self.lanes = {}, [], []
        self.group = None   # cores per chip on a mesh of chips (Machine(chip_cores=))
        self.cores = None   # (lo, hi): finish builds the tables of these cores only

    def map_of(self, dst_space, parts, lanes=None):
        """lanes: a register narrower than its space (--sliced) takes only its first `lanes` lanes on each core."""
        if lanes is not None:
            ln = self.lay.lane[dst_space.name]
            parts = [(s, np.asarray(d)[ln[d] < lanes], np.asarray(p)[ln[np.asarray(d)] < lanes]) for s, d, p in parts]
        key = (dst_space.name, lanes) + tuple((self.prog.space[s].name, d.tobytes(), np.asarray(p).tobytes())
                                              for s, d, p in parts)
        if key not in self.key2id:
            self.key2id[key] = len(self.maps)
            self.maps.append((dst_space, [(self.prog.space[s], np.asarray(d), np.asarray(p)) for s, d, p in parts]))
            self.lanes.append(lanes)
        return self.key2id[key]

    def finish(self, hrun=False):
        lay, P = self.lay, self.lay.P
        cells = self.prog.cells.name
        remote = [set() for _ in range(P)]
        remote_x = {}      # other spaces read across cores (multigrid levels): one more halo plan each
        self.flat = []
        for dsp, parts in self.maps:
            n = dsp.n
            which = np.full(n, -1)
            src = np.zeros(n, np.int64)
            for k, (ssp, d, p) in enumerate(parts):
                assert ssp is not SCALAR, "scalar gather source"
                which[d] = k
                src[d] = p
            own_d = lay.owner[dsp.name]
            rows = []
            for k, (ssp, _, _) in enumerate(parts):
                e = np.nonzero(which == k)[0]
                so = lay.owner[ssp.name][src[e]]
                loc = so == own_d[e]
                if not loc.all():
                    if ssp.name != cells and ssp.name not in remote_x:
                        remote_x[ssp.name] = [set() for _ in range(P)]
                    rs = remote if ssp.name == cells else remote_x[ssp.name]
                    for i, c in zip(own_d[e[~loc]], src[e[~loc]]):
                        rs[i].add(int(c))
                rows.append((k, e.astype(np.int32), loc, src[e].astype(np.int32)))
            del which, src      # each row keeps its own sources: the unwritten lanes are the elements no row names
            self.flat.append((dsp, rows))
        # halo layout: plan 0 is the cell space (the only plan of a one-space program, encoded as before), plan q > 0
        # one per other space read across cores. A plan has its own receive lists, senders and halo slots.
        assert not (hrun and self.group), "hrun is single-chip only (its gap lanes break the 16-B sender-chip groups)"
        self.plans = [self._plan(cells, remote, hrun)] + [self._plan(nm, remote_x[nm], False) for nm in sorted(remote_x)]
        self.plan_of = {pl["space"]: q for q, pl in enumerate(self.plans)}
        pl = self.plans[0]
        self.halo_cells, self.pos, self.halo_pos = pl["cells"], pl["posa"], pl["pos"]
        self.hlanes, self.hstart = pl["hlanes"], pl["hstart"]
        # receive words a slot of plan q needs: its widest core's halo (a coarse level's halo is far narrower than the
        # cells'; one hmax for every slot was most of the halo buffer on the multigrid builds)
        self.hmax_q = [pl["hmax"] for pl in self.plans]
        self.hmax = max(self.hmax_q)
        # per map: arrays over all written (core, lane): core, lane, kind, part, idx
        # cores=(lo, hi) builds rows for those cores only (a chip of a mesh lowered on its own: the same rows, in the
        # same order, as building all and selecting)
        self.tab = []
        lo, hi = self.cores or (0, P)
        for m in range(len(self.flat)):
            (dsp, rows), self.flat[m] = self.flat[m], None     # freed as the tables grow
            own_d, lane_d = lay.owner[dsp.name], lay.lane[dsp.name]
            unw = np.ones(dsp.n, bool)
            C, L, K, Pt, I = [], [], [], [], []
            for k, e, loc, se in rows:
                unw[e] = False
                if self.cores:
                    sel = (own_d[e] >= lo) & (own_d[e] < hi)
                    e, loc, se = e[sel], loc[sel], se[sel]
                ssp = self.maps[len(self.tab)][1][k][0]
                C.append(own_d[e]); L.append(lane_d[e]); Pt.append(np.full(len(e), k))
                K.append(np.where(loc, 0, 1))
                idx = lay.lane[ssp.name][se].copy()
                hp = self.plans[self.plan_of[ssp.name]]["pos"] if not loc.all() else None
                for j in np.nonzero(~loc)[0]:
                    idx[j] = hp[own_d[e[j]]][int(se[j])]
                I.append(idx)
            z = np.nonzero(unw)[0]
            if self.lanes[m] is not None:
                z = z[lane_d[z] < self.lanes[m]]
            if self.cores:
                z = z[(own_d[z] >= lo) & (own_d[z] < hi)]
            del unw
            # narrow rows (15 B, was 40 B in int64): readers compare, index and subtract; Encoded widens before packing.
            # Unwritten lanes (kind 3, most rows of the multigrid maps) are only (zc, zl): core and lane, 4 B a row.
            cat = lambda xs, t: np.concatenate(xs).astype(t) if xs else np.zeros(0, t)
            assert P < 1 << 15
            zt = np.uint16 if len(z) == 0 or lane_d[z].max() < 1 << 16 else np.uint32   # > 65,536 lanes: encode.WIDE
            self.tab.append(dict(core=cat(C, np.int32), lane=cat(L, np.int32), kind=cat(K, np.int8),
                                 part=cat(Pt, np.int16), idx=cat(I, np.int32), zc=own_d[z].astype(np.int16),
                                 zl=lane_d[z].astype(zt), nparts=len(self.maps[len(self.tab)][1])))
        self.flat = None
        pl = self.plans[0]
        self.send, self.recv_off, self.senders, self.receivers = pl["send"], pl["recv_off"], pl["senders"], pl["receivers"]

    def _plan(self, name, remote, hrun):
        """Halo plan of one space: per receiver core i its remote elements grouped by sender, each group in the
        sender's lane order (a face row the sender holds as consecutive lanes lands in consecutive halo slots, so the
        send and the receiver's patch can move it as one run), and the send lists sender j -> receiver i: (j's lanes,
        i's halo position). With chips (group = cores per chip) every sender chip's run starts on a 16-B boundary,
        where a fabric write of the run lands."""
        lay, P = self.lay, self.lay.P
        own, lc = lay.owner[name], lay.lane[name]
        hc = [np.array(sorted(r, key=lambda c: (own[c], lc[c])), np.int64) for r in remote]
        posa = []
        for h in hc:
            q = np.arange(len(h))
            if self.group and len(h):
                ch = own[h] // self.group
                start = np.r_[True, ch[1:] != ch[:-1]]
                gap = np.zeros(len(h), np.int64)
                k = 0
                for t in range(len(h)):
                    if start[t] and t:
                        k = -(-(t + gap[t - 1]) // 4) * 4 - t
                    gap[t] = k
                q = q + gap
            posa.append(q)
        pos = [dict(zip(h.tolist(), q.tolist())) for h, q in zip(hc, posa)]
        # hrun: a run of >= HRUN consecutive sender lanes starts at a halo position equal to its first lane mod 4 (the
        # sender also sends the lanes just before it into the gap), so a receiver whose destination lanes agree mod 4
        # moves the run with one NoC copy (16-byte agreement); hmax rounds up to 4 words
        hlanes, hstart = [None] * P, [None] * P
        if hrun:
            for i, h in enumerate(hc):
                lanes, ps, st = [], {}, {}
                o, l = own[h].tolist(), lc[h].tolist()
                k = 0
                while k < len(h):
                    j = o[k]
                    if k == 0 or o[k - 1] != j:
                        st[j] = len(lanes)
                    e = k + 1
                    while e < len(h) and o[e] == j and l[e] == l[e - 1] + 1:
                        e += 1
                    if e - k >= HRUN:
                        g = (l[k] - len(lanes)) % 4
                        lanes += [max(l[k] - g + q, 0) for q in range(g)]
                    for c, x in zip(h[k:e].tolist(), l[k:e]):
                        ps[c] = len(lanes)
                        lanes.append(x)
                    k = e
                pos[i], hlanes[i], hstart[i] = ps, lanes, st
        hmax = max(1, max((len(x) if x is not None else (int(q[-1]) + 1 if len(q) else 0)
                           for x, q in zip(hlanes, posa)), default=0))
        if hrun or self.group:   # 16-B granules: hrun's aligned runs, the fabric writes of sender-chip groups
            hmax = -(-hmax // 4) * 4
        send = defaultdict(list)   # (j, i) -> lanes
        recv_off = {}
        for i, h in enumerate(hc):
            for j in np.unique(own[h]) if len(h) else []:
                sel = np.nonzero(own[h] == j)[0]
                if hlanes[i] is None:
                    send[(int(j), i)] = lc[h[sel]]
                    recv_off[(int(j), i)] = int(posa[i][sel[0]])
                    assert np.all(np.diff(posa[i][sel]) == 1)
                else:   # hrun: the group spans its first cell's position to its last one's, gap lanes included
                    assert np.all(np.diff(sel) == 1)
                    a, b = hstart[i][int(j)], pos[i][int(h[sel[-1]])]
                    send[(int(j), i)] = np.asarray(hlanes[i][a:b + 1], np.int64)
                    recv_off[(int(j), i)] = a
        senders, receivers = defaultdict(list), defaultdict(list)
        for (j, i) in send:
            senders[i].append(j)
            receivers[j].append(i)
        return dict(space=name, cells=hc, pos=pos, posa=posa, hlanes=hlanes, hstart=hstart, hmax=hmax, send=send,
                    recv_off=recv_off, senders=senders, receivers=receivers)

    @staticmethod
    def drop(t, cover=None):
        """Table t without the rows `cover` selects (a mask over the written rows), or without its unwritten lanes."""
        if cover is None:
            return dict(t, zc=np.zeros(0, t["zc"].dtype), zl=np.zeros(0, t["zl"].dtype))   # not [:0]: a view keeps its base
        return {key: (v[~cover] if key in ROWS else v) for key, v in t.items()}

    def needs_halo(self, mid):
        t = self.tab[mid]
        return set(t["part"][t["kind"] == 1].tolist())


# =================================================================================================================
# fusion of elementwise runs into Dest blocks
# =================================================================================================================
class _Fuser:
    """Turns a run of ew/copy IR instructions into blocks. Values are registers (an IR register's value does not
    outlive its redefinition, so a register names one value at a time)."""

    def __init__(self, run, live_after, imm, nslot=NSLOT):
        self.run, self.live_after, self.imm, self.nslot = run, live_after, imm, nslot
        self.uses = defaultdict(list)       # reg -> positions that read it
        self.defpos = defaultdict(list)
        for k, ins in enumerate(run):
            for a in (ins[3:] if ins[0] == "ew" else () if ins[0] == "bcast" else ins[2:3]):
                self.uses[a].append(k)
            self.defpos[ins[1]].append(k)
        self.blocks = []
        self._new()

    def _new(self):
        self.slot = [None] * self.nslot      # reg or ('imm', reg)
        self.dirty = set()
        self.uops, self.stores = [], []

    def used_after(self, r, k):
        """r's current value is read after position k (in the run, before r is redefined, or after the run)."""
        nd = self.defpos.get(r, [])
        j = bisect.bisect_right(nd, k)
        next_def = nd[j] if j < len(nd) else len(self.run)
        u = self.uses.get(r, [])
        i = bisect.bisect_right(u, k)
        if i < len(u) and u[i] <= next_def:
            return True
        return next_def == len(self.run) and r in self.live_after

    def _end(self, k):
        for s, r in enumerate(self.slot):
            if r is not None and r in self.dirty and self.used_after(r, k - 1):
                self.stores.append((s, r))
        if self.uops:
            if not any(u[0] in ("ld", "rvf") for u in self.uops):
                self.uops.insert(0, ("ld", self.nslot - 1, "__anchor__"))   # orders PACK after UNPACK's waits
            self.blocks.append(("blk", self.uops, self.stores))
        self._new()

    def _where(self, r):
        for s, x in enumerate(self.slot):
            if x == r:
                return s
        return None

    def _free_slots(self, k, keep):
        """Slots that may be overwritten now: empty, or clean values (or values dead after k) not in `keep`."""
        out = []
        for s, x in enumerate(self.slot):
            if x is None:
                out.append(s)
            elif x in keep:
                continue
            elif x not in self.dirty or not self.used_after(x, k):
                out.append(s)
        return out

    def _need(self, k, ins):
        if ins[0] == "bcast":   # read from L1 by rvf, not loaded
            return [], [], 1
        op = ins[0] == "ew" and ins[2]
        args = list(dict.fromkeys(ins[3:] if ins[0] == "ew" else ins[2:3]))
        missing = [a for a in args if self._where(a) is None]
        # can the output take an operand's slot?
        reuse = any(a == ins[1] or not self.used_after(a, k) for a in args)
        extra = 0
        if op == "pow":
            extra = 1
        return missing, args, (0 if reuse else 1) + extra

    def add(self, k, ins):
        missing, args, nout = self._need(k, ins)
        free = self._free_slots(k, set(args))
        if len(missing) + nout > len(free):
            self._end(k)
            missing, args, nout = self._need(k, ins)
            free = self._free_slots(k, set(args))
            assert len(missing) + nout <= len(free), ins
        for a in missing:
            s = free.pop(0)
            if a in self.imm:
                self.uops.append(("fill", s, self.imm[a]))
            else:
                self.uops.append(("ld", s, a))
            self.slot[s] = a
            self.dirty.discard(a)
        sl = {a: self._where(a) for a in args}
        out = ins[1]
        # output slot: an operand that dies here, else a free slot
        dying = [a for a in args if a == out or not self.used_after(a, k)]

        def take_out():
            if dying:
                return sl[dying[0]]
            return free.pop(0)
        if ins[0] == "bcast":
            s = free.pop(0)
            self.uops.append(("rvf", s, ins[2]))
        elif ins[0] == "copy":
            a = ins[2]
            if dying:
                s = sl[a]
            else:
                s = free.pop(0)
                self.uops.append(("cpd", sl[a], s))
        else:
            op = ins[2]
            if op in BINARY:
                s = take_out()
                self.uops.append(("op", op, s, sl[ins[3]], sl[ins[4]]))
            elif op in CMP:
                s = take_out()
                self.uops.append(("op", "sub", s, sl[ins[3]], sl[ins[4]]))
                self.uops.append(("un", CMP[op], s))
            elif op == "where":
                s = take_out()
                self.uops.append(("op", "where", s, sl[ins[3]], sl[ins[4]], sl[ins[5]]))
            elif op in UNARY:
                a = ins[3]
                if dying:
                    s = sl[a]
                else:
                    s = free.pop(0)
                    self.uops.append(("cpd", sl[a], s))
                self.uops.append(("un", op, s))
            elif op == "pow":
                x, y = ins[3], ins[4]
                yv = self.imm.get(y)
                s = take_out() if x in dying else free.pop(0)
                if yv == 2.0:
                    self.uops.append(("op", "mul", s, sl[x], sl[x]))
                elif yv == 4.0:
                    self.uops.append(("op", "mul", s, sl[x], sl[x]))
                    self.uops.append(("op", "mul", s, s, s))
                elif yv == 0.5:
                    if s != sl[x]:
                        self.uops.append(("cpd", sl[x], s))
                    self.uops.append(("un", "sqrt", s))
                else:   # exp(y log x)
                    if s != sl[x]:
                        self.uops.append(("cpd", sl[x], s))
                    self.uops.append(("un", "log", s))
                    self.uops.append(("op", "mul", s, s, sl[y]))
                    self.uops.append(("un", "exp", s))
            else:
                raise NotImplementedError(op)
        # the output replaces whatever was in s
        self.slot[s] = out
        self.dirty.add(out)
        # an operand whose slot was taken is gone from Dest; other copies of `out` in Dest are stale
        for t, x in enumerate(self.slot):
            if t != s and x == out:
                self.slot[t] = None

    def finish(self):
        for k, ins in enumerate(self.run):
            self.add(k, ins)
        self._end(len(self.run))
        return self.blocks


# =================================================================================================================
# the lowering
# =================================================================================================================

def _block_deps(blks):
    """Uses, defs and predecessors of a run's blocks: read/write, write/read and write/write on every register."""
    n = len(blks)
    U = [set(_mir_uses(b)) - {"__anchor__"} for b in blks]
    D = [set(_mir_defs(b)) for b in blks]
    pred = [set() for _ in range(n)]
    lastw, reads = {}, defaultdict(list)
    for j in range(n):
        for r in U[j]:
            if r in lastw:
                pred[j].add(lastw[r])
        for r in D[j]:
            if r in lastw:
                pred[j].add(lastw[r])
            pred[j].update(reads[r])
        for r in U[j]:
            reads[r].append(j)
        for r in D[j]:
            lastw[r], reads[r] = j, []
    return U, D, pred


def _dfs_order(blks, rev=False):
    """Post-order depth-first walk from the run's sinks: each block follows right after the values it needs, so an
    accumulator chain is emitted term by term and a term's inputs die before the next term's are made."""
    n = len(blks)
    _, _, pred = _block_deps(blks)
    hassucc = {i for p in pred for i in p}
    seen, out = set(), []
    for root in [j for j in range(n) if j not in hassucc] + list(range(n)):
        stack = [(root, False)]
        while stack:
            j, done = stack.pop()
            if done:
                out.append(j)
                continue
            if j in seen:
                continue
            seen.add(j)
            stack.append((j, True))
            # push in reverse so preds are visited in the order the recursive walk visits them
            stack += [(i, False) for i in sorted(pred[j], reverse=not rev) if i not in seen]
    return [blks[j] for j in out]

_MOVABLE = ("ew", "gather", "reduce", "copy", "bsum")


def _lsched(code, space, log=print):
    """Reorder each straight-line run of ew/gather/reduce/copy/bsum to cut the peak number of live vector registers.
    The trace emits numpy's whole-array order (all S gathers of an ELL product, then all S products, then the sums),
    so every slot's term is live at once; a post-order walk from the run's sinks finishes one term into the
    accumulator before making the next. Same instructions on the same registers, only reordered under RAW/WAR/WAW,
    so results are bit-identical. Per run, keeps the order (original, walk, reversed walk) with the lowest peak."""
    n = len(code)
    vec = lambda r: space[r] is not SCALAR
    live_end = {}
    ends = [i for i in range(n) if code[i][0] in _MOVABLE and (i + 1 == n or code[i + 1][0] not in _MOVABLE)]
    lo = liveness(code, at=set(ends))
    out, i, nrun, p0, p1 = [], 0, 0, 0, 0
    while i < n:
        if code[i][0] not in _MOVABLE:
            out.append(code[i]); i += 1
            continue
        j = i
        while j < n and code[j][0] in _MOVABLE:
            j += 1
        blk = code[i:j]
        live = {r for r in lo[j - 1] if vec(r)}

        def peak(order):
            L, m = set(live), len(live)
            for ins in reversed(order):
                u, d = _uses_defs(ins)
                L -= set(d)
                L |= {r for r in u if vec(r)}
                m = max(m, len(L))
            return m
        best, pb = blk, peak(blk)
        q0 = pb
        if len(blk) > 2:
            pred, lastw, reads = [set() for _ in blk], {}, defaultdict(list)
            for k, ins in enumerate(blk):
                u, d = _uses_defs(ins)
                for r in u:
                    if r in lastw:
                        pred[k].add(lastw[r])
                for r in d:
                    if r in lastw:
                        pred[k].add(lastw[r])
                    pred[k].update(reads[r])
                for r in u:
                    reads[r].append(k)
                for r in d:
                    lastw[r], reads[r] = k, []
            hassucc = {a for p in pred for a in p}
            for rev in (False, True):
                seen, o = set(), []
                for root in [k for k in range(len(blk)) if k not in hassucc] + list(range(len(blk))):
                    stack = [(root, False)]
                    while stack:
                        k, done = stack.pop()
                        if done:
                            o.append(k)
                            continue
                        if k in seen:
                            continue
                        seen.add(k)
                        stack.append((k, True))
                        stack += [(a, False) for a in sorted(pred[k], reverse=not rev) if a not in seen]
                cand = [blk[k] for k in o]
                pc = peak(cand)
                if pc < pb:
                    best, pb = cand, pc
        nrun, p0, p1 = nrun + 1, max(p0, q0), max(p1, pb)
        out += best
        i = j
    log(f"lsched: {nrun} runs, peak live vectors {p0} -> {p1}")
    return out


def face_counts(mesh):
    """Faces of every cell, boundary faces included: the cell's ELL row length (ttfluid.solver.ell.Ell)."""
    nI = mesh.nInternalFaces
    return np.bincount(np.concatenate([mesh.owner[:nI], mesh.neighbour] + [p.faceCells for p in mesh.patches]),
                       minlength=mesh.nCells)


def sliced_lanes(owner, counts, P, Q=0):
    """Lanes of a partition with every core's cells sorted by face count, widest first (then by index): ELL slot k's
    real entries are then lanes [0, n_k) of each core, n_k = the core's cells with more than k faces. Q > 0 (layout
    rcbq<Q>): each run of equal face count is quarter-interleaved within itself (_quarter), so index neighbours are a
    4-lane shift and the [0, n_k) sets do not change."""
    lane = np.zeros(len(owner), np.int64)
    order = np.lexsort((np.arange(len(owner)), -counts, owner))
    cnt = np.bincount(owner, minlength=P)
    pos = np.arange(len(owner)) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    if Q:
        key = owner[order] * (counts.max() + 1) + counts[order]
        start = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        run = np.diff(np.r_[start, len(order)])
        m = np.arange(len(order)) - np.repeat(start, run)
        pos = pos - m + _quarter(m, np.repeat(run, run), Q)
    lane[order] = pos
    return lane


def gmax(x):
    """A count over the whole mesh (lower/spmd.py: the maximum over every chip's process)."""
    return x


def slot_extents(code, space, cells, slotfold, L, T):
    """Tiles of each cells register the program reads (backward, flow-insensitive, to a fixed point).

    Every read needs the whole vector except the slot operand of step k of a fold over ELL slots (slotfold), which
    needs only the first L[k] tiles: the cells with a k-th face, under sliced_lanes. Its other lanes are ELL padding,
    neutral to the fold. A register no one reads needs 0."""
    D = {}

    def need(r, t):
        if isinstance(r, int) and space[r] is cells and D.get(r, 0) < t:
            D[r] = t
            return True
        return False
    changed = True
    while changed:
        changed = False
        for ins in reversed(code):
            op = ins[0]
            if op in ("ew", "copy"):
                d = ins[1]
                t = D.get(d, 0) if space[d] is cells else T
                k = slotfold.get(d) if op == "ew" else None
                for i, a in enumerate(ins[3:] if op == "ew" else ins[2:3]):
                    changed |= need(a, min(t, L[k]) if k is not None and i == 1 else t)
            else:
                for a in _uses_defs(ins)[0]:
                    changed |= need(a, T)
                if op == "state":
                    changed |= need(ins[1], T)
    return D


class Machine:
    def __init__(self, prog, mesh, P, nslot=NSLOT, log=print, layout="rcb", tiles=None, l1_vec=None, fuse=False, scratch_cap=32,
                 span_homes=False, hoist=False, cap_min=False, sched=False, amg_vec=False, ptiles=False, bcast_run=False, joint_red=False, fold1=False,
                 gsplit=False, fusepart=False, bgather=False, wplace=False, mad=False, bscal=False, hsplit=False, bloc=0, hrun=False, chip_cores=None, red2=None, chip_line=False, redm=False, pseg=False, dedup=False, narrow=False, lsched=False, sliced=False, cse_ir=False, hlate=False, hre=0):
        # two-stage reductions: fold the chip's partials, then the K chip sums; no joint reductions then. Needed when
        # K * P partials do not fit one fold tile, and the default from 8 chips (r13 2x4: 520.3 vs 546.9 ms/it)
        self.red2 = bool(chip_cores) and ((P > LANES or P >= 8 * chip_cores) if red2 is None else red2)
        # redm: move independent sums next to each other first (_adjacent_reduces), then joint_red groups them
        joint_red = (joint_red or redm) and not self.red2
        self.redm = joint_red
        self.prog, self.P, self.nslot, self.bcast_run, self.joint_red = prog, P, nslot, bcast_run, joint_red
        self.fold1, self.gsplit, self.fusepart, self._log = fold1, gsplit, fusepart, log
        self.bgather, self.bloc, self.hrun = bgather, bloc, hrun
        self.bscal = bscal
        self.hsplit = hsplit   # NCRISC copies and sends the second half of each halo (vm_ncrisc N_HALO)
        assert not (hsplit and chip_cores), "hsplit is single-chip only (the gateway path sends from BRISC)"
        self.wplace = wplace
        self.hlate = hlate     # a halo goes out right before its first gather, not after its source's last write
        self.hre = hre         # >0: a halo last read more than hre MIR back is sent again, not held in its slot
        self.pseg = pseg
        self.narrow_on, self.narrow, self.narrow_slot, self.pre_h, self.narrow_home = narrow, set(), {}, {}, set()   # narrow: bf16 DRAM pages for eligible constants (_narrow)
        self.coff, self.cpages, self.hpages, self.dslots = {}, 0, 0, []
        self.F = prog.dtype.type
        # constants are preloaded: hoist their definitions (the trace emits them where first used, maybe in a loop)
        pre = [i for i in prog.code if i[0] in ("const", "imm", "state")]
        code = pre + [i for i in prog.code if i[0] not in ("const", "imm", "state")]
        code = fold_constants(code, prog.space, prog.dtype)
        if cse_ir:
            n0 = len(code)
            code = cse(code)[0]
            log(f"cse: {n0} -> {len(code)} instructions")
        code = dce(code)
        self.sliced, self.ext = sliced, {}
        if sliced:
            code, D = self._sliced(code, prog, mesh, P, tiles, chip_cores if chip_line else None, log)
        n0 = sum(1 for i in code if i[0] == "copy")
        code, self.ren = coalesce(code, prog.space)
        for r, t in (D.items() if sliced else ()):     # a merged register spans its widest member
            self.ext[self.ren(r)] = max(self.ext.get(self.ren(r), 0), t)
        code = dce(code)
        log(f"IR: {len(code)} instructions, copies {n0} -> {sum(1 for i in code if i[0] == 'copy')}")
        if lsched:
            code = _lsched(code, prog.space, log)
        if self.redm:
            code = _adjacent_reduces(code)
        self.code = code
        self.space = list(prog.space)
        self.imm = {ins[1]: self.F(ins[2]) for ins in code if ins[0] == "imm"}
        self.pre = {ins[1]: ins[2] for ins in code if ins[0] in ("const", "state")}
        # a constant copy coalescing merged with a variable (Var(zeros) of a Krylov loop) is written later: it is not
        # a constant any more, and fetching it from DRAM would read the initial value after the loop wrote the tile
        written = {d for ins in code if ins[0] not in ("const", "state") for d in _uses_defs(ins)[1]}
        self.const_regs = {ins[1] for ins in prog.code if ins[0] == "const"} - written
        self.prog_pre = {ins[1]: ins[2] for ins in prog.code if ins[0] in ("const", "state")}
        if layout in ("lattice", "grid"):
            from .lattice import lattice_coords, lattice_lanes, grid_lanes
            X = lattice_coords(mesh)
            assert X is not None, "mesh is not a lattice"
            if layout == "grid":
                nI = mesh.nInternalFaces
                own, lane, self.box, self.grid = grid_lanes(X, P, mesh.owner[:nI], mesh.neighbour)
                log("grid layout: grid %s, axes %s, box %s" % (self.grid + (self.box,)))
            else:
                own, lane, self.box = lattice_lanes(X, P, chips=P // chip_cores if chip_cores and chip_line else None)
            self.lay = Layout(prog, own, P, lane, amg_vec=amg_vec, ptiles=ptiles)
        elif layout.startswith("rows"):   # rows<K>: K x strips cut along y, one row width on every core (lattice.row_lanes)
            from .lattice import lattice_coords, strip_owner, row_lanes
            X = lattice_coords(mesh)
            assert X is not None, "mesh is not a lattice"
            own = strip_owner(X, P, int(layout[4:] or 1))
            self.lay = Layout(prog, own, P, row_lanes(X, own, P), amg_vec=amg_vec, ptiles=ptiles)
        elif layout.startswith("rcbq") and not sliced:   # rcbq<Q>: rcb cores, quarter-interleaved lanes (quarter_lanes)
            own = rcb(mesh.C, P)
            self.lay = Layout(prog, own, P, quarter_lanes(own, P, int(layout[4:])), amg_vec=amg_vec, ptiles=ptiles)
        else:
            own = self._own if sliced else rcb(mesh.C, P, chips=P // chip_cores if chip_cores and chip_line else None)
            Q = int(layout[4:]) if layout.startswith("rcbq") else 0
            self.lay = Layout(prog, own, P, sliced_lanes(own, face_counts(mesh), P, Q) if sliced else None, amg_vec=amg_vec, ptiles=ptiles)
        self.T = max(self.lay.T, tiles or 1)   # tiles: force more tiles per vector than needed (testing)
        self.Ts = {n: (self.T if t == self.lay.T else t) for n, t in self.lay.Ts.items()}   # T: the slot stride
        assert not sliced or layout == "rcb" or layout.startswith("rcbq"), "--sliced sorts rcb lanes"
        if sliced:      # the prefix property every narrow register relies on (spmd: lanes come from prep --sliced)
            fc, ln = face_counts(mesh), self.lay.lane[prog.cells.name]
            for k, Lk in enumerate(self.slotL):
                assert not (fc > k).any() or ln[fc > k].max() < Lk * LANES, f"--sliced: slot {k} is not a lane prefix (prep --sliced?)"
        self.maps = Maps(prog, self.lay)
        self.maps.group = chip_cores
        self.masks = {}
        assert not self.red2 or (chip_cores <= LANES and P // chip_cores <= LANES)
        assert self.red2 or P <= LANES, "K * P partials exceed one fold tile"
        # fixed zeroed tiles, alternating per reduction (triples under joint_red, a second stage under red2)
        self.fold = [self._tmp(SCALAR) for _ in range(6 if joint_red else 4 if self.red2 else 2)]
        for r in self.fold:
            self.pre[r] = None
        self.nred = 0
        self._lower()
        if fusepart:
            self._fuse_partials()
        if mad:
            self._mad()
        self._fetches()
        self.maps.finish(hrun)
        assert not (hsplit and len(self.maps.plans) > 1), "hsplit reads plan 0 only (vm_ncrisc N_HALO)"
        self._split_gathers()
        self._halos()
        self.scratch_cap, self.span_homes, self.hoist, self.cap_min = scratch_cap, span_homes, hoist, cap_min
        self.sched = sched
        if fuse and self.T > 1:
            self._segments()
            self._scratch()
        self._allocate()
        self.dedup = dedup
        self._pre_place = (list(self.mir), dict(self.phys), self.ntiles, self.nvec, self.nsmall)
        self.place(l1_vec, log)

    def place(self, l1_vec, log=lambda *a: None):
        """(Re)do the L1/DRAM placement and the syncs for l1_vec vector slots in L1 (None: all)."""
        mir, phys, self.ntiles, self.nvec, self.nsmall = self._pre_place
        self.mir, self.phys = list(mir), dict(phys)
        self._place(l1_vec)
        if self.dedup:
            self._dedup()
        if self.gsplit:
            self._gather_split()
        if self.bgather:
            self._b_gathers()
        if self.bscal:
            self._b_scalars()
        if self.narrow_on:
            self._narrow()
        self._sync()
        log(self.summary())

    def _sliced(self, code, prog, mesh, P, tiles, chips, log):
        """--sliced: each core's cells sorted widest first, so ELL slot k is a lane prefix of L_k tiles; every cells
        register gets the tiles it is read over (slot_extents), and a fold step over a slot narrower than its result
        becomes copy + in-place update of the first L_k tiles (the copy coalesces away when the accumulator dies there)."""
        self._own = own = rcb(mesh.C, P, chips=P // chips if chips else None)
        fc = face_counts(mesh)
        T = max(-(-gmax(int(np.bincount(own, minlength=P).max())) // LANES), tiles or 1)
        self.slotL = L = [max(1, -(-gmax(int(np.bincount(own[fc > k], minlength=P).max())) // LANES))
                          for k in range(gmax(int(fc.max())))]
        D = slot_extents(code, prog.space, prog.cells, prog.slotfold, L, T)
        out, nf = [], 0
        for ins in code:
            d = ins[1] if ins[0] == "ew" else None
            if d in prog.slotfold and len(ins) == 5 and prog.space[ins[3]] is prog.cells and D.get(ins[4], T) < D.get(d, 0):
                out += [("copy", d, ins[3]), ("ew", d, ins[2], d, ins[4])]
                nf += 1
            else:
                out.append(ins)
        ews = [i for i in code if i[0] == "ew" and prog.space[i[1]] is prog.cells]
        log(f"sliced: slot tiles {L} of T {T}; {nf} fold steps in place; cells ew {len(ews)}, op-tiles "
            f"{sum(min([D.get(x, T) for x in i[3:] if prog.space[x] is prog.cells] + [D.get(i[1], 0)]) for i in ews)} vs {len(ews) * T}")
        return out, D

    def loop(self, ins):
        """Tile passes of an ew / copy: its narrowest vector (a sliced fold step runs over its slot operand only)."""
        return min(self.size(x) for x in [ins[1]] + list(ins[2:3] if ins[0] == "copy" else ins[3:]) if self.is_vec(x))

    # ---- registers
    def is_vec(self, r):
        """A vector: T tiles per core (cells, wide patches). Everything else (scalars, patch vectors) is one tile."""
        return isinstance(r, int) and self.space[r].name in self.lay.wide

    def size(self, r):
        if not self.is_vec(r):
            return 1
        return max(1, self.ext[r]) if r in self.ext else self.Ts[self.space[r].name]

    def tiles(self, blks):
        """Tile-loop count of a block or of a fused segment's blocks: every vector it touches has the same tile count."""
        # a stacked register read by a block that writes no stacked register (bsum, L1 or DRAM loads) is read a
        # block at a time: it loops over its base's tiles. An elementwise op on stacked registers loops over all of it.
        sto = lambda x: any(isinstance(r, int) and self.space[r].stack for _, r in x[2])
        stk = lambda x, u: u[0] in ("lds", "ldr") and u[2] is not None and isinstance(u[2], int) and \
            self.space[u[2]].stack and not sto(x)
        regs = [u[2] for x in blks for u in x[1] if u[0] in ("ld", "lds", "rvf") and not stk(x, u)]
        regs += [r for x in blks for _, r in x[2]]
        regs += [u[2] for x in blks for u in x[1] if u[0] == "ldr" and u[2] is not None and not stk(x, u)]  # patch sources
        ts = {self.size(r) for r in regs if r != "__anchor__" and self.is_vec(r)}
        ts |= {self.Ts[b.name] if b.name in self.lay.wide else 1
               for b in (self.space[u[2]].stack[0] for x in blks for u in x[1] if stk(x, u))}
        if self.sliced and ts:      # a block runs over its narrowest vector; wider ones are read at their first tiles
            return min(ts)
        assert len(ts) <= 1, f"a block mixes vectors of {sorted(ts)} tiles"
        return ts.pop() if ts else self.T

    def _tmp(self, space):
        self.space.append(space)
        return len(self.space) - 1

    def _mask(self, sp):
        if sp.name not in self.masks:
            r = self._tmp(sp)
            self.masks[sp.name] = r
            self.pre[r] = np.ones(sp.n, self.F)      # preloaded; padding lanes are 0 by the scatter
        return self.masks[sp.name]

    # ---- IR -> MIR
    def _lower(self):
        code = self.code
        # read at a run's last instruction only: one followed by anything but ew/copy (or the last one)
        n = len(code)
        live_out = liveness(code, at={i for i in range(n) if i + 1 == n or code[i + 1][0] not in ("ew", "copy")})
        mir = []
        run, run_end = [], None
        self.halo_of = {}

        def fuse(ins_list, live):
            if ins_list:
                mir.extend(_Fuser(list(ins_list), live, self.imm, self.nslot).finish())

        def flush():
            if not run:
                return
            if self.T == 1:
                fuse(run, live_out[run_end])
            else:
                # a vector block runs once per tile, so a one-tile update inside it would run T times: those
                # instructions go first in their own blocks; a scalar write that a pending vector instruction
                # still reads (or writes) ends the group
                groups, S, V = [], [], []
                for ins in run:
                    if not self.is_vec(ins[1]):      # scalars and patch vectors: one tile, run once
                        if ins[1] in _ews_reads(V) | {x[1] for x in V}:
                            groups += [S, V]
                            S, V = [], []
                        S.append(ins)
                    else:
                        V.append(ins)
                groups += [S, V]
                # vectors of different tile counts (multigrid levels) never share a block: one V group per count,
                # each in program order (an ew reads only its own space and one-tile values, so they are independent)
                split = []
                for g in groups:
                    ts = list(dict.fromkeys(self.loop(x) for x in g)) if g and self.is_vec(g[0][1]) else []
                    if len(ts) > 1:
                        for x in g:
                            assert all(self.size(y) == self.size(x[1]) or self.sliced and self.size(y) >= self.loop(x)
                                       for y in _ews_reads([x]) if self.is_vec(y)), x
                        if self.sliced:   # a fold step reads the ops around it: consecutive runs, in order
                            for x in g:
                                if split and split[-1] and split[-1][0] in g and self.loop(split[-1][0]) == self.loop(x):
                                    split[-1].append(x)
                                else:
                                    split.append([x])
                        else:
                            split += [[x for x in g if self.loop(x) == t] for t in ts]
                    else:
                        split.append(g)
                groups = split
                for i, g in enumerate(groups):
                    fuse(g, set(live_out[run_end]) | _ews_reads([x for h in groups[i + 1:] for x in h]))
            run.clear()
        skip = set()
        for idx, ins in enumerate(code):
            if idx in skip:
                continue
            op = ins[0]
            if op in ("ew", "copy"):
                run.append(ins)
                run_end = idx
                continue
            flush()
            if op in ("imm", "const", "state"):
                continue
            if op == "gather":
                dsp = self.space[ins[1]]
                mid = self.maps.map_of(dsp, ins[2], self.size(ins[1]) * LANES if self.is_vec(ins[1]) and self.size(ins[1]) < self.T else None)
                mir.append(("gather", ins[1], mid, [(s, None) for s, _, _ in ins[2]]))
            elif op == "bsum":
                # S tile blocks of the stacked register summed in one block: block k read by a tile-shifted load
                _, S, w = self.space[ins[2]].stack
                mir.append(("blk", [("lds", 0, ins[2], 0)] + [u for k in range(1, S) for u in
                            (("lds", 1, ins[2], k * w), ("op", "add", 0, 0, 1))], [(0, ins[1])]))
            elif op == "reduce":
                # joint_red: up to three adjacent independent sums leave in one cross-core reduction ('red2')
                group = [ins]
                while self.joint_red and len(group) < 3 and idx + len(group) < len(code):
                    nxt = code[idx + len(group)]
                    if nxt[0] != "reduce" or any(nxt[3] == g[1] for g in group):
                        break
                    group.append(nxt)
                pair = len(group) > 1
                skip.update(range(idx + 1, idx + len(group)))
                k = self.nred % 2
                self.nred += 1
                folds = self.fold[3 * k:3 * k + 3] if self.joint_red else [self.fold[k]]
                parts, tmps = [], []
                for g in group:
                    assert g[2] == "sum", g
                    x = g[3]
                    sp = self.space[x]
                    assert sp is not SCALAR
                    part = self._tmp(SCALAR)
                    partial = [("ld", 0, x), ("ld", 1, self._mask(sp)), ("fill", 2, 0.0),
                               ("op", "where", 0, 1, 0, 2), ("fold", 0)]
                    if self.is_vec(x) and self.T > 1 and self.fold1 and not self.fusepart:   # fusepart folds once itself
                        # fold1: the masked tiles add lane by lane into one tile, folded once (not once per tile)
                        acc = self._tmp(SCALAR)
                        mir.append(("blk", [("ld", self.nslot - 1, "__anchor__"), ("fill", 0, 0.0)], [(0, acc)]))
                        mir.append(("blk", partial[:-1] + [("ld", 1, acc), ("op", "add", 0, 0, 1)], [(0, acc)]))
                        mir.append(("blk", [("ld", 0, acc), ("fold", 0)], [(0, part)]))
                        parts.append(part)
                        tmps.append(self._tmp(SCALAR))
                        continue
                    if self.is_vec(x) and self.T > 1:     # per-tile partials accumulate into part
                        mir.append(("blk", [("ld", self.nslot - 1, "__anchor__"), ("fill", 0, 0.0)], [(0, part)]))
                        partial += [("ld", 1, part), ("op", "add", 0, 0, 1)]
                    mir.append(("blk", partial, [(0, part)]))
                    parts.append(part)
                    tmps.append(self._tmp(SCALAR))
                if pair:
                    mir.append(("red2",) + tuple(x for f, p in zip(folds, parts) for x in (f, p)))
                elif self.red2:   # scope 1: this chip's partials; scope 2: the K chip sums
                    mir.append(("red", folds[0], parts[0], 1))
                    mir.append(("blk", [("ld", 0, folds[0]), ("fold", 0)], [(0, tmps[0])]))
                    folds = [self.fold[2 + k]]
                    mir.append(("red", folds[0], tmps[0], 2))
                    tmps = [self._tmp(SCALAR)]
                else:
                    mir.append(("red", folds[0], parts[0]))
                mir.append(("blk", [u for i in range(len(group)) for u in (("ld", i, folds[i]), ("fold", i))],
                            [(i, tmps[i]) for i in range(len(group))]))
                for g, tmp in zip(group, tmps):
                    if self.bcast_run:   # the broadcast opens the next fused run: its consumers share the block
                        run.append(("bcast", g[1], tmp))
                        run_end = idx + len(group) - 1
                    else:
                        mir.append(("blk", [("rvf", 0, tmp)], [(0, g[1])]))
            elif op == "out":
                r = ins[2]
                if r in self.imm:
                    t = self._tmp(SCALAR)
                    mir.append(("blk", [("ld", self.nslot - 1, "__anchor__"), ("fill", 0, self.imm[r])], [(0, t)]))
                    r = t
                mir.append(("out", ins[1], r))
            elif op == "jumpif":
                if ins[1] in self.imm:
                    if self.imm[ins[1]] != 0:
                        mir.append(("jump", ins[2]))
                    continue
                mir.append(("jumpif", ins[1], ins[2]))
            elif op in ("label", "jump"):
                mir.append(ins)
            elif op == "keep":
                mir.append(("keep", [r for r in ins[1] if r not in self.imm]))
            else:
                raise NotImplementedError(op)
        flush()
        self.mir = mir

    # ---- fusepart: a reduction's per-tile partial (mask, fold, accumulate) joins the fused block that computes the
    # summand, which then never goes to L1: per partial one load and one store fewer per tile, the mask loaded once
    # per block, and one block fewer.
    @staticmethod
    def _slots(u):
        k = u[0]
        if k in ("ld", "fill", "fold", "rvf", "lds", "ldr"):
            return [u[1]]
        if k == "op":
            return list(u[2:])
        if k == "cpd":
            return [u[1], u[2]]
        if k == "un":
            return [u[2]]
        return None

    # ---- mad: a multiply whose product is only read by the next add or subtract of the block becomes one SFPU pass
    # (C_MAD: o + x y, C_MSB: o - x y; addcmul) instead of two, the product never taking a Dest slot of its own.
    @staticmethod
    def _rw(u):
        k = u[0]
        if k in ("ld", "fill", "lds", "ldr", "rvf"):
            return (), (u[1],)
        if k in ("fold",):
            return (u[1],), (u[1],)
        if k == "op":
            return tuple(u[3:]), (u[2],)
        if k == "cpd":
            return (u[1],), (u[2],)
        if k == "un":
            return (u[2],), (u[2],)
        raise NotImplementedError(k)

    def _mad(self):
        n = 0
        for b in self.mir:
            if b[0] != "blk":
                continue
            U, stored = b[1], {s for s, _ in b[2]}
            i = 0
            while i < len(U):
                u = U[i]
                if u[0] != "op" or u[1] != "mul":
                    i += 1
                    continue
                s, x, y = u[2], u[3], u[4]
                j, fuse, sink = i + 1, None, True
                while j < len(U):
                    rd, wr = self._rw(U[j])
                    if s in rd or s in wr:
                        v = U[j]
                        if v[0] == "op" and v[1] in ("add", "sub") and v[3] != v[4] and s in v[3:5] and \
                                (v[1] == "add" or v[4] == s):
                            fuse = j
                        break
                    if x in wr or y in wr:
                        sink = False
                    j += 1
                if fuse is not None:
                    v = U[fuse]
                    o, out = (v[4] if v[3] == s else v[3]), v[2]
                    live = False
                    if out != s:
                        for w in U[fuse + 1:]:
                            rd, wr = self._rw(w)
                            if s in rd or s in wr:
                                live = s in rd
                                break
                        else:
                            live = s in stored
                    # hoist (the operands were reused before the add): the add's other operand keeps its value and
                    # its output is untouched between the multiply and the add, so the add can run at the multiply
                    if not live and not sink:
                        sink = None
                        if all(o not in self._rw(w)[1] and out not in self._rw(w)[0] + self._rw(w)[1]
                               for w in U[i + 1:fuse]):
                            sink = False
                    if not live and sink is not None:
                        mad = ("op", "mad" if v[1] == "add" else "msb", out, x, y, o)
                        if sink:
                            U[fuse] = mad
                            del U[i]
                        else:
                            U[i] = mad
                            del U[fuse]
                        n += 1
                        continue
                i += 1
        self._log("mad: %d multiply-adds fused" % n)

    def _fuse_partials(self):
        mir = self.mir

        def partial(x):
            if x[0] != "blk" or len(x[2]) != 1 or x[2][0][0] != 0:
                return None
            u = x[1]
            if len(u) < 5 or u[0][:2] != ("ld", 0) or u[1][:2] != ("ld", 1) or u[2] != ("fill", 2, 0.0) or \
                    u[3] != ("op", "where", 0, 1, 0, 2) or u[4] != ("fold", 0):
                return None
            if len(u) == 5:
                return u[0][2], u[1][2], None, x[2][0][1]
            if len(u) == 7 and u[5][:2] == ("ld", 1) and u[6] == ("op", "add", 0, 0, 1) and u[5][2] == x[2][0][1]:
                return u[0][2], u[1][2], x[2][0][1], x[2][0][1]
            return None
        uses = Counter(r for x in mir for r in _mir_uses(x) if not isinstance(r, tuple))
        merged = {}          # producer index -> (free slots, [(x slot, part, accumulate)])
        drop, moved = set(), []
        for j, x in enumerate(mir):
            p = partial(x)
            if p is None:
                continue
            xr, mask, acc, part = p
            if uses[xr] != 1:
                continue
            i = j - 1
            while i >= 0 and mir[i][0] == "blk" and not any(r == xr for _, r in mir[i][2]):
                i -= 1
            if i < 0 or mir[i][0] != "blk" or i in drop:
                continue
            prod = mir[i]
            sx = next(s for s, r in prod[2] if r == xr)
            if i not in merged:
                # the appended uops run after every uop of the block and the stores read their slots at its end:
                # every slot the block does not store is free by then
                if any(self._slots(u) is None for u in prod[1]):
                    continue
                stored = {s for s, _ in prod[2]}
                free = [s for s in range(self.nslot) if s not in stored]
                if len(free) < 3:
                    continue
                merged[i] = (free[:3], mask, [])
            elif merged[i][1] != mask:
                continue
            # the blocks in between must not touch the accumulator (its zero init moves above the producer)
            if acc is not None:
                init = j - 1
                if not (mir[init][0] == "blk" and mir[init][2] == [(0, acc)] and
                        mir[init][1] == [("ld", self.nslot - 1, "__anchor__"), ("fill", 0, 0.0)]):
                    continue
                moved.append((i, init))
                drop.add(init)
            merged[i][2].append((sx, part, acc, xr))
            drop.add(j)
        if not merged:
            return
        out = []
        for i, x in enumerate(mir):
            if i in drop:
                continue
            for k, init in moved:
                if k == i:
                    out.append(mir[init])
            if i in merged:
                (m, z, tq), mask, parts = merged[i]
                uops = list(x[1]) + [("ld", m, mask), ("fill", z, 0.0)]
                once = self.fold1 and all(acc is not None for _, _, acc, _ in parts)
                for sx, part, acc, _ in parts:
                    # fold1: the masked tiles add lane by lane into part, folded once after the block
                    uops += [("op", "where", sx, m, sx, z)] + ([] if once else [("fold", sx)])
                    if acc is not None:
                        uops += [("ld", tq, acc), ("op", "add", sx, sx, tq)]
                xr = {r for _, _, _, r in parts}
                x = ("blk", uops, [(s, r) for s, r in x[2] if r not in xr] + [(sx, part) for sx, part, _, _ in parts])
                if once:
                    out.append(x)
                    x = ("blk", [u for k, (_, part, _, _) in enumerate(parts) for u in (("ld", k, part), ("fold", k))],
                         [(k, part) for k, (_, part, _, _) in enumerate(parts)])
            out.append(x)
        self._log("fusepart: %d partials into %d blocks" % (sum(len(v[2]) for v in merged.values()), len(merged)))
        self.mir = out

    # ---- constants live in DRAM; B fetches one into a tile before its first use in a straight-line segment.
    # Constants read inside an inner (Krylov) loop stay resident.
    def _fetches(self, ahead=6):
        mir = self.mir
        labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
        depth = np.zeros(len(mir), np.int64)
        for j, ins in enumerate(mir):
            if ins[0] in ("jump", "jumpif"):
                t = labels[ins[-1]]
                if t < j:
                    depth[t:j + 1] += 1
        consts = {r for r, v in self.pre.items() if v is not None and r not in self.masks.values()
                  and r in self.const_regs}
        resident = set()
        for j, ins in enumerate(mir):
            if depth[j] >= 2:
                resident |= set(_mir_uses(ins)) & consts
        stream = consts - resident
        out, fetched, seg_start = [], {}, 0
        self.fetched_from = {}
        for ins in mir:
            op = ins[0]
            if op in ("label", "jump", "jumpif"):
                fetched.clear()
                out.append(ins)
                seg_start = len(out)
                continue
            need = [r for r in dict.fromkeys(_mir_uses(ins)) if r in stream and r not in fetched]
            for r in need:
                t = self._tmp(self.space[r])
                self.fetched_from[t] = r
                fetched[r] = t
                out.insert(max(seg_start, len(out) - ahead), ("fetch", t, r))
            if fetched:
                ins = _rename_uses(ins, fetched)
            out.append(ins)
        self.mir = out
        self.streamed = sorted(stream)
        self.resident_consts = sorted(resident)
        # DRAM constant pages: per core, every constant's own tile count back to back (page core * cpages + coff + t)
        self.coff, self.cpages = {}, 0
        for r in self.streamed:
            self.coff[r], self.cpages = self.cpages, self.cpages + self.size(r)

    # ---- a gather whose entries are mostly one lane offset (lattice layout) or zeros becomes a compute block (offset
    # load or zero fill) plus an NCRISC patch of the rest; a scalar NCRISC copy costs ~32 cycles a value, the block
    # ~730 cycles a tile, so the split pays from ~24 covered values per core.
    def _split_gathers(self, min_per_core=24):
        cells = self.prog.cells.name
        self.base = {}
        for mid, t in enumerate(self.maps.tab):
            dsp, parts = self.maps.maps[mid]
            best, cover, ncov = None, None, len(t["zc"])
            for k, (ssp, _, _) in enumerate(parts):
                if ssp.name != cells or dsp.name != cells:
                    continue
                sel = (t["part"] == k) & (t["kind"] == 0)
                d = t["idx"] - t["lane"]
                vals, cnt = np.unique(d[sel & (d % 4 == 0)], return_counts=True)
                if len(cnt) and cnt.max() > ncov:
                    best, cover = (k, int(vals[np.argmax(cnt)])), sel & (d == vals[np.argmax(cnt)])
                    ncov = int(cover.sum())
            if ncov < min_per_core * self.P:
                continue
            self.base[mid] = best             # None: zero fill
            self.maps.tab[mid] = Maps.drop(t, cover)
        out = []
        for ins in self.mir:
            if ins[0] == "gather" and ins[2] in self.base:
                b = self.base[ins[2]]
                # a fill-only block needs the anchor load: it orders PACK after UNPACK's waits
                u = [("ld", self.nslot - 1, "__anchor__"), ("fill", 0, 0.0)] if b is None else \
                    [("lds", 0, ins[3][b[0]][0], b[1])]
                out.append(("blk", u, [(0, ins[1])]))
                if len(self.maps.tab[ins[2]]["core"]) + len(self.maps.tab[ins[2]]["zc"]):
                    out.append(ins)
                continue
            out.append(ins)
        self.mir = out

    # ---- halos: one push per (straight-line segment, source value), hoisted to just after the source's last write
    def _segments(self):
        """Segment tile-loop fusion, step 1: wrap each maximal run of vector blocks that can run tile by tile as
        ('seg', blocks). A run ends at a non-block instruction it cannot move past, at a shifted read of a value the run
        defined, and at a definition of a value the run read shifted. A one-tile block independent of the run moves
        before it (the reduction's partial-sum reset), as does a fetch whose tile the run has not touched.
        With self.hoist, a gather / halo / red also stops ending runs: one independent of the run (reads nothing the
        run defines, writes nothing it touched) moves before it; one that only reads what the run defines (a halo push
        or reduction of a run value, a gather of one) moves after it, and the run continues until a block touches what
        such a deferred instruction reads or writes. Order among moved instructions is kept, so values are unchanged."""
        reads = lambda b: {u[2] for u in b[1] if u[0] in ("ld", "lds", "rvf") and u[2] != "__anchor__"}
        out, pre, cur, post = [], [], [], []
        defs, rd, shifted = set(), set(), set()
        pdefs, puses = set(), set()      # what the deferred instructions write / read
        self.cuts = Counter()       # why runs ended (census only)

        def close():
            out.extend(pre)
            out.extend([("seg", list(cur))] if len(cur) > 1 else cur)
            out.extend(post)
            for x in (pre, cur, post, defs, rd, shifted, pdefs, puses):
                x.clear()
            nonlocal npatch
            npatch = 0
        mir, skip, npatch = self.mir, False, 0
        labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
        depth = np.zeros(len(mir) + 1, int)     # pseg 2, 3: only in the outer iteration's body, not in a solve's loop
        for j, ins in enumerate(mir):
            if ins[0] in ("jump", "jumpif") and labels.get(ins[-1], j) < j:
                depth[labels[ins[-1]]:j + 1] += 1
        for i, ins in enumerate(mir):
            if skip:
                skip = False
                continue
            nxt = mir[i + 1] if i + 1 < len(mir) else None
            split = ins[0] == "blk" and nxt is not None and nxt[0] == "gather" and nxt[2] in self.base and \
                any(x == nxt[1] for _, x in ins[2])      # the first half of a split gather stays with its patch
            if split and self.pseg and not (self.pseg in (2, 3) and depth[i] > 1):
                # pseg: the pair joins the run as one page the reader builds (shifted source or zeros, residue patched
                # in), when nothing it reads is written by the run; the reader keeps one patch per stream
                u, dst = ins[1][-1], nxt[1]
                src = u[2] if u[0] == "lds" else None
                srcs = {x for x, _ in nxt[3]} | ({src} if src is not None else set())
                if (not npatch or self.pseg >= 3) and not srcs & (defs | pdefs) and dst not in defs | rd | shifted | pdefs | puses:
                    page = ("blk", [("ldr", 0, src, u[3] if src is not None else 0, nxt[2], nxt[3])], [(0, dst)])
                    if cur and self.tiles([page]) != self.tiles(cur):     # one tile count per segment, as below
                        self.cuts["tiles"] += 1
                        close()
                    cur.append(page)
                    defs.add(dst)
                    rd |= srcs
                    shifted |= srcs
                    npatch, skip = 1, True
                    self.cuts["pseg"] += 1
                    continue
            if ins[0] == "blk" and not split:
                r, d = reads(ins), {x for _, x in ins[2]}
                if any(self.is_vec(x) for x in r | d):
                    if cur and self.tiles([ins]) != self.tiles(cur):     # one tile count per segment
                        self.cuts["tiles"] += 1
                        close()
                    if {u[2] for u in ins[1] if u[0] == "lds"} & defs or d & shifted:
                        self.cuts["shift-dep" if cur else "none"] += 1
                        close()
                    elif r & pdefs or d & (pdefs | puses):
                        self.cuts["deferred-dep"] += 1
                        close()
                    cur.append(ins)
                    defs |= d
                    rd |= r
                    shifted |= {u[2] for u in ins[1] if u[0] == "lds"}
                    continue
                if cur and not (r & (defs | pdefs)) and not (d & (defs | rd | pdefs | puses)):
                    pre.append(ins)
                    continue
            elif ins[0] == "fetch" and cur and ins[1] not in defs | rd | pdefs | puses:
                pre.append(ins)
                continue
            elif self.hoist and cur and ins[0] in ("gather", "halo", "red", "red2"):
                u, d = set(_mir_uses(ins)), set(_mir_defs(ins))
                if not d & (defs | rd | pdefs | puses):
                    if not u & (defs | pdefs):
                        pre.append(ins)
                        self.cuts["hoisted-" + ins[0]] += 1
                    else:
                        post.append(ins)
                        pdefs |= d
                        puses |= u
                        self.cuts["deferred-" + ins[0]] += 1
                    continue
            if cur:
                k = "split-gather" if split else "blk-dep" if ins[0] == "blk" else ins[0]
                if ins[0] == "gather":
                    src = {x for x, _ in ins[3]}
                    k += "-dep" if src & defs or ins[1] in defs | rd else "-indep"
                self.cuts[k] += 1
            close()
            out.append(ins)
        close()
        self.mir = out

    def _schedule(self, blks):
        """Reorder a run's blocks (dependences kept: read/write, write/read, write/write on every register) so values
        die early: at each step the ready block that frees the most of the run's values net of the ones it creates,
        ties in program order. Every block computes what it did, so results are unchanged; fewer values live at once
        means fewer cap cuts, and each cut stores and re-reads the values crossing it."""
        n = len(blks)
        if n < 3:
            return blks
        U, D, pred = _block_deps(blks)
        succ = [[] for _ in range(n)]
        for j in range(n):
            for i in pred[j]:
                succ[i].append(j)
        npred = [len(p) for p in pred]
        left = Counter(r for j in range(n) for r in U[j])     # unscheduled in-run reads per register
        born = {r for j in range(n) for r in D[j] if self.is_vec(r)}
        ready, out = [j for j in range(n) if npred[j] == 0], []
        while ready:
            def score(j):
                frees = sum(1 for r in U[j] if r in born and left[r] == sum(1 for x in [j] if r in U[x]))
                return (sum(1 for r in D[j] if r in born) - frees, j)
            j = min(ready, key=score)
            ready.remove(j)
            out.append(blks[j])
            for r in U[j]:
                left[r] -= 1
            for k in succ[j]:
                npred[k] -= 1
                if npred[k] == 0:
                    ready.append(k)
        assert len(out) == n
        return out

    def _cap_runs(self):
        """Cut a run where more than scratch_cap scratch values would be live across a block boundary (each one is a
        tile of L1 on every core, whatever T is): a value the run defines and reads again further on is a scratch
        value until that read. The values crossing a cut are stored to their homes and the next piece reads them."""
        cap, out = self.scratch_cap, []
        for ins in self.mir:
            if ins[0] != "seg" or cap is None:
                out.append(ins)
                continue
            if not self.sched:
                pieces = self._cut(ins[1], cap)
            elif self.sched == 1 or len(ins[1]) < 3:
                pieces = self._cut(self._schedule(ins[1]), cap)
            elif self.sched == 3:
                # sched 3: sched 2's three orders, each cut where the fewest values cross (min-store cut points under
                # the same cap rule, not the first boundary over it), the order whose cuts store the fewest
                pieces = min((self._cut_dp(o, cap) for o in (self._schedule(ins[1]), _dfs_order(ins[1]),
                                                             _dfs_order(ins[1], rev=True))),
                             key=lambda q: (self._crossing(q), len(q)))
            else:
                # sched 2: of the greedy order and the two depth-first orders, the one whose cuts store the fewest
                # values (each crossing value is a vector store plus a re-read per later piece that reads it)
                pieces = min((self._cut(o, cap) for o in (self._schedule(ins[1]), _dfs_order(ins[1]),
                                                          _dfs_order(ins[1], rev=True))),
                             key=lambda q: (self._crossing(q), len(q)))
            self.cuts["cap"] += len(pieces) - 1
            out += [("seg", q) if len(q) > 1 else q[0] for q in pieces]
        self.mir = out

    def _cut(self, blks, cap):
        """Split an ordered run into pieces where more than cap scratch values would be live across a boundary."""
        last = {}
        for j, b in enumerate(blks):
            for r in _mir_uses(b):
                last[r] = j
        pieces, start, born = [], 0, {}
        live = lambda k: sum(1 for r, d in born.items() if d < k and last.get(r, -1) >= k)
        j = 0
        while j < len(blks):
            if live(j) > cap:
                # cap_min: cut where the fewest of this piece's values cross (each is stored and re-read as a
                # whole vector), not at the first boundary over the cap (where cap of them cross)
                k = j if not self.cap_min else min(range(start + 1, j + 1), key=lambda k: (live(k), -k))
                pieces.append(blks[start:k])
                start, born, j = k, {}, k
            for _, r in blks[j][2]:
                if self.is_vec(r):
                    born[r] = j
            j += 1
        pieces.append(blks[start:])
        return pieces

    def _cut_dp(self, blks, cap):
        """Split an ordered run into pieces, each with at most cap scratch values live across any of its block
        boundaries (the rule _cut applies), at the cut points that minimise the values crossing a cut: dynamic
        programme over piece starts; a value born in a piece and read at or after the piece end is stored."""
        n = len(blks)
        last = {}
        for j, b in enumerate(blks):
            for r in _mir_uses(b):
                last[r] = j
        births = [[r for _, r in b[2] if self.is_vec(r)] for b in blks]
        INF = float("inf")
        best, arg = [0] + [INF] * n, [0] * (n + 1)
        for a in range(n):
            if best[a] == INF:
                continue
            bylast, live = Counter(), 0
            for k in range(a + 1, n + 1):
                for r in births[k - 1]:
                    if last.get(r, -1) >= k:
                        bylast[last[r]] += 1
                        live += 1
                c = best[a] + live
                if c < best[k]:
                    best[k], arg[k] = c, a
                if k < n and live > cap:
                    break
                live -= bylast.pop(k, 0)
        cuts, k = [], n
        while k > 0:
            cuts.append(k)
            k = arg[k]
        bounds = [0] + cuts[::-1]
        return [blks[a:b] for a, b in zip(bounds, bounds[1:])]

    def _crossing(self, pieces):
        """Vector values a piece defines and a later piece of the same run reads (each one is stored, then re-read)."""
        n, later = 0, set()
        for q in reversed(pieces):
            defs = {r for b in q for _, r in b[2] if self.is_vec(r)}
            n += len(defs & later)
            later |= {r for b in q for r in _mir_uses(b)}
        return n

    def _scratch(self):
        """Segment tile-loop fusion, step 2: a vector a run defines and nobody reads after it lives in a one-tile
        scratch register (one tile per pass instead of T tiles in L1 or DRAM). A vector the run defines that is read
        after it AND later in the run is stored twice, to its home and to a scratch shadow the run reads (so no read
        in a run goes through a home the run wrote)."""
        self._cap_runs()
        mir = self.mir
        succ = _mir_succ(mir)
        ud = [(set(_mir_uses(i)), set(_mir_defs(i))) for i in mir]
        live_in = [set() for _ in mir]
        changed = True
        while changed:
            changed = False
            for i in range(len(mir) - 1, -1, -1):
                o = set().union(*[live_in[j] for j in succ[i]])
                inn = (o - ud[i][1]) | ud[i][0]
                if inn != live_in[i]:
                    live_in[i], changed = inn, True
        for i, ins in enumerate(mir):
            if ins[0] != "seg":
                continue
            after = set().union(*[live_in[j] for j in succ[i]])
            blks, ren = ins[1], {}
            out = []
            for j, b in enumerate(blks):
                uops = [(u[0], u[1], ren.get(u[2], u[2])) + u[3:] if u[0] in ("ld", "rvf") else u for u in b[1]]
                later = {r for x in blks[j + 1:] for r in _mir_uses(x)}
                sts = []
                for slot, r in b[2]:
                    ren.pop(r, None)
                    if not self.is_vec(r):
                        sts.append((slot, r))
                    elif r not in after:
                        ren[r] = self._tmp(SCRATCH)
                        sts.append((slot, ren[r]))
                    elif r in later:
                        ren[r] = self._tmp(SCRATCH)
                        sts += [(slot, r), (slot, ren[r])]
                    else:
                        sts.append((slot, r))
                out.append(("blk", uops, sts))
            mir[i] = ("seg", out)

    def _halos(self):
        mir = self.mir
        out = []
        seg = []           # current straight-line segment (index into out)
        pushed = {}        # x -> halo register, valid in this segment while x is not rewritten
        last_def = {}      # x -> position in `out` after which x was last written in this segment
        lastrd = {}        # x -> position in `out` of the last gather that read x's halo (hre)
        seg_start = 0
        self.halo_regs = []
        for ins in mir:
            op = ins[0]
            if op in ("label", "jump", "jumpif"):
                pushed.clear()
                last_def.clear()
                out.append(ins)
                seg_start = len(out)
                continue
            if op == "gather":
                need = self.maps.needs_halo(ins[2])
                parts = []
                for k, (s, _) in enumerate(ins[3]):
                    h = None
                    if k in need:
                        if self.hre and s in pushed and len(out) - lastrd[s] > self.hre:
                            del pushed[s]     # the source is unchanged: sending it again frees the slot in between
                        if s not in pushed:
                            h = len(self.halo_regs)
                            self.halo_regs.append(s)
                            pos = len(out) if self.hlate else last_def.get(s, seg_start)
                            if self.hlate and out and out[-1][0] == "blk" and any(x == ins[1] for _, x in out[-1][2]):
                                # the first half of a split gather: the pair must stay adjacent (_place builds one page
                                # from it; apart, the gather patches its residue onto a zero page and drops the local half)
                                assert s not in _mir_defs(out[-1])
                                pos -= 1
                            out.insert(pos, ("halo", h, s))
                            # positions after pos moved by one
                            for y in last_def:
                                if last_def[y] >= pos and y != s:
                                    last_def[y] += 1
                            pushed[s] = h
                        h = pushed[s]
                        lastrd[s] = len(out)
                    parts.append((s, h))
                ins = ("gather", ins[1], ins[2], parts)
            out.append(ins)
            for d in _mir_defs(ins):
                pushed.pop(d, None)
                last_def[d] = len(out)
        self.mir = out

    # ---- tile allocation
    def _allocate(self):
        mir = self.mir
        n = len(mir)
        succ = _mir_succ(mir)
        ud = [(_mir_uses(i), _mir_defs(i)) for i in mir]
        live_in = [set() for _ in range(n)]
        live_out = [set() for _ in range(n)]
        changed = True
        while changed:
            changed = False
            for i in range(n - 1, -1, -1):
                o = set()
                for s in succ[i]:
                    o |= live_in[s]
                u, d = ud[i]
                inn = (o - set(d)) | set(u)
                if o != live_out[i] or inn != live_in[i]:
                    live_out[i], live_in[i] = o, inn
                    changed = True
        tiles = set()
        for u, d in ud:
            tiles |= set(u) | set(d)
        tiles.discard("__anchor__")
        live0 = live_in[0] if n else set()
        adj = defaultdict(set)
        for (u, d), lo in zip(ud, live_out):
            for x in d:
                for y in lo:
                    if y != x:
                        adj[x].add(y)
                        adj[y].add(x)
        # a fused run: pass t touches tile t of every vector it reads or writes, and a one-tile scratch value dies in
        # the pass that made it, so two values share a home unless their spans in the pass overlap (a block may read
        # one and write the other). A value live into the run starts before it, one live out of it ends after it,
        # and one read shifted (lds reads the neighbouring tiles, written in other passes) spans the whole run
        for i, ins in enumerate(mir):
            if ins[0] == "seg":
                k = len(ins[1])
                span = {r: (-1, -1) for r in ud[i][0] if not isinstance(r, tuple)}
                for j, x in enumerate(ins[1]):
                    for r in _mir_uses(x) + _mir_defs(x):
                        if r != "__anchor__" and not isinstance(r, tuple):
                            span[r] = (span[r][0], j) if r in span else (j, j)
                    for u in x[1]:
                        if u[0] == "lds":
                            span[u[2]] = (-1, k)
                        elif u[0] == "ldr":      # pseg: the reader builds the page from any tile of its sources
                            for r in ([u[2]] if u[2] is not None else []) + [y for y, _ in u[5]]:
                                span[r] = (-1, k)
                span.pop("__anchor__", None)     # an lds of the anchor (its value is never used) has a fixed home
                for r in span:
                    if r in live_out[i]:
                        span[r] = (span[r][0], k)
                for a in span:
                    for b in span:
                        if a != b and ((not self.span_homes and not (self.space[a] is SCRATCH and self.space[b] is SCRATCH))
                                       or not (span[a][1] <= span[b][0] or span[b][1] <= span[a][0])):
                            adj[a].add(b)
        # a gather between two spaces never runs in place: N writes destination lanes while it still reads source
        # lanes, and two spaces' lane maps overlap (a multigrid prolongation amg1 -> cells into its own source tile
        # overwrote aggregates before reading them; MachineEmu builds the result aside and could not see it)
        for ins in mir:
            if ins[0] == "gather":
                for src, _ in ins[3]:
                    if src != ins[1] and self.space[src] is not self.space[ins[1]]:
                        adj[ins[1]].add(src)
                        adj[src].add(ins[1])
        # everything live at entry (preloaded consts, state, masks) is live together
        entry = sorted(x for x in (live0 | set(self.pre)) & tiles if isinstance(x, int))
        for a in entry:
            for b in entry:
                if a != b:
                    adj[a].add(b)
        # the fold tiles keep their zero lanes: nothing else may ever use them
        for r in self.fold:
            for y in tiles:
                if y != r and not isinstance(y, tuple):
                    adj[r].add(y)
                    adj[y].add(r)
        # the halo registers are a separate pool
        halos = {x for x in tiles if isinstance(x, tuple)}
        # pools by tile count: one-tile registers in slots 0.., then vectors in slots of their own tile count (T-tile
        # slots first, then each narrower multigrid size), so a one-tile coarse value never takes a T-tile slot
        slot = {}
        pool = lambda r: self.T > 1 and self.is_vec(r)
        cls = lambda r: self.size(r) if pool(r) else 0
        order = entry + [t for t in sorted(tiles - set(entry) - halos, key=lambda r: -len(adj[r]))]
        for r in order:
            used = {slot[y] for y in adj[r] if y in slot and cls(y) == cls(r)}
            c = 0
            while c in used:
                c += 1
            slot[r] = c
        nsmall = 1 + max((c for r, c in slot.items() if not pool(r)), default=-1)
        sizes = sorted({cls(r) for r in slot if pool(r)}, reverse=True)
        gid, self.vsize0 = {}, []
        for n in sizes:
            for c in range(1 + max(c for r, c in slot.items() if pool(r) and cls(r) == n)):
                gid[n, c] = len(self.vsize0)
                self.vsize0.append(n)
        voff = np.concatenate([[0], np.cumsum(self.vsize0)]).astype(int)
        phys = {r: (nsmall + int(voff[gid[cls(r), c]]) if pool(r) else c) for r, c in slot.items()}
        self.ntiles = nsmall + int(voff[-1])
        self.nsmall, self.nvec = nsmall, len(self.vsize0)
        self.vslots = [(nsmall + int(o), n) for o, n in zip(voff, self.vsize0)]
        # fixed tiles: the anchor and the two zeroed fold tiles per parity. The anchor's value is never used: a
        # one-tile preloaded register nothing writes (a vector, a mask included, may move to DRAM in _place; a
        # written tile would add sync waits)
        wr = {r for ins in mir for r in _mir_defs(ins)}
        ro = sorted(phys[r] for r in self.pre if r in phys and not pool(r) and r not in wr and r not in self.fold)
        assert nsmall > 0
        phys["__anchor__"] = ro[0] if ro else 0
        # halo slots, colored per plan into disjoint ranges: a slot's credit and data counters assume one sender set
        plan = {h: self.maps.plan_of[self.space[self.halo_regs[h[1]]].name] for h in halos}
        hphys, base = {}, 0
        for q in range(len(self.maps.plans)):
            hq = [h for h in halos if plan[h] == q]
            for h in sorted(hq, key=lambda r: -len(adj[r])):
                used = {hphys[y] for y in adj[h] if y in hphys and plan[y] == q}
                c = base
                while c in used:
                    c += 1
                hphys[h] = c
            base = max([hphys[h] + 1 for h in hq], default=base)
        self.halo_plan = {hphys[h]: plan[h] for h in halos}
        # diagnostic: the MIR point holding the most plan-0 (finest) halo registers live, and where they were defined
        cnt = [sum(1 for x in lo if x in halos and plan[x] == 0) for lo in live_out]
        i0 = int(np.argmax(cnt)) if cnt else 0
        hdef = {x: j for j, (u, d) in enumerate(ud) for x in d if x in halos}
        pa = getattr(self.prog, "phase_at", {})
        ph = [pa[ins[1]] for ins in mir[:i0 + 1] if ins[0] == "label" and ins[1] in pa]
        self.halo_peak = (max(cnt, default=0), i0, sorted(hdef.get(x, -1) for x in live_out[i0] if x in halos and plan[x] == 0),
                          ph[-1] if ph else None)
        # and who holds them: per peak halo, its source register and the MIR points of the gathers that read it
        self.halo_peak_readers = sorted((hdef.get(x, -1), self.halo_regs[x[1]], [j for j, (u, _) in enumerate(ud) if x in u])
                                        for x in live_out[i0] if x in halos and plan[x] == 0)
        self.nhalo = max(hphys.values()) + 1 if hphys else 0
        assert self.nhalo <= NH, f"{self.nhalo} halo slots (hcnt[{NH}] on device)"
        # halo buffer: slot s at word hoff[s], sized to its plan's widest halo. Slot ids stay disjoint per plan (credit
        # and data counters assume one sender set), but two slots whose registers never interfere share words: a
        # receiver credits slot s only at its HALO for s, after its own earlier reads, so nothing still read is
        # overwritten (the invariant the per-plan coloring above already relies on). First fit, widest first.
        if len(self.maps.plans) > 1:
            sz = [self.maps.hmax_q[self.halo_plan.get(s, 0)] for s in range(self.nhalo)]
            inter = [set() for _ in range(self.nhalo)]
            for h in halos:
                inter[hphys[h]].update(hphys[y] for y in adj[h] if y in hphys and hphys[y] != hphys[h])
            at = {}
            for s in sorted(range(self.nhalo), key=lambda s: -sz[s]):
                o = 0
                for lo, hi in sorted((at[t], at[t] + sz[t]) for t in inter[s] if t in at):
                    if o + sz[s] <= lo:
                        break
                    o = max(o, hi)
                at[s] = o
            self.hoff = [at[s] for s in range(self.nhalo)]
            self.hwords_plan = sum(sz)
            self.hwords = max([at[s] + sz[s] for s in at], default=1)
        else:
            self.hoff = [s * self.maps.hmax for s in range(self.nhalo + 1)]
            self.hwords = self.hwords_plan = max(self.hoff[-1], 1)
        self.phys, self.hphys = phys, hphys
        self.adj = adj
        self.live_entry = live0

    # ---- DRAM homes. With l1_vec set (and T > 1), only that many vector slots stay in L1; the rest live in DRAM, each
    # as many pages per core as its widest register's tiles, packed: phys DBASE + the slot's tile offset, page core *
    # hpages + offset + tile. A DRAM-homed fetched constant is read in place from its DRAM constant pages (phys KBASE +
    # the constant's tile offset coff, page core * cpages + offset + tile; no fetch). Compute reaches a DRAM vector only through rings: a
    # load becomes ("ldr", slot, reg, shift, mid, halos), whose page the reader (agent N) builds (the source tile
    # read at the shift, or zeros, then the gather residue of map `mid` patched in), and a store goes to the writer
    # (agent B). A split gather (offset/zero block + residue gather) touching a DRAM vector becomes one ldr block.
    def _place(self, l1_vec):
        self.dram, self.alias, self.ndram, self.hpages, self.dslots = set(), {}, 0, 0, []
        self.ovl, self.ovdev, self.ovwid, self.locmap = {}, {}, {}, {}
        vsize = self.vsize0
        self.vslots = [(self.nsmall + o, n) for o, n in zip(np.cumsum([0] + vsize[:-1]), vsize)] if vsize else []
        # l1_vec counts T-tile slots: a budget of l1_vec * T tiles, so narrower multigrid slots pack more per budget
        if l1_vec is None or self.T == 1 or l1_vec * self.T >= sum(vsize):
            return
        mir, phys, T = self.mir, self.phys, self.T
        # a use weighs the passes its loops typically make: the trace's hint per loop (a pressure solve runs hundreds,
        # a smoother a few), 10 for a loop without one; this picks what lives in L1 and never changes results
        labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
        trips = getattr(self.prog, "trips", {})
        passes = np.ones(len(mir))
        for j, ins in enumerate(mir):
            if ins[0] in ("jump", "jumpif") and labels[ins[-1]] < j:
                passes[labels[ins[-1]]:j + 1] *= trips.get(ins[-1], 10)
        isv = lambda r: r != "__anchor__" and not isinstance(r, tuple) and self.is_vec(r)
        rweight = Counter()
        for j, ins in enumerate(mir):
            regs = []
            if ins[0] in ("blk", "seg"):
                regs = [r for b in ([ins] if ins[0] == "blk" else ins[1])
                        for r in [u[2] for u in b[1] if u[0] in ("ld", "lds", "ldr") and u[2] is not None] +
                        [y for u in b[1] if u[0] == "ldr" for y, _ in u[5]] + [r for _, r in b[2]]]
            elif ins[0] == "gather":
                regs = [ins[1]] + [x for x, _ in ins[3]]
            elif ins[0] == "halo":
                regs = [ins[2]]
            for r in regs:
                if r in phys and isv(r):
                    rweight[r] += passes[j]
        if self.wplace:
            # per register, heaviest first: the lowest of l1_vec L1 colours no interfering register holds, else a
            # DRAM colour the same way (a slot-wise pick leaves a loop's L1 slots idle through the outer body). An L1
            # colour is T tiles wide whatever the registers in it (a narrower multigrid value fills the front of its
            # slot); DRAM colours are per tile count, so a coarse level never pays a fine slot's pages
            # a fetched constant read by one instruction stays in place in DRAM: a colour would add a fetch copy
            # before that one read (K10 gave 103 such temps colours and its p solve ran 25 % slower than K8)
            nread = Counter(r for ins in mir for r in set(_mir_uses(ins)))
            wr = {r for ins in mir if ins[0] != "fetch" for r in _mir_defs(ins)}
            inplace = {r for r in getattr(self, "fetched_from", {}) if r not in wr and nread[r] <= 1}
            vslot, col, dcol = {}, {}, {}
            order = sorted((r for r in phys if isv(r) and r not in inplace), key=lambda r: (-rweight[r], str(r)))
            ovl = self._overlay(order, T, isv) if self.wplace >= 2 else {}
            for r in order:
                if r in ovl:
                    vslot[r] = ("ovl", ovl[r])
                    continue
                used = {col[y] for y in self.adj[r] if y in col}
                c = next((c for c in range(l1_vec) if c not in used), None)
                if c is not None:
                    col[r] = vslot[r] = c
                else:
                    n = self.size(r)
                    used = {dcol[y] for y in self.adj[r] if y in dcol}
                    d = 0
                    while (n, d) in used:
                        d += 1
                    dcol[r] = vslot[r] = (n, d)
            vslot = vslot.get
            keep = list(range(1 + max(col.values(), default=-1)))
            vsize = {c: T for c in keep}
            vsize.update({c: c[0] for c in set(dcol.values())})
            dslot = {c: i for i, c in enumerate(sorted(set(dcol.values()), key=lambda c: (-c[0], c[1])))}
        else:
            slot_at = {b: c for c, (b, _) in enumerate(self.vslots)}
            vslot = lambda r: slot_at[phys[r]]
            weight = Counter()
            for r, x in rweight.items():
                weight[vslot(r)] += x
            # greedy by weight under the budget: a DRAM use costs vsize pages, so a slot kept in L1 saves weight[c]
            # pages per tile it takes (equal sizes: the l1_vec heaviest slots)
            keep, used = [], 0
            for c in sorted(range(self.nvec), key=lambda c: -weight[c]):
                if used + vsize[c] <= l1_vec * T:
                    keep.append(c)
                    used += vsize[c]
            keep.sort()
            dslot = {c: i for i, c in enumerate(c for c in range(self.nvec) if c not in keep)}
        koff = np.concatenate([[0], np.cumsum([vsize[c] for c in keep])]).astype(int)
        newc = {c: int(koff[i]) for i, c in enumerate(keep)}
        dord = sorted(dslot, key=dslot.get)
        dpos = np.concatenate([[0], np.cumsum([vsize[c] for c in dord])]).astype(int)
        doff = {c: int(dpos[i]) for i, c in enumerate(dord)}
        alias = {}
        writes = set()
        for ins in mir:
            writes |= set(_mir_defs(ins)) - ({ins[1]} if ins[0] == "fetch" else set())
        for r in list(phys):
            if r == "__anchor__" or isinstance(r, tuple) or not self.is_vec(r):
                continue
            c = vslot(r)
            if self.wplace >= 2 and r in ovl:
                offs = sorted(set(ovl.values()))
                phys[r] = ovl[r]
                self.ovl[r] = self.nsmall + int(koff[-1]) + T * offs.index(ovl[r])
            elif c in newc:
                phys[r] = self.nsmall + newc[c]
            elif r in self.fetched_from and r not in writes:
                alias[r] = KBASE + self.coff[self.fetched_from[r]]
                phys[r] = alias[r]
                self.dram.add(r)
            else:
                phys[r] = DBASE + doff[c]
                self.dram.add(r)
        self.ntiles = self.nsmall + int(koff[-1])
        self.nvec, self.ndram, self.hpages = len(newc), len(dslot), int(dpos[-1])
        self.dslots = [(DBASE + doff[c], vsize[c]) for c in dslot]
        self.vslots = [(self.nsmall + newc[c], vsize[c]) for c in keep]
        self.alias = alias
        # overlaid colours: memu keeps each in its own array at a virtual base (ovl) past the L1 slots and poisons the
        # one-tile tiles it covers on device (ovdev); syncs treat every tile of an overlapping group as one location
        for r, v in self.ovl.items():
            self.ovdev[v] = phys[r]
        if self.ovl:
            wid = Counter()
            for r in self.ovl:
                wid[phys[r]] = max(wid[phys[r]], min(self.size(r), T))
            self.ovwid = {v: wid[o] for v, o in self.ovdev.items()}
            grp, hi = [], -1
            for lo in sorted(wid):
                if lo >= hi:
                    grp.append(lo)
                hi = max(hi, lo + wid[lo])
                for t in range(lo, lo + wid[lo]):
                    self.locmap[t] = grp[-1]
            self._log("wplace 2: %d vector registers overlaid on one-tile slots (%d offsets), %d stay in %d L1 colours" % (
                len(self.ovl), len(wid), sum(1 for r in phys if r not in self.ovl and isv(r) and r not in self.dram), len(keep)))
        D = self.dram

        def blk(ins, nxt):
            """A block with its DRAM-homed loads made ldr pages: (new block, instructions consumed)."""
            dst = ins[2][0][1] if len(ins[2]) == 1 else None
            u = ins[1][-1]
            split = (dst is not None and u[0] in ("lds", "fill") and
                     (len(ins[1]) == 1 or (len(ins[1]) == 2 and ins[1][0][2] == "__anchor__")))
            src = u[2] if u[0] == "lds" else None
            if split and (dst in D or src in D):
                if nxt is not None and nxt[0] == "gather" and nxt[1] == dst and nxt[2] in self.base:
                    # the split gather's two halves become one page the reader builds
                    return ("blk", [("ldr", 0, src, u[3] if src is not None else 0, nxt[2], nxt[3])], ins[2]), 2
                return ("blk", [("ldr", 0, src, u[3] if src is not None else 0, None, ())], ins[2]), 1
            return ("blk", [("ldr", v[1], v[2], v[3] if v[0] == "lds" else 0, None, ())
                            if v[0] in ("ld", "lds") and v[2] in D else v for v in ins[1]], ins[2]), 1
        out = []
        i = 0
        while i < len(mir):
            ins = mir[i]
            nxt = mir[i + 1] if i + 1 < len(mir) else None
            if ins[0] == "fetch" and ins[1] in alias:
                i += 1
                continue
            if ins[0] == "blk":
                ins, k = blk(ins, nxt)
                if k > 1:
                    out.append(ins)
                    i += k
                    continue
            elif ins[0] == "seg":
                ins = ("seg", [blk(b, None)[0] for b in ins[1]])
            elif ins[0] == "gather" and ins[1] in D:
                # residue only (or never split): the reader patches the gather onto a zero page
                out.append(("blk", [("ldr", 0, None, 0, ins[2], ins[3])], [(0, ins[1])]))
                i += 1
                continue
            out.append(ins)
            i += 1
        self.mir = out

    def _overlay(self, regs, T, isv):
        """wplace 2: home a vector register (heaviest first) on consecutive one-tile slots that no register live with it
        uses (the one-tile pool is mostly momentum's patch values, idle while the solves run). Never beside a register
        that shares a tile loop with it, a block's or a fused run's (the loop touches tile t of a vector in pass t, so
        a one-tile value written in pass 0 lands on tile k before pass k reads it), never on the anchor or the fold tiles. {register: first slot}; the rest go to the L1 colours or DRAM."""
        phys, adj, ns = self.phys, self.adj, self.nsmall
        cocc = defaultdict(set)
        for ins in self.mir:
            if ins[0] in ("blk", "seg"):
                rr = {r for b in ([ins] if ins[0] == "blk" else ins[1])
                      for r in _mir_uses(b) + _mir_defs(b) if r != "__anchor__" and not isinstance(r, tuple)}
                for r in rr:
                    cocc[r] |= rr
        small = lambda y: y in phys and y != "__anchor__" and not isinstance(y, tuple) and not isv(y) and phys[y] < ns
        fixed = {phys["__anchor__"]} | {phys[r] for r in self.fold if r in phys}
        out = {}
        for r in regs:
            n = min(self.size(r), T)
            near = adj[r] | cocc[r]
            busy = set(fixed) | {phys[y] for y in near if small(y)}
            for y in near:
                if y in out:
                    busy.update(range(out[y], out[y] + min(self.size(y), T)))
            o = next((o for o in range(ns - n + 1) if busy.isdisjoint(range(o, o + n))), None)
            if o is not None:
                out[r] = o
        return out

    # ---- narrow: an in-place constant every streamed read of which is an unpatched reader page at a lane shift % 8 == 0
    # (so the bf16 page keeps 16-B unpack alignment) gets a bf16 shadow in DRAM (constant index narrow_slot[r]) that its
    # reader pages stream: half their bytes. Compute widens it to fp32 in Dest (C_LDRH, exact). B_FETCH copies into L1
    # still read the fp32 pages. pre_h holds the bf16 (nearest even) values memu uses for the in-place reads; the fp32
    # program (emu.py, prog_pre) stays the reference the rounding is graded against.
    def _narrow(self):
        src = {t: self.fetched_from[t] for t in self.alias}
        why = defaultdict(set)   # excluded constant -> reasons
        for ins in self.mir:
            blks = [ins] if ins[0] == "blk" else [b for b in ins[1] if b[0] == "blk"] if ins[0] == "seg" else []
            for b in blks:
                for u in b[1]:
                    if u[0] != "ldr":
                        continue
                    for x, _ in u[5]:
                        if x in src:
                            why[src[x]].add("part")
                    if u[2] in src and u[4] is not None:
                        why[src[u[2]]].add("patched")
                    if u[2] in src and u[3] % 8:
                        why[src[u[2]]].add("shift")
            if ins[0] == "gather":
                for x, _ in ins[3]:
                    if x in src:
                        why[src[x]].add("gather")
        self.narrow = set(src.values()) - set(why)
        self.narrow_slot, self.cpages = {}, sum(self.size(r) for r in self.streamed)   # bf16 shadows after the fp32 ones
        for r in sorted(self.narrow):
            self.narrow_slot[r], self.cpages = self.cpages, self.cpages + self.size(r)
        self.pre_h = {}
        for r in self.narrow:
            self.pre_h[r] = bf16_bits(self.pre[r])   # uint16 bits, not an fp32 copy of every narrowed constant
            assert not np.any((self.pre_h[r] & 0x7F80) == 0x7F80)   # finite
        cnt = Counter(k for r in set(why) & set(src.values()) for k in why[r])
        self._log(f"narrow: {len(self.narrow)} of {len(set(src.values()))} in-place constants bf16 in DRAM, excluded by "
                  + (", ".join(f"{k} {n}" for k, n in sorted(cnt.items())) or "nothing"))
        self.narrow_home = set()
        if self.narrow_on >= 2:
            self._narrow_homes()

    # ---- narrow 2: a DRAM-homed vector written only in the outer body and read inside a solve loop (the matrix
    # coefficients and preconditioner diagonals, never a field, a Krylov vector or a preloaded value), read only as
    # unpatched reader pages at a lane shift % 8 == 0 and stored only through the writer ring, is packed to bf16 (C_STWH)
    # and read back widened (C_LDRH): half its bytes both ways. memu rounds its stores the same way. narrow 3 drops the
    # read-in-a-solve condition (every outer temporary): on pitz_r4 that moved the U solve from 3 to 25 iterations,
    # because fluxes and face gradients cancel in the divergence sums and bf16 leaves their rounding as the result.
    def _narrow_homes(self):
        from .census import region_of
        reg = region_of(self.mir)
        cand = {r for r in self.dram if r not in self.alias}
        state = {self.ren(r) for _, c in self.prog.state for r in np.ravel(c)}
        why = defaultdict(set)
        for r in cand:
            if r in state:
                why[r].add("state")
            if r in self.pre:
                why[r].add("pre")
        insolve = set()
        for j, ins in enumerate(self.mir):
            if ins[0] in ("blk", "seg"):
                for b in ([ins] if ins[0] == "blk" else ins[1]):
                    if b[0] != "blk":
                        continue
                    for u in b[1]:
                        if u[0] == "ldr" and reg[j].startswith("solve"):
                            insolve.add(u[2])
                        if u[0] == "ldr":
                            for x, _ in u[5]:
                                if x in cand:
                                    why[x].add("part")
                            if u[2] in cand and (u[4] is not None or u[3] % 8):
                                why[u[2]].add("patched" if u[4] is not None else "shift")
                        elif u[0] in ("ld", "lds") and u[2] in cand:
                            why[u[2]].add("ld")
                    for _, r in b[2]:
                        if r in cand and reg[j].startswith("solve"):
                            why[r].add("solve")
            else:
                for r in set(_mir_uses(ins)) | set(_mir_defs(ins)):
                    if r in cand:
                        why[r].add(ins[0])
        if self.narrow_on == 2:
            for r in cand - insolve:
                why[r].add("notsolve")
        self.narrow_home = cand - set(why)
        if os.environ.get("CFD_NARROW_DEBUG"):
            st = Counter(reg[j] for j, ins in enumerate(self.mir))
            self._log(f"narrow 2 debug: regions {dict(st)}, state in dram {len(state & self.dram)}, cand {len(cand)}")
        cnt = Counter(k for r in why for k in why[r])
        self._log(f"narrow 2: {len(self.narrow_home)} of {len(cand)} DRAM-homed vectors bf16, excluded by "
                  + (", ".join(f"{k} {n}" for k, n in sorted(cnt.items())) or "nothing"))

    # ---- sync insertion (on physical tiles)
    def _dedup(self):
        """dedup: a fused run whose blocks read one DRAM page (same register, shift and patch) more than once streams
        it once: a copy block ahead of the first reader puts it in a one-tile scratch, every reader loads that. The
        scratch tiles go after the one-tile registers (vector homes move up), coloured by their span in the run."""
        key = lambda u: (u[2], u[3], u[4], tuple(u[5]))
        segs, need, saved = [], 0, 0
        for i, ins in enumerate(self.mir):
            if ins[0] != "seg":
                continue
            blks = ins[1]
            uses = defaultdict(list)
            for j, b in enumerate(blks):
                if b[0] == "blk":
                    for u in b[1]:
                        if u[0] == "ldr":
                            uses[key(u)].append(j)
            dup = {k: (v[0], v[-1]) for k, v in uses.items() if len(v) > 1}
            saved += sum(len(v) - 1 for v in uses.values())
            if not dup:
                continue
            col, ends = {}, []     # interval colouring over block positions: copy block at first, last reader at end
            for k, (a, z) in sorted(dup.items(), key=lambda kv: kv[1]):
                c = next((c for c, e in enumerate(ends) if e < a), None)
                if c is None:
                    ends.append(z)
                    c = len(ends) - 1
                ends[c] = z
                col[k] = c
            need = max(need, len(ends))
            segs.append((i, dup, col))
        if not segs:
            return
        base, K, T = self.nsmall, need, self.T
        for r, p in self.phys.items():
            if base <= p < DBASE:
                self.phys[r] = p + K
        self.nsmall += K
        self.ntiles += K
        for i, dup, col in segs:
            regs = {k: self._tmp(SCRATCH) for k in dup}
            for k, r in regs.items():
                self.phys[r] = base + col[k]
            out = []
            for j, b in enumerate(self.mir[i][1]):
                for k, (a, _) in dup.items():
                    if a == j:
                        out.append(("blk", [("ldr", 0) + k[:3] + (k[3],)], [(0, regs[k])]))
                if b[0] == "blk":
                    b = ("blk", [("ld", u[1], regs[key(u)]) if u[0] == "ldr" and key(u) in regs else u
                                 for u in b[1]], b[2])
                out.append(b)
            self.mir[i] = ("seg", out)
        self._log("dedup: %d runs, %d pages now streamed once, %d reads fewer per tile pass, %d scratch tiles" % (
            len(segs), sum(len(d) for _, d, _ in segs), saved, K))

    def _gather_split(self):
        """gsplit: in a stretch of gathers (each with its shifted-load block), every gather patches its local residue
        and zeros in place (mode 1) and its halo values after the stretch's last gather (mode 2), so NCRISC does all
        local work before it waits for any neighbour's halo."""
        def lds_blk(x):
            return x[0] == "blk" and all(u[0] in ("lds", "fill") or (u[0] == "ld" and u[2] == "__anchor__") for u in x[1])

        def remote(x):
            return any(h is not None for _, h in x[3]) and len(x) == 4
        out, i, mir = [], 0, self.mir
        while i < len(mir):
            j = i
            while j < len(mir) and (lds_blk(mir[j]) or mir[j][0] in ("gather", "tok")):
                j += 1
            win = mir[i:j] if j > i else [mir[i]]
            if sum(1 for x in win if x[0] == "gather" and remote(x)) >= 2:
                tail, ng = [], 0
                for x in win:
                    if x[0] == "gather" and remote(x):
                        # bloc K: BRISC patches the local part of every K-th one (mode 4), NCRISC its halo values
                        onb = self.bloc and ng % self.bloc == 0 and self.phys[x[1]] < DBASE and \
                            all(self.phys[s] < DBASE for s, _ in x[3])
                        ng += 1
                        out.append(x + (4 if onb else 1,))
                        tail.append(x + (2,))
                    else:
                        out.append(x)
                out += tail
            else:
                out += win
            i = max(j, i + 1)
        self.mir = out

    S_OPS = ("add", "sub", "mul", "div", "max", "min", "where", "mad", "msb")
    S_UN = ("neg", "abs", "gtz", "gez", "ltz", "lez", "eqz", "nez")

    def _b_scalars(self):
        """bscal: a block of one-tile (scalar) registers runs on BRISC on lane 0 in soft-float instead of as whole-tile
        SFPU passes on the reduction's latency path (a broadcast it reads becomes a lane-0 load). BRISC writes lane 0
        only, so each output some compute block, gather or halo reads as a whole tile is broadcast right after it by a
        compute block of rvf's (alpha and beta of a Krylov step; the exit flag and carried scalars need none)."""
        phys = self.phys
        small = lambda r: isinstance(r, int) and not self.is_vec(r) and phys[r] < DBASE

        def ok(x):
            if x[0] != "blk" or not x[2] or not all(small(r) for _, r in x[2]):
                return False
            for u in x[1]:
                if u[0] not in ("ld", "fill", "rvf", "cpd", "un", "op"):
                    return False
                if u[0] in ("ld", "rvf") and not (u[2] != "__anchor__" and small(u[2])):
                    return False
                if (u[0] == "op" and u[1] not in self.S_OPS) or (u[0] == "un" and u[1] not in self.S_UN):
                    return False
            return True
        cand = {i for i, x in enumerate(self.mir) if ok(x)}

        def tile_reads(x):
            if x[0] == "seg":
                return {r for b in x[1] for r in tile_reads(b)}
            if x[0] == "blk":
                return {u[2] for u in x[1] if u[0] in ("ld", "lds") and u[2] != "__anchor__"} | \
                       {r for u in x[1] if u[0] == "ldr" for r in ([u[2]] if u[2] is not None else []) + [s for s, _ in u[5]]}
            if x[0] == "gather":
                return {s for s, _ in x[3]}
            if x[0] == "halo":
                return {x[2]}
            return set()
        # lane-uniform registers (every lane holds the value): fill, rvf and per-lane math on uniform inputs. BRISC
        # computes lane 0 only, so a block whose output is read as a whole tile must produce a uniform value there.
        nonu = set()

        def blk_nonu(b):
            slot = {}
            for u in b[1]:
                k = u[0]
                if k in ("fill", "rvf"):
                    slot[u[1]] = True
                elif k == "ld":
                    slot[u[1]] = u[2] != "__anchor__" and u[2] not in nonu and small(u[2])
                elif k in ("lds", "ldr", "fold"):
                    slot[u[1]] = False
                elif k == "cpd":
                    slot[u[2]] = slot.get(u[1], False)
                elif k == "un":
                    slot[u[2]] = slot.get(u[2], False)
                elif k == "op":
                    slot[u[2]] = all(slot.get(s, False) for s in u[3:] if isinstance(s, int))
            return {r for s, r in b[2] if not slot.get(s, False)}
        while True:
            n0 = len(nonu)
            for x in self.mir:
                if x[0] == "blk":
                    nonu |= blk_nonu(x)
                elif x[0] == "seg":
                    for b in x[1]:
                        nonu |= blk_nonu(b)
                elif x[0] in ("red", "red2"):
                    nonu |= set(x[1::2])
                elif x[0] in ("gather", "fetch"):
                    nonu.add(x[1])
            if len(nonu) == n0:
                break
        nwork = {i: sum(u[0] in ("op", "un", "cpd") for u in self.mir[i][1]) for i in cand}
        conv = set(cand)
        while True:  # a block left on compute reads whole tiles too, so iterate to a fixed point
            reads = set()
            for i, x in enumerate(self.mir):
                if i not in conv:
                    reads |= tile_reads(x)
            # a broadcast back costs about what the SFPU pass it saves does
            keep = {i for i in conv if nwork[i] > sum(r in reads for _, r in self.mir[i][2])
                    and not any(r in reads and r in nonu for _, r in self.mir[i][2])}
            if keep == conv:
                break
            conv = keep
        out, nbc = [], 0
        for i, x in enumerate(self.mir):
            if i not in conv:
                out.append(x)
                continue
            out.append(("bscal", [("ld",) + u[1:3] if u[0] == "rvf" else u for u in x[1]], x[2]))
            bc = [(k, r) for k, r in enumerate(r_ for _, r_ in x[2] if r_ in reads)]
            if bc:
                out.append(("blk", [("rvf", k, r) for k, r in bc], bc))
                nbc += len(bc)
        self._log("bscal: %d scalar blocks on BRISC, %d broadcasts back" % (len(conv), nbc))
        self.mir = out

    def _b_gathers(self):
        """bgather: a gather with no halo entries whose sources and destination are L1 tiles runs on BRISC (mode 3),
        which otherwise waits; NCRISC keeps the halo-reading ones."""
        out, nb = [], 0
        for x in self.mir:
            if x[0] == "gather" and len(x) == 4 and self.phys[x[1]] < DBASE and \
                    all(self.phys[s] < DBASE and h is None for s, h in x[3]) and \
                    not (self.maps.tab[x[2]]["kind"] == 1).any():
                x = x + (3,)
                nb += 1
            out.append(x)
        self._log("bgather: %d gathers on BRISC" % nb)
        self.mir = out

    def _sync(self):
        mir = self.mir
        phys = self.phys
        shared = set()
        acc = []
        lm = getattr(self, "locmap", {})
        for ins in mir:
            a = _mir_access(ins, phys)
            if lm:
                a = [(ag, lm.get(loc, loc), k) for ag, loc, k in a]
            if self.hsplit and ins[0] == "halo":
                a.append(("N", phys[ins[2]], "R"))
            acc.append(a)
            for ag, loc, k in a:
                if ag in "BN":
                    shared.add(loc)
        n = len(mir)
        labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
        succ = _mir_succ(mir)
        pred = defaultdict(list)
        for i, ss in enumerate(succ):
            for s in ss:
                pred[s].append(i)
        need = [set() for _ in range(n)]      # syncs to put before instruction i
        state_in = [None] * n
        state_out = [None] * n
        work = list(range(n))
        inq = set(work)
        while work:
            i = work.pop(0)
            inq.discard(i)
            st = set()
            for p in pred[i]:
                if state_out[p] is not None:
                    st |= state_out[p]
            state_in[i] = st
            cur = set(st)
            for s in sorted(need[i]):
                cur = _apply_sync(cur, s)
            for ag, loc, k in acc[i]:
                for s in _needed(cur, ag, loc, k, shared):
                    need[i].add(s)
                    cur = _apply_sync(cur, s)
            for ag, loc, k in acc[i]:
                if loc in shared or ag == "P":
                    for y in "UPBN":
                        if y != ag and (loc in shared or y == "U"):
                            cur.add((loc, ag, k, y))
            cur = frozenset(cur)
            if cur != state_out[i]:
                state_out[i] = cur
                for s in succ[i]:
                    if s not in inq:
                        work.append(s)
                        inq.add(s)
        if os.environ.get("CFD_SYNC_ALL") == "1":    # debugging: every agent pair synced before every MIR op
            every = {("C", "B"), ("C", "N"), ("B", "C"), ("N", "C"), ("B", "N"), ("N", "B"), ("P", "U")}
            need = [every if ins[0] in ("blk", "gather", "halo", "red", "out", "fetch") else need[i]
                    for i, ins in enumerate(mir)]
        out = []
        for i, ins in enumerate(mir):
            for s in sorted(need[i]):
                out.append(("tok",) if s == ("P", "U") else ("sync",) + s)
            out.append(self._seg_toks(ins) if ins[0] == "seg" else ins)
        self.mir = out

    def _seg_toks(self, ins):
        """Inside a fused run, a block that unpacks an L1 tile an earlier block of the same pass packed waits for a
        pack token; a tile read at or before the block that writes it is carried to the next pass (a partial), which
        a token at the end of the pass covers."""
        phys = self.phys
        rd = lambda b: {phys[u[2]] for u in b[1] if u[0] in ("ld", "lds", "rvf") and phys[u[2]] < DBASE}
        wr = lambda b: {phys[r] for _, r in b[2] if phys[r] < DBASE}
        items, packed, seen_rd, carried = [], set(), set(), False
        for b in ins[1]:
            r = rd(b)
            if r & packed:
                items.append(("tok",))
                packed.clear()
            seen_rd |= r
            carried |= bool(wr(b) & seen_rd)
            packed |= wr(b)
            items.append(b)
        if carried:
            items.append(("tok",))
        return ("seg", items)

    def summary(self):
        c = Counter(i[0] for i in self.mir)
        nb = [len(i[1]) for i in self.mir if i[0] == "blk"]
        return (f"MIR {len(self.mir)}: {dict(c)}; uops/block {np.mean(nb):.1f}; tiles {self.ntiles} ({self.nsmall} "
                f"one-tile + {self.nvec} vectors, {self.ntiles - self.nsmall} tiles), DRAM vectors {self.ndram} + {len(self.alias)} in-place consts, "
                f"ldr {sum(u[0] == 'ldr' for i in self.mir if i[0] == 'blk' for u in i[1])}, halo slots "
                f"{self.nhalo} x {self.maps.hmax} ({self.hwords} words, {self.hwords_plan} unaliased; plan hmax {self.maps.hmax_q}); maps {len(self.maps.maps)}; peak fine halos live {self.halo_peak[0]} at MIR {self.halo_peak[1]} (defined at {self.halo_peak[2]}, phase index {self.halo_peak[3]}); P {self.P}, cells/core "
                f"{self.lay.count[self.prog.cells.name].max()}; peak halo readers (sent at, source, gathers at, allocation-time MIR) "
                f"{[(d, self.space[s].name, rd) for d, s, rd in getattr(self, 'halo_peak_readers', [])]}")


# ---- MIR helpers
def _mir_uses(ins):
    op = ins[0]
    if op == "seg":       # the values the run reads before writing them
        out, d = [], set()
        for x in ins[1]:
            out += [r for r in _mir_uses(x) if r not in d]
            d |= set(_mir_defs(x))
        return out
    if op == "blk":
        out = []
        for u in ins[1]:
            if u[0] in ("ld", "rvf", "lds"):
                out.append(u[2])
            elif u[0] == "ldr":
                out += ([u[2]] if u[2] is not None else []) + [s for s, h in u[5]] + \
                       [("H", h) for s, h in u[5] if h is not None]
        return out
    if op == "gather":
        return [s for s, h in ins[3]] + [("H", h) for s, h in ins[3] if h is not None]
    if op == "halo":
        return [ins[2]]
    if op == "red":
        return [ins[2]]
    if op == "red2":
        return list(ins[2::2])
    if op in ("out",):
        return [ins[2]]
    if op == "jumpif":
        return [ins[1]]
    if op == "keep":
        return list(ins[1])
    return []


def _mir_defs(ins):
    op = ins[0]
    if op == "seg":
        return [r for x in ins[1] for r in _mir_defs(x)]
    if op == "blk":
        return [r for _, r in ins[2]]
    if op == "gather":
        return [ins[1]]
    if op == "halo":
        return [("H", ins[1])]
    if op == "red":
        return [ins[1]]
    if op == "red2":
        return list(ins[1::2])
    if op == "fetch":
        return [ins[1]]
    return []


def _rename_uses(ins, ren):
    f = lambda r: ren.get(r, r)
    op = ins[0]
    if op == "blk":
        return ("blk", [(u[0], u[1], f(u[2])) + u[3:] if u[0] in ("ld", "rvf", "lds") else
                        (u[:2] + (f(u[2]) if u[2] is not None else None,) + u[3:5] + ([(f(s), h) for s, h in u[5]],))
                        if u[0] == "ldr" else u for u in ins[1]], ins[2])
    if op == "gather":
        return ("gather", f(ins[1]), ins[2], [(f(s), h) for s, h in ins[3]])
    if op == "red" and len(ins) > 4:
        return (op, ins[1], f(ins[2]), ins[3], ins[4], f(ins[5]))
    if op in ("red", "out"):
        return (op, ins[1], f(ins[2])) + tuple(ins[3:])
    if op == "red2":
        return (op,) + tuple(x if i % 2 == 0 else f(x) for i, x in enumerate(ins[1:]))
    if op == "keep":
        return ("keep", [f(r) for r in ins[1]])
    return ins


def _adjacent_reduces(code, window=64):
    """Trace-level half of the pairing: reduce A ... reduce B with only ew/copy between and B not depending on A
    becomes (the between that does not need A) A B (the between that does). Lowering then puts nothing but B's partial
    between the two reds, so fusion cannot glue A's scalar consumers to B's operand, and joint_red groups them."""
    out, i = [], 0
    while i < len(code):
        a = code[i]
        if a[0] != "reduce":
            out.append(a)
            i += 1
            continue
        dep, rd, wr, dins, ind, hit = {a[1]}, {a[3]}, set(), [], [], None
        k = i + 1
        while k < len(code) and k <= i + window:
            b = code[k]
            u, d = _uses_defs(b)
            u, d = set(u), set(d)
            if b[0] == "reduce":
                if not u & dep and not d & (rd | wr | dep):
                    hit = b
                break
            if b[0] not in ("ew", "copy", "imm"):
                break
            if u & dep:
                dins.append(b)
                dep |= d
                wr |= d
                rd |= u
            elif d & (rd | wr | dep):
                break
            else:
                ind.append(b)
            k += 1
        if hit is None or not dins:
            out.append(a)
            i += 1
            continue
        out += ind + [a, hit] + dins
        i = k + 1
    return out


def _mir_succ(mir):
    labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
    out = []
    for i, ins in enumerate(mir):
        nxt = [i + 1] if i + 1 < len(mir) else []
        if ins[0] == "jump":
            out.append([labels[ins[1]]])
        elif ins[0] == "jumpif":
            out.append(nxt + [labels[ins[2]]])
        else:
            out.append(nxt)
    return out


def _mir_access(ins, phys):
    """[(agent, physical tile, 'R'/'W')]; agents U P (compute unpack / pack), B, N. A DRAM-homed vector is read by
    the reader (N) and written by the writer (B)."""
    op = ins[0]
    a = []
    if op == "seg":
        return [x for b in ins[1] for x in _mir_access(b, phys)]
    if op == "blk":
        for u in ins[1]:
            if u[0] in ("ld", "rvf", "lds"):
                a.append(("U", phys[u[2]], "R"))
            elif u[0] == "ldr":
                for r in ([u[2]] if u[2] is not None else []) + [s for s, h in u[5]]:
                    a.append(("N", phys[r], "R"))
        for _, r in ins[2]:
            a.append(("B" if phys[r] >= DBASE else "P", phys[r], "W"))
    elif op == "gather":
        mode = ins[4] if len(ins) > 4 else 0
        ag = "B" if mode in (3, 4) else "N"     # bgather (3) and bloc (4) run on BRISC
        for s, h in ins[3]:
            a.append((ag, phys[s], "R"))
        # a split gather's parts patch disjoint lanes: "Q" orders against every access but another patch
        a.append((ag, phys[ins[1]], "Q" if mode in (1, 2, 4) else "W"))
    elif op == "bscal":
        a += [("B", phys[u[2]], "R") for u in ins[1] if u[0] == "ld"] + [("B", phys[r], "W") for _, r in ins[2]]
    elif op == "halo":
        a.append(("B", phys[ins[2]], "R"))
    elif op == "red":
        a.append(("B", phys[ins[2]], "R"))
        a.append(("B", phys[ins[1]], "W"))
    elif op == "red2":
        a += [("B", phys[r], "R") for r in ins[2::2]] + [("B", phys[r], "W") for r in ins[1::2]]
    elif op == "out":
        a.append(("B", phys[ins[2]], "R"))
    elif op == "fetch":
        a.append(("B", phys[ins[1]], "W"))
    elif op == "jumpif":
        for ag in "UBN":
            a.append((ag, phys[ins[1]], "R"))
    return a


def _apply_sync(st, s):
    """Remove the facts a sync resolves. s = (X, Y): 'C' as X covers U and P facts, as Y it covers both too."""
    X, Y = s
    xs = ("U", "P") if X == "C" else (X,)
    ys = ("U", "P") if Y == "C" else (Y,)
    if s == ("P", "U"):
        xs, ys = ("P",), ("U",)
    return {f for f in st if not (f[1] in xs and f[3] in ys)}


def _needed(st, ag, loc, k, shared):
    out = set()
    for (l, x, kx, y) in st:
        if l != loc or y != ag or x == ag:
            continue
        if k not in "WQ" and kx not in "WQ" or k == kx == "Q":
            continue
        if x == "U" and ag == "P":
            continue          # PACK always follows UNPACK
        if x == "P" and ag == "U":
            out.add(("P", "U"))
        elif x in "UP" and ag in "BN":
            out.add(("C", ag))
        elif x in "BN" and ag in "UP":
            out.add((x, "C"))
        else:
            out.add((x, ag))
    return out
