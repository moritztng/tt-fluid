"""Generated SFPU blocks (LEVERS 13): cut the arithmetic of one MIR block into straight-line bodies that run as one
replay-buffer program per tile instead of one stock LLK call per op.

A MIR block is a uop list over Dest slots plus its stores. Here it becomes a value flow (every load, constant and op
result is a value), and plan() walks it in program order and groups consecutive fusable ops into bodies:

* a body reads its operands from Dest (one SFPLOAD each), keeps every intermediate in an LREG and writes back only
  the values something after it still reads (one SFPSTORE each);
* a load between two ops is hoisted in front of the body while a Dest slot is free, so a chain that reloads a slot
  does not end the body. Loads keep their relative order (the reader's ring is consumed in program order);
* ops the stock library implements as a long sequence (div, sqrt, tanh, log, exp, fold) stay stock words and end a body.

The arithmetic and its order are unchanged, so memu keeps executing the original uop list and stays the reference.

Cost model measured on Blackhole (state/flx/rows/kernel-unit-costs.md): a body is BODY_FIX + 32 * instructions cycles
per tile, a Dest read, a Dest write and an arithmetic instruction each counting one."""
from collections import namedtuple

# SFPU instructions per fusable op, in the form that is bitwise the stock tile op (mad / msb keep the stock call's
# multiply by +-1; where = setcc, mov, encc; max / min = one two-cycle swap; a comparison builds 0.0 / 1.0)
FUS = {"add": 1, "sub": 1, "mul": 1, "mad": 2, "msb": 2, "where": 3, "max": 2, "min": 2, "neg": 1, "abs": 1,
       "gtz": 4, "gez": 4, "ltz": 4, "lez": 4, "eqz": 4, "nez": 4}
BODY_FIX = 17       # cycles per body and tile outside the instructions (measured)
CYC = 32            # cycles per SFPU instruction over one tile (32 steps of 32 lanes, measured 1.00 per step)
# stock tile ops, cycles per fp32 tile, measured on Blackhole (state/flx/rows/kernel-unit-costs.md, runs u6 and u8);
# sub is the add with another mode bit, the other comparisons are gtz with another test. EST: sqrt, tanh, log, exp
# (not measured), fold (flx-asm's hood profile, in situ).
STOCK = {"add": 231.8, "sub": 231.8, "mul": 231.8, "mad": 305.8, "msb": 305.8, "where": 164.8,
         "max": 144.5, "min": 144.5, "div": 1099.8, "neg": 167.8, "abs": 167.8, "sqrt": 721.0, "tanh": 900.0,
         "log": 900.0, "exp": 900.0, "gtz": 335.8, "gez": 335.8, "ltz": 335.8, "lez": 335.8, "eqz": 335.8,
         "nez": 335.8, "fill": 107.8, "cpd": 135.8, "fold": 3500.0}
MEASURED = {"add", "sub", "mul", "mad", "msb", "where", "max", "min", "div", "neg", "abs", "gtz", "gez", "ltz", "lez",
            "eqz", "nez", "fill", "cpd"}
# a body issued through the load macro (SFPLOADMACRO, runs u7 and u8): one load-style instruction per cycle, each
# scheduling up to one op per SFPU sub-unit. Measured: 17 + 32 * 2 + 3 for an in-place add, 17 + 32 * 3 + 5 for
# mul then add from three operands. PIPE_FLUSH and PIPE_CFG (instructions to set templates and sequences) are EST.
PIPE_FLUSH = 5
PIPE_CFG = 9

Ev = namedtuple("Ev", "kind name out ins uop")    # kind: L load, K constant, A fusable op, N stock op


def flow(uops, stores):
    """Value flow of one block: events in program order and, per value, the indices of the events that read it
    (len(events) for a store). Slot copies (cpd) are aliases and cost nothing here."""
    slot, ev, uses, nv = {}, [], {}, 0

    def new():
        nonlocal nv
        nv += 1
        uses[nv] = []
        return nv

    def rd(s, at):
        v = slot[s]
        uses[v].append(at)
        return v

    for u in uops:
        k, at = u[0], len(ev)
        if k in ("ld", "lds", "ldr", "rvf"):
            slot[u[1]] = v = new()
            ev.append(Ev("L", k, v, (), u))
        elif k == "fill":
            slot[u[1]] = v = new()
            ev.append(Ev("K", "fill", v, (), u))
        elif k == "cpd":
            slot[u[2]] = slot[u[1]]
        elif k == "op":
            ins = tuple(rd(s, at) for s in u[3:])
            slot[u[2]] = v = new()
            ev.append(Ev("A" if u[1] in FUS else "N", u[1], v, ins, u))
        elif k == "un":
            ins = (rd(u[2], at),)
            slot[u[2]] = v = new()
            ev.append(Ev("A" if u[1] in FUS else "N", u[1], v, ins, u))
        elif k == "fold":
            ins = (rd(u[1], at),)
            slot[u[1]] = v = new()
            ev.append(Ev("N", "fold", v, ins, u))
        else:
            raise NotImplementedError(k)
    outs = []
    for s, r in stores:
        outs.append((rd(s, len(ev)), r))
    return ev, uses, outs


def _lregs(ops, const, uses, last):
    """LREGs a body needs at its fullest: operands are loaded at first use and dropped after their last use inside
    the body, constants hold one LREG throughout, a result that leaves the body holds its LREG to the end."""
    inside = {e.out for e in ops}
    end = {}
    for i, e in enumerate(ops):
        for v in e.ins:
            end[v] = i
    live, peak = set(), 0
    ncon = len({v for e in ops for v in e.ins if v in const})
    for i, e in enumerate(ops):
        live |= {v for v in e.ins if v not in const}
        peak = max(peak, len(live) + ncon + 1)        # + the result being written
        live -= {v for v in e.ins if end[v] == i and not (v in inside and any(a > last for a in uses[v]))}
        live.add(e.out)
        if not any(a > i for a in [end.get(e.out, -1)]) and not any(a > last for a in uses[e.out]):
            live.discard(e.out)
    return max(peak, len(live) + ncon)


def plan(uops, stores, slots=4, lregs=8, maxlen=32, hoist=True):
    """[(kind, payload)] in execution order: ("load", Ev), ("stock", Ev), ("fill", value id: a constant something
    outside a body reads, materialized by the stock fill), ("body", dict(ops, ins, outs, n))."""
    ev, uses, outs = flow(uops, stores)
    n = len(ev)
    const = {e.out for e in ev if e.kind == "K"}
    live = lambda v, pos: any(a > pos for a in uses[v])
    res, out, done_fill = set(), [], set()

    def need(v):                    # a constant read from Dest: fill it once
        if v in const and v not in done_fill:
            done_fill.add(v)
            res.add(v)
            out.append(("fill", v))

    i = 0
    while i < n:
        e = ev[i]
        if e.kind == "K":
            i += 1
        elif e.kind == "L":
            out.append(("load", e))
            res.add(e.out)
            res = {v for v in res if live(v, i)}
            i += 1
        elif e.kind == "N":
            for v in e.ins:
                need(v)
            out.append(("stock", e))
            res.add(e.out)
            res = {v for v in res if live(v, i)}
            i += 1
        else:
            ops, hoisted, j, best = [], [], i, None
            while j < n:
                f = ev[j]
                if f.kind == "K":
                    j += 1
                    continue
                if f.kind == "L":
                    if hoist and len(res) + len(hoisted) < slots:
                        hoisted.append(f)
                        j += 1
                        continue
                    break
                if f.kind == "N":
                    break
                trial = ops + [f]
                inside = {x.out for x in trial}
                ins = {v for x in trial for v in x.ins if v not in inside and v not in const}
                outs_ = {x.out for x in trial if live(x.out, j)}
                stay = {v for v in res | {h.out for h in hoisted} if live(v, j)}
                ninstr = len(ins) + sum(FUS[x.name] for x in trial) + len(outs_)
                if len(stay) + len(outs_) > slots or ninstr > maxlen or _lregs(trial, const, uses, j) > lregs:
                    break
                ops = trial
                best = (j, len(hoisted), ins, outs_, ninstr)
                j += 1
            if best is None:        # not even one op fits (its result has no slot): the stock call, in place
                for v in e.ins:
                    need(v)
                out.append(("stock", e))
                res.add(e.out)
                res = {v for v in res if live(v, i)}
                i += 1
                continue
            last, nh, ins, outs_, ninstr = best
            for h in hoisted[:nh]:
                out.append(("load", h))
                res.add(h.out)
            out.append(("body", dict(ops=ops, ins=ins, outs=outs_, n=ninstr, last=last)))
            res |= outs_
            res = {v for v in res if live(v, last)}
            i = last + 1
    for v, _ in outs:
        need(v)
    return out, ev, uses


def cost(pl):
    """(cycles per tile of the planned block's ops, of the same ops as stock calls, bodies, instructions in bodies,
    ops inside bodies, ops left stock). Loads and stores are the same in both and are not counted."""
    blk = stock = 0.0
    nb = ni = nin = nst = 0
    for k, p in pl:
        if k == "body":
            blk += BODY_FIX + CYC * p["n"]
            stock += sum(STOCK[x.name] for x in p["ops"])
            nb += 1
            ni += p["n"]
            nin += len(p["ops"])
        elif k == "stock":
            blk += STOCK[p.name]
            stock += STOCK[p.name]
            nst += 1
        elif k == "fill":
            blk += STOCK["fill"]
    return blk, stock, nb, ni, nin, nst


def pipe_cost(pl, uses, n):
    """EST, lower bound of a body through the load macro: one cycle per row for every Dest operand and for every result
    that cannot be stored onto the slot of an operand that dies in the body. Assumes every op rides on a load (true
    for chains within 4 instruction templates; trees and long bodies will need plain instructions on top)."""
    blk, i = 0.0, 0
    for k, p in pl:
        if k == "body":
            dead = sum(1 for v in p["ins"] if not any(a > p["last"] for a in uses[v]))
            rows = max(len(p["ins"]), 1) + max(0, len(p["outs"]) - dead)
            blk += BODY_FIX + PIPE_FLUSH + PIPE_CFG + CYC * rows
        elif k == "stock":
            blk += STOCK[p.name]
        elif k == "fill":
            blk += STOCK["fill"]
    return blk


def stock_cost(uops):
    """Cycles per tile of every arithmetic uop of a block as the VM runs it today (fill and cpd included)."""
    c = 0.0
    for u in uops:
        if u[0] in ("op", "un"):
            c += STOCK[u[1]]
        elif u[0] in ("fill", "cpd", "fold"):
            c += STOCK[u[0]]
    return c


# ---- body programs --------------------------------------------------------------------------------------------------
# program(spec) turns one planned body (blklower: ins / outs with Dest slots, ops over values, constants) into the
# inline words the VM's C_BLK executes: [ncfg | L << 8 | nrep << 16 | nfl << 24, config words, row words]. The kernel
# pushes the config words once per tile, records the row words into the replay buffer and replays them nrep times
# (device/kernels/blk_exec.h). Instructions are kept symbolic, (mnemonic, TT_OP_ arguments...), until the last step;
# sim() executes the symbolic row on numpy with the pipeline's read latencies and check() compares it with the ops
# evaluated one by one, so a schedule is replayed against the plain evaluation before it may run on a chip.
import numpy as np

FP32, AM_STAY, AM_STEP = 3, 7, 6    # SFPLOAD / SFPSTORE mode; ADDR_MOD_7 (no step) and ADDR_MOD_6 (dest += 2), blk_init
C0, C1 = 9, 10                      # LCONST_0, LCONST_1
NE0, GTE0, EQ0 = 2, 4, 6            # SFPSETCC_MOD1_LREG_*
# the six comparisons as the stock tile ops compile (vm_compute.cpp, objdump of the TRISC1 kernel): the SFPSETCC modes
# and the value of the lanes they select (the other lanes get the other one of 0.0 / 1.0)
CMP = {"eqz": ((EQ0,), 1), "nez": ((EQ0,), 0), "gez": ((GTE0,), 1), "ltz": ((GTE0,), 0), "gtz": ((GTE0, NE0), 1), "lez": ((GTE0, NE0), 0)}
NEG1 = "-1.0"                       # the msb's multiplier, a constant LREG of its own (stock: addcmul's value)
ENCC = ("sfpencc", 3, 0, 0, 10)     # every lane on again, as compiled SFPI code ends a v_if
MADS = ("sfpmad", "sfpadd", "sfpmul")
NLANE = 32


class Reject(Exception):
    pass


def _f32bits(x):
    return int(np.float32(x).view(np.uint32))


def _plain(spec, lregs=8):
    """One instruction per cycle: (cfg, row). Operands are loaded when first needed (or a cycle early, into the
    wait for a multiply-add), every op is its stock instruction sequence, results are stored when they are ready."""
    ins, outs, const, ops = dict(spec["ins"]), dict(spec["outs"]), spec["const"], spec["ops"]
    last, first = {}, {}
    for i, o in enumerate(ops):
        for v in o[2:]:
            last[v] = i
            first.setdefault(v, i)
    cfg, row = [], []
    free, reg, ready = list(range(lregs)), {}, {}
    const = dict(const)
    if any(o[0] == "msb" for o in ops):
        const[NEG1] = -1.0
    for v, x in const.items():
        b = _f32bits(x)
        if b == 0:
            reg[v] = C0
        elif b == 0x3f800000:
            reg[v] = C1
        else:
            if not free:
                raise Reject("lregs")
            reg[v] = r = free.pop(0)
            cfg += [("sfploadi", r, 0, b >> 16)] if b & 0xffff == 0 else [("sfploadi", r, 8, b >> 16), ("sfploadi", r, 10, b & 0xffff)]
    unloaded = sorted((v for v in ins if v in first), key=lambda v: first[v])
    pend, stored = [], set()
    cur = [0]

    def alloc():
        if not free:
            raise Reject("lregs")
        return free.pop(0)

    def emit(*x):
        row.append(tuple(x))
        return len(row)

    def load(v, r=None):
        reg[v] = r = alloc() if r is None else r
        ready[r] = emit("sfpload", r, FP32, AM_STAY, ins[v] * 64)
        unloaded.remove(v)

    def storable(v):
        return ready[reg[v]] <= len(row) and not any(ins[u] == outs[v] for u in unloaded)

    def store(v):
        emit("sfpstore", reg[v], FP32, AM_STAY, outs[v] * 64)
        pend.remove(v)
        stored.add(v)
        if last.get(v, -1) < cur[0]:
            free.append(reg.pop(v))

    def wait(rs):
        while any(ready.get(r, 0) > len(row) for r in rs):
            v = next((v for v in pend if storable(v)), None)
            if v is not None:
                store(v)
            elif unloaded and len(free) > 2:
                load(unloaded[0])
            else:
                emit("sfpnop")

    def need(v):
        if v not in reg:
            load(v)
        return reg[v]

    def dies(v, i):
        return v not in const and last[v] == i and (v not in outs or v in stored)

    def own(v, i, others=()):
        """A register holding v that this op may overwrite: v's own when v dies here, else a copy."""
        if dies(v, i) and v not in others:
            return reg.pop(v)
        r = alloc()
        wait([reg[v]])
        ready[r] = emit("sfpmov", 0, reg[v], r, 0)
        return r

    for i, (name, out, *a) in enumerate(ops):
        cur[0] = i
        if name in ("add", "mul"):
            ra, rb = need(a[0]), need(a[1])
            wait([ra, rb])
            r = reg.pop(a[0]) if dies(a[0], i) else reg.pop(a[1]) if dies(a[1], i) and a[1] != a[0] else alloc()
            ready[r] = (emit("sfpadd", C1, ra, rb, r, 0) if name == "add" else emit("sfpmul", ra, rb, C0, r, 0)) + 1
        elif name == "sub":     # 1 a - b: the add with its addend negated
            ra, rb = need(a[0]), need(a[1])
            wait([ra, rb])
            r = reg.pop(a[0]) if dies(a[0], i) else reg.pop(a[1]) if dies(a[1], i) and a[1] != a[0] else alloc()
            ready[r] = emit("sfpadd", C1, ra, rb, r, 2) + 1
        elif name in ("mad", "msb"):    # d +- b c as addcmul runs it: t = b * (+-1), then c t + d in one multiply-add
            rb, rc, rd = need(a[0]), need(a[1]), need(a[2])
            wait([rb])
            t = alloc()
            ready[t] = emit("sfpmul", rb, C1 if name == "mad" else reg[NEG1], C0, t, 0) + 1
            wait([rc, t, rd])
            r = t
            ready[r] = emit("sfpmad", rc, t, rd, r, 0) + 1
        elif name in CMP:
            ra = need(a[0])
            wait([ra])
            r = reg.pop(a[0]) if dies(a[0], i) else alloc()
            modes, hit = CMP[name]
            for md in modes:
                emit("sfpsetcc", 0, ra, 0, md)
            emit("sfpmov", 0, C1 if hit else C0, r, 0)
            emit("sfpcompc", 0, 0, 0, 0)
            emit("sfpmov", 0, C0 if hit else C1, r, 0)
            ready[r] = emit(*ENCC)
        elif name in ("neg", "abs"):
            ra = need(a[0])
            wait([ra])
            r = reg.pop(a[0]) if dies(a[0], i) else alloc()
            ready[r] = emit("sfpmov", 0, ra, r, 1) if name == "neg" else emit("sfpabs", 0, ra, r, 1)
        elif name == "where":
            c, x, y = a
            rc = need(c)
            need(x)
            late = y in unloaded and last[y] == i and first[y] == i and y != c and y != x and a.count(y) == 1
            if not late:
                need(y)
            r = own(x, i, others=(c, y))
            wait([rc, r] + ([] if late else [reg[y]]))
            emit("sfpsetcc", 0, rc, 0, EQ0)
            if late:    # the false value comes straight from Dest, under the lane mask
                emit("sfpload", r, FP32, AM_STAY, ins[y] * 64)
                unloaded.remove(y)
            else:
                emit("sfpmov", 0, reg[y], r, 0)
            ready[r] = emit(*ENCC)
        elif name in ("max", "min"):
            need(a[0])
            need(a[1])
            if a[0] == a[1]:
                r = own(a[0], i)
            else:
                rd, rc = own(a[0], i, others=(a[1],)), own(a[1], i, others=(a[0],))
                wait([rd, rc])
                emit("sfpswap", 0, rc, rd, 1)       # rd = min, rc = max, two cycles
                ready[rd] = ready[rc] = emit("sfpnop")
                r = rc if name == "max" else rd
                free.append(rd if name == "max" else rc)
        else:
            raise Reject(name)
        for v in set(a):
            if v in reg and dies(v, i):
                free.append(reg.pop(v))
        reg[out] = r
        if out in outs:
            pend.append(out)
        elif out not in last:
            raise Reject("dead op")
    cur[0] = len(ops)
    while pend:
        v = pend[0]
        wait([reg[v]])
        if v in pend:
            store(v)
    k = max(j for j, x in enumerate(row) if x[0] in ("sfpload", "sfpstore"))
    row[k] = row[k][:3] + (AM_STEP,) + row[k][4:]
    return cfg, row


def program(spec, form="plain", lregs=8, sym=False):
    """The inline words of one body, or None when this form cannot run it (the block then keeps its stock words)."""
    try:
        cfg, row = _plain(spec, lregs)
    except Reject:
        return None
    u = 1 if len(row) >= 4 else 2
    if len(row) * u > 32:
        return None
    if sym:
        return cfg, row
    from . import tensix
    enc = lambda x: getattr(tensix, x[0])(*x[1:])
    return [len(cfg) | (len(row) * u) << 8 | (NLANE // u) << 16 | 2 << 24] + [enc(x) for x in cfg] + [enc(x) for x in row] * u


def _ord(x):        # the SFPU's total order on fp32 bit patterns: sign-magnitude, -NaN < -inf < -0 < +0 < +inf < +NaN
    b = x.view(np.uint32).astype(np.int64)
    return np.where(b >> 31, -(b & 0x7fffffff) - 1, b)


PRIM = {
    "add": lambda a, b: (a.astype(np.float64) + b).astype(np.float32),
    "mul": lambda a, b: (a.astype(np.float64) * b + 0.0).astype(np.float32),
    "neg": lambda a: (a.view(np.uint32) ^ np.uint32(0x80000000)).view(np.float32),
    "abs": lambda a: np.where(np.isnan(a), a, (a.view(np.uint32) & np.uint32(0x7fffffff)).view(np.float32)),
    "eq0": lambda a: (a.view(np.uint32) & np.uint32(0x7fffffff)) == 0,
    "gte0": lambda a: (a.view(np.uint32) >> 31) == 0,
    "sub": lambda a, b: (a.astype(np.float64) - b).astype(np.float32),
    "mad1": lambda a, b, c: (a.astype(np.float64) * b + c).astype(np.float32),
    "max": lambda a, b: np.where(_ord(a) >= _ord(b), a, b),
    "min": lambda a, b: np.where(_ord(a) >= _ord(b), b, a),
}


def sim(cfg, row, slots, nrow=2, strict=True):
    """Run the symbolic program on numpy: slots {slot: array (nrow, lanes)} -> the slots after. Every read of a
    register before its value has landed (a multiply-add lands two cycles after its issue, a swap likewise) and every
    Dest access inside a lane-mask window but the conditional load raises."""
    slots = {s: a.copy() for s, a in slots.items()}
    n = next(iter(slots.values())).shape[1]
    R = {C0: np.zeros(n, np.float32), C1: np.ones(n, np.float32)}
    on = np.ones(n, bool)
    ready = {}

    def rd(r, t):
        if strict and ready.get(r, 0) > t:
            raise AssertionError("register %d read at cycle %d, lands at %d" % (r, t, ready[r]))
        return R[r]

    def wr(r, x, t, lat=1):
        assert r not in (C0, C1) and not (strict and ready.get(r, 0) > t + lat)
        R[r] = np.where(on, x, R[r]) if r in R else np.where(on, x, np.float32(np.nan))
        ready[r] = t + lat

    for x in cfg:
        assert x[0] == "sfploadi"
        b = R[x[1]].view(np.uint32).copy() if x[1] in R else np.zeros(n, np.uint32)
        b = np.full(n, x[3] << 16, np.uint32) if x[2] == 0 else (b & 0xffff) | np.uint32(x[3] << 16) if x[2] == 8 else (b & 0xffff0000) | np.uint32(x[3])
        R[x[1]] = b.view(np.float32)
    keep = set(R)
    t = 0
    for k in range(nrow):
        for r in list(R):       # nothing but the constants survives a row by contract: poison the rest
            if r not in keep:
                R[r] = np.full(n, np.nan, np.float32)
        for x in row:
            op = x[0]
            if op == "sfpload":
                wr(x[1], slots[x[4] // 64][k], t)
            elif op == "sfpstore":
                assert on.all(), "store under a lane mask"
                slots[x[4] // 64][k] = rd(x[1], t)
            elif op == "sfpadd":
                assert x[1] == C1 and x[5] in (0, 2)
                wr(x[4], PRIM["sub" if x[5] else "add"](rd(x[2], t), rd(x[3], t)), t, 2)
            elif op == "sfpmul":
                assert x[3] == C0 and x[5] == 0
                wr(x[4], PRIM["mul"](rd(x[1], t), rd(x[2], t)), t, 2)
            elif op == "sfpmad":
                assert x[5] == 0
                wr(x[4], PRIM["mad1"](rd(x[1], t), rd(x[2], t), rd(x[3], t)), t, 2)
            elif op == "sfpmov":
                wr(x[3], PRIM["neg"](rd(x[2], t)) if x[4] == 1 else rd(x[2], t), t)
            elif op == "sfpabs":
                wr(x[3], PRIM["abs"](rd(x[2], t)), t)
            elif op == "sfpsetcc":
                on = on & {EQ0: PRIM["eq0"], NE0: lambda a: ~PRIM["eq0"](a), GTE0: PRIM["gte0"]}[x[4]](rd(x[2], t))
            elif op == "sfpcompc":
                on = ~on
            elif op == "sfpencc":
                assert x == ENCC
                on = np.ones(n, bool)
            elif op == "sfpswap":
                c, d = rd(x[2], t), rd(x[3], t)
                wr(x[3], PRIM["min"](c, d), t, 2)
                wr(x[2], PRIM["max"](c, d), t, 2)
            elif op != "sfpnop":
                raise NotImplementedError(op)
            t += 1
        assert on.all(), "a row must end with every lane enabled"
    return slots


def ref(spec, slots, nrow=2):
    """The body's ops evaluated one by one on the same primitives: {slot: array} of the outputs."""
    env = {v: slots[s] for v, s in spec["ins"]}
    n = next(iter(slots.values())).shape
    env.update({v: np.full(n, x, np.float32) for v, x in spec["const"].items()})
    one, zero = np.float32(1), np.float32(0)
    for name, out, *a in spec["ops"]:
        x = [env[v] for v in a]
        if name == "where":
            env[out] = np.where(PRIM["eq0"](x[0]), x[2], x[1])
        elif name in CMP:
            modes, hit = CMP[name]
            sel = PRIM["gte0"](x[0]) & ~PRIM["eq0"](x[0]) if len(modes) == 2 else PRIM["eq0" if modes[0] == EQ0 else "gte0"](x[0])
            env[out] = np.where(sel, one if hit else zero, zero if hit else one).astype(np.float32)
        elif name in ("mad", "msb"):
            env[out] = PRIM["mad1"](x[1], PRIM["mul"](x[0], np.float32(1 if name == "mad" else -1) + np.zeros_like(x[0])), x[2])
        else:
            env[out] = PRIM[name](*x)
    return {s: env[v] for v, s in spec["outs"]}


def check(spec, seed=0, **kw):
    """True when the generated program computes the body bitwise as its ops do, on random data with zeros of both
    signs, infinities and NaN mixed in; None when no program is generated."""
    p = program(spec, sym=True, **kw)
    if p is None:
        return None
    rng = np.random.default_rng(seed)
    special = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e-30, -3.5], np.float32)
    slots = {}
    for s in {s for _, s in spec["ins"]} | {s for _, s in spec["outs"]}:
        a = rng.standard_normal((2, 64)).astype(np.float32)
        m = rng.random((2, 64)) < 0.4
        a[m] = special[rng.integers(0, len(special), m.sum())]
        slots[s] = a
    got, want = sim(*p, slots), ref(spec, slots)
    return all(np.array_equal(got[s].view(np.uint32), want[s].view(np.uint32)) for s in want)


_ACC = {}


def accept(spec, **kw):
    """The encoder's test of a planned body: a program exists and its replay computes what the ops do. Cached per
    body (slots, ops and constants)."""
    names = {}
    k = (tuple(sorted(kw.items())), tuple((names.setdefault(v, len(names)), s) for v, s in spec["ins"]),
         tuple((name, names.setdefault(out, len(names))) + tuple(names.setdefault(v, len(names)) for v in a) for name, out, *a in spec["ops"]),
         tuple((names.get(v), s) for v, s in spec["outs"]), tuple(sorted((names.get(v, -1), _f32bits(x)) for v, x in spec["const"].items())))
    if k not in _ACC:
        with np.errstate(all="ignore"):
            _ACC[k] = check(spec, **kw) is True
    return _ACC[k]
