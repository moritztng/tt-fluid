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
# stock tile ops, cycles per fp32 tile. Measured: add sub mul mad msb where (kernel-unit-costs.md); fold from flx-asm's
# hood profile. The rest are EST from the instruction count of the stock loop (17 + 32 n) until they are measured.
STOCK = {"add": 231.8, "sub": 231.8, "mul": 231.8, "mad": 305.8, "msb": 305.8, "where": 164.8,
         "max": 209.0, "min": 209.0, "div": 817.0, "neg": 145.0, "abs": 145.0, "sqrt": 721.0, "tanh": 900.0,
         "log": 900.0, "exp": 900.0, "gtz": 273.0, "gez": 273.0, "ltz": 273.0, "lez": 273.0, "eqz": 273.0,
         "nez": 273.0, "fill": 81.0, "cpd": 113.0, "fold": 3500.0}
MEASURED = {"add", "sub", "mul", "mad", "msb", "where", "fold"}

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
            out.append(("body", dict(ops=ops, ins=ins, outs=outs_, n=ninstr)))
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


def stock_cost(uops):
    """Cycles per tile of every arithmetic uop of a block as the VM runs it today (fill and cpd included)."""
    c = 0.0
    for u in uops:
        if u[0] in ("op", "un"):
            c += STOCK[u[1]]
        elif u[0] in ("fill", "cpd", "fold"):
            c += STOCK[u[0]]
    return c
