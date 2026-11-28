"""Lower one MIR block to generated SFPU bodies (LEVERS 13): blockgen.plan cuts the block's arithmetic into bodies,
this module gives every value that lives in Dest a real slot and returns the block as the encoder emits it.

lower_block(uops, stores, nslot) -> (items, stores2). An item is a uop in the original format with a re-assigned
Dest slot, or ("body", spec):

    spec = dict(ins=[(value, slot)],            operands read from Dest, in order of first use
                outs=[(value, slot)],           results written back to Dest, in op order
                ops=[(name, out_value, *in_values)],   operand order as in the uop: binary (a, b), where (cond, x, y),
                                                mad / msb (b, c, d) for d +- b c, unary (x,)
                const={value: float})           operands that are constants (an LREG immediate, never in Dest)

Values are blockgen.flow's ids. A body reads all its operands of a row before it stores that row's results, so a
result may take the slot of an operand that dies in the body. Stock unary ops and fold work in place: an input that
is still read later is copied (cpd) into a free slot first and the copy is consumed. The original cpd uops are
aliases in flow(), so the lowered block carries only the copies it needs.

Loads keep their relative order and their tails (the reader's ring is consumed in program order). The arithmetic and
its order are unchanged: memu keeps running the original uops and stays the reference. selfcheck() evaluates the
original and the lowered block on the same random tiles with memu's op table and compares every store bitwise.

    python -m ttfluid.resident.blklower MCACHE.pkl[.zst] [--nslot N] [--no-hoist] [--seed S]"""
import numpy as np

from . import blockgen as bg

F = np.float32
LOADS = ("ld", "lds", "ldr", "rvf")


class Infeasible(Exception):
    pass


def lower_block(uops, stores, nslot, lregs=8, maxlen=32, hoist=True, info=None, accept=None, keep=False):
    """(items, stores2). A block without a body, one the allocation cannot fit into nslot slots, or one with a body
    that accept(spec) turns down (the generator cannot build its program) comes back unchanged (the same uop and store
    lists). info, when given, counts: lowered, body, nobody, fallback, and per cause fallback:<why> (a turned-down
    body is fallback:rejected, its bodies are counted in rejected bodies). keep: a load stays in the slot the
    original uop named when that slot is free; default the lowest free slot (fewer distinct bodies)."""
    info = info if info is not None else {}

    def bump(k, v=1):
        info[k] = info.get(k, 0) + v

    try:
        pl, ev, uses = bg.plan(uops, stores, slots=nslot, lregs=lregs, maxlen=maxlen, hoist=hoist)
    except NotImplementedError as e:
        bump("fallback")
        bump("fallback:uop %s" % e)
        return list(uops), list(stores)
    nbody = sum(1 for k, _ in pl if k == "body")
    if not nbody:
        bump("nobody")
        return list(uops), list(stores)
    try:
        items, stores2 = _allocate(pl, ev, uses, bg.flow(uops, stores)[2], nslot, keep, bump)
    except Infeasible as e:
        bump("fallback")
        bump("fallback:%s" % e)
        return list(uops), list(stores)
    if accept is not None:
        no = sum(1 for u in items if u[0] == "body" and not accept(u[1]))
        if no:
            bump("fallback")
            bump("fallback:rejected")
            bump("rejected bodies", no)
            return list(uops), list(stores)
    bump("lowered")
    bump("body", nbody)
    return items, stores2


def _allocate(pl, ev, uses, outs, nslot, keep=False, bump=lambda *a: None):
    pos = {e.out: i for i, e in enumerate(ev)}           # value -> the event that makes it
    const = {e.out: e.uop[2] for e in ev if e.kind == "K"}
    done = set()                                         # executed events (hoisted loads run ahead of their index)
    live = lambda v: any(a not in done for a in uses[v])     # a store reads at len(ev): never done
    at, free, items = {}, set(range(nslot)), []          # at: value -> its Dest slot

    def alloc(pref=None):
        if not free:
            raise Infeasible("no free slot")
        s = pref if pref in free else min(free)
        free.discard(s)
        return s

    def sweep():
        for v in [v for v in at if not live(v)]:
            free.add(at.pop(v))

    def slot(v):
        if v not in at:
            raise Infeasible("operand not in Dest")
        return at[v]

    for k, p in pl:
        if k == "load":
            u = p.uop
            s = alloc(u[1] if keep else None)
            items.append((u[0], s) + tuple(u[2:]))
            done.add(pos[p.out])
            at[p.out] = s
        elif k == "fill":
            s = alloc()
            items.append(("fill", s, const[p]))
            at[p] = s
        elif k == "stock":
            u = p.uop
            src = [slot(v) for v in p.ins]
            done.add(pos[p.out])
            if u[0] in ("un", "fold"):                   # in place: a still-read input is copied first
                v, s = p.ins[0], src[0]
                if live(v):
                    s2 = alloc()
                    items.append(("cpd", s, s2))
                    s = s2
                else:
                    del at[v]
                items.append(("un", u[1], s) if u[0] == "un" else ("fold", s))
            else:                                        # may write any slot, an input's included
                sweep()
                # the result keeps the original's relation to the operands where it can: onto operand k if the
                # original wrote onto operand k and that operand dies here, else a slot no operand sits in
                k = [j for j, q in enumerate(u[3:]) if q == u[2]]
                if k and src[k[0]] in free:
                    s = alloc(src[k[0]])
                elif free - set(src):
                    s = alloc(min(free - set(src)))
                else:
                    s = alloc()
                if (s in src) != bool(k) or (k and s != src[k[0]]):
                    bump("stock results placed unlike the original")
                items.append(("op", u[1], s) + tuple(src))
            at[p.out] = s
        elif k == "body":
            ops = p["ops"]
            inside = {x.out for x in ops}
            order = []
            for x in ops:
                for v in x.ins:
                    if v not in inside and v not in const and v not in order:
                        order.append(v)
            assert set(order) == set(p["ins"]), (order, p["ins"])
            ins = [(v, slot(v)) for v in order]
            done.update(pos[x.out] for x in ops)
            dying = {v: s for v, s in ins if not live(v)}
            sweep()                                      # operands that die here give their slots to the results
            fresh = free - set(dying.values())           # free slots no operand of this body sits in
            dep, res, hazard = {}, [], []
            for x in ops:
                dep[x.out] = set().union(*[dep[v] if v in inside else {v} for v in x.ins if v not in const])
                if x.out not in p["outs"]:
                    continue
                # in place on an operand the result is computed from (the load of that operand precedes the store by
                # construction, and a load macro stores at its own address for free): the one used last; else a slot
                # no operand of this body sits in; else the slot of an unrelated dying operand, flagged
                own = [v for v in dep[x.out] if v in dying and dying[v] in free]
                if own:
                    s = dying[max(own, key=order.index)]
                elif fresh & free:
                    s = min(fresh & free)
                elif free:
                    s = min(free)
                    hazard.append((x.out, [v for v, q in dying.items() if q == s][0]))
                else:
                    raise Infeasible("no free slot")
                free.discard(s)
                at[x.out] = s
                res.append((x.out, s))
            assert len(res) == len(p["outs"])
            spec = dict(ins=ins, outs=res, ops=[(x.name, x.out) + tuple(x.ins) for x in ops],
                        const={v: const[v] for x in ops for v in x.ins if v in const})
            if hazard:      # (result, operand): the result takes the slot of an operand it does not depend on
                spec["hazard"] = hazard
            items.append(("body", spec))
        else:
            raise AssertionError(k)
        sweep()
        if len(at) > nslot:
            raise Infeasible("more values than slots")
    return items, [(slot(v), r) for v, r in outs]


# ---- numpy evaluators (memu's op table), for the self-check ---------------------------------------------------------

def _source(u, env, nload):
    """The tile a load delivers: ld reads env[reg]; lds / ldr / rvf are opaque and keyed by their place in the block's
    load order, which the lowering must keep (the tail is asserted equal in both evaluators through env['#loads'])."""
    seen = env.setdefault("#loads", {})
    tail = (u[0],) + tuple(repr(x) for x in u[2:])
    assert seen.setdefault(nload, tail) == tail, "load %d differs: %s vs %s" % (nload, seen[nload], tail)
    if u[0] == "ld":
        return env[u[2]]
    return env[("#opaque", nload)]


def _op(name, a):
    from .memu import BIN, UN
    with np.errstate(all="ignore"):
        if name == "where":
            return np.where(a[0] != 0, a[1], a[2]).astype(F)
        if name in ("mad", "msb"):                       # one rounding, as memu
            pr = a[0].astype(np.float64) * a[1]
            return (a[2] + pr if name == "mad" else a[2] - pr).astype(F)
        if name in BIN:
            return BIN[name](a[0], a[1]).astype(F)
        return UN[name](a[0]).astype(F)


def _fold(x):
    sm = x.reshape(32, 32).sum(0, dtype=F).sum(0, dtype=F)       # columns, then rows
    y = np.full(x.shape, np.nan, F)
    y[0] = sm
    return y


def eval_items(items, stores2, env):
    """{reg: tile} of every store of a lowered (or original: no body items) block. env: reg -> tile for ld, and
    ("#opaque", k) -> tile for the k-th load of the block when it is an lds / ldr / rvf."""
    D, nload = {}, 0
    for u in items:
        k = u[0]
        if k in LOADS:
            D[u[1]] = _source(u, env, nload)
            nload += 1
        elif k == "fill":
            D[u[1]] = np.full(env["#shape"], u[2], F)
        elif k == "cpd":
            D[u[2]] = D[u[1]]
        elif k == "op":
            D[u[2]] = _op(u[1], [D[s] for s in u[3:]])
        elif k == "un":
            D[u[2]] = _op(u[1], [D[u[2]]])
        elif k == "fold":
            D[u[1]] = _fold(D[u[1]])
        elif k == "body":
            sp = u[1]
            val = {v: D[s] for v, s in sp["ins"]}        # every operand of the row is read before a result lands
            for v, c in sp["const"].items():
                val[v] = np.full(env["#shape"], c, F)
            for o in sp["ops"]:
                val[o[1]] = _op(o[0], [val[v] for v in o[2:]])
            for v, s in sp["outs"]:
                D[s] = val[v]
        else:
            raise NotImplementedError(k)
    return [(r, D[s]) for s, r in stores2]


def eval_uops(uops, stores, env):
    return eval_items(uops, stores, env)


def _tiles(rng, n=1024):
    x = (rng.standard_normal(n) * np.float32(10.0) ** rng.integers(-3, 4, n)).astype(F)
    r = rng.random(n)
    x[r < 0.15] = 0.0
    x[(r >= 0.15) & (r < 0.22)] = -0.0
    x[(r >= 0.22) & (r < 0.27)] = 1.0
    x[(r >= 0.27) & (r < 0.30)] = -1.0
    return x


def check_block(uops, stores, nslot, rng, info=None, **policy):
    """(items, stores2, mismatching stores). The same random tiles through the original and the lowered block."""
    items, stores2 = lower_block(uops, stores, nslot, info=info, **policy)
    env = {"#shape": (1024,)}
    nload = 0
    for u in uops:
        if u[0] in LOADS:
            if u[0] == "ld":
                env.setdefault(u[2], _tiles(rng))
            elif u[0] == "rvf":
                env[("#opaque", nload)] = np.full(1024, _tiles(rng, 1)[0], F)
            else:
                env[("#opaque", nload)] = _tiles(rng)
            nload += 1
    a = eval_uops(uops, stores, env)
    env.pop("#loads")
    b = eval_items(items, stores2, env)
    bad = sum(1 for (ra, xa), (rb, xb) in zip(a, b) if ra != rb or not np.array_equal(xa.view(np.uint32), xb.view(np.uint32)))
    return items, stores2, bad + abs(len(a) - len(b))


def body_key(sp, slots=True):
    """A body up to the naming of its values: op names, operand wiring, constants (and Dest slots if slots)."""
    n = {}
    for v, _ in sp["ins"]:
        n[v] = ("i", len(n))
    for o in sp["ops"]:
        for v in o[2:]:
            if v in sp["const"]:
                n.setdefault(v, ("k", float(sp["const"][v])))
    ops = []
    for j, o in enumerate(sp["ops"]):
        ops.append((o[0],) + tuple(n[v] for v in o[2:]))
        n[o[1]] = ("t", j)
    return (tuple(s for _, s in sp["ins"]) if slots else len(sp["ins"]), tuple(ops),
            tuple((n[v][1], s) if slots else n[v][1] for v, s in sp["outs"]))


def selfcheck(m, nslot=None, seed=0, log=print, **policy):
    """Every blk of the Machine's MIR (a seg's blks too): lower, evaluate both forms on random tiles, compare every
    store bitwise. Returns the counters; per tile-it numbers weigh a block by its passes as tools/blkprice.py does."""
    from collections import Counter
    nslot = nslot or m.nslot
    rng = np.random.default_rng(seed)
    info, c, w = {}, Counter(), Counter()
    keys, keys0 = Counter(), Counter()
    for ins in m.mir:
        if ins[0] not in ("blk", "seg"):
            continue
        blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"]
        bt = m.tiles(blks)
        for b in blks:
            vec = any(u[0] == "ldr" or (u[0] in ("ld", "lds", "rvf") and u[2] != "__anchor__" and m.is_vec(u[2]))
                      for u in b[1]) or any(m.is_vec(x) for _, x in b[2])
            passes = bt if (ins[0] == "seg" or vec) else 1
            items, stores2, bad = check_block(b[1], b[2], nslot, rng, info=info, **policy)
            c["blocks"] += 1
            c["mismatching blocks"] += bad > 0
            c["mismatching stores"] += bad
            c["stores"] += len(b[2])
            if bad and c["mismatching blocks"] <= 3:
                log("MISMATCH", b[1], b[2], "->", items, stores2)
            w["block tiles"] += passes
            w["cpd before"] += passes * sum(u[0] == "cpd" for u in b[1])
            w["cpd after"] += passes * sum(u[0] == "cpd" for u in items)
            w["fill before"] += passes * sum(u[0] == "fill" for u in b[1])
            w["fill after"] += passes * sum(u[0] == "fill" for u in items)
            w["sfpu words before"] += passes * sum(u[0] in ("op", "un", "fold", "fill", "cpd") for u in b[1])
            w["sfpu words after"] += passes * sum(u[0] in ("op", "un", "fold", "fill", "cpd", "body") for u in items)
            for u in items:
                if u[0] == "body":
                    w["bodies"] += passes
                    w["body ops"] += passes * len(u[1]["ops"])
                    w["body ins"] += passes * len(u[1]["ins"])
                    w["body outs"] += passes * len(u[1]["outs"])
                    w["body consts"] += passes * len(u[1]["const"])
                    keys[body_key(u[1])] += passes
                    keys0[body_key(u[1], False)] += passes
                    si = dict(u[1]["ins"])
                    w["body outs in place"] += passes * sum(s in si.values() for _, s in u[1]["outs"])
                    w["body hazards"] += passes * len(u[1].get("hazard", ()))
                    assert len(set(si.values())) == len(si) and len({s for _, s in u[1]["outs"]}) == len(u[1]["outs"])
                    assert all(0 <= s < nslot for s in list(si.values()) + [s for _, s in u[1]["outs"]])
                elif u[0] in ("op", "un", "fold"):
                    w["stock ops after"] += passes
    T = m.T
    log("T %d nslot %d (machine %d) policy %s" % (T, nslot, m.nslot, policy))
    log("blocks %d: lowered %d (bodies %d), without a body %d, fallbacks %d %s, rejected bodies %d" % (
        c["blocks"], info.get("lowered", 0), info.get("body", 0), info.get("nobody", 0), info.get("fallback", 0),
        {k: v for k, v in info.items() if k.startswith("fallback:")}, info.get("rejected bodies", 0)))
    log("stores %d, mismatching stores %d in %d blocks; stock results placed unlike the original %d (static sites)" % (
        c["stores"], c["mismatching stores"], c["mismatching blocks"], info.get("stock results placed unlike the original", 0)))
    log("per tile-it: " + ", ".join("%s %.1f" % (k, v / T) for k, v in sorted(w.items())))
    log("distinct bodies: %d with their Dest slots, %d without; the 40 commonest with slots cover %.1f %% of body tiles" % (
        len(keys), len(keys0), 100.0 * sum(v for _, v in keys.most_common(40)) / max(sum(keys.values()), 1)))
    return dict(c=dict(c), info=info, per_tile_it={k: v / T for k, v in w.items()}, distinct=len(keys), distinct_noslot=len(keys0))


if __name__ == "__main__":
    import argparse
    import io

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--nslot", type=int)
    ap.add_argument("--no-hoist", action="store_true")
    ap.add_argument("--keep", action="store_true", help="a load stays in the slot the original uop named when it is free")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import cloudpickle
    f = open(a.path, "rb")
    if a.path.endswith(".zst"):
        import zstandard
        f = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True, closefd=True), 1 << 24)
    with f:
        LOWER, prog, m = cloudpickle.load(f)
    print(a.path.split("/")[-1])
    r = selfcheck(m, a.nslot, a.seed, hoist=not a.no_hoist, keep=a.keep)
    raise SystemExit(1 if r["c"].get("mismatching stores") else 0)
