"""Control flow, liveness and footprint of a traced program."""
from collections import Counter




PURE = ("ew", "gather", "bsum", "reduce", "copy")     # instructions whose only effect is their result register


def uses_defs(ins):
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


def successors(code):
    labels = {ins[1]: i for i, ins in enumerate(code) if ins[0] == "label"}
    succ = []
    for i, ins in enumerate(code):
        nxt = [i + 1] if i + 1 < len(code) else []
        if ins[0] == "jump":
            succ.append([labels[ins[1]]])
        elif ins[0] == "jumpif":
            succ.append(nxt + [labels[ins[2]]])
        else:
            succ.append(nxt)
    return succ


def live_outs(code, strong=False):
    """Yields (i, registers live after instruction i) for i = n-1 .. 0, at the fixed point of the backward dataflow.
    strong: an instruction without side effects whose result is dead does not make its operands live (one solve of
    what repeated dead-code elimination converges to, plus dead loop-carried cycles). Only the jump targets' live-in
    sets are stored; each yielded set is fresh, so a caller that keeps none of them holds O(targets) sets, not O(n)."""
    n = len(code)
    pure = [strong and ins[0] in PURE for ins in code]
    succ = successors(code)
    ud = [uses_defs(ins) for ins in code]
    tin = {s: set() for i, ss in enumerate(succ) for s in ss if s != i + 1}

    def sweep(emit):
        changed, nxt = False, set()
        for i in range(n - 1, -1, -1):
            out = set()
            for s in succ[i]:
                out |= nxt if s == i + 1 else tin[s]
            u, d = ud[i]
            inn = out if pure[i] and d[0] not in out else (out - set(d)) | set(u)
            if i in tin and inn != tin[i]:
                tin[i] = inn
                changed = True
            if emit:
                yield i, out
            nxt = inn
        return changed

    def run(emit):
        g = sweep(emit)
        try:
            while True:
                next(g)
        except StopIteration as e:
            return e.value
    while run(False):
        pass
    yield from sweep(True)


def liveness(code, strong=False, at=None):
    """live_out[i]: registers live after instruction i (live_outs, stored); with `at`, stored only for i in at (None
    elsewhere)."""
    live_out = [None] * len(code)
    for i, out in live_outs(code, strong):
        if at is None or i in at:
            live_out[i] = out
    return live_out


def footprint(prog):
    """Vectors held per space: constants, state, and the peak of everything live at once."""
    code, space = prog.code, prog.space
    kinds = Counter()
    for ins in code:
        if ins[0] in ("const", "state"):
            kinds[(ins[0], space[ins[1]].name)] += 1
    peak = Counter()
    for _, lo in live_outs(code):
        c = Counter(space[r].name for r in lo)
        for k, v in c.items():
            peak[k] = max(peak[k], v)
    return kinds, peak
