"""Reference executor of a traced program: numpy fp32 over global vectors. It defines what the device must compute;
the device VM is graded against it and against the solver it was traced from."""
from collections import Counter

import numpy as np

from .trace import SCALAR

EW = {
    "add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "max": np.maximum, "min": np.minimum,
    "pow": np.power, "neg": np.negative, "abs": np.abs, "sqrt": np.sqrt, "tanh": np.tanh, "log": np.log,
    "exp": np.exp,
    "gt": np.greater, "ge": np.greater_equal, "lt": np.less, "le": np.less_equal, "eq": np.equal,
    "ne": np.not_equal, "where": lambda c, a, b: np.where(c != 0, a, b),
}


class Emulator:
    def __init__(self, prog):
        self.p = prog
        self.F = F = prog.dtype.type
        self.labels = {ins[1]: i for i, ins in enumerate(prog.code) if ins[0] == "label"}
        self.regs = [None] * len(prog.space)
        self.history = []
        self.counts = Counter()
        self.record = None
        self.record_from = 0      # reg -> first value written (debugging a lowering)

    def _new(self, r):
        sp, F = self.p.space[r], self.F
        return np.zeros((), F) if sp is SCALAR else np.zeros(sp.n, F)

    def run(self, max_steps=None, log=None):
        code, regs, labels, F = self.p.code, self.regs, self.labels, self.F
        pc, row = 0, {}
        steps = 0
        with np.errstate(all="ignore"):
            while pc < len(code):
                ins = code[pc]
                op = ins[0]
                self.counts[op if op != "ew" else "ew:" + ins[2]] += 1
                pc += 1
                if op == "ew":
                    regs[ins[1]] = F(EW[ins[2]](*[regs[a] for a in ins[3:]])) if self.p.space[ins[1]] is SCALAR \
                        else EW[ins[2]](*[regs[a] for a in ins[3:]]).astype(F, copy=False)
                elif op == "gather":
                    out = self._new(ins[1])
                    for src, dst, pos in ins[2]:
                        v = regs[src]
                        out[dst] = v if v.ndim == 0 else v[pos]
                    regs[ins[1]] = out
                elif op == "bsum":
                    base, S, _ = self.p.space[ins[2]].stack
                    regs[ins[1]] = regs[ins[2]].reshape(S, base.n).sum(axis=0, dtype=F) if S > 1 else regs[ins[2]].copy()
                elif op == "reduce":
                    a = regs[ins[3]]
                    regs[ins[1]] = F({"sum": np.sum, "max": np.max, "min": np.min}[ins[2]](a))
                elif op == "copy":
                    regs[ins[1]] = regs[ins[2]].copy()
                if self.record is not None and len(self.history) >= self.record_from and op in ("ew", "gather", "bsum", "reduce", "copy") and ins[1] not in self.record:
                    self.record[ins[1]] = np.copy(regs[ins[1]])
                if op in ("ew", "gather", "bsum", "reduce", "copy"):
                    pass
                elif op in ("const", "state"):
                    regs[ins[1]] = ins[2].astype(F).copy()
                elif op == "imm":
                    regs[ins[1]] = F(ins[2])
                elif op == "jumpif":
                    if regs[ins[1]] != 0:
                        pc = labels[ins[2]]
                elif op == "jump":
                    pc = labels[ins[1]]
                elif op == "out":
                    row[ins[1]] = float(regs[ins[2]])
                    if ins[1] == "outer":
                        self.history.append(row)
                        if log:
                            log(row)
                        row = {}
                        steps += 1
                        if max_steps and steps >= max_steps:
                            return self.history
        return self.history

    def state(self):
        """name -> fp32 array (n,) + trailing, the current values of the state registers."""
        out = {}
        for name, c in self.p.state:
            arrs = [self.regs[r] for r in c.ravel()]
            a = np.stack(arrs, -1).reshape(arrs[0].shape + c.shape) if c.ndim else arrs[0]
            out[name] = a
        return out
