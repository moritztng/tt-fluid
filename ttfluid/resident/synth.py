"""A small traced program over a real mesh's cells that uses every machine feature the solver does: a neighbour
gather across cores (halo), a streamed constant, every unary and comparison, where, a global sum, outputs and a
data-dependent loop. It is the first thing the device VM runs.

    prog, mesh = synth(case, iters)
"""
import numpy as np

from ttfluid.resident.program import Program
from ttfluid.resident.trace import SCALAR, Space, Tracer


def synth(solver, iters=5):
    m, ops = solver.mesh, solver.ops
    tr = Tracer(np.float32)
    n = m.nCells
    cells = Space("cells", n)
    rng = np.random.default_rng(0)
    x = tr.reg(cells)
    tr.code.append(("state", x, rng.uniform(-1, 1, n).astype(np.float32)))
    w = tr.const(rng.uniform(0.1, 0.4, n), cells)
    it = tr.reg(SCALAR)
    tr.copy(it, tr.imm(0.0))
    imm = tr.imm
    L, END = tr.new_label(), tr.new_label()
    tr.label(L)
    nb = tr.gather(cells, [(x, np.arange(n), ops.cnbr[:, 0])])
    y = tr.ew("add", tr.ew("mul", x, imm(0.5)), tr.ew("mul", nb, w))
    pos = tr.ew("gt", y, imm(0.0))
    a = tr.ew("where", pos, tr.ew("sqrt", tr.ew("abs", y)), tr.ew("neg", y))
    b = tr.ew("exp", tr.ew("mul", tr.ew("log", tr.ew("add", tr.ew("abs", y), imm(1.0))), imm(0.5)))
    c = tr.ew("tanh", tr.ew("mul", y, imm(0.3)))
    d = tr.ew("max", tr.ew("min", y, imm(0.25)), imm(-0.25))
    e = tr.ew("add", tr.ew("ge", y, imm(0.1)), tr.ew("le", y, imm(-0.1)))
    f = tr.ew("add", tr.ew("lt", y, imm(0.0)), tr.ew("ne", tr.ew("eq", y, imm(0.0)), imm(0.0)))
    z = tr.ew("add", tr.ew("add", tr.ew("mul", a, imm(0.01)), tr.ew("mul", b, imm(0.01))),
              tr.ew("add", tr.ew("mul", c, imm(0.01)), tr.ew("mul", d, imm(0.01))))
    z = tr.ew("add", tr.ew("add", z, tr.ew("mul", tr.ew("sub", e, f), imm(0.001))), y)
    s = tr.reduce("sum", tr.ew("mul", z, z))
    xn = tr.ew("div", z, tr.ew("sqrt", tr.ew("div", s, imm(float(n)))))
    tr.out("s", s)
    it2 = tr.ew("add", it, imm(1.0))
    tr.copy(it, it2)
    tr.copy(x, xn)
    tr.out("outer", it)
    tr.jumpif(tr.ew("lt", it, imm(float(iters))), L)
    tr.label(END)
    tr.code.append(("keep", [x]))
    return Program(tr, cells, {}, [("x", np.array(x, object))], ["s", "outer"], [])
