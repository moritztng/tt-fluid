"""BROKEN ARM of the reorder check (tools/ordercheck.py): the lowering with --order rcm, where every table follows
the banded lane order and ONE cell field does not: its values stay laid out in the cell-id lane order, as a
permutation applied to the tables and not to that field would leave them. ordercheck must then FAIL.

    python -m ttfluid.tools.order_broken <run_mesh arguments with --order rcm --emuonly --ref OUT.ref>

The field is the first cell register with a non-uniform start value, state fields first; its name is printed."""
import runpy
import sys

import numpy as np

from ttfluid.resident import machine as M
from ttfluid.resident import memu

rec = {}
_lanes = M.sliced_lanes


def lanes(owner, counts, P, Q=0, banded=None):
    lane = _lanes(owner, counts, P, Q, banded)
    if banded is not None:
        rec.update(own=np.asarray(owner), idx=_lanes(owner, counts, P, Q), band=lane)
    return lane


_init = memu.MachineEmu.__init__


def init(self, m, dtype=None):
    assert rec, "order_broken: run it with --order rcm"
    own, idx, band = rec["own"], rec["idx"], rec["band"]
    n = len(own)
    at = np.full((m.P, int(idx.max()) + 1), -1, np.int64)
    at[own, idx] = np.arange(n)      # the cell at (core, lane) in the cell-id order
    src = at[own, band]              # whose value a banded lane holds when the field was not permuted
    assert (src >= 0).all()
    ok = lambda r: r in m.pre and m.pre[r] is not None and m.space[r] is m.prog.cells and np.shape(m.pre[r]) == (n,) and np.ptp(m.pre[r]) > 0
    named = [(nm, m.ren(r)) for nm, c in m.prog.state for r in np.asarray(c).ravel()]
    name, r = next(((nm, r) for nm, r in named if ok(r)), None) or next((("const", r) for r in sorted(m.pre) if ok(r)))
    m.pre[r] = np.asarray(m.pre[r])[src]
    print("order_broken: register %d (%s) left in the cell-id lane order, %d of %d cells hold another cell's value"
          % (r, name, int((src != np.arange(n)).sum()), n), flush=True)
    _init(self, m, dtype)


M.sliced_lanes = lanes
memu.MachineEmu.__init__ = init
sys.argv = ["run_mesh"] + sys.argv[1:]
runpy.run_module("ttfluid.device.run_mesh", run_name="__main__")
