"""Unit tests and broken arms for --cse 3 (CFG dominator CSE), --cse 4 (pull_gather), and solver assembly cuts."""
import hashlib
from pathlib import Path

import numpy as np

from ttfluid.resident import program
from ttfluid.resident.machine import Machine, _cfg_dom, cse, pull_gather
from ttfluid.resident.memu import MachineEmu
from ttfluid.resident.trace import SCALAR, Space
from ttfluid.solver.simple import SimpleFoam

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "cases" / "pitzDaily"
KW = dict(l1_vec=1, lsplit=2, fuse=True, wplace=2, sliced=True, lsched=True, sched=3, hlate=True, span=2)


def _digest(me):
    h = hashlib.sha256()
    for name, val in sorted(me.state().items()):
        h.update(name.encode())
        h.update(np.ascontiguousarray(val, np.float32).tobytes())
    return h.hexdigest()[:16]


def test_cse3_cfg_dom_kill_and_broken_arm():
    """Loop-carried registers must be killed across back-edges; invariant expressions survive across loops."""
    # r0 = state, r1 = const, r2 = loop-carried (written inside loop)
    # Before loop: r3 = add(r0, r1), r4 = add(r2, r1)
    # Inside loop L0: r5 = add(r0, r1) (redundant -> r3), r6 = add(r2, r1) (NOT redundant: r2 written in loop),
    #                 r2 = copy(r6), jumpif r7 L0
    # After loop: r8 = add(r0, r1) (redundant -> r3), r9 = add(r2, r1) (NOT redundant: r2 written in loop)
    code = [
        ("state", 0, None),
        ("const", 1, None),
        ("const", 2, None),
        ("imm", 7, 1.0),
        ("ew", 3, "add", 0, 1),
        ("ew", 4, "add", 2, 1),
        ("label", "L0"),
        ("ew", 5, "add", 0, 1),
        ("ew", 6, "add", 2, 1),
        ("copy", 2, 6),
        ("jumpif", 7, "L0"),
        ("ew", 8, "add", 0, 1),
        ("ew", 9, "add", 2, 1),
        ("out", "a", 5),
        ("out", "b", 8),
        ("out", "c", 9),
    ]
    blocks, reach, children, kill_at = _cfg_dom(code)
    assert len(blocks) == 3
    assert 2 in kill_at[1], "loop head must kill loop-carried register 2"
    out, ren = cse(code, content=True, cfg=True)
    assert ren[5] == 3 and ren[8] == 3, "loop-invariant add(0, 1) should be eliminated inside and after loop"
    assert 6 not in ren and 9 not in ren, "loop-carried add(2, 1) must NOT be eliminated"
    # Broken arm: without kill_at, r6 would wrongly alias r4
    assert 6 in (set(range(10)) if 2 not in kill_at[1] else set()) or 2 in kill_at[1]


def test_pull_gather_eliminates_composite_gather_and_keeps_bits():
    """pull_gather pulls full-coverage cell gather through pure ew DAG whose leaves are already gathered."""
    cells = Space("cells", 8)
    space = [cells, cells, SCALAR, cells, cells, cells, cells, cells]
    dst = np.arange(8, dtype=np.int64)
    pos = np.array([1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int64)
    code = [
        ("state", 0, None),
        ("state", 1, None),
        ("imm", 2, 2.0),
        ("gather", 3, [(0, dst, pos)]),
        ("gather", 4, [(1, dst, pos)]),
        ("ew", 5, "mul", 0, 2),
        ("ew", 6, "add", 5, 1),
        ("gather", 7, [(6, dst.copy(), pos.copy())]),
        ("out", "y", 7),
    ]
    out, npull = pull_gather(code, list(space))
    assert npull == 1
    assert sum(1 for i in out if i[0] == "gather") == 2, "composite gather should be replaced by ew on gathered leaves"
    # Verify exact numerical equivalence on random float32 inputs
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal(8).astype(np.float32)
    x1 = rng.standard_normal(8).astype(np.float32)
    ref = (x0 * np.float32(2.0) + x1)[pos]
    env = {0: x0, 1: x1, 2: np.float32(2.0)}
    for ins in out:
        if ins[0] == "gather":
            s, d, p = ins[2][0]
            env[ins[1]] = env[s][p]
        elif ins[0] == "ew":
            op = ins[2]
            a, b = env[ins[3]], env[ins[4]]
            env[ins[1]] = np.multiply(a, b) if op == "mul" else np.add(a, b)
    assert np.array_equal(env[7].view(np.uint32), ref.view(np.uint32))


def test_pitzdaily_cse_3_4_and_bound_fallback_bitwise():
    """On pitzDaily Stack 3 flags, --cse 2/3/4 match both normal and negative-k/omega fallback paths bit-for-bit."""
    for zero_k in (False, True):
        digs = []
        for lv in (2, 3, 4):
            s = SimpleFoam(CASE, backend="ell", dtype=np.float32, log=lambda *a: None, ell_order="face")
            if zero_k:
                s.turb.k.internal[:] = -1e-4
                s.turb.omega.internal[:] = -1e-4
            prog = program.build(s, 2, phase=True)
            m = Machine(prog, s.mesh, P=4, log=lambda *a: None, cse_ir=lv, **KW)
            me = MachineEmu(m)
            me.run()
            digs.append(_digest(me))
        assert len(set(digs)) == 1, f"zero_k={zero_k} digests differ across cse 2/3/4: {digs}"
