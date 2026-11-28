"""Tests for ttfluid.tools.deadpass_census (lever 27 dead tile-pass census)."""

import numpy as np

from ttfluid.tools.deadpass_census import (
    ONE_U32,
    ZERO_NEG,
    ZERO_POS,
    analyze_region,
    run_self_tests,
)


def test_builtin_self_test():
    """All 4 built-in synthetic cases (chained folds, clobber guard, broken check, split gather)."""
    run_self_tests()


def test_dense_split_gather_kills_preceding_lds():
    """When a split gather uses a dense twin (is_dense=True), it overwrites all 1024 lanes so the prior lds is dead."""
    const_uni = np.full(4, -1, dtype=np.int64)
    # Map 0: dense twin (is_d=True) covering tile 0 & tile 1
    map_counts = {
        0: (
            np.array([1024, 1024], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            np.array([0, 0], dtype=np.int32),
            True,
        )
    }
    events = [
        (0, "solve3", "p_solve", ("blk_run", 2, [(
            [("lds", 0, ("L1", 10, True), 4)],
            [(0, ("L1", 5, True))],
        )])),
        (1, "solve3", "p_solve", ("gather", 5, 0, [(("L1", 10), None)], 0)),
        (2, "solve3", "p_solve", ("blk_run", 2, [(
            [("ld", 0, ("L1", 5, True)), ("ld", 1, ("L1", 20, True)), ("op", "mul", 2, 0, 1)],
            [(2, ("HOME", 30, True))],
        )])),
    ]
    prog = dict(
        name="dense_split",
        core=0,
        T=2,
        cells=2048,
        cpages=4,
        const_uni=const_uni,
        init_ro={},
        state_locs={("HOME", 30), ("HOME", 31)},
        map_counts=map_counts,
        events=events,
        flags={},
    )
    r = analyze_region(prog, care=False)
    row = r["rows"]["solve3:p_solve"]
    assert row["passes"] == 4
    assert row["passes_dead"] == 2  # both lds passes overwritten by dense gather


def test_ldr_home_bypass_to_const_keeps_const_count_exact():
    """Bypassing a HOME load to a CONST origin must not inflate const_pages_live above const_pages."""
    const_uni = np.array([-1, -1, ZERO_POS, ONE_U32], dtype=np.int64)
    events = [
        # Copy non-uniform const[0..1] into HOME[10..11] via mul by const[3]=1.0 (exact!)
        (0, "outer", "mom", ("blk_run", 2, [(
            [
                ("ldr", 0, ("CONST", 0), 0, None, [], False, False),
                ("ldr", 1, ("CONST", 2), 0, None, [], False, False),
                ("op", "add", 2, 0, 1),
            ],
            [(2, ("HOME", 10, True))],
        )])),
        # Read HOME[10..11] in next block -> tile 0 bypasses to CONST[0] under care=True
        (1, "outer", "mom", ("blk_run", 2, [(
            [("ldr", 0, ("HOME", 10), 0, None, [], False, False)],
            [(0, ("HOME", 20, True))],
        )])),
    ]
    prog = dict(
        name="home_bypass_const",
        core=0,
        T=2,
        cells=2048,
        cpages=4,
        const_uni=const_uni,
        init_ro={},
        state_locs={("HOME", 20), ("HOME", 21)},
        map_counts={},
        events=events,
        flags={},
    )
    r = analyze_region(prog, care=True)
    row = r["rows"]["outer:mom"]
    assert row["const_pages"] == 4
    assert row["const_pages_uniform"] == 2
    assert row["const_pages_dead"] == 3  # const[2], const[3] uniform + block-0 read of const[0] bypassed
    assert row["const_pages_nonuni_dead"] == 1
    assert row["stream_pages"] == 6 and row["stream_pages_dead"] == 3  # 3 live: const[1] + 2 reads in blk 1
    assert row["st_home_dead"] == 1  # tile 0 copy store HOME[10] eliminated!


def test_signed_zero_exact_vs_algebraic():
    """x + (+0.0) and x * 0.0 are dead only under care=True; x + (-0.0) and where(1.0, x, y) are dead even at care=False."""
    const_uni = np.array([ZERO_POS, ZERO_NEG, ONE_U32, ZERO_POS], dtype=np.int64)
    events = [
        (0, "outer", "mom", ("blk_run", 1, [(
            [
                ("ldr", 0, ("HOME", 0), 0, None, [], False, False),
                ("ldr", 1, ("CONST", 0), 0, None, [], False, False),  # +0.0
                ("ldr", 2, ("CONST", 1), 0, None, [], False, False),  # -0.0
                ("ldr", 3, ("CONST", 2), 0, None, [], False, False),  # 1.0
                ("op", "add", 4, 0, 1),   # x + (+0.0): algebraic only
                ("op", "add", 5, 4, 2),   # + (-0.0): exact!
                ("op", "where", 6, 3, 5, 1),  # where(1.0, ..., +0.0): exact!
            ],
            [(6, ("HOME", 10, True))],
        )])),
    ]
    prog = dict(
        name="signed_zero",
        core=0,
        T=1,
        cells=1024,
        cpages=4,
        const_uni=const_uni,
        init_ro={},
        state_locs={("HOME", 10)},
        map_counts={},
        events=events,
        flags={},
    )
    r_ex = analyze_region(prog, care=False)
    r_al = analyze_region(prog, care=True)
    assert r_ex["rows"]["outer:mom"]["op_tiles_dead"] == 2  # add(-0.0) + where(1.0)
    assert r_al["rows"]["outer:mom"]["op_tiles_dead"] == 3  # + add(+0.0)
