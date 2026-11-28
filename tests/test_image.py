"""Per-chip images: the sparse read of a chip array and the state round trip (host only, no device)."""
import time

import numpy as np

from ttfluid.lower import image_dev
from ttfluid.lower.image import GAP, LANES, Image, Pages, _ChipFiles, load_state, par_zero
from ttfluid.resident.machine import DBASE


def test_read_into_reads_what_the_file_stores(tmp_path):
    rng = np.random.default_rng(0)
    a = np.zeros((2048, LANES), np.float32)            # 8 MB: holes far wider than GAP between the data
    for row in (0, 1, 2, 5, 40, 1000, 1001 + GAP // 4096, 2047):   # neighbours, gaps under GAP, one over it
        a[row] = rng.standard_normal(LANES)
    Image({"scalars": {"K": 1}}, {0: {"home": a}}).write_chip(tmp_path, 0)
    out = np.zeros(a.shape, np.uint32)
    _ChipFiles(tmp_path / "chip0").read_into("home", out)
    assert np.array_equal(out, a.view(np.uint32))


def test_put_state_is_the_inverse_of_state():
    rng = np.random.default_rng(1)
    P, n, nt, hp = 3, 5000, 4, 6                         # cores, cells, arena tiles and home pages per core
    core = rng.integers(0, P, n).astype(np.int32)
    lane = np.zeros(n, np.int32)
    for c in range(P):                                   # a core's cells in a shuffled order of its lanes
        m = np.flatnonzero(core == c)
        lane[m] = rng.permutation(2 * LANES)[:len(m)]
    regs = [["cells", 1, 2], ["cells", DBASE + 2, 2], ["cells", DBASE + 4, 2]]   # one arena-homed, two DRAM-homed
    shared = dict(scalars=dict(K=1, P=P), spaces={"cells": n}, const_state={},
                  state=[["U", [3], regs], ["p", [], [["cells", DBASE, 2]]]])
    img = Image(shared, {0: {"ids/cells": np.arange(n), "core/cells": core, "lane/cells": lane}})
    state = {"U": rng.standard_normal((n, 3)).astype(np.float32), "p": rng.standard_normal(n).astype(np.float32)}
    A, H = np.zeros((P, nt, LANES), np.float32), np.zeros((P, hp, LANES), np.float32)
    img.put_state(0, A, H, state)
    back = img.state(lambda k: A, lambda k: H)
    assert all(np.array_equal(back[k], state[k]) for k in state)
    assert np.count_nonzero(A) + np.count_nonzero(H) == 4 * n


def lay(sg, k, W):
    """upload_placed on the host: what W workers per part (kernels/place.cpp) leave in a chip's array, from the staged
    pages, the maps and the arguments they get."""
    out = np.full(sg.full, 0xDEADBEEF, np.uint32)        # whatever the DRAM held before
    for p in (sg.part(j) for j in range(sg.parts)):
        stage = np.zeros(p.shape, np.uint32)
        p.fill(k, stage)
        m, args = image_dev.page_map(p.on(k), W, p.size)
        for w, first, count, staged in args:
            for j in range(count):
                out[p.first + first + j] = stage[staged] if m[w, j >> 5] >> (j & 31) & 1 else 0
                staged += int(m[w, j >> 5] >> (j & 31) & 1)
    return out


def test_staged_upload_lays_the_array_a_full_upload_writes(tmp_path):
    rng = np.random.default_rng(2)
    P, hp, n = 3, 700, 5000                              # cores, home pages per core, cells
    home = np.zeros((P * hp, LANES), np.float32)
    for row in (0, 1, 2, 5, 699, 700, 1400, 1401 + GAP // 4096, P * hp - 1):
        home[row] = rng.standard_normal(LANES)
    core = rng.integers(0, P, n).astype(np.int32)
    lane = np.zeros(n, np.int32)
    for c in range(P):
        m = np.flatnonzero(core == c)
        lane[m] = rng.permutation(2 * LANES)[:len(m)]
    regs = [["cells", DBASE + 1, 2], ["cells", DBASE + 300, 2], ["cells", DBASE + 698, 2]]   # over stored pages and holes
    shared = dict(scalars=dict(K=1, P=P, ntiles=4, tab_stream=False), spaces={"cells": n}, const_state={},
                  state=[["U", [3], regs], ["p", [], [["cells", DBASE + 40, 2]]]])
    Image(shared, {0: {"home": home, "ids/cells": np.arange(n), "core/cells": core, "lane/cells": lane}}).write_chip(tmp_path, 0)
    img = Image(shared, {0: tmp_path / "chip0"})
    state = {"U": rng.standard_normal((n, 3)).astype(np.float32), "p": rng.standard_normal(n).astype(np.float32)}
    for st in (None, state):
        fill = lambda k, out, row=None: image_dev.chip_fill(img, k, P, "home", out, st, row)
        full = np.zeros(home.shape, np.uint32)
        fill(0, full)
        assert st is None or not np.array_equal(full, home.view(np.uint32))
        pages = lambda k: image_dev.chip_pages(img, k, P, "home", st)
        sg = image_dev.Staged([0], full.shape, fill, pages).mark()
        assert sg.sparse and sg.parts == 1 and 9 <= sg.shape[0] < len(full) // 20      # a row lies across two blocks of its file
        for cap in (None, 10 * 4096, 3 * 4096, 4096):    # one part; parts of at most 10 and 3 staged pages; as many as there may be
            sg = image_dev.Staged([0], full.shape, fill, pages, cap).mark()
            assert sg.parts == 1 if cap is None else sg.parts > 1 and (sg.parts == 64 or sg.rows.max() * 4096 <= cap)
            for W in (2 * P, 7, 1):
                assert np.array_equal(lay(sg, 0, W), full)
            ahead = image_dev.Ahead(sg)                  # the parts' host buffers, in order, two in hand at most
            for j, p in enumerate(sg.pieces()):
                stage = np.zeros(p.shape, np.uint32)
                p.fill(0, stage)
                assert np.array_equal(ahead.next(), stage)
                if j + 2 < sg.parts:                     # part j + 1 is built, part j + 2 waits for done()
                    while not ahead.built.qsize():
                        time.sleep(0.001)
                    time.sleep(0.02)
                    assert ahead.built.qsize() == 1
                ahead.done()
            sg.on[0][1] = False                          # the gate can fail: a page that holds values marked zero
            assert not np.array_equal(lay(sg, 0, 2 * P), full)


def scan(X, W, HPG, ranges):
    """kernels/scan.cpp on the host: the page map W workers write for a chip's homes X (pages, LANES)."""
    m, args = image_dev.page_map(np.zeros(len(X), bool), W)
    for w, first, count, _ in args:
        c = first // HPG
        while c * HPG < first + count:
            for lo, hi in ranges:
                for p in range(max(first, c * HPG + lo), min(first + count, c * HPG + hi)):
                    if X[p].any():
                        m[w, (p - first) >> 5] |= np.uint32(1 << ((p - first) & 31))
            c += 1
    return m


def collect(X, m, on, W, S):
    """kernels/place.cpp backwards on the host: the stage of S + 1 rows the workers fill from the marked pages."""
    stage = np.full((S + 1, LANES), 0xDEADBEEF, np.uint32)
    for w, first, count, staged in image_dev.page_map(on, W)[1]:
        for j in range(count):
            if m[w, j >> 5] >> (j & 31) & 1:
                stage[staged] = X[first + j]
                staged += 1
    stage[S] = 0
    return stage


def test_state_download_reads_only_the_pages_that_hold_a_value():
    rng = np.random.default_rng(2)
    P, n, nt, hp = 3, 5000, 4, 8
    core = rng.integers(0, P, n).astype(np.int32)
    lane = np.zeros(n, np.int32)
    for c in range(P):
        m = np.flatnonzero(core == c)
        lane[m] = rng.permutation(2 * LANES)[:len(m)]
    regs = [["cells", 1, 2], ["cells", DBASE + 2, 2], ["cells", DBASE + 4, 2]]
    shared = dict(scalars=dict(K=1, P=P), spaces={"cells": n}, const_state={},
                  state=[["U", [3], regs], ["p", [], [["cells", DBASE, 2]]]])
    img = Image(shared, {0: {"ids/cells": np.arange(n), "core/cells": core, "lane/cells": lane}})
    state = {"U": rng.standard_normal((n, 3)).astype(np.float32), "p": np.zeros(n, np.float32)}
    state["U"][:, 2] = 0                                  # two of the three DRAM-homed registers hold nothing
    A, H = np.zeros((P, nt, LANES), np.float32), np.zeros((P, hp, LANES), np.float32)
    img.put_state(0, A, H, state)
    H[:, 6:] = 7.0                                        # pages past the state: not the download's to read
    ranges = image_dev.state_pages(img)
    assert ranges == [(0, 6)]
    X = H.view(np.uint32).reshape(P * hp, LANES)
    for W in (2 * P, 7, 1):
        m = scan(X, W, hp, ranges)
        on = image_dev.map_pages(m, len(X))
        S = int(on.sum())
        assert S == 2 * P                                 # of 6 * P state pages
        row = np.where(on, np.cumsum(on) - 1, S).reshape(P, hp)
        pages = Pages(collect(X, m, on, W, S), row)
        back = img.state(lambda k: A, lambda k: pages.view(np.float32))
        assert all(np.array_equal(back[k], state[k]) for k in state)
        m[0, 0] ^= np.uint32(1 << 2)                      # a page the scan missed: the state has to differ
        on = image_dev.map_pages(m, len(X))
        S = int(on.sum())
        pages = Pages(collect(X, m, on, W, S), np.where(on, np.cumsum(on) - 1, S).reshape(P, hp), zero_row=S)
        back = img.state(lambda k: A, lambda k: pages.view(np.float32))
        assert not np.array_equal(back["U"], state["U"])


def test_load_state_mmap_and_compressed(tmp_path):
    rng = np.random.default_rng(7)
    arrays = {
        "p": rng.standard_normal(5000).astype(np.float32),
        "U": rng.standard_normal((5000, 3)).astype(np.float32),
        "phi": np.asfortranarray(rng.standard_normal((2000, 5)).astype(np.float32)),
    }
    p_raw, p_cmp = tmp_path / "raw.npz", tmp_path / "cmp.npz"
    np.savez(p_raw, **arrays, hist="[]")
    np.savez_compressed(p_cmp, **arrays, hist="[]")
    for path, expect_mmap in ((p_raw, True), (p_cmp, False)):
        got = load_state(path)
        assert set(got) == set(arrays)
        for k, v in arrays.items():
            assert np.array_equal(got[k], v)
            assert got[k].dtype == v.dtype
        if expect_mmap:
            assert not got["p"].flags.writeable


def test_contiguous_layout_roundtrip_parts_and_zero_skip(tmp_path):
    rng = np.random.default_rng(9)
    P, hp, nt = 5, 40, 4
    # Core 2 is empty; chip 1's boundary space is empty; lanes are strictly 0..n_c-1 per core (real lowering layout)
    counts0 = [1500, 2100, 0, 500, 1100]
    counts1 = [1300, 900, 1024, 0, 1400]
    n0, n1 = sum(counts0), sum(counts1)
    n_cells = n0 + n1
    perm = rng.permutation(n_cells)
    ids0, ids1 = perm[:n0], perm[n0:]

    def make_cl(counts):
        c = np.repeat(np.arange(P, dtype=np.int32), counts)
        l = np.concatenate([np.arange(cnt, dtype=np.int32) for cnt in counts])
        shuf = rng.permutation(len(c))
        return c[shuf], l[shuf]

    c0, l0 = make_cl(counts0)
    c1, l1 = make_cl(counts1)
    b_n = 77
    bc0 = np.zeros(b_n, np.int32)
    bl0 = np.arange(b_n, dtype=np.int32)
    regs_u = [["cells", 0, 3], ["cells", DBASE + 1, 3], ["cells", DBASE + 4, 3]]
    shared = dict(
        scalars=dict(K=2, P=P, ntiles=nt, tab_stream=False),
        spaces={"cells": n_cells, "inlet": b_n},
        const_state={"k/0": np.float32(0.25)},
        state=[
            ["U", [3], regs_u],
            ["p", [], [["cells", DBASE + 8, 3]]],
            ["bval", [], [["inlet", DBASE + 11, 1]]],
            ["k", [], [["cells", None, None]]],
        ],
    )
    home0 = np.zeros((P * hp, LANES), np.float32)
    home1 = np.zeros((P * hp, LANES), np.float32)
    home0[0] = rng.standard_normal(LANES)
    home1[P * hp - 1] = rng.standard_normal(LANES)
    Image(shared, {
        0: {"home": home0, "ids/cells": ids0, "core/cells": c0, "lane/cells": l0,
            "ids/inlet": np.arange(b_n), "core/inlet": bc0, "lane/inlet": bl0},
        1: {"home": home1, "ids/cells": ids1, "core/cells": c1, "lane/cells": l1,
            "ids/inlet": np.empty(0, np.int64), "core/inlet": np.empty(0, np.int32), "lane/inlet": np.empty(0, np.int32)},
    }).write(tmp_path)
    img = Image.load(tmp_path)
    assert img._layout(0, "cells")[0] is True
    assert img._layout(1, "inlet")[0] is True
    state = {
        "U": rng.standard_normal((n_cells, 3)).astype(np.float32),
        "p": np.zeros(n_cells, np.float32),
        "bval": rng.standard_normal(b_n).astype(np.float32),
        "k": np.full(n_cells, 0.25, np.float32),
    }
    state["U"][:, 2] = 0.0
    # Split across many parts so cores straddle part boundaries, and verify pooled Ahead + place match full fill
    fill = lambda k, out, row=None: image_dev.chip_fill(img, k, P, "home", out, state, row)
    pages = lambda k: image_dev.chip_pages(img, k, P, "home", state)
    full = [np.zeros((P * hp, LANES), np.uint32) for _ in range(2)]
    for k in range(2):
        fill(k, full[k])
    sg = image_dev.Staged([0, 1], (P * hp, LANES), fill, pages, cap=8 * 4096).mark()
    assert sg.sparse and sg.parts >= 4
    ahead = image_dev.Ahead(sg, alive=2)
    seen_ids = set()
    for p in sg.pieces():
        got = ahead.next()
        seen_ids.add(id(got.base if got.base is not None else got))
        ref = image_dev.host_buffer(2, [0, 1], p.shape, p.fill)
        assert np.array_equal(got, ref)
        ahead.done()
    assert len(seen_ids) <= 2
    for k in range(2):
        assert np.array_equal(lay(sg, k, 2 * P), full[k])
    # Full round trip through scan + collect + Pages(..., zero_row=S)
    A = [np.zeros((P, nt, LANES), np.float32) for _ in range(2)]
    for k in range(2):
        img.put_state(k, A[k], None, state)
    ranges = image_dev.state_pages(img)
    p_out = []
    for k in range(2):
        m = scan(full[k], 2 * P, hp, ranges)
        on = image_dev.map_pages(m, len(full[k]))
        S = int(on.sum())
        p_out.append(Pages(collect(full[k], m, on, 2 * P, S),
                           np.where(on, np.cumsum(on) - 1, S).reshape(P, hp), zero_row=S).view(np.float32))
    back = img.state(lambda k: A[k], lambda k: p_out[k])
    assert all(np.array_equal(back[k], state[k]) for k in state)
    # par_zero smoke test on > 4 MB array
    big = np.ones((2048, LANES), np.uint32)
    par_zero(big)
    assert not big.any()
