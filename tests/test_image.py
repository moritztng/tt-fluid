"""Per-chip images: the sparse read of a chip array and the state round trip (host only, no device)."""
import time

import numpy as np

from ttfluid.lower import image_dev
from ttfluid.lower.image import GAP, LANES, Image, Pages, _ChipFiles
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
        pages = Pages(collect(X, m, on, W, S), np.where(on, np.cumsum(on) - 1, S).reshape(P, hp))
        back = img.state(lambda k: A, lambda k: pages.view(np.float32))
        assert not np.array_equal(back["U"], state["U"])
