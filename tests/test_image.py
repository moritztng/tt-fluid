"""Per-chip images: the sparse read of a chip array and the state round trip (host only, no device)."""
import numpy as np

from ttfluid.lower.image import GAP, LANES, Image, _ChipFiles
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


def lay(sg, k, W, stage):
    """kernels/place.cpp on the host: what W workers leave in a chip's array, from the map and the arguments they get."""
    m, args = sg.work(k, W)
    out = np.full(sg.full, 0xDEADBEEF, np.uint32)        # whatever the DRAM held before
    for w, first, count, staged in args:
        for j in range(count):
            out[first + j] = stage[staged] if m[w, j >> 5] >> (j & 31) & 1 else 0
            staged += int(m[w, j >> 5] >> (j & 31) & 1)
    return out


def test_staged_upload_lays_the_array_a_full_upload_writes(tmp_path):
    from ttfluid.lower import image_dev
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
        sg = image_dev.Staged([0], full.shape, fill, lambda k: image_dev.chip_pages(img, k, P, "home", st))
        assert sg.sparse and 9 <= sg.shape[0] < len(full) // 20      # a row lies across two blocks of its file
        stage = np.zeros(sg.shape, np.uint32)
        sg.fill(0, stage)
        for W in (2 * P, 7, 1):
            assert np.array_equal(lay(sg, 0, W, stage), full)
        on = sg.on[0].copy()                             # the gate can fail: one staged page marked zero
        sg.on[0][np.flatnonzero(on)[3]] = False
        assert not np.array_equal(lay(sg, 0, 2 * P, stage), full)
