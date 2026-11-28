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
