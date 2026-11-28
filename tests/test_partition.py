"""The chip-level cut of a mesh: the strips of a line of chips share faces with their neighbours only."""
import numpy as np

from ttfluid.resident import chipplan
from ttfluid.resident.machine import rcb


def _line(n, extra=()):
    """n cells in a row, each sharing a face with the next, plus extra (cell, cell) faces."""
    xyz = np.stack([np.arange(n, dtype=float), np.zeros(n), np.zeros(n)], 1)
    fo = np.array(list(range(n - 1)) + [a for a, _ in extra])
    fn = np.array(list(range(1, n)) + [b for _, b in extra])
    return xyz, (fo, fn)


def test_strips_touch_only_their_neighbours():
    # cell 5 (strip 0) shares a face with cell 25 (strip 2): a coarse cell beside a strip thinner than itself
    xyz, faces = _line(80, extra=[(5, 25)])
    chip = rcb(xyz, 8, chips=8, faces=faces)   # one core per chip: the part is the chip
    assert np.abs(chip[faces[0]] - chip[faces[1]]).max() == 1
    assert chip[5] == 1 and np.bincount(chip, minlength=8).min() >= 9
    for R, C in ((1, 8), (2, 4)):   # a line of chips, and a Galaxy tray's 2x4 ladder
        chipplan.place_blocks(chipplan.chip_adjacency(chip, faces[0], faces[1], 8), 8, R, C)


def test_strips_are_the_equal_cut_when_no_face_reaches_across():
    xyz, faces = _line(80)
    assert np.array_equal(rcb(xyz, 8, chips=8, faces=faces), np.arange(80) // 10)
