"""A run from images resolves `--bpatch auto` like the lowering would: the lowering records the local residue counts
in the image, and the device set-up hands them to run_mesh. Images lowered before that keep K = 0."""
import subprocess
import sys
from pathlib import Path

import pytest

from ttfluid.lower import image_dev
from ttfluid.lower.image import Image
from ttfluid.resident.tabstream import bpatch_auto

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("--arch bh --mesh 1,2 --layout rcb --l1 0 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream --amg --amg-stack "
         "--amg-vec --amg-smoothed 0 --amg-depth 0 --lsched --phase --sliced --fuse --cse 1 --hlate 1 --strips --cores 2").split()


def test_bpatch_auto_threshold():
    assert bpatch_auto([6400, 100], 64.0) == (4, 64.0)
    assert bpatch_auto([6399, 100], 64.0)[0] == 0
    assert bpatch_auto([0, 0], 64.0) == (0, 0.0)
    assert bpatch_auto(None, 64.0) == (0, 0.0)       # no streamed tables, or images without the counts


def test_images_carry_the_counts(tmp_path, monkeypatch):
    d = tmp_path / "img"
    r = subprocess.run([sys.executable, "-m", "ttfluid.device.run_mesh", "case", "3", "--case", str(ROOT / "cases" / "pitzDaily"), *FLAGS,
                        "--emuonly", "--images", str(d), "--min-free-gb", "0"], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    img = Image.load(d)
    loc = img.shared["info"]["loc"]
    assert len(loc) == 2 and all(isinstance(x, int) for x in loc) and 0 < loc[1] <= loc[0]
    lower = dict(img.shared["flags"])
    E = image_dev.setup(d, lower, img.s["K"], img.s["P"], 1, 2)[2]
    assert list(E.tab_stream["loc"]) == loc          # what run_mesh hands to bpatch_auto on a run from the images
    assert bpatch_auto(E.tab_stream["loc"], 64.0)[0] == (4 if loc[0] >= 64 * loc[1] else 0)

    del img.shared["info"]["loc"]                    # images lowered before the counts were recorded
    monkeypatch.setattr(Image, "load", staticmethod(lambda _d: img))
    E = image_dev.setup(d, lower, img.s["K"], img.s["P"], 1, 2)[2]
    assert E.tab_stream is True and not isinstance(E.tab_stream, dict)
    with pytest.raises(TypeError):
        E.tab_stream["loc"]
