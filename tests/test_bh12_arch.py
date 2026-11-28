"""`--arch bh12` lowers for the 12x10 (120-core) Blackhole Galaxy grid, keeps Blackhole defaults (gshare 6/3, fabric
flags), passes emulator + devsim + tabstream checks, and rejects a P=120 image loaded against P=110 (`--arch bh`)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ttfluid import cli
from ttfluid.lower import image_dev
from ttfluid.lower.image import Image

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("--layout rcb --tiles 2 --l1 1 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream --lsched --phase "
         "--sliced --fuse --cse 1 --hlate 1 --strips").split()


def _run(*extra, what="case", n="2"):
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    base = [sys.executable, "-m", "ttfluid.device.run_mesh", what, str(n)]
    if what == "case":
        base += ["--case", str(ROOT / "cases" / "pitzDaily")]
    return subprocess.run(
        [*base, *FLAGS, *extra, "--emuonly", "--min-free-gb", "0"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_bh12_image_roundtrip_and_grid_mismatch_rejected(tmp_path):
    img_dir = tmp_path / "img_bh12"
    r = _run("--arch", "bh12", "--mesh", "1,1", "--tscheck", "--images", str(img_dir))
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "P 120," in r.stdout
    assert "tabstream check: streamed walk == cursor walk" in r.stdout

    img = Image.load(img_dir)
    assert img.s["P"] == 120
    assert img.s["K"] == 1

    # Loading with P=120 succeeds; loading the same P=120 image with P=110 (`--arch bh`) must fail (broken arm).
    lower = dict(img.shared["flags"])
    _, m, _, _, _, _, _, _, _ = image_dev.setup(img_dir, lower, K=1, P=120, R=1, C=1)
    assert m.T == 2
    with pytest.raises(AssertionError):
        image_dev.setup(img_dir, lower, K=1, P=110, R=1, C=1)


def test_bh12_emulates_120_cores_and_devsim_passes():
    r = _run("--arch", "bh12", "--mesh", "1,1", "--emu-check")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "P 120," in r.stdout
    last_line = r.stdout.strip().splitlines()[-1]
    assert " last " in last_line
    last = json.loads(last_line.split(" last ", 1)[1])
    assert last["outer"] == 2.0
    assert 0.0 < last["p"] < 0.2 and 0.0 < last["U"] < 0.8

    # Full protocol simulator (devsim) on 2-chip mesh with bh12 and streamed dense tables.
    ds = _run("--arch", "bh12", "--mesh", "1,2", "--cores", "2", "--tabdense", "1", "--span", "2", "--tscheck", "--devsim")
    assert ds.returncode == 0, ds.stdout[-2000:] + ds.stderr[-2000:]
    assert "tabstream check: streamed walk == cursor walk" in ds.stdout
    assert "devsim ok" in ds.stdout


def test_bh12_gshare_and_invalid_arch_rejected():
    src = (ROOT / "ttfluid" / "device" / "run_mesh.py").read_text()
    assert "bh12=((12, 10)," in src
    assert 'a.arch.startswith("bh")' in src

    bad_arch = _run("--arch", "bh99", "--mesh", "1,1")
    assert bad_arch.returncode == 2
    assert "invalid choice: 'bh99'" in bad_arch.stderr

    bad_dense = _run("--arch", "bh12", "--mesh", "1,1", "--cores", "2", "--tabdense", "1", "--span", "2",
                     "--tscheck", "--tabdense-break", "8")
    assert bad_dense.returncode != 0
    assert "dense index differs" in bad_dense.stderr


def test_cli_accepts_bh12_with_bh_fabric_flags(monkeypatch):
    seen = {}

    def fake_run_module(mod, run_name=None):
        seen["mod"] = mod
        seen["argv"] = list(sys.argv)

    monkeypatch.setattr(cli.runpy, "run_module", fake_run_module)
    cli.main(["check", str(ROOT / "cases" / "pitzDaily"), "--arch", "bh12", "--chips", "8", "-n", "2"])
    assert seen["mod"] == "ttfluid.device.run_mesh"
    argv = seen["argv"]
    assert argv[argv.index("--arch") + 1] == "bh12"
    assert argv[argv.index("--mesh") + 1] == "2,4"
    assert "--gw" in argv and "--links" in argv and "--rlink" in argv
    assert "--block2d" in argv and "--strips" in argv
    assert "--emuonly" in argv
    assert cli.GALAXY_BLOCK[8] == "0,1,2,3,4,5,6,7"

    with pytest.raises(SystemExit) as exc:
        cli.main(["check", str(ROOT / "cases" / "pitzDaily"), "--arch", "bh99"])
    assert exc.value.code == 2
