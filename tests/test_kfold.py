"""`--kfold` moves no value: on pitzDaily the emulated state and history are the control's bit for bit, with the fold
table coding pages on every core, and the device simulator agrees that the reader may skip them. The emulator fills a
coded page from the table words the device gets, so a table that codes a non-uniform page (`--kfold-broken 1`) changes
the state. On a device the gate is the state compare against a run without the flag."""
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("--arch bh --layout rcb --l1 1 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream --lsched --phase "
         "--sliced --fuse --cse 1 --hlate 1 --strips --cores 2 --mesh 1,1").split()


def run(ref, *extra):
    r = subprocess.run([sys.executable, "-m", "ttfluid.device.run_mesh", "case", "2", "--case", str(ROOT / "cases" / "pitzDaily"),
                        *FLAGS, *extra, "--emuonly", "--min-free-gb", "0", "--ref", str(ref)], cwd=ROOT, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    with open(ref, "rb") as f:
        hist, state = pickle.load(f)
    return r.stdout, hist, {k: np.asarray(v).tobytes() for k, v in state.items()}


def test_folded_constant_pages_keep_the_state_bitwise(tmp_path):
    _, hist, state = run(tmp_path / "ctl")
    out, h, s = run(tmp_path / "kf", "--kfold", "1", "--devsim")
    coded = re.search(r"kfold: table at arena tile \d+ \(1 tile\), (\d+) constant pages a core, coded per core min (\d+)", out)
    assert coded and int(coded.group(2)) > 0, "no page was coded on some core: the test did not exercise the fold"
    assert "devsim ok" in out
    assert h == hist and s == state
    out, h, s = run(tmp_path / "kx", "--kfold", "1", "--kfold-broken", "1")
    assert "BROKEN" in out
    assert s != state, "a non-uniform page coded as uniform left the state unchanged: the emulator does not read the table"
