"""Lever 27 Phase 1 (--deadpass, dense split-gather lds skip, coded-only --kfold C_LDK):
on pitzDaily (T=6, t_full=5 on 2 cores) every optimization is bitwise-exact against the control, passes DevSim and
tabstream check on 1 chip and on 2 chips (--mesh 1,2), and both broken arms (--deadpass-broken 1: flipped live bitmap
bit on core 0; --deadpass-broken 2: dropped tail cell_mask) fail the state compare."""
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ttfluid.resident.deadpass import dp_live_core, dp_live_vec, dp_pack_core

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("--arch bh --layout rcb --l1 1 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream --lsched --phase "
         "--sliced --fuse --cse 1 --hlate 1 --strips --cores 2").split()


def run(ref, *extra, mesh="1,1", check_rc=True):
    cmd = [sys.executable, "-m", "ttfluid.device.run_mesh", "case", "2", "--case", str(ROOT / "cases" / "pitzDaily"),
           *FLAGS, "--mesh", mesh, *extra, "--emuonly", "--min-free-gb", "0"]
    if ref is not None:
        cmd += ["--ref", str(ref)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if check_rc:
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    if ref is None or not Path(ref).exists():
        return r.returncode, r.stdout, r.stderr, None, None
    with open(ref, "rb") as f:
        hist, state = pickle.load(f)
    return r.returncode, r.stdout, r.stderr, hist, {k: np.asarray(v).tobytes() for k, v in state.items()}


def test_dp_pack_and_decode_roundtrip():
    # Empty table
    w0 = dp_pack_core([], 1)
    assert int(w0[0]) == (0 | (1 << 16))
    # Multi-word bitmasks (nw=3, T=78 like Small k1), uint8 palette
    masks = [0, 1, (1 << 77) | (1 << 33) | 1, (1 << 78) - 1, 1, (1 << 32) | (1 << 64)]
    w = dp_pack_core(masks, 78)
    assert (int(w[0]) >> 31) == 0
    for idx, mb in enumerate(masks):
        dpid = idx + 1
        for t in (0, 1, 31, 32, 33, 63, 64, 77):
            assert dp_live_core(w, dpid, t) == bool((mb >> t) & 1)
    # Vectorized decode across 2 cores with different tables
    w2 = dp_pack_core([mb ^ 1 for mb in masks], 78)
    L = max(len(w), len(w2))
    dp_arr = np.zeros((2, L), np.uint32)
    dp_arr[0, :len(w)] = w
    dp_arr[1, :len(w2)] = w2
    for idx, mb in enumerate(masks):
        for t in (0, 1, 32, 77):
            got = dp_live_vec(dp_arr, idx + 1, t)
            assert got.tolist() == [bool((mb >> t) & 1), bool(((mb ^ 1) >> t) & 1)]
    # >256 distinct palette entries triggers uint16 index mode
    big = [i for i in range(300)]
    wb = dp_pack_core(big, 10)
    assert (int(wb[0]) >> 31) == 1
    for idx in (0, 1, 255, 256, 299):
        for t in range(10):
            assert dp_live_core(wb, idx + 1, t) == bool((idx >> t) & 1)


def test_deadpass_bitwise_and_broken_arms(tmp_path):
    _, _, _, hist, state = run(tmp_path / "ctl")
    # Phase 1: --deadpass 1 + --tabdense 1 + --span 2 + --kfold 1 + --devsim + --tscheck
    _, out, _, h, s = run(tmp_path / "dp", "--deadpass", "1", "--tabdense", "1", "--span", "2",
                          "--kfold", "1", "--tscheck", "--devsim", "--emu-check")
    m = re.search(r"deadpass: table at arena tile \d+ \((\d+) tile, max \d+ words, nw \d+\), (\d+) masked loops, (\d+)/(\d+) dead tile-passes, (\d+) cell_mask splits", out)
    assert m, out[-2000:]
    assert int(m.group(1)) == 1, "expected DP table to fit in 1 tile on pitzDaily"
    assert int(m.group(2)) > 0 and int(m.group(3)) > 0, "expected dead tile-passes on pitzDaily"
    assert int(m.group(5)) > 0, "expected cell_mask two-range splits on pitzDaily"
    assert "tabstream check: streamed walk == cursor walk" in out
    assert "devsim ok" in out
    assert h == hist and s == state, "Phase 1 must match control bitwise"

    # Verify Item 1 (--tabdense split-gather lds skip) and Item 4 (coded-only C_LDK)
    lm = re.search(r"tabdense: skipped (\d+) dense-overwritten split-gather lds blocks", out)
    assert lm and int(lm.group(1)) > 0, out[-2000:]
    km = re.search(r"coded regs (\d+)", out)
    assert km and int(km.group(1)) > 0, out[-2000:]

    # Broken arm 1: flip one live tile bit per dp_loop on core 0 -> passes DevSim (no deadlock) and changes state
    rc1, out1, _, _, s1 = run(tmp_path / "brk1", "--deadpass", "1", "--deadpass-broken", "1", "--devsim", check_rc=False)
    assert "BROKEN" in out1, out1[-2000:]
    assert "devsim ok" in out1, out1[-2000:]
    assert rc1 != 0 or s1 != state, "flipped live tile bit left state unchanged"

    # Broken arm 2: drop tail cell_mask -> must diverge (NaN, rc=4) or change state
    rc2, out2, _, _, s2 = run(tmp_path / "brk2", "--deadpass", "1", "--deadpass-broken", "2", check_rc=False)
    assert "BROKEN" in out2, out2[-2000:]
    assert rc2 != 0 or s2 != state, "dropped tail cell_mask left state unchanged"


@pytest.mark.parametrize("mesh", ["1,1", "1,2"])
def test_deadpass_multichip_spmd(mesh):
    _, out0, _, _, _ = run(None, mesh=mesh)
    last0 = out0.strip().splitlines()[-1]
    _, out1, _, _, _ = run(None, "--deadpass", "1", "--tabdense", "1", "--span", "2", "--kfold", "1",
                           "--tscheck", "--emu-check", mesh=mesh)
    last1 = out1.strip().splitlines()[-1]
    assert "tabstream check: streamed walk == cursor walk" in out1
    assert last1 == last0
