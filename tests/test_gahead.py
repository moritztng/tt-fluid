"""--gahead (vm_ncrisc.cpp): the reader's gather read-ahead, its own kernel text, against a direct gather on the host.

The mock NoC lands a read only at a barrier (the page is poison until then), so a page used too early, a wrong run end
or a slot evicted while a part still points at it shows as a wrong lane. Mode 2 drops the read-ahead's barrier and has
to fail: the test can fail."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def waits(r):
    return int(r.stdout.split("waits=")[1].split()[0])


@pytest.mark.skipif(shutil.which("g++") is None, reason="needs g++")
@pytest.mark.parametrize("ngc,gar", [(14, 24), (14, 3), (2, 24), (3, 8), (40, 24), (1, 4)])
def test_gahead_host(tmp_path, ngc, gar):
    src, exe = tmp_path / "h.cpp", tmp_path / "h"
    subprocess.run([sys.executable, str(ROOT / "tests" / "gahead_harness.py"), str(ROOT / "ttfluid/device/kernels/vm_ncrisc.cpp"), str(src)], check=True)
    subprocess.run(["g++", "-std=c++17", "-O1", "-w", f"-DNGCV={ngc}", f"-DGARV={gar}", "-o", str(exe), str(src)], check=True)
    run = lambda mode, seed: subprocess.run([str(exe), str(mode), str(seed), "1500"], capture_output=True, text=True)
    off, on, on2, broken = run(0, 1), run(1, 1), run(1, 2), run(2, 1)
    assert off.returncode == 0 and on.returncode == 0 and on2.returncode == 0, off.stdout + on.stdout + on2.stdout
    assert waits(on) <= waits(off)
    if ngc > 1:
        assert waits(on) < waits(off) or ngc == 2, on.stdout + off.stdout
        assert broken.returncode == 1, "the read-ahead without its barrier passed: " + broken.stdout
