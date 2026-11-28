"""`--tabdense` and `--span` move no value: on pitzDaily the emulator's last row is the control's, on one chip and on
two, the streamed-table check (`--tscheck`) decodes every dense table back to its map, and a table with swapped lanes
(`--tabdense-break`) fails that check. The emulator does not read the streamed tables, so `--tscheck` is the CPU gate
for a dense table's content; on a device the gate is the state compare against a run without the flags."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("--arch bh --layout rcb --l1 1 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream --lsched --phase "
         "--sliced --fuse --cse 1 --hlate 1 --strips --cores 2").split()
DENSE = "--tabdense 1 --span 2 --tscheck".split()


def run(*extra):
    return subprocess.run([sys.executable, "-m", "ttfluid.device.run_mesh", "case", "2", "--case", str(ROOT / "cases" / "pitzDaily"),
                           *FLAGS, *extra, "--emuonly", "--min-free-gb", "0"], cwd=ROOT, capture_output=True, text=True, timeout=900)


def last_row(r):
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("mesh", ["1,1", "1,2"])
def test_dense_tables_and_span_keep_the_result(mesh):
    ref = last_row(run("--mesh", mesh))
    r = run("--mesh", mesh, *DENSE, "--emu-check")
    assert last_row(r) == ref
    served = re.search(r"span: (\d+) gathers read the L1 span", r.stdout)
    dense = re.search(r"tabdense 1: \d+ twin maps, (\d+) of \d+ \(core, twin\) tables dense", r.stdout)
    assert served and int(served.group(1)) > 0, "no gather read the span: the test did not exercise it"
    assert dense and int(dense.group(1)) > 0, "no dense table was built: the test did not exercise it"
    assert "tabstream check: streamed walk == cursor walk" in r.stdout


@pytest.mark.parametrize("mesh", ["1,1", "1,2"])
@pytest.mark.parametrize("l1", ["2", "1"])
def test_span3_shares_colour_0_and_keeps_the_result(mesh, l1):
    # --tiles 20 forces T=20 > nsmall/3 so vectors spill to DRAM and --span 3 exercises both colour 0 sharing and Krylov loops
    ref = last_row(run("--mesh", mesh, "--tiles", "20", "--l1", l1, "--lsplit", "2", "--cse", "2"))
    r = run("--mesh", mesh, "--tiles", "20", "--l1", l1, "--lsplit", "2", "--cse", "2",
            "--tabdense", "1", "--span", "3", "--tscheck", "--emu-check")
    assert last_row(r) == ref
    served = re.search(r"span 3: (\d+) gathers read colour 0", r.stdout)
    dense = re.search(r"tabdense 1: \d+ twin maps, (\d+) of \d+ \(core, twin\) tables dense", r.stdout)
    assert served and int(served.group(1)) > 0, "no gather read colour 0: the test did not exercise --span 3"
    assert dense and int(dense.group(1)) > 0, "no dense table was built under --span 3"
    assert "tabstream check: streamed walk == cursor walk" in r.stdout


def test_a_broken_dense_table_fails_the_check():
    r = run("--mesh", "1,1", *DENSE, "--tabdense-break", "8")
    assert r.returncode != 0
    assert "dense index differs" in r.stderr
    r3 = run("--mesh", "1,1", "--tiles", "20", "--l1", "2", "--lsplit", "2", "--cse", "2",
             "--tabdense", "1", "--span", "3", "--tscheck", "--tabdense-break", "8")
    assert r3.returncode != 0
    assert "dense index differs" in r3.stderr
