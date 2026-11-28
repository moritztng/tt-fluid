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


def test_a_broken_dense_table_fails_the_check():
    r = run("--mesh", "1,1", *DENSE, "--tabdense-break", "8")
    assert r.returncode != 0
    assert "dense index differs" in r.stderr


def test_amg_nonsplit_dense_tables_and_broken_arm():
    amg = "--amg --amg-kind matched --amg-stack".split()
    ref = last_row(run("--mesh", "1,1", *amg))
    r = run("--mesh", "1,1", *amg, *DENSE, "--emu-check")
    assert last_row(r) == ref
    assert "tabstream check: streamed walk == cursor walk" in r.stdout
    # confirm non-split and zero-fill 1-part L1 gathers in p_solve went through dense tables
    m = re.search(r"gcensus p_solve\s+gather 1-part l1\s+ops\s+\d+\s+local entries a core\s+(\d+)\s+through a dense table\s+(\d+)", r.stdout)
    assert m and int(m.group(2)) > 0, r.stdout[-2000:]
    # confirm --span 2 also serves multi-gather DRAM sources in AMG (l1=0 forces DRAM fine vectors)
    r0 = run("--mesh", "1,1", "--l1", "0", *amg, *DENSE, "--emu-check")
    assert last_row(r0) == ref
    assert "tabstream check: streamed walk == cursor walk" in r0.stdout
    bad = run("--mesh", "1,1", *amg, *DENSE, "--tabdense-break", "8")
    assert bad.returncode != 0
    assert "dense index differs" in bad.stderr


@pytest.mark.parametrize("mesh", ["1,1", "1,2"])
def test_amg_span3_pseg5_and_broken_arm(mesh):
    amg = "--l1 2 --lsplit 2 --cse 2 --amg --amg-kind matched --amg-stack".split()
    ref = last_row(run("--mesh", mesh, *amg))
    extra = "--span 3 --pseg 5 --bpatch 6 --tpat 3 --tabdense 1 --tscheck --devsim --emu-check".split()
    r = run("--mesh", mesh, *amg, *extra)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    got = [ln for ln in r.stdout.splitlines() if ln.startswith("red2 ")]
    assert got and got[-1] == ref
    assert "tabstream check: streamed walk == cursor walk" in r.stdout
    assert "devsim ok" in r.stdout
    sp = re.search(r"span 3: (\d+) gathers read colour 0", r.stdout)
    assert sp and int(sp.group(1)) > 0, r.stdout[-2000:]
    up = re.search(r"gcensus U_solve\s+page 1-part l1, window same l1\s+ops\s+\d+\s+local entries a core\s+\d+\s+through a dense table\s+(\d+)", r.stdout)
    assert up and int(up.group(1)) > 0, r.stdout[-2000:]
    if mesh == "1,1":
        bad = run("--mesh", mesh, *amg, *extra, "--tabdense-break", "8")
        assert bad.returncode != 0
        assert "dense index differs" in bad.stderr

