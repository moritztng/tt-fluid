"""`--order boxu --tabrun RMIN` encodes qualifying dense tiles as run-copy command streams with balanced 2T/3T splits
and halo/zero don't-care bridging: on pitzDaily (1x1 and 1x2) `--tscheck` verifies 1T, 2T and 3T decoders match
bit-identically on every core, map and tile; `--tabrun-break` fails `--tscheck` loudly; and `--order boxu` matches
the CPU `SimpleFoam(ell_order='dir')` reference."""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLAGS = ("--arch bh --layout rcb --l1 1 --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 1024 --tabstream --lsched --phase "
         "--sliced --fuse --cse 1 --hlate 1 --strips --cores 2").split()
RUN = "--order boxu --tabdense 1 --span 2 --tabrun 4 --tscheck".split()


def run(*extra):
    return subprocess.run(
        [sys.executable, "-m", "ttfluid.device.run_mesh", "case", "2", "--case", str(ROOT / "cases" / "pitzDaily"),
         *FLAGS, *extra, "--emuonly", "--min-free-gb", "0"],
        cwd=ROOT, capture_output=True, text=True, timeout=900,
    )


def last_row(r):
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


@pytest.mark.parametrize("mesh", ["1,1", "1,2"])
@pytest.mark.parametrize("weights", ["1,1,1", "3,3,2"])
def test_runcopy_tables_match_dense_and_cursor_walk(mesh, weights):
    ref = last_row(run("--mesh", mesh, "--order", "boxu", "--tabdense", "1", "--span", "2"))
    r = run("--mesh", mesh, *RUN, "--run-weights", weights, "--emu-check")
    assert last_row(r) == ref
    m = re.search(r"tabrun 4 \(weights [^)]+\): (\d+) of (\d+) dense tiles run-copy", r.stdout)
    assert m and int(m.group(1)) > 0, f"no run-copy tile was built:\n{r.stdout[-1000:]}"
    assert "tabstream check: streamed walk == cursor walk" in r.stdout


def test_a_broken_runcopy_table_fails_the_check():
    r = run("--mesh", "1,1", *RUN, "--tabrun-break", "4")
    assert r.returncode != 0
    assert "dense index differs" in r.stderr
