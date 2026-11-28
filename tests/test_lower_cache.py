"""The staged lowering cache (ttfluid/lower/cache.py) must not change what the device runs: an image lowered through
it, cold or warm, is byte-equal to one lowered without it. And the gate can fail: a key that leaves out one of its
inputs serves a stale stage, and the same comparison catches it."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# pitzDaily on two Blackhole chips of four cores each: two tiles a vector, DRAM homes, streamed tables, as a large case
FLAGS = ("--arch bh --mesh 1,2 --cores 4 --layout rcb --strips --wplace 2 --gw 1 --nsh 1 --page 16 --nwin 8192 --tabstream --tsort 1 "
         "--lsched --phase --sliced --fuse --cse 1 --hlate 1 --l1 1 --min-free-gb 1").split()


def lower(d, *extra, n=50):
    """Lower into directory d; returns ({file or shared array: sha256}, the lowering's output)."""
    r = subprocess.run([sys.executable, "-m", "ttfluid.device.run_mesh", "case", str(n), *FLAGS, "--case", str(ROOT / "cases" / "pitzDaily"),
                        "--images", str(d), *map(str, extra)], cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED="0"),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    sha = lambda b: hashlib.sha256(b).hexdigest()
    files = {str(f.relative_to(d)): sha(f.read_bytes()) for f in sorted(Path(d).rglob("*")) if f.is_file() and f.name != "shared.npz"}
    z = np.load(Path(d) / "shared.npz")   # a zip: its bytes carry the time of writing, its arrays do not
    files.update({"shared.npz/" + k: sha(str(z[k].dtype).encode() + str(z[k].shape).encode() + z[k].tobytes()) for k in z.files})
    assert len(files) > 20
    return files, r.stdout


def test_cached_lowering_is_byte_equal(tmp_path):
    cache = tmp_path / "cache"
    cold, out = lower(tmp_path / "cold", "--cache", cache)
    assert "cache: trace built" in out
    assert cold == lower(tmp_path / "plain")[0]
    warm, out = lower(tmp_path / "warm", "--cache", cache, "--pseg", 3)     # one flag of the machine changed
    assert "cache: trace read" in out
    plain = lower(tmp_path / "plain_pseg", "--pseg", 3)[0]
    assert plain != cold
    assert warm == plain


def test_key_missing_an_input_is_caught(tmp_path):
    cache = tmp_path / "cache"
    lower(tmp_path / "n50", "--cache", cache, "--cache-omit", "n")
    stale, out = lower(tmp_path / "n60", "--cache", cache, "--cache-omit", "n", n=60)   # the iteration cap is not in the key
    assert "cache: trace read" in out
    assert stale != lower(tmp_path / "plain_n60", n=60)[0]
    good, out = lower(tmp_path / "n60_good", "--cache", cache, n=60)
    assert "cache: trace built" in out and good == lower(tmp_path / "plain_n60b", n=60)[0]


def test_sparse_write_is_np_save(tmp_path):
    """Image files are written with their zero blocks as holes; their bytes are np.save's."""
    from ttfluid.lower.image import _save_sparse
    rng = np.random.default_rng(0)
    arrays = [np.zeros(0, np.float32), np.arange(5, dtype=np.int64), np.zeros((3, 5000), np.uint32), np.ones((2, 3, 1024), np.float32),
              rng.integers(0, 2, (7, 4099)).astype(np.uint8) * (rng.random((7, 1)) > .5),
              np.where(rng.random((64, 4096)) > .9, 1, 0).astype(np.float32) * (np.arange(64) % 5 == 0)[:, None]]
    for a in arrays:
        np.save(tmp_path / "a.npy", a)
        _save_sparse(tmp_path / "b.npy", a)
        assert (tmp_path / "a.npy").read_bytes() == (tmp_path / "b.npy").read_bytes()
    assert (tmp_path / "b.npy").stat().st_blocks * 512 < arrays[-1].nbytes / 2   # four rows in five are holes


def preflight(*extra):
    r = subprocess.run([sys.executable, "-m", "ttfluid.device.run_mesh", "case", "50", *FLAGS, "--case", str(ROOT / "cases" / "pitzDaily"),
                        "--preflight", *map(str, extra)], cwd=ROOT, env=dict(os.environ, PYTHONHASHSEED="0"), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_preflight_refuses_before_the_image(tmp_path):
    """--preflight ends on the refusals a lowering would end on and writes nothing; with --cache the second refusal of a
    line comes from its kept sizes, before the lowering; an accepted line then lowers to the uncached image."""
    c, d = tmp_path / "cache", tmp_path / "img"
    rc, out = preflight("--images", d)
    assert rc == 0 and "preflight ok: CB bytes" in out and not d.exists(), out[-2000:]
    cb = int(out.split("preflight ok: CB bytes ")[1].split()[0])
    rc, out = preflight("--cache", c, "--l1-bytes", cb - 1)
    assert rc != 0 and "CB bytes %d over --l1-bytes %d" % (cb, cb - 1) in out and "lowered to that before" not in out, out[-2000:]
    rc, out = preflight("--cache", c, "--l1-bytes", cb - 1)
    assert rc != 0 and "lowered to that before" in out and "[t" not in out.split("sizes of this line")[1], out[-2000:]
    rc, out = preflight("--cache", c, "--l1-bytes", cb)
    assert rc == 0 and "preflight ok: CB bytes %d of %d" % (cb, cb) in out, out[-2000:]
    rc, out = preflight("--cache", c, "--min-free-gb", 10 ** 6)   # no disk has a million GB free
    assert rc != 0 and "refusing to write" in out, out[-2000:]
    rc, out = preflight("--cache", c, "--l1-bytes", cb)
    assert rc == 0 and "lowered to that before" in out, out[-2000:]
    assert lower(tmp_path / "a", "--cache", c, "--l1-bytes", cb)[0] == lower(tmp_path / "b", "--l1-bytes", cb)[0]
