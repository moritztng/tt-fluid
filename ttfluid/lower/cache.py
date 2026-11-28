"""Staged cache of a lowering: a stage whose inputs did not change is read back instead of rebuilt.

    cache = Cache("~/flx/cache")
    mesh, prog = cache.get("trace", inputs, build)     # build() runs only when no entry has these inputs

A stage is keyed by a digest of everything it reads: its flags, the content of its input files, the source of the
modules it runs, the interpreter's hash seed and the numpy version. The entry is the pickled value with every numpy
array out of band in one flat file, which a hit maps into memory copy-on-write: no read, no second copy, and the pages
are the kernel's to evict. An image lowered through the cache is byte-equal to one lowered without it
(tests/test_lower_cache.py; the same test turns red when a key leaves out one of its inputs).

Entries are directories <root>/<stage>-<key>/ with value.pkl, buffers.bin and inputs.json (what the key was made of,
so two entries of one stage can be diffed). A build is written beside the cache and renamed in, so a killed lowering
leaves no half entry and two lowerings that miss at once both succeed.
"""
import hashlib
import json
import mmap
import os
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

FORMAT = 1   # bump when the entry layout changes: older entries are then never hit
SKIP = ("postProcessing", "dynamicCode")   # case sub-directories no solver input lives in


def _file_digest(p):
    h = hashlib.blake2b(digest_size=16)
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def tree_digest(path):
    """Content digest of a file, or of every file under a directory (names and bytes; an OpenFOAM case)."""
    path = Path(path)
    if path.is_file():
        return _file_digest(path)
    files = sorted(p for p in path.rglob("*") if p.is_file() and not any(
        q in SKIP or q.startswith(("processor", "log")) for q in p.relative_to(path).parts))
    with ThreadPoolExecutor(8) as ex:
        dig = list(ex.map(_file_digest, files))
    return hashlib.blake2b(json.dumps([[str(p.relative_to(path)), d] for p, d in zip(files, dig)]).encode(), digest_size=16).hexdigest()


def source_digest(names):
    """Digest of the source files of these modules or packages (dotted names; a package means every .py in it)."""
    import importlib.util
    files = []
    for n in names:
        spec = importlib.util.find_spec(n)
        files += [f for d in spec.submodule_search_locations or () for f in Path(d).glob("*.py")] or [Path(spec.origin)]
    return hashlib.blake2b(b"".join(f.name.encode() + f.read_bytes() for f in sorted(set(files))), digest_size=16).hexdigest()


def process_inputs():
    """What every stage depends on besides its own inputs: iteration order of sets (the hash seed) and the arithmetic."""
    return dict(format=FORMAT, hashseed=os.environ.get("PYTHONHASHSEED", "random:%d" % os.getpid()), numpy=np.__version__,
                python=sys.version.split()[0])


def _view(root, offset, shape, strides, dtype):
    return np.ndarray(shape, dtype, buffer=root, offset=offset, strides=strides)


class _Pickler(pickle.Pickler):
    """A view is stored as its place in the array that owns the memory, so arrays that shared memory share it again
    after the load. The lowering relies on that: machine.cse matches gather index arrays by their address (_aid)."""

    def reducer_override(self, a):
        if type(a) is not np.ndarray or a.base is None or a.dtype.hasobject:
            return NotImplemented
        root = a
        while type(root.base) is np.ndarray:
            root = root.base
        if root.base is not None or not (root.flags.c_contiguous or root.flags.f_contiguous) or not root.flags.owndata:
            return NotImplemented
        off = a.__array_interface__["data"][0] - root.__array_interface__["data"][0]
        return _view, (root, off, a.shape, a.strides, a.dtype)


class Cache:
    def __init__(self, root, log=print, omit=()):
        """omit: input names left out of every key. Only for the test that proves the gate can fail."""
        self.root, self.log, self.omit = Path(root).expanduser(), log, set(omit)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, stage, inputs):
        inputs = {k: v for k, v in dict(inputs, **process_inputs()).items() if k not in self.omit}
        return hashlib.blake2b(json.dumps([stage, inputs], sort_keys=True, default=str).encode(), digest_size=10).hexdigest(), inputs

    def get(self, stage, inputs, build):
        key, inputs = self.key(stage, inputs)
        d = self.root / f"{stage}-{key}"
        t0 = time.time()
        if (d / "value.pkl").exists():
            value = self._load(d)
            os.utime(d)   # last use, for prune()
            self.log("cache: %s read from %s in %.1f s" % (stage, d, time.time() - t0), flush=True)
            return value
        value = build()
        t1 = time.time()
        tmp = self.root / f".{stage}-{key}.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir()
        try:
            n = self._store(tmp, value)
            (tmp / "inputs.json").write_text(json.dumps(inputs, indent=1, sort_keys=True, default=str))
            os.rename(tmp, d)
        except OSError:   # another lowering stored the same entry first (rename onto a full directory), or the disk is full
            shutil.rmtree(tmp, ignore_errors=True)
            if not (d / "value.pkl").exists():
                raise
        self.log("cache: %s built in %.1f s, stored to %s in %.1f s (%.2f GB)" % (stage, t1 - t0, d, time.time() - t1, n / 1e9), flush=True)
        return value

    @staticmethod
    def _store(d, value):
        """value.pkl + buffers.bin: every out-of-band buffer at a 4-KB boundary, its offset and length in the index."""
        bufs = []
        with open(d / "value.pkl", "wb") as f:
            _Pickler(f, protocol=5, buffer_callback=bufs.append).dump(value)
        index, off = [], 0
        with open(d / "buffers.bin", "wb") as f:
            for b in bufs:
                m = b.raw()
                f.seek(off)
                f.write(m)
                index.append((off, m.nbytes))
                off = -(-(off + max(m.nbytes, 1)) // 4096) * 4096   # an empty array keeps an address of its own
            f.truncate(off)
        (d / "index.json").write_text(json.dumps(index))
        return off

    @staticmethod
    def _load(d):
        index = json.loads((d / "index.json").read_text())
        with open(d / "buffers.bin", "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY) if index and index[-1][0] + index[-1][1] else None
        view = memoryview(mm) if mm is not None else memoryview(b"")
        with open(d / "value.pkl", "rb") as f:
            return pickle.load(f, buffers=[view[o:o + n] for o, n in index])
