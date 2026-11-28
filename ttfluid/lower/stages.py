"""Where a lowering spends its wall clock and its memory: seconds, resident GB at the end and peak resident GB of
every stage, one JSON line per stage.

    st = Stages("stages.jsonl")
    with st("partition"):
        ...
    st.methods(Machine, "machine", ["_lower", "_allocate"])   # time these methods as stages machine._lower, ...

The peak is the kernel's own (VmHWM, reset at the start of every stage through /proc/self/clear_refs), so it sees a
temporary that a sampler would miss. Stages nest: a parent's peak covers its children. Off (path None) it costs nothing.
"""
import functools
import json
import os
import time
from contextlib import contextmanager

GB = 1 << 30


def _status(key):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) * 1024
    return 0


def _reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


class Stages:
    def __init__(self, path=None, log=print):
        self.path, self.log, self.t0, self.open = path, log, time.time(), []   # open: [name, start, peak so far] per running stage
        self.seen = 0   # the process's peak before the last reset
        if path:
            open(path, "w").close()

    @contextmanager
    def __call__(self, name):
        if not self.path:
            yield
            return
        for s in self.open:   # the reset below forgets the peak the parents have seen so far
            s[2] = max(s[2], _status("VmHWM"))
        self.seen = max(self.seen, _status("VmHWM"))
        _reset_peak()
        me = [name, time.time(), 0]
        self.open.append(me)
        try:
            yield
        finally:
            self.open.pop()
            peak = max(me[2], _status("VmHWM"))
            row = dict(stage=".".join([s[0] for s in self.open] + [name]), s=round(time.time() - me[1], 2), at=round(me[1] - self.t0, 1),
                       rss_gb=round(_status("VmRSS") / GB, 2), peak_gb=round(peak / GB, 2), load=round(os.getloadavg()[0], 1))
            with open(self.path, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.log("[stage] %-34s %8.1f s  rss %6.2f GB  peak %6.2f GB" % (row["stage"], row["s"], row["rss_gb"], row["peak_gb"]), flush=True)
            for s in self.open:
                s[2] = max(s[2], peak)

    def peak_gb(self):
        """Peak resident GB of the process so far (the per-stage resets do not lose it)."""
        return max(self.seen, _status("VmHWM")) / GB

    def methods(self, cls, names):
        """Time the methods (or a module's functions) `names` as stages Class.name, nested under the stage that calls them."""
        if not self.path:
            return
        for n in names:
            f = getattr(cls, n)

            def timed(*a, _f=f, _n=getattr(cls, "__name__", "").rsplit(".", 1)[-1] + "." + n, **k):
                with self(_n):
                    return _f(*a, **k)
            setattr(cls, n, functools.wraps(f)(timed))
