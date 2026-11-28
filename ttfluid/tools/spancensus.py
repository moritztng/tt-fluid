"""Static census of the gather pages by source vector over a lowered Machine (a run_mesh --mcache file): what an L1
span per DRAM-homed source vector would have to hold, and how often it would be filled.

    python3 -m ttfluid.tools.spancensus MACHINE.pkl[.zst] [--spans 1 2 3 4] [--json OUT]

A tile loop is one `blk` or one fused `seg`. A gather page is an `ldr` unit with a map: the reader builds it from the
window (its source read at the map's shift) and the residue patched in from the part sources. Counted per phase for
one SIMPLE iteration: every loop body is walked twice, the first pass weighted 1 and the second by the trace's trip
hint minus one (10 where the trace gives none, as machine.py _place does); the outermost loop counts its second
pass only. Trip hints are hints: the page counts order the phases, they are not a timing.

For S spans, each holding one whole DRAM-homed vector (T tiles), kept by least-recent use over that order: fills
(T pages each), against the window pages the same gathers read today (one per page and tile). A store to a vector
drops its span. `need` is the number of distinct DRAM-homed vectors one loop gathers from: all of them have to be
resident while the loop runs, so a loop with need > S keeps its extra sources on today's path.
"""
import argparse, io, json, pickle
from collections import OrderedDict, defaultdict

from ttfluid.resident.program import PHASES


def load(path):
    f = open(path, "rb")
    if path.endswith(".zst"):
        import zstandard
        f = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True, closefd=True), 1 << 24)
    with f:
        return pickle.load(f)[:3]


def walk(mir, trips):
    """(instruction index, weight) in execution order for one SIMPLE iteration, loop bodies unrolled twice."""
    labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
    end = {}   # label index -> index of its last backward jump
    for j, ins in enumerate(mir):
        if ins[0] in ("jump", "jumpif") and ins[-1] in labels and labels[ins[-1]] < j:
            end[labels[ins[-1]]] = max(end.get(labels[ins[-1]], -1), j)
    outer = max(end, key=lambda i: end[i] - i) if end else None

    def run(lo, hi, w, out):
        i = lo
        while i < hi:
            if i in end and end[i] < hi:
                e, n = end[i], trips.get(mir[i][1], 10)
                first, second = (0.0, w) if i == outer else (w, w * max(n - 1, 0))
                run(i + 1, e + 1, first, out)
                run(i + 1, e + 1, second, out)
                i = e + 1
            else:
                out.append((i, w))   # weight 0 (the outermost loop's first pass) still warms the spans
                i += 1
        return out

    return run(0, len(mir), 1.0, [])


def census(prog, m, spans=(1, 2, 3, 4)):
    mir, D, T = m.mir, m.dram, m.T
    alias = getattr(m, "alias", {})
    pa = getattr(prog, "phase_at", {})
    ph_of, cur = [], None
    for ins in mir:
        if ins[0] == "label" and ins[1] in pa:
            cur = pa[ins[1]]
        ph_of.append(cur)
    info = {}   # tile loop -> (gather pages, window pages from DRAM, plain DRAM loads, DRAM gather sources, stores)
    for i, ins in enumerate(mir):
        if ins[0] not in ("blk", "seg"):
            continue
        blks = [ins] if ins[0] == "blk" else ins[1]
        g = wd = pl = l1g = 0
        src = OrderedDict()
        for b in blks:
            for u in b[1]:
                if u[0] != "ldr":
                    continue
                if u[4] is None:
                    pl += u[2] in D
                    continue
                ds = [s for s, _ in u[5] if s in D]
                if not ds:
                    l1g += 1
                    continue
                g += 1
                wd += u[2] is not None and u[2] in D
                for s in ds:
                    src[s] = None
        info[i] = (g, wd, pl, tuple(src), tuple(r for b in blks for _, r in b[2]), l1g)
    out = {}
    order = walk(mir, getattr(prog, "trips", {}))
    for S in spans:
        lru, st = OrderedDict(), defaultdict(lambda: defaultdict(float))
        for i, w in order:
            if i not in info:
                continue
            g, wd, pl, src, stores, l1g = info[i]
            r = st[ph_of[i]]
            r["loops"] += w
            r["plain"] += w * pl
            r["l1_gather_pages"] += w * l1g
            if g:
                r["gloops"] += w
                r["gather_pages"] += w * g
                r["window_pages"] += w * wd
                r["need_%d" % min(len(src), 6)] += w
                r["const_src"] += w * sum(s in alias for s in src)
                for s in src[:S]:   # the first S sources of the loop are spanned, the rest stay on today's path
                    if s in lru:
                        lru.move_to_end(s)
                    else:
                        r["fills"] += w
                        lru[s] = None
                        while len(lru) > S:
                            lru.popitem(last=False)
                r["pages_spanned"] += w * sum(1 for b in ([mir[i]] if mir[i][0] == "blk" else mir[i][1]) for u in b[1]
                                              if u[0] == "ldr" and u[4] is not None and any(s in src[:S] for s, _ in u[5])
                                              and all(s in src[:S] or s not in D for s, _ in u[5]))
            for x in stores:
                lru.pop(x, None)
        out[S] = {("top" if k is None else PHASES[k]): dict(v) for k, v in st.items()}
    return out, T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine")
    ap.add_argument("--spans", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--json")
    a = ap.parse_args()
    _, prog, m = load(a.machine)
    out, T = census(prog, m, a.spans)
    print("T %d tiles a vector, %d DRAM-homed registers (%d constants read in place), %d MIR instructions" % (
        T, len(m.dram), len(getattr(m, "alias", {})), len(m.mir)))
    base = out[a.spans[0]]
    print("per SIMPLE it and core, in tile loops (x T = pages):")
    print("%-12s %9s %9s %11s %11s %9s %9s  need 1/2/3/4/5/6+ (loops)" % (
        "phase", "loops", "gather", "gather pg", "window pg", "L1 gath", "plain"))
    for ph in list(PHASES) + ["top"]:
        r = base.get(ph)
        if not r:
            continue
        print("%-12s %9.0f %9.0f %11.0f %11.0f %9.0f %9.0f  %s" % (
            ph, r.get("loops", 0), r.get("gloops", 0), r.get("gather_pages", 0), r.get("window_pages", 0),
            r.get("l1_gather_pages", 0), r.get("plain", 0), "/".join("%.0f" % r.get("need_%d" % k, 0) for k in range(1, 7))))
    for S in a.spans:
        print("S = %d span(s): fills and the gather pages they serve (page-loops; x T = pages)" % S)
        for ph in list(PHASES) + ["top"]:
            r = out[S].get(ph)
            if not r or not r.get("gloops"):
                continue
            print("  %-12s fills %9.0f  pages served %11.0f of %11.0f  (window pages today %11.0f)" % (
                ph, r.get("fills", 0), r.get("pages_spanned", 0), r.get("gather_pages", 0), r.get("window_pages", 0)))
        tot = lambda k: sum(r.get(k, 0) for r in out[S].values())
        print("  %-12s fills %9.0f  pages served %11.0f of %11.0f  (window pages today %11.0f)" % (
            "all", tot("fills"), tot("pages_spanned"), tot("gather_pages"), tot("window_pages")))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"T": T, "spans": {str(S): v for S, v in out.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
