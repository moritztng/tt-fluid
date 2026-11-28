"""What the constant pages of the outer body hold, by content.

    python -m ttfluid.tools.const_census MCACHE [--core 0,7] [--json out.json]

MCACHE is run_mesh's --mcache (LOWER, prog, m). A streamed constant lives in DRAM as its own tiles per core (page
core * cpages + coff + t) and a block reads it in place (an ldr uop), one 4 KB page per tile pass. This tool classes
every such page of a core by its bits and weighs each class by the page reads of one outer iteration.

Classes, first match wins:
  zero      every lane +0.0
  uniform   one bit pattern
  two       two bit patterns (masks, signs)
  copy      the bytes of an earlier constant page of the same core
  derived   the constant equals op(a) or op(a, b) of other constants in every element of its space (neg, 0 - a,
            abs, 1 - a, a * a, mul, add, sub, div; found on a sample, confirmed on the whole space), in numpy fp32
  few       at most 16 bit patterns
  bf16      every lane has its low 16 bits clear (exact in 16 bits)
  fp32      none of these
A shifted read (a split gather's offset load) is counted under `shifted` and classed by its source page.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from ttfluid.resident.census import region_of
from ttfluid.resident.machine import KBASE, LANES
from ttfluid.resident.program import PHASES
from ttfluid.tools.solve_census import load

CLASSES = ("zero", "uniform", "two", "copy", "derived", "few", "bf16", "fp32")
UNARY = {"neg": np.negative, "0-a": lambda a: np.float32(0) - a, "abs": np.abs, "1-a": lambda a: np.float32(1) - a,
         "a*a": lambda a: a * a}
BINARY = {"mul": (np.multiply, True), "add": (np.add, True), "sub": (np.subtract, False), "div": (np.divide, False)}


def reads(prog, m):
    """Constant reads of the outer body: [(const register, shift, patched, tile passes, phase, MIR index, in a run)],
    plus the outer body's B_FETCH copies [(const register, tiles, phase)]."""
    mir, reg = m.mir, region_of(m.mir)
    labels_ph = {L: PHASES[i] for L, i in getattr(prog, "phase_at", {}).items()}
    ph, out, fetch = "top", [], []
    for j, ins in enumerate(mir):
        if ins[0] == "label" and ins[1] in labels_ph:
            ph = labels_ph[ins[1]]
        if reg[j] != "outer":
            continue
        if ins[0] == "fetch":
            fetch.append((m.fetched_from[ins[1]], m.size(ins[1]), ph))
        if ins[0] not in ("blk", "seg"):
            continue
        blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"]
        bt = m.tiles(blks)
        for b in blks:
            for u in b[1]:
                if u[0] == "ldr" and u[2] is not None and m.phys[u[2]] >= KBASE:
                    out.append((m.fetched_from[u[2]], u[3], u[4] is not None, bt, ph, j, ins[0] == "seg"))
    return out, fetch


def derivations(m, regs, sample=8192, seed=0, log=print):
    """{r: (op, a[, b])} for constants that are exactly an fp32 op of other constants over their whole space."""
    rng = np.random.default_rng(seed)
    by_space = defaultdict(list)
    for r in regs:
        by_space[m.space[r].name].append(r)
    found = {}
    with np.errstate(all="ignore"):
        for name, rs in by_space.items():
            n = m.space[rs[0]].n
            idx = np.sort(rng.choice(n, min(n, sample), replace=False))
            V = np.stack([np.asarray(m.pre[r], np.float32)[idx] for r in rs])
            W = rng.integers(1, 1 << 62, V.shape[1], dtype=np.uint64) | np.uint64(1)
            h = lambda X: (np.ascontiguousarray(X, np.float32).view(np.uint32).astype(np.uint64) * W).sum(-1)
            target = defaultdict(list)
            for k, x in enumerate(h(V)):
                target[int(x)].append(k)
            full = lambda r: np.asarray(m.pre[r], np.float32)
            same = lambda r, X: np.array_equal(full(r).view(np.uint32), np.ascontiguousarray(X, np.float32).view(np.uint32))
            # an operand must not be derived itself when the match is made: no cycles
            for opn, f in UNARY.items():
                for k, x in enumerate(h(f(V))):
                    for t in target.get(int(x), ()):
                        r, a = rs[t], rs[k]
                        if t != k and r not in found and a not in found and same(r, f(full(a))):
                            found[r] = (opn, a)
            for opn, (f, comm) in BINARY.items():
                for i in range(len(rs)):
                    hs = h(f(V[i][None, :], V[i:] if comm else V))
                    for k in np.nonzero(np.isin(hs, np.fromiter(target, np.uint64, len(target)).astype(hs.dtype)))[0]:
                        j = int(k) + (i if comm else 0)
                        for t in target[int(hs[k])]:
                            r, a, b = rs[t], rs[i], rs[j]
                            if r in (a, b) or r in found or a in found or b in found:
                                continue
                            if same(r, f(full(a), full(b))):
                                found[r] = (opn, a, b)
            log(f"derivations in {name}: {sum(1 for r in rs if r in found)} of {len(rs)} constants")
    return found


def census(prog, m, core, derived, log=print):
    rd, fetch = reads(prog, m)
    consts = sorted({r for r, *_ in rd}, key=lambda r: m.coff[r])
    cells = int(m.lay.count[prog.cells.name][core])
    sel = {}
    pages, seen = {}, {}
    cls, two_vals, nvals, pagekey = {}, {}, {}, {}
    at, uni = {}, {}     # page of the core's constant array -> (constant, tile); uniform page -> its bit pattern
    for r in sorted(m.coff, key=m.coff.get):     # every stored constant: a read may land on a neighbour's page
        sp = m.space[r]
        if sp.name not in sel:
            k = np.nonzero(m.lay.owner[sp.name] == core)[0]
            sel[sp.name] = (k, m.lay.lane[sp.name][k])
        k, ln = sel[sp.name]
        n = m.size(r)
        a = np.zeros(n * LANES, np.float32)
        keep = ln < n * LANES
        a[ln[keep]] = np.asarray(m.pre[r], np.float32)[k[keep]]
        bits = a.view(np.uint32).reshape(n, LANES)
        for t in range(n):
            p = bits[t]
            key = p.tobytes()
            u = np.unique(p)
            nvals[(r, t)] = len(u)
            if not p.any():
                c = "zero"
            elif len(u) == 1:
                c = "uniform"
                uni[(r, t)] = int(u[0])
            elif len(u) == 2:
                c = "two"
                two_vals[(r, t)] = tuple(float(x) for x in u.view(np.float32))
            elif key in seen:
                c = "copy"
            elif r in derived:
                c = "derived"
            elif len(u) <= 16:
                c = "few"
            elif not (p & 0xFFFF).any():
                c = "bf16"
            else:
                c = "fp32"
            seen.setdefault(key, (r, t))
            pagekey[(r, t)] = seen[key]
            cls[(r, t)] = c
            at[m.coff[r] + t] = (r, t)
    # page reads per outer iteration: a read in a tile loop of bt passes reads the constant's first bt tiles (a
    # --sliced block loops over its narrowest vector)
    # A one-tile constant in a tile loop is read every pass.
    nread, nreg = Counter(), Counter()
    by_phase, cls_phase = Counter(), defaultdict(Counter)
    shifted = patched = overrun = beyond = 0
    for r, shift, pat, bt, ph, j, inseg in rd:
        nreg[r] += 1
        n = m.size(r)
        for t in range(bt):
            # the reader fetches page coff + t whatever the constant's own tile count (vm_ncrisc.cpp dpage), so a
            # pass beyond it lands on the pages stored behind it: classed as what is fetched
            k = at.get(m.coff[r] + t)
            if k is None:
                beyond += 1      # past the core's last constant page
                continue
            overrun += t >= n
            nread[k] += 1
            cls_phase[ph][cls[k]] += 1
        by_phase[ph] += bt
        shifted += bt * (shift != 0)
        patched += bt * pat
    distinct, weighted = Counter(), Counter()
    for k, c in cls.items():
        distinct[c] += nread[k] > 0
        weighted[c] += nread[k]
    # what L1 homes for the most-read pages would take off DRAM: pages that need storage, most read first
    hot = sorted((nread[k] for k, c in cls.items() if c not in ("zero", "uniform")), reverse=True)
    hot_cum = {n: int(sum(hot[:n])) for n in (8, 16, 32, 64, 128, 256, 512)}
    total = sum(weighted.values())
    kb = lambda pg: 4.0 * pg / cells
    # how often one page is read per iteration
    mult = Counter(nread[k] for k in cls)
    # the same (constant, shift) read more than once inside one fused run
    run_dup = 0
    per_run = defaultdict(Counter)
    for r, shift, pat, bt, ph, j, inseg in rd:
        if inseg:
            per_run[j][(r, shift, pat)] += bt
    for j, c in per_run.items():
        for key, v in c.items():
            n1 = m.tiles([x for x in m.mir[j][1] if x[0] == "blk"])
            run_dup += v - n1
    pairs = Counter(v for v in two_vals.values())
    reg_cls, reg_w = defaultdict(Counter), defaultdict(Counter)
    for (r, t), c in cls.items():
        if nread[(r, t)]:
            reg_cls[r][c] += 1
            reg_w[r][c] += nread[(r, t)]
    never = sum(1 for k in cls if not nread[k])
    pal = Counter()
    for k, v in uni.items():
        pal[v] += nread[k]
    pal = [(float(np.uint32(v).view(np.float32)), f"{v:08x}", int(n)) for v, n in pal.most_common() if n]
    # read pages that are distinct by content (zero and uniform pages aside): what a shared home would store
    homes = len({pagekey[k] for k, c in cls.items() if nread[k] and c not in ("zero", "uniform")})
    out = dict(core=core, cells=cells, T=int(m.T), consts=len(consts), streamed=len(m.streamed), cpages=int(m.cpages),
               reads=len(rd), page_reads=int(total), kb_cell_it=kb(total),
               by_class={c: dict(pages=int(distinct[c]), page_reads=int(weighted[c]), kb_cell_it=kb(weighted[c]),
                                 share=weighted[c] / max(total, 1)) for c in CLASSES},
               by_phase={p: int(v) for p, v in by_phase.items()},
               shifted_page_reads=int(shifted), patched_page_reads=int(patched),
               reads_per_page={int(k): int(v) for k, v in sorted(mult.items())},
               run_duplicate_page_reads=int(run_dup),
               fetch_pages=int(sum(n for _, n, _ in fetch)),
               two_pairs=[[list(k), int(v)] for k, v in pairs.most_common(12)],
               consts_whole={c: int(sum(1 for r in consts if set(reg_cls[r]) == {c})) for c in CLASSES},
               derived={str(r): [d[0]] + [int(x) for x in d[1:]] for r, d in derived.items() if r in set(consts)},
               pages_never_read=int(never), distinct_content_pages=int(homes), overrun_page_reads=int(overrun),
               beyond_page_reads=int(beyond), uniform_values=pal[:64], uniform_distinct=len(pal),
               by_class_phase={p: {c: int(v) for c, v in d.items()} for p, d in cls_phase.items()},
               l1_home_page_reads=hot_cum,
               regs={str(r): dict(space=m.space[r].name, tiles=int(m.size(r)), reads=int(nreg[r]),
                                  cls={c: int(v) for c, v in reg_cls[r].items()},
                                  page_reads={c: int(v) for c, v in reg_w[r].items()}) for r in consts})
    log(f"core {core}: {cells} cells, T {m.T}, {len(consts)} constants read in the outer body ({len(m.streamed)} "
        f"streamed, {m.cpages} pages per core), {len(rd)} reads = {total} page reads per it = {kb(total):.2f} KB/cell-it")
    log("  class     pages   page reads  KB/cell-it  share")
    for c in CLASSES:
        log(f"  {c:8s} {distinct[c]:6d} {weighted[c]:11d} {kb(weighted[c]):10.2f}  {100 * weighted[c] / max(total, 1):5.1f} %")
    exact = sum(weighted[c] for c in ("zero", "uniform", "two", "copy", "derived"))
    log(f"  exact classes (zero, uniform, two, copy, derived): {100 * exact / max(total, 1):.1f} % of the page reads")
    log(f"  by phase: {dict(by_phase)}; shifted {shifted}, reader-patched {patched}, B_FETCH pages {out['fetch_pages']}")
    log(f"  pages stored {len(cls)}, never read in the outer body {never}; distinct by content among the read "
        f"non-zero, non-uniform pages: {homes}")
    log(f"  page reads the N most-read stored pages take (L1 homes): {hot_cum}; reads past a constant's own tiles "
        f"{overrun} (classed as the page fetched), past the core's pages {beyond}")
    log(f"  uniform pages: {len(pal)} distinct values; page reads by value: {[(v, n) for v, _, n in pal[:16]]}")
    for p, d in cls_phase.items():
        if sum(d.values()) > 1000:
            log(f"  {p}: " + ", ".join(f"{c} {d[c]}" for c in CLASSES if d[c]))
    log(f"  reads of one page per it (reads: pages): {dict(sorted(mult.items()))}")
    log(f"  page reads a fused run repeats (same constant, shift and patch in one run): {run_dup}")
    log(f"  two-valued pages by value pair: {pairs.most_common(8)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcache")
    ap.add_argument("--core", default="0")
    ap.add_argument("--json")
    ap.add_argument("--derive", type=int, default=1)
    a = ap.parse_args()
    LOWER, prog, m = load(a.mcache)
    rd, _ = reads(prog, m)
    regs = sorted({r for r, *_ in rd})
    derived = derivations(m, sorted(m.streamed)) if a.derive else {}
    derived = {r: d for r, d in derived.items() if r in set(regs)}
    res = [census(prog, m, c, derived) for c in map(int, a.core.split(","))]
    if a.json:
        with open(a.json, "w") as f:
            json.dump(dict(mcache=a.mcache, lower={k: v for k, v in LOWER.items() if isinstance(v, (int, float, str, bool, type(None)))},
                           cores=res), f)


if __name__ == "__main__":
    main()
