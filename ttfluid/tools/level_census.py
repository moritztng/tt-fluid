"""Gather levels of the traced SIMPLE iteration, and what each level moves: today's schedule against a level-fused one.

    python -m ttfluid.tools.level_census MCACHE [--core 0] [--gcache -1] [--scratch 32,64,128,256,0] [--json out.json]

MCACHE is run_mesh's --mcache (LOWER, prog, m). The outer body (the outer loop minus its Krylov loops) is cut into
straight-line regions at every branch. Inside a region a value's LEVEL is the number of barriers on its longest path
from the region's inputs. A barrier is anything that needs ALL tiles of a value computed in this region before one tile
can be read: a gather or a reader-patched page of it, a shifted load (a split gather), a halo message, a sum over
tiles. All blocks of one level can run in ONE tile loop: no block of the level reads another tile of anything the level
computes.

Per level, in 4 KB pages per core and outer iteration:
  today   the pages of the schedule as lowered (tools/solve_census.py kinds: const, vec, store, gmiss, tab), and the
          Dest traffic: loads from DRAM and from L1, DRAM stores, op tiles.
  fused   the level as one tile loop with S one-tile scratch values: every input (constant or vector of an earlier
          level) is read once, every value a later level, a solve or the next iteration reads is stored once, the rest
          lives in scratch. Over S live values the one with the furthest next use leaves (an input is read again; a
          value of the level is stored and read back). S 0 = no limit. Gather sources are read once per level and
          source through the gather cache.
The fused numbers are a bound from the block order as lowered, not a built schedule.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ttfluid.resident.census import _misses, region_of
from ttfluid.resident.machine import DBASE, KBASE
from ttfluid.resident.program import PHASES
from ttfluid.tools.solve_census import load

KINDS = ("const", "vec", "store", "gmiss", "tab")


def pieces(mir):
    """Straight-line pieces of the program: MIR index lists with no branch and no branch target inside."""
    targets = {ins[-1] for ins in mir if ins[0] in ("jump", "jumpif")}
    out, cur = [], []
    for j, ins in enumerate(mir):
        if ins[0] in ("jump", "jumpif") or (ins[0] == "label" and ins[1] in targets):
            if cur:
                out.append(cur)
            cur = [j] if ins[0] == "jumpif" else []      # the branch reads its scalar
            if cur:
                out.append(cur)
                cur = []
        else:
            cur.append(j)
    if cur:
        out.append(cur)
    return out


def pertile(m, r):
    """A value with one instance per tile: a vector, or a fused run's scratch register."""
    return isinstance(r, int) and (m.is_vec(r) or m.space[r].name == "scratch")


def loop(m, x):
    """Tile-loop count of a block: Machine.tiles, 1 for a block of one-tile values only."""
    regs = [u[2] for u in x[1] if u[0] in ("ld", "lds", "rvf", "ldr") and u[2] is not None] + [r for _, r in x[2]]
    return m.tiles([x]) if any(pertile(m, r) for r in regs) or any(u[0] == "ldr" for u in x[1]) else 1


def passes(m, ins, x):
    """Tile-loop passes of block x of MIR instruction ins: a fused run loops all its blocks with the run's tile count."""
    return m.tiles([y for y in ins[1] if y[0] == "blk"]) if ins[0] == "seg" else loop(m, x)


def units_of(m, mir, idxs):
    """[(j, k, x, level, reads, writes)] of one straight-line piece. x: a block (k-th of a fused run), a gather, halo,
    red, red2, fetch or out. reads and writes are VALUES (register, version): version 0 is the piece's input, every
    write makes a new one, so a reused scratch register does not chain the levels."""
    ver, lvl, out = {}, {}, []
    val = lambda r: (r, ver.get(r, 0))
    for j in idxs:
        ins = mir[j]
        items = [x for x in ins[1] if x[0] == "blk"] if ins[0] == "seg" else [ins]
        for k, x in enumerate(items):
            op, rd, acc = x[0], [], False
            if op == "blk":
                for u in x[1]:
                    if u[0] in ("ld", "rvf"):
                        rd.append((u[2], False))
                    elif u[0] == "lds":
                        rd.append((u[2], True))
                    elif u[0] == "ldr":
                        if u[2] is not None:
                            rd.append((u[2], u[3] != 0))
                        for s, h in u[5]:
                            rd.append((s, True))
                            if h is not None:
                                rd.append((("H", h), False))
                df = [r for _, r in x[2] if r != "__anchor__"]
                acc = passes(m, ins, x) > 1   # a one-tile value written by a tile loop is a sum over tiles
            elif op == "gather":
                for s, h in x[3]:
                    rd.append((s, True))
                    if h is not None:
                        rd.append((("H", h), False))
                df = [x[1]]
            elif op == "halo":
                rd, df = [(x[2], True)], [("H", x[1])]
            elif op == "red":
                rd, df = [(x[2], False)], [x[1]]
            elif op == "red2":
                rd, df = [(r, False) for r in x[2::2]], list(x[1::2])
            elif op == "fetch":
                df = [x[1]]
            elif op == "out":
                rd, df = [(x[2], False)], []
            else:
                continue
            rv = [(val(r), b) for r, b in rd]
            L = max([lvl[v] + b for v, b in rv if v in lvl], default=0)
            wv = []
            for r in df:
                ver[r] = ver.get(r, 0) + 1
                lvl[val(r)] = L + (1 if acc and not pertile(m, r) else 0)
                wv.append(val(r))
            out.append((j, k, x, L, [v for v, _ in rv], wv))
    return out, ver


class Level:
    def __init__(self):
        self.c = Counter()
        self.phases = Counter()
        self.ev = []            # the level's vector traffic in block order: ("u", value) reads, ("d", value) writes
        self.gath = defaultdict(list)    # gather source value -> [(map id, part)]


def census(prog, m, core, gcache, tsort, budgets):
    from ttfluid.resident.machine import _mir_defs, _mir_uses
    mir, reg = m.mir, region_of(m.mir)
    labels_ph = {L: PHASES[i] for L, i in getattr(prog, "phase_at", {}).items()}
    ph_at, ph = [], "top"
    for ins in mir:
        if ins[0] == "label" and ins[1] in labels_ph:
            ph = labels_ph[ins[1]]
        ph_at.append(ph)
    ovl = getattr(m, "ovl", {})
    # scr: a fused run's one-tile scratch; ovl and l1: vectors with an L1 home today, left where they are
    home = lambda r: "scr" if m.space[r].name == "scratch" else "ovl" if r in ovl else \
        ("const" if m.phys[r] >= KBASE else "dram") if r in m.dram else "l1"
    vec = lambda r: isinstance(r, int) and r in m.phys and pertile(m, r)
    vsz = {}
    size = lambda v: m.size(v[0]) if m.is_vec(v[0]) else vsz.get(v, 1)
    tab = m.maps.tab
    gm = {}

    def entries(mid, k):
        """(destination tile, source tile, source index) of the core's local entries of part k of a gather map."""
        if (mid, k) not in gm:
            t = tab[mid]
            e = (t["core"] == core) & (t["part"] == k) & (t["kind"] == 0)
            gm[(mid, k)] = (t["lane"][e] >> 10, t["idx"][e] >> 10, t["idx"][e])
        return gm[(mid, k)]

    def gmiss(mid, parts):
        tot = 0
        for k, (s, _) in enumerate(parts):
            if m.phys[s] >= DBASE and s not in ovl:
                d, st, idx = entries(mid, k)
                q = np.lexsort((idx, d)) if tsort else np.arange(len(d))
                tot += _misses(st[q].tolist(), gcache)
        t = tab[mid]
        return tot, 4 * (int((t["core"] == core).sum()) + int((t["zc"] == core).sum())) / 4096

    # a register some straight-line piece reads before writing it is live across pieces: its last write in a piece stays
    exposed = set()
    for idxs in pieces(mir):
        d = set()
        for j in idxs:
            exposed |= {r for r in _mir_uses(mir[j]) if r not in d}
            d |= set(_mir_defs(mir[j]))
    lev, readers, last = {}, defaultdict(set), {}
    ri = -1
    for idxs in pieces(mir):
        if not idxs or reg[idxs[0]] != "outer":
            continue
        ri += 1
        us, ver = units_of(m, mir, idxs)
        for r, n in ver.items():
            last[(ri, r)] = n
        for j, k, x, L, rv, wv in us:
            g = lev.setdefault((ri, L), Level())
            c = g.c
            g.phases[ph_at[j]] += 1
            for v in rv:
                readers[(ri, v)].add(L)
            V = dict((v[0], v) for v in rv)       # register -> the value this unit reads
            if x[0] == "blk":
                bt = passes(m, mir[j], x)
                c["blocks"] += bt
                for u in x[1]:
                    if u[0] in ("ld", "rvf", "lds"):
                        c["ld_l1"] += bt
                        if vec(u[2]):
                            g.ev.append(("u", V[u[2]]))
                    elif u[0] == "ldr":
                        c["ld_dram"] += bt
                        if u[2] is not None:
                            c["const" if m.phys[u[2]] >= KBASE else "vec"] += bt
                            g.ev.append(("u", V[u[2]]))
                        if u[4] is not None:
                            a, b = gmiss(u[4], u[5])
                            c["gmiss"] += a
                            c["tab"] += b
                            for kk, (s, _) in enumerate(u[5]):
                                g.gath[V[s]].append((u[4], kk))
                    elif u[0] == "op":
                        c["op_tiles"] += bt
                for _, r in x[2]:
                    if r != "__anchor__":
                        c["store" if r in m.dram else "st_l1"] += bt
                for v in wv:
                    if vec(v[0]):
                        vsz[v] = bt
                        g.ev.append(("d", v))
            elif x[0] == "gather":
                if len(x) < 5 or x[4] in (1, 3, 4):
                    a, b = gmiss(x[2], x[3])
                    c["gmiss"] += a
                    c["tab"] += b
                    c["gathers"] += 1
                    for kk, (s, _) in enumerate(x[3]):
                        g.gath[V[s]].append((x[2], kk))
                if vec(x[1]):
                    g.ev.append(("d", wv[0]))
            elif x[0] in ("halo", "red", "red2"):
                c[x[0]] += 1

    rows = []
    for (ri, L), g in sorted(lev.items()):
        c = g.c
        today = {k: c[k] for k in KINDS}
        fg = 0        # gather sources of the level, each read once through the cache in destination-tile order
        for (s, _), lst in g.gath.items():
            if not vec(s) or home(s) in ("l1", "ovl", "scr"):
                continue
            d = np.concatenate([entries(mid, k)[0] for mid, k in lst])
            st = np.concatenate([entries(mid, k)[1] for mid, k in lst])
            q = np.lexsort((st, d))
            fg += _misses(st[q].tolist(), gcache)
        ev = [(k, v) for k, v in g.ev if home(v[0]) not in ("l1", "ovl")]
        nxt = defaultdict(list)
        for i, (k, v) in enumerate(ev):
            if k == "u":
                nxt[v].append(i)
        # stored: read by another level of the piece, or the piece's last write of a register that is live across pieces
        outs = {v for k, v in ev if k == "d" and (readers[(ri, v)] - {L} or (v[1] == last[(ri, v[0])] and v[0] in exposed))}
        fused = {}
        for S in budgets:
            pages, held, indram, mine, ptr = Counter(), {}, set(), set(), defaultdict(int)

            def keep(v, i):
                """v wants a scratch slot after event i; over the budget the furthest next use leaves."""
                us = nxt[v]
                while ptr[v] < len(us) and us[ptr[v]] <= i:
                    ptr[v] += 1
                if ptr[v] == len(us):
                    held.pop(v, None)
                    return
                held[v] = us[ptr[v]]
                if S and len(held) > S:
                    e = max(held, key=held.get)
                    del held[e]
                    if e in mine and e not in indram:     # a value of the level leaves scratch: stored once
                        indram.add(e)
                        pages["spill"] += size(e)
            peak = 0
            for i, (k, v) in enumerate(ev):
                if k == "u":
                    if v not in held:
                        pages["spill" if v in mine else "const" if home(v[0]) == "const" else "vec"] += size(v)
                    keep(v, i)
                else:
                    mine.add(v)
                    if v in outs:
                        pages["store"] += size(v)
                        indram.add(v)
                    keep(v, i)
                peak = max(peak, len(held))
            pages["gmiss"], pages["tab"] = fg, c["tab"]
            fused[S] = dict(pages=dict(pages), total=sum(pages.values()), peak=peak)
        rows.append(dict(region=ri, level=L, phases=dict(g.phases), today=today, total=sum(today.values()),
                         work={k: c[k] for k in ("blocks", "ld_dram", "ld_l1", "store", "st_l1", "op_tiles", "gathers", "halo", "red", "red2")},
                         fused=fused))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mcache")
    ap.add_argument("--core", type=int, default=0)
    ap.add_argument("--gcache", type=int, default=-1, help="gather cache tiles; -1: T + 1, a whole vector")
    ap.add_argument("--scratch", default="32,64,128,256,0", help="one-tile scratch budgets of the fused schedule, 0 = no limit")
    ap.add_argument("--json")
    a = ap.parse_args()
    LOWER, prog, m = load(a.mcache)
    gc = a.gcache if a.gcache >= 0 else m.T + 1
    budgets = [int(x) for x in a.scratch.split(",")]
    rows = census(prog, m, a.core, gc, bool(LOWER.get("tsort", 1)), budgets)
    cpc = int(m.lay.count[prog.cells.name][a.core])
    kb = lambda p: p * 4096 / cpc / 1024
    print(f"core {a.core}: {cpc} cells, T {m.T}, gcache {gc}; outer body, pages per core per outer it; KB/cell-it in the last lines")
    print("%-3s %-3s %-22s %7s | %7s %7s %7s %6s %6s %8s | %7s %7s %7s %7s | " % (
        "reg", "lvl", "phases", "blocks", *KINDS, "today", "ld_dram", "ld_l1", "st_l1", "op_til")
        + " ".join("%8s" % ("S=%d" % s if s else "S=inf") for s in budgets) + " | peak")
    tot, ftot, wtot = Counter(), Counter(), Counter()
    for r in rows:
        if not r["total"] and not r["work"]["blocks"]:
            continue
        w = r["work"]
        tot.update(r["today"])
        wtot.update(w)
        for s in budgets:
            ftot[s] += r["fused"][s]["total"]
        ph = ",".join(p for p, _ in sorted(r["phases"].items(), key=lambda kv: -kv[1]))[:22]
        print("%-3d %-3d %-22s %7d | " % (r["region"], r["level"], ph, w["blocks"])
              + " ".join("%7d" % r["today"][k] for k in KINDS[:3]) + " %6d %6d %8d | " % (r["today"]["gmiss"], r["today"]["tab"], r["total"])
              + "%7d %7d %7d %7d | " % (w["ld_dram"], w["ld_l1"], w["st_l1"], w["op_tiles"])
              + " ".join("%8d" % r["fused"][s]["total"] for s in budgets) + " | %d" % r["fused"][budgets[-1]]["peak"])
    allp = sum(tot.values())
    print("%-30s %7d | " % ("all", wtot["blocks"]) + " ".join("%7d" % tot[k] for k in KINDS[:3]) + " %6d %6d %8d | " % (tot["gmiss"], tot["tab"], allp)
          + "%7d %7d %7d %7d | " % (wtot["ld_dram"], wtot["ld_l1"], wtot["st_l1"], wtot["op_tiles"])
          + " ".join("%8d" % ftot[s] for s in budgets))
    print("KB/cell-it: today %.1f; level-fused " % kb(allp) + ", ".join("%s %.1f" % ("S=%d" % s if s else "S=inf", kb(ftot[s])) for s in budgets))
    nlev = Counter(r["region"] for r in rows if r["work"]["blocks"])
    print("levels with blocks per region: " + " ".join("%d:%d" % kv for kv in sorted(nlev.items())))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(cells=cpc, T=m.T, gcache=gc, today=dict(tot), today_kb=kb(allp),
                                                fused_kb={str(s): kb(ftot[s]) for s in budgets}, rows=rows), indent=1, default=int))


if __name__ == "__main__":
    main()
