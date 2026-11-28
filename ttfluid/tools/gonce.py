"""flx-asm: gather once (DECISIONS D36.2). Per outer it on one core of a machine cache: what the outer body patches
through maps today, against the same body with every gather moved down to the fields ("gather the fields once and
rebuild behind the gather").
  today        every (source value, map) read: patched values = table entries, patched pages = destination tiles
  dedup        the same (source value, map) read once
  fields       each gathered value is rebuilt on the gathered side from gathers of the fields it is a function of: one
               gather per distinct (field version, map chain), and the blocks between are run again per chain (op-tiles)
A chain longer than one map is a gather of a gather: it needs a composed table (a wider stencil).
A field version = a value the outer body did not compute elementwise before the read (state, or a solve's output).
python -m ttfluid.tools.gonce MCACHE [CORE]"""
import sys
from collections import Counter, defaultdict
import numpy as np
from ttfluid.tools.solve_census import load
from ttfluid.resident.census import region_of
from ttfluid.resident.machine import _mir_defs
CORE = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LOWER, prog, m = load(sys.argv[1])
mir, reg = m.mir, region_of(m.mir)
pa = getattr(prog, "phase_at", {})
_ent = {}
def ent(mid):          # (entries, destination tiles) of a map on this core
    if mid not in _ent:
        t = m.maps.tab[mid]; s = t["core"] == CORE
        _ent[mid] = (int(s.sum()), int(np.unique(t["lane"][s] >> 10).size))
    return _ent[mid]
LOADS = ("ld", "lds", "ldr", "rvf")
def is_vec(r):
    try: return isinstance(r, int) and m.is_vec(r)
    except Exception: return False
ver = Counter()
node = {}              # (reg, ver) -> ("blk", nops, [(operand key, map or None)]) | ("gather", [(src key, mid)]); absent = field
phase, rows = "init", []
kinds = Counter()
for j, ins in enumerate(mir):
    if ins[0] == "label" and ins[1] in pa: phase = pa[ins[1]]
    blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"] if ins[0] == "seg" else []
    if not reg[j].startswith("outer"):
        for d in _mir_defs(ins):
            if isinstance(d, int): ver[d] += 1
        continue
    if ins[0] == "gather":
        srcs = [(s, ver[s]) for s, _ in ins[3] if s is not None]
        for k in srcs: rows.append((phase, k, ins[2]))
        d = ins[1]; ver[d] += 1; node[(d, ver[d])] = ("gather", [(k, ins[2]) for k in srcs])
        continue
    for b in blks:
        deps, nops = [], 0
        for u in b[1]:
            kinds[u[0]] += 1
            if u[0] == "ldr" and u[4] is not None:
                srcs = [s for s, _ in u[5]] if u[5] else [u[2]]
                for s in sorted(set(x for x in srcs if x is not None)):
                    k = (s, ver[s]); rows.append((phase, k, u[4])); deps.append((k, u[4]))
            elif u[0] in LOADS:
                if u[2] != "__anchor__" and is_vec(u[2]): deps.append(((u[2], ver[u[2]]), "S" if u[0] == "lds" else None))
            elif u[0] not in ("st", "stw", "sth", "stwh"): nops += 1
        for _, x in b[2]:
            ver[x] += 1; node[(x, ver[x])] = ("blk", nops, deps)
    if not blks:
        for d in _mir_defs(ins):
            if isinstance(d, int): ver[d] += 1
# cone of a read (value k through chain suf): the field gathers it needs and the blocks to run again behind them
G, Rb, seen = set(), {}, set()
def cone(k, suf):
    stack = [(k, suf)]
    while stack:
        k, suf = stack.pop()
        if (k, suf) in seen: continue
        seen.add((k, suf))
        n = node.get(k)
        if n is None: G.add((k, suf)); continue
        if n[0] == "gather":
            for s, mid in n[1]: stack.append((s, (mid,) + suf))
            continue
        Rb[(k, suf)] = n[1]
        for s, mp in n[2]: stack.append((s, suf if mp is None else (mp,) + suf))
tv = Counter(); tp = Counter(); dv = Counter(); dp = Counter(); first = set(); cls = Counter()
for ph, k, mid in rows:
    e, p = ent(mid); tv[ph] += e; tp[ph] += p
    c = "field" if k not in node else node[k][0]
    cls[c] += e
    if (k, mid) not in first: first.add((k, mid)); dv[ph] += e; dp[ph] += p
    cone(k, (mid,))
def last(suf):         # the map whose destination side the value lives on; "S" (a lane shift) keeps the side
    ms = [x for x in suf if x != "S"]
    return ms[-1]
gv = Counter(); gp = Counter(); gn = Counter(); shifted = 0
for k, suf in G:
    ms = tuple(x for x in suf if x != "S"); e, p = ent(ms[-1]); L = min(len(ms), 3)
    gv[L] += e; gp[L] += p; gn[L] += 1; shifted += "S" in suf
rt = sum(nops * ent(last(suf))[1] for (k, suf), nops in Rb.items())
phs = []
for r in rows:
    if r[0] not in phs: phs.append(r[0])
TV, TP = sum(tv.values()), sum(tp.values())
print("%s T %d core %d | outer body, per outer it | unit kinds %s" % (sys.argv[1].split("/")[-1], m.T, CORE, dict(kinds.most_common(12))))
print("  today   patched values %10d  patched pages %8d  reads %d | by source: %s" % (TV, TP, len(rows), ", ".join("%s %.1f %%" % (c, 100 * v / max(TV, 1)) for c, v in cls.most_common())))
for ph in phs: print("    %-12s values %10d pages %8d | dedup values %10d pages %8d" % (ph, tv[ph], tp[ph], dv[ph], dp[ph]))
DV, DP = sum(dv.values()), sum(dp.values())
print("  dedup   patched values %10d (%.1f %%)  patched pages %8d (%.1f %%)" % (DV, 100 * DV / max(TV, 1), DP, 100 * DP / max(TP, 1)))
GV, GP = sum(gv.values()), sum(gp.values())
print("  fields  gathered values %9d (%.1f %% of today)  pages %8d (%.1f %%)  gathers %d | by chain length: %s | with a lane shift in the chain: %d" % (
    GV, 100 * GV / max(TV, 1), GP, 100 * GP / max(TP, 1), len(G), ", ".join("%s%d: %d gathers, %d values, %d pages" % ("" if L < 3 else ">=", L, gn[L], gv[L], gp[L]) for L in sorted(gn)), shifted))
print("          blocks run again behind the gathers: %d (block, chain) pairs, %d op-tiles per outer it; distinct field versions gathered: %d" % (
    len(Rb), rt, len({k for k, _ in G})))
# ---- the best a single table can do: per map, the fewest vectors to gather so that every value read through the map
# today can be rebuilt elementwise behind it (min vertex cut of the direct-load DAG; a block with a patched or shifted
# operand, a fold, or a standalone gather cannot be moved behind the map without a composed table, so it is a leaf)
def opaque(k):
    n = node.get(k)
    return n is None or n[0] == "gather" or n[3]
# second pass over the blocks for the opaque flag (kept apart so the walk above stays as it is)
ver2 = Counter(); flag = {}
for j, ins in enumerate(mir):
    blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"] if ins[0] == "seg" else []
    if not reg[j].startswith("outer") or not blks:
        for d in _mir_defs(ins):
            if isinstance(d, int): ver2[d] += 1
        continue
    for b in blks:
        op = any((u[0] == "ldr" and u[4] is not None) or u[0] in ("lds", "fold", "rvf") for u in b[1])
        for _, x in b[2]:
            ver2[x] += 1; flag[(x, ver2[x])] = op
assert all(ver2[r] == ver[r] for r in set(ver) | set(ver2)), "the two walks disagree on versions"
for k, n in list(node.items()):
    if n[0] == "blk": node[k] = n + (flag[k],)
def mincut(sinks):
    ids = {}; adj = []           # vertex v -> in 2i, out 2i + 1
    def vid(k):
        if k not in ids: ids[k] = len(ids); adj.append([]); adj.append([])
        return ids[k]
    E = []                       # [to, cap]
    def edge(a, b, c):
        adj[a].append(len(E)); E.append([b, c]); adj[b].append(len(E)); E.append([a, 0])
    INF = 1 << 30
    stack = list(sinks); done = set()
    while stack:
        k = stack.pop()
        if k in done: continue
        done.add(k); vid(k)
        if not opaque(k):
            for s, mp in node[k][2]: stack.append(s)
    S = 2 * len(ids); T = S + 1; adj.append([]); adj.append([])
    for k in done:
        i = ids[k]; edge(2 * i, 2 * i + 1, 1)
        if opaque(k) or not node[k][2]: edge(2 * i + 1, T, INF)
        else:
            for s, mp in node[k][2]: edge(2 * i + 1, 2 * ids[s], INF)
    for k in sinks: edge(S, 2 * ids[k], INF)
    flow = 0
    while True:                  # augmenting paths by BFS (unit vertex capacities: at most len(sinks) rounds)
        par = {S: None}; q = [S]
        for a in q:
            if a == T: break
            for ei in adj[a]:
                b, c = E[ei]
                if c > 0 and b not in par: par[b] = ei; q.append(b)
        if T not in par: break
        a = T
        while par[a] is not None:
            ei = par[a]; E[ei][1] -= 1; E[ei ^ 1][1] += 1; a = E[ei ^ 1][0]
        flow += 1
    cut = [k for k, i in ids.items() if 2 * i in par and 2 * i + 1 not in par]
    above = [k for k, i in ids.items() if 2 * i + 1 in par]
    return flow, cut, above
bym = defaultdict(set)
for k, mid in first: bym[mid].add(k)
CV = CP = RT = NS = NC = 0; lines = []
for mid in sorted(bym):
    e, p = ent(mid); f, cut, above = mincut(sorted(bym[mid]))
    assert f == len(cut), (mid, f, len(cut))
    r = sum(node[k][1] for k in above if k in node and node[k][0] == "blk")
    CV += f * e; CP += f * p; RT += r * p; NS += len(bym[mid]); NC += f
    lines.append("map %d: %d entries, %d pages, vectors read %d -> %d, rebuilt blocks %d (%d ops)" % (mid, e, p, len(bym[mid]), f, len(above), r))
print("  cut     patched values %10d (%.1f %% of today, %.1f %% of dedup)  patched pages %8d (%.1f %% of today)  vectors gathered %d -> %d | op-tiles run again %d per outer it" % (
    CV, 100 * CV / max(TV, 1), 100 * CV / max(DV, 1), CP, 100 * CP / max(TP, 1), NS, NC, RT))
for l in sorted(lines, key=lambda s: -int(s.split(": ")[1].split()[0]))[:8]: print("    " + l)
