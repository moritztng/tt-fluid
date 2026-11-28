"""flx-asm: the gathered values of the outer body (one SIMPLE iteration) on one core, counted by what is read.

A gathered value is one table entry: (source register version, source lane or halo index) copied to a destination
lane by a patched load or a GATHER op. Per phase this prints
  total      table entries per outer it (the scalar copies the reader, BRISC and UNPACK do)
  halo       entries whose source is a halo slot; zero = lanes the map leaves unwritten
  dup-map    entries of a (source version, map, part) that an earlier op of the same outer it already gathered
  dup-value  entries whose (source version, destination space, destination lane, source) an earlier op already
             gathered, whatever map id it came through (maps that overlap or are equal on this core)
  by source  field (state or solve output, not written in the outer body before the read), derived from fields
             only, derived from gathered values, gathered (a gather of a gather)
and writes every read with its source's defining site (trace.SRC, when the cache has it) to OUT.json so the
tables can be regrouped without loading the machine cache again.

    python -m ttfluid.tools.gval MCACHE[.zst] OUT.json [CORE]
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from ttfluid.resident.census import region_of
from ttfluid.resident.machine import _mir_defs
from ttfluid.resident.program import PHASES
from ttfluid.tools.solve_census import load


def main(path, out, core=0):
    LOWER, prog, m = load(path)
    mir, reg = m.mir, region_of(m.mir)
    pa = getattr(prog, "phase_at", {})
    src_site = getattr(prog, "src", {}) or {}
    slotfold = getattr(prog, "slotfold", {}) or {}
    sname = {}
    for name, arr in getattr(prog, "state", []):
        for r in np.asarray(arr, object).ravel():
            if isinstance(r, (int, np.integer)):
                sname[int(r)] = name

    mc = {}

    def mrows(mid):
        if mid not in mc:
            t = m.maps.tab[mid]
            sel = t["core"] == core
            mc[mid] = dict(lane=t["lane"][sel].astype(np.int64), kind=t["kind"][sel].astype(np.int64),
                           part=t["part"][sel].astype(np.int64), idx=t["idx"][sel].astype(np.int64),
                           zero=int((t["zc"] == core).sum()), nparts=int(t.get("nparts", 1)))
        return mc[mid]

    def is_vec(r):
        try:
            return m.is_vec(r)
        except Exception:
            return False

    ver = Counter()          # register -> writes so far
    dfn = {}                 # (reg, ver) -> ("solve",) | ("blk", bases) | ("gather", (src, ver), mid)

    def bases(key):
        d = dfn.get(key)
        if d is None or d[0] == "solve":
            return {("field", key[0])}
        if d[0] == "gather":
            return {("G", d[1][0], d[2])}
        return d[1]

    phase, rows, odd = "init", [], Counter()

    def read(ph, j, op, mid, srcs, single, tiles, dst):
        """One op's reads through map mid: srcs = [(register, shift)] per part, or single = the one source."""
        t = mrows(mid)
        if srcs:
            if len(srcs) != t["nparts"]:
                odd["parts != sources"] += 1
            per = defaultdict(list)
            for k, (s, sh) in enumerate(srcs):
                if s is not None:
                    per[s].append(k)
        else:
            per = {single: None}
        for s, parts in per.items():
            if s is None:
                continue
            rows.append(dict(ph=ph, j=j, op=op, s=int(s), v=ver[s], mid=int(mid), parts=parts, tiles=int(tiles),
                             dst=dst))

    for j, ins in enumerate(mir):
        if ins[0] == "label" and ins[1] in pa:
            phase = PHASES[pa[ins[1]]]
        outer = reg[j].startswith("outer")
        blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"] if ins[0] == "seg" else []
        if not outer:
            for d in _mir_defs(ins):
                if isinstance(d, int):
                    ver[d] += 1
                    dfn[(d, ver[d])] = ("solve",)
            continue
        if ins[0] == "gather":
            srcs = list(ins[3])
            read(phase, j, "gather", ins[2], srcs, None, m.size(ins[1]), int(ins[1]))
            d = ins[1]
            s0 = srcs[0][0] if srcs and srcs[0][0] is not None else None
            ver[d] += 1
            dfn[(d, ver[d])] = ("gather", (s0, ver[s0]) if s0 is not None else (None, 0), ins[2])
            continue
        for b in blks:
            bs = set()
            outs = [int(x) for _, x in b[2]]
            for u in b[1]:
                if u[0] == "ldr":
                    if u[4] is not None:
                        srcs = list(u[5]) if u[5] else None
                        read(phase, j, "patch", u[4], srcs, u[2], m.tiles([b]), outs[0] if outs else -1)
                        bs.add(("G", srcs[0][0] if srcs else u[2], u[4]))
                    elif u[2] is not None and is_vec(u[2]):
                        bs |= bases((u[2], ver[u[2]]))
                elif u[0] in ("ld", "lds", "rvf") and u[2] != "__anchor__" and isinstance(u[2], int) and is_vec(u[2]):
                    if u[0] == "lds":
                        bs.add(("S", u[2]))
                    else:
                        bs |= bases((u[2], ver[u[2]]))
            for x in outs:
                ver[x] += 1
                dfn[(x, ver[x])] = ("blk", frozenset(bs))
        if not blks:
            for d in _mir_defs(ins):
                if isinstance(d, int):
                    ver[d] += 1
                    dfn[(d, ver[d])] = ("solve",)

    # ---- per read: entries, class of the source, and the keys of its values ------------------------------------
    dsid = {}

    def dst_space(mid):
        try:
            return m.maps.maps[mid][0].name
        except Exception:
            return "map%d" % mid

    def cls(key):
        d = dfn.get(key)
        if d is None:
            return "field"
        if d[0] == "solve":
            return "solve-output"
        if d[0] == "gather":
            return "gathered"
        kinds = {x[0] for x in d[1]}
        return "derived:fields-only" if kinds <= {"field"} else "derived:has-gathered" if "G" in kinds \
            else "derived:shifted"

    seen_map, keys_by_src = set(), defaultdict(list)
    for r in rows:
        t = mrows(r["mid"])
        sel = np.ones(len(t["lane"]), bool) if r["parts"] is None else np.isin(t["part"], r["parts"])
        r["e"] = int(sel.sum())
        r["halo"] = int((t["kind"][sel] == 1).sum())
        r["zero"] = t["zero"]
        key = (r["s"], r["v"])
        r["cls"] = cls(key)
        r["name"] = sname.get(r["s"]) or src_site.get(r["s"]) or ""
        r["slot"] = slotfold.get(r["s"], slotfold.get(r["dst"], -1))
        d = dfn.get(key)
        r["nbase"] = len(d[1]) if d is not None and d[0] == "blk" else 0
        r["fbase"] = sorted({sname.get(x[1], str(x[1])) for x in d[1] if x[0] == "field"})[:12] \
            if d is not None and d[0] == "blk" else []
        mk = (key, r["mid"], tuple(r["parts"]) if r["parts"] is not None else None)
        r["dupmap"] = r["e"] if mk in seen_map else 0
        seen_map.add(mk)
        ds = dsid.setdefault(dst_space(r["mid"]), len(dsid))
        r["dsp"] = dst_space(r["mid"])
        k = (((ds * 2 + t["kind"][sel]) << 44) | (t["lane"][sel] << 22) | t["idx"][sel])
        keys_by_src[key].append((len(keys_by_src[key]), r, k))
    for key, lst in keys_by_src.items():
        seen = np.zeros(0, np.int64)
        for _, r, k in lst:
            new = np.setdiff1d(k, seen)
            r["dupval"] = int(len(k) - len(new))
            seen = np.union1d(seen, new)

    # ---- tables ---------------------------------------------------------------------------------------------------
    phs = []
    for r in rows:
        if r["ph"] not in phs:
            phs.append(r["ph"])
    cells = None
    try:
        cells = int((m.lay.owner[prog.cells.name] == core).sum())
    except Exception:
        pass
    cs = ["field", "solve-output", "derived:fields-only", "derived:has-gathered", "derived:shifted", "gathered"]
    print(path.split("/")[-1], "T", m.T, "core", core, "cells", cells, "| src sites", len(src_site), "| odd", dict(odd))
    tot = Counter()
    for ph in phs + ["all"]:
        rr = [r for r in rows if ph in ("all", r["ph"])]
        e = sum(r["e"] for r in rr)
        line = "  %-9s total %9d ops %5d | halo %8d | dup-map %8d (%4.1f %%) | dup-value %8d (%4.1f %%)" % (
            ph, e, len(rr), sum(r["halo"] for r in rr), sum(r["dupmap"] for r in rr),
            100 * sum(r["dupmap"] for r in rr) / max(e, 1), sum(r["dupval"] for r in rr),
            100 * sum(r["dupval"] for r in rr) / max(e, 1))
        if cells:
            line += " | per cell %6.1f, after dup-value %6.1f" % (e / cells, (e - sum(r["dupval"] for r in rr)) / cells)
        print(line)
        print("            " + " | ".join("%s %d (%.1f %%)" % (c, sum(r["e"] for r in rr if r["cls"] == c),
                                                             100 * sum(r["e"] for r in rr if r["cls"] == c) / max(e, 1))
                                          for c in cs if any(r["cls"] == c for r in rr)))
    byname = defaultdict(lambda: [0, 0, 0, set()])
    for r in rows:
        b = byname[(r["cls"], r["name"])]
        b[0] += r["e"]; b[1] += r["dupval"]; b[2] += 1; b[3].add(r["ph"])
    print("  by source (class, name or site): entries, dup-value, ops, phases")
    for (c, nm), b in sorted(byname.items(), key=lambda x: -x[1][0])[:60]:
        print("    %-22s %-44s %9d %9d %5d  %s" % (c, nm[:44], b[0], b[1], b[2], ",".join(sorted(b[3]))))
    json.dump(dict(cache=path, core=core, T=int(m.T), cells=cells, phases=phs, nsites=len(src_site),
                   rows=[{k: v for k, v in r.items()} for r in rows]), open(out, "w"))
    print("  wrote", out, len(rows), "reads")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0)
