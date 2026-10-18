"""DRAM traffic of a placed Machine program, by region, from memu's execution counts (MachineEmu.pcc).

Per core and per outer iteration: pages the reader builds for ldr loads (one 4 KB page per tile), pages the writer
stores to DRAM homes, gather values read from DRAM-homed sources (as gather-cache misses of 4 KB, simulated for one
core with the device's round-robin cache), and residue table bytes (streamed from DRAM under tabstream). Regions:
the outer body (the outer loop minus its inner loops) and each inner (Krylov) loop, named by its solve index.
"""
from collections import Counter, defaultdict

import numpy as np

from .machine import DBASE, KBASE


def loops(mir):
    labels = {ins[1]: i for i, ins in enumerate(mir) if ins[0] == "label"}
    return sorted({(labels[ins[-1]], j) for j, ins in enumerate(mir)
                   if ins[0] in ("jump", "jumpif") and labels[ins[-1]] < j})


def region_of(mir):
    """MIR index -> region name: 'top', 'outer', or 'solveK' (K-th inner loop in program order)."""
    ls = loops(mir)
    outer = max(ls, key=lambda l: l[1] - l[0]) if ls else None
    inner = [l for l in ls if l != outer]
    # a pbicgstab has a loop with an inner exit; nested inner loops count with their outermost inner loop
    tops = [l for l in inner if not any(o != l and o[0] <= l[0] and l[1] <= o[1] for o in inner)]
    out = []
    for j in range(len(mir)):
        k = next((i for i, l in enumerate(tops) if l[0] <= j <= l[1]), None)
        if k is not None:
            out.append(f"solve{k}")
        elif outer and outer[0] <= j <= outer[1]:
            out.append("outer")
        else:
            out.append("top")
    return out


def _misses(srcs, ngc):
    """Round-robin cache of ngc tiles over a sequence of source tiles: misses."""
    if ngc == 0:
        return len(srcs)
    tags, nxt, miss = [None] * ngc, 0, 0
    for t in srcs:
        if t not in tags:
            tags[nxt] = t
            nxt = (nxt + 1) % ngc
            miss += 1
    return miss


class Census:
    def __init__(self, m, pcc, n_outer, core=0, gcache=8):
        self.m, self.core, self.gcache = m, core, gcache
        T, D = m.T, m.dram
        reg = region_of(m.mir)
        self.rows = defaultdict(Counter)
        self.regs = defaultdict(Counter)    # region -> (r|w, register) -> vector passes (one per T pages)
        self._gm = {}
        for j, ins in enumerate(m.mir):
            n = pcc.get(j, 0)
            if not n:
                continue
            r = self.rows[reg[j]]
            blks = [ins] if ins[0] == "blk" else [x for x in ins[1] if x[0] == "blk"] if ins[0] == "seg" else []
            for b in blks:
                vec = any(u[0] == "ldr" or (u[0] in ("ld", "lds") and u[2] != "__anchor__" and m.is_vec(u[2]))
                          for u in b[1]) or any(m.is_vec(x) for _, x in b[2])
                bt = m.tiles(blks)
                passes = bt if (vec or ins[0] == "seg") else 1
                r["c_tiles"] += n * passes
                for u in b[1]:
                    if u[0] == "ldr":
                        r["rd_pages"] += n * bt / (2 if u[2] in getattr(m, "narrow_home", ()) else 1)
                        self.regs[reg[j]][("r", u[2])] += n
                        if u[2] is not None and m.phys[u[2]] >= KBASE:
                            r["rd_const"] += n * bt
                            if m.fetched_from.get(u[2]) in getattr(m, "narrow", ()):
                                r["rd_pages"] -= n * bt / 2     # a bf16 page is half a page of bytes
                                r["rd_narrow"] += n * bt
                        if u[4] is not None:
                            self._patch(r, n, u[4], u[5])
                    elif u[0] in ("ld", "lds") and u[2] != "__anchor__" and m.is_vec(u[2]):
                        r["l1_tiles"] += n * passes
                for _, x in b[2]:
                    if x in D:
                        r["wr_pages"] += n * bt / (2 if x in getattr(m, "narrow_home", ()) else 1)
                        self.regs[reg[j]][("w", x)] += n
                        r["dram_stores"] += n
            if ins[0] == "gather" and (len(ins) < 5 or ins[4] in (1, 3, 4)):   # gsplit: count a split gather once
                self._patch(r, n, ins[2], ins[3])
            elif ins[0] in ("red", "red2", "halo"):
                r[ins[0]] += n
            if ins[0] == "seg":
                r["runs"] += n
        self.n = n_outer

    def _patch(self, r, n, mid, parts):
        """Gather residue: table bytes for this core, DRAM-source values as cache misses (pages)."""
        m, c = self.m, self.core
        t = m.maps.tab[mid]
        sel = t["core"] == c
        r["tab_bytes"] += n * 4 * (int(sel.sum()) + int((t["zc"] == c).sum()))
        key = (mid, tuple(s for s, _ in parts))
        if key not in self._gm:
            g = 0
            order = np.lexsort((t["lane"][sel], t["part"][sel]))
            kind, part, idx = t["kind"][sel][order], t["part"][sel][order], t["idx"][sel][order]
            for k, (s, _) in enumerate(parts):
                if m.phys[s] < DBASE:
                    continue
                e = (part == k) & (kind == 0)
                g += _misses((idx[e] >> 10).tolist(), self.gcache)
            self._gm[key] = g
        r["g_pages"] += n * self._gm[key]

    def table(self, cells_per_core=None):
        """Lines: per region, per outer iteration and core: pages read (ldr + gather misses), written, KB/cell."""
        m = self.m
        cpc = cells_per_core or int(m.lay.count[m.prog.cells.name].mean())
        tot = Counter()
        for r in self.rows.values():
            tot.update(r)
        out = []
        hdr = "%-8s %9s %9s %8s %8s %9s %8s %7s %6s %5s" % ("region", "ldr pg", "store pg", "gmiss pg", "tab KB",
                                                            "KB/cell", "c tiles", "stores", "share", "runs")
        out.append(hdr)

        def kb(r):
            return (4096 * (r["rd_pages"] + r["wr_pages"] + r["g_pages"]) + r["tab_bytes"]) / cpc / 1024 / self.n
        allkb = kb(tot) or 1
        for name in sorted(self.rows, key=lambda k: -kb(self.rows[k])):
            r = self.rows[name]
            out.append("%-8s %9d %9d %8d %8.0f %9.2f %8d %7d %5.1f%% %5d" % (
                name, r["rd_pages"] / self.n, r["wr_pages"] / self.n, r["g_pages"] / self.n,
                r["tab_bytes"] / self.n / 1024, kb(r), r["c_tiles"] / self.n, r["dram_stores"] / self.n,
                100 * kb(r) / allkb, r["runs"] / self.n))
        out.append("%-8s %9d %9d %8d %8.0f %9.2f %8d %7d" % (
            "total", tot["rd_pages"] / self.n, tot["wr_pages"] / self.n, tot["g_pages"] / self.n,
            tot["tab_bytes"] / self.n / 1024, kb(tot), tot["c_tiles"] / self.n, tot["dram_stores"] / self.n))
        self.totals = tot
        return out

    def reg_lines(self, region):
        """Per outer iteration in one region: DRAM vector reads and writes by register, largest first."""
        c = self.regs[region]
        return ["%s %-24s %10.1f" % (rw, x, n / self.n) for (rw, x), n in c.most_common()]


def prof(hdr, log=print):
    """CFD_PROF device census from the per-core headers (run_vm / run_mesh --prof): means over cores of the cycles and
    counts per op of every interpreter thread and agent, their share of the run, and the parts of STREAM, GATHER, HALO."""
    import json
    # census, means over cores: cycles per op of each interpreter thread / agent and their share of the run
    CN = ["END", "ACQ", "COMMIT", "REL", "LD", "ST", "FILL", "RVF", "add", "sub", "mul", "div", "max", "min", "WHERE",
          "CPD", "neg", "abs", "sqrt", "tanh", "log", "exp", "gtz", "gez", "ltz", "lez", "eqz", "nez", "FOLD", "TOK",
          "SIGB", "SIGN", "WAITB", "WAITN", "JMP", "BRT", "LDS", "TBEG", "TEND", "LDR+STW"]   # 40 slots: STW counts with LDR
    BN = ["SCAL", "SIGC", "WAITC", "SIGN", "WAITN", "HALO", "RED", "OUT", "FETCH", "JMP", "BRT", "SINK"]
    NN = ["END", "SIGC", "WAITC", "SIGB", "WAITB", "GATHER", "HALO", "JMP", "BRT", "STREAM"]
    h = hdr.astype(np.float64)
    for lo, hi in ((64, 80), (96, 112), (128, 248), (288, 296)):   # cycle words are stored in 16-cycle units
        h[:, lo:hi] *= 16
    def tab(base, cbase, names):
        cyc, cnt = h[:, base:base + len(names)].mean(0), h[:, cbase:cbase + len(names)].mean(0)
        tot = cyc.sum()
        return dict(total=tot, ops={nm: dict(cyc=round(float(c)), n=round(float(k)), per=round(float(c / max(k, 1)), 1),
                                              share=round(float(c / tot), 4))
                                    for nm, c, k in zip(names, cyc, cnt) if k})
    c = dict(unpack=tab(128, 248, CN), math=tab(168, 248, CN), pack=tab(208, 248, CN),
                         b=tab(64, 80, BN), n=tab(96, 112, NN), run_ticks_core0=int(hdr[0, 10]))
    # parts of STREAM, not in its total: free-slot wait, read barriers, patched pages (barrier + residue)
    c["n_stream"] = {nm: dict(cyc=round(float(h[:, 106 + i].mean())), n=round(float(h[:, 122 + i].mean())))
                                 for i, nm in enumerate(("slot_wait", "barrier", "patch"))}
    # parts (hdr 288 + i cycles, 296 + i counts): the patch's barrier and residue entries, GATHER's halo wait and scatters
    part = lambda i: dict(cyc=round(float(h[:, 288 + i].mean())), n=round(float(h[:, 296 + i].mean())))
    c["n_stream"]["patch_barrier"] = part(0)
    c["n_stream"]["patch_entries"] = dict(cyc=0, n=part(1)["cyc"])
    c["n_gather"] = dict(halo_wait=part(2), scatter=part(3))
    c["n_stream"].update(ent_l1=part(4), ent_dram=part(5), ent_halo=part(6), ent_zero=part(7))
    c["b_halo"] = {nm: dict(cyc=round(float(h[:, 76 + i].mean())), n=round(float(h[:, 92 + i].mean())))
                               for i, nm in enumerate(("credit_wait", "copy", "wbar"))}
    log("census parts", json.dumps(dict(n_gather=c["n_gather"], b_halo=c["b_halo"])), flush=True)
    for k in ("unpack", "math", "pack", "b", "n"):
        t = c[k]
        top = sorted(t["ops"].items(), key=lambda kv: -kv[1]["cyc"])[:10]
        log("census %-6s total %.3g cyc: " % (k, t["total"]) +
              ", ".join("%s %.1f%% (%d x %.0f)" % (nm, 100 * v["share"], v["n"], v["per"]) for nm, v in top), flush=True)
    log("census n STREAM parts: " + ", ".join("%s %.3g cyc (%d)" % (nm, v["cyc"], v["n"])
                                                for nm, v in c["n_stream"].items()), flush=True)
    return c
