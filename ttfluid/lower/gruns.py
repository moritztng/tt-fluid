"""Gather run census: how many gathered values per contiguous source run, per p-solve map and level, on real images.

    python3 gruns.py SAVE.pkl [--images DIR --chips 0,8,16,31] [--out runs.json]
    run_mesh.py ... --gruns OUT.json      (in process after the encoding, then exit: no pickle of a large case)

SAVE: a run_mesh --save (its maps, part counts, MIR); --images: per-chip images of the SAME lowering (map ids follow the
mesh's patches, so a save of another mesh does not name them). Weights: gathers of each map inside p_solve's CG loop
body (once per p iteration) and in the rest of p_solve (once per solve), level from the destination space.
A run: consecutive entries of one local section (one source part) whose destination lane and source position
both step by one, inside one destination page tile and one source tile. 'al4': runs also cut where lane and position
disagree mod 4 (what a 16-byte NoC copy needs). Halo sections are counted the same way, by receive-buffer position.
"""
import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

def runs(ent, al4):
    """run lengths of one section's entries (lane << 16 | pos, sorted by lane)"""
    if not len(ent):
        return np.zeros(0, np.int64)
    e = ent.astype(np.int64)
    ln, ps = e >> 16, e & 0xFFFF
    step = (np.diff(ln) == 1) & (np.diff(ps) == 1) & (np.diff(ln >> 10) == 0) & (np.diff(ps >> 10) == 0)
    start = np.flatnonzero(np.r_[True, ~step])
    rl = np.diff(np.r_[start, len(e)])
    if al4:     # lane and pos step together inside a run, so a run is aligned or not as a whole; unaligned -> singles
        ok = (ln[start] % 4) == (ps[start] % 4)
        rl = np.r_[rl[ok], np.ones(int(rl[~ok].sum()), np.int64)]
    return rl


def weights(m, prog):
    """per map: gathers per p it (inside p_solve's CG loop body), per solve (the rest of p_solve), level written"""
    from ttfluid.resident.program import PHASES
    mir, phase_at = m.mir, getattr(prog, "phase_at", {}) or {}
    PS, NM = PHASES.index("p_solve"), len(m.maps.tab)
    lab = {ins[1]: j for j, ins in enumerate(mir) if ins[0] == "label"}
    p0 = min(j for j, ins in enumerate(mir) if ins[0] == "label" and phase_at.get(ins[1]) == PS)
    p1 = min([j for j, ins in enumerate(mir) if ins[0] == "label" and ins[1] in phase_at and j > p0] or [len(mir)])
    body = np.zeros(len(mir), bool)
    for j in range(p0, p1):
        ins = mir[j]
        if ins[0] in ("jump", "jumpif") and p0 <= lab.get(ins[-1], j) < j:
            body[lab[ins[-1]]:j + 1] = True
    w_it, w_sol, level = np.zeros(NM), np.zeros(NM), {}
    for j in range(p0, p1):
        ins = mir[j]
        if ins[0] != "gather" or (len(ins) > 4 and ins[4] == 2):     # gsplit mode 2 is the halo tail of the same gather
            continue
        mid = ins[2]
        (w_it if body[j] else w_sol)[mid] += 1
        mt = re.match(r"amgs(\d+)", m.space[ins[1]].name)
        level[mid] = int(mt.group(1)) if mt else 0
    return w_it, w_sol, level


def census(m, prog, rows, log=print):
    """rows: iterable of (L1 index row, blob row) of the tabstream tables, one per core. Returns the summary dict."""
    w_it, w_sol, level = weights(m, prog)
    nparts = [t["nparts"] for t in m.maps.tab]
    NM = len(nparts)
    log(f"{NM} maps; p_solve CG body gathers {int(w_it.sum())} over {int((w_it > 0).sum())} maps, once per solve {int(w_sol.sum())}")
    acc = defaultdict(lambda: defaultdict(lambda: np.zeros(65, np.int64)))
    cores = 0
    for L, Bi in rows:
        assert int(L[0]) == NM, (int(L[0]), NM)
        for mid in range(NM):
            if not (w_it[mid] or w_sol[mid]):
                continue
            q, tm = int(L[1 + mid]) & 0xFFFFF, int(L[1 + mid]) >> 20
            kinds = ["l", "h"] * nparts[mid] + ["z"]
            base, nh = int(L[q]), len(kinds) * tm + 1
            w = np.asarray(L[q + 2:q + 2 + (nh + 1) // 2], np.int64)
            H = base + np.stack([w & 0xFFFF, w >> 16], 1).ravel()[:nh]
            for s, kind in enumerate(kinds[:-1]):
                ent = np.asarray(Bi[H[s * tm]:H[(s + 1) * tm]])     # a section's tiles are back to back
                for key, al in (("r", False), ("r4", True)):
                    rl = runs(ent, al)
                    if len(rl):
                        acc[(mid, kind)][key] += np.bincount(np.minimum(rl, 64), minlength=65)
                        acc[(mid, kind)][key + "n"][:2] += (len(rl), int(rl.sum()))     # exact: the histogram caps at 64
        cores += 1
    out = {"cores": cores, "maps": {}, "levels": {}}
    tot = defaultdict(lambda: np.zeros(4))
    for (mid, kind), hs in sorted(acc.items()):
        r, v = (int(x) for x in hs["rn"][:2])
        r4 = int(hs["r4n"][0])
        out["maps"][f"{mid}{kind}"] = dict(level=level.get(mid), it=int(w_it[mid]), solve=int(w_sol[mid]), values=v / cores,
                                           runs=r / cores, runs4=r4 / cores, hist=hs["r"][:17].tolist())
        tot[(level.get(mid), kind)] += np.array([w_it[mid] * v, w_it[mid] * r, w_it[mid] * r4, w_sol[mid] * v]) / cores
    log("per core per p it (CG body), by level and section kind (l local, h halo):")
    log(f"{'level':>5} {'kind':>4} {'values':>10} {'runs':>10} {'val/run':>7} {'runs al4':>10} {'val/run4':>8} {'per-solve vals':>14}")
    for (lv, kind), t in sorted(tot.items(), key=lambda x: (x[0][0] if x[0][0] is not None else 9, x[0][1])):
        log(f"{lv!s:>5} {kind:>4} {t[0]:10.0f} {t[1]:10.0f} {t[0] / max(t[1], 1):7.3f} {t[2]:10.0f} {t[0] / max(t[2], 1):8.3f} {t[3]:14.0f}")
        out["levels"][f"{lv}{kind}"] = dict(values=t[0], runs=t[1], runs4=t[2], per_solve=t[3])
    V, R, R4 = (sum(t[i] for t in tot.values()) for i in range(3))
    log(f"all: {V:.0f} values in {R:.0f} runs ({V / max(R, 1):.3f} per run), {R4:.0f} 16-B-aligned runs "
        f"({V / max(R4, 1):.3f} per run) per core per p it")
    out["all"] = dict(values=V, runs=R, runs4=R4)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("save")
    ap.add_argument("--images", help="read the entries from these images (same flags, same mesh) instead of the save")
    ap.add_argument("--chips", default="0,8,16,31")
    ap.add_argument("--out")
    a = ap.parse_args()
    with open(a.save, "rb") as f:
        LOWER, prog, m, E = pickle.load(f)
    if a.images:
        def rows():
            for k in [int(x) for x in a.chips.split(",")]:
                d = Path(a.images, f"chip{k}")
                T, B = np.load(d / "tab.npy", mmap_mode="r"), np.load(d / "blob.npy", mmap_mode="r")
                yield from zip(T, B)
    else:
        rows = lambda: zip(E.tab, E.tab_blob)
    out = census(m, prog, rows())
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
