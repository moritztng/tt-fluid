"""Residue tables streamed from DRAM instead of held in L1.

Today every core holds its whole gather table in L1 (encode.py _tables): per map and part the local and halo residue
entries (lane << 16 | idx, sorted by destination lane), then the zero lanes; NCRISC walks them with one cursor per
section, a tile at a time. At 65M cells on 31 chips that is ~0.9 MB per core.

Streamed layout, per core:
  L1:   [nmaps, map_off | Tm << 20...]; per map: the map's base offset into the core's DRAM blob, a mask of the
        parts that read a halo (bit k), then the sections' tile bounds (loc and halo of each part, then the zeros;
        sections are back to back in the blob, so section k tile t starts at base + H[k * Tm + t]) as uint16 offsets
        from base, two per word, nsec * Tm + 1 of them; then the halo section exactly as today. Tm: tiles up to the
        map's last entry on this core (a multigrid level's map writes a one- or few-tile vector); tiles past Tm have
        no entries.
  DRAM: the sections' entries back to back, in the order the L1 index names them.
Because each section is sorted by lane, tile t's entries of a section are blob[idx[t]:idx[t + 1]]: no cursor state,
and the reader can prefetch tile t + 1 of the patched map while it patches tile t (two staging buffers of one
tile's words; a lane is patched at most once per map, so one tile needs at most 1024 words over all its sections).

Encoded(m, tab_stream=True) uses it: E.tab becomes the L1 part, E.tab_blob the DRAM part, E.tab_halo the halo
section's offset in the new L1 table. stream(E) builds the layout from an Encoded program's cursor tables; check(E, S) replays NCRISC's cursor walk on the L1 table and the
streamed walk on the blob for every core, map and tile and asserts they patch the same lanes from the same sources.
"""
import numpy as np

from .machine import NH
from . import encode as _enc     # _enc.WIDE: wide entries (encode.py), read at call time


def _xs():
    return 22 if _enc.WIDE else 16


def _xm():
    return (1 << _xs()) - 1


def _cut(kind, x):
    """an entry as the device reads it: WIDE keeps the low 32 bits (the lane within its tile), zero lanes mod 1024"""
    if not _enc.WIDE:
        return x
    return x & 1023 if kind == "z" else x & 0xFFFFFFFF
RUNSTATS = None   # list: per local section, a bincount of run lengths (capped at 64); set by run_vm --runstats


def _sections(tab, mid, nparts):
    """[(kind, entries)] of map mid in one core's L1 table, kind 'l' local, 'h' halo, 'z' zero lanes."""
    q = int(tab[1 + mid])
    out, counts = [], []
    for _ in range(nparts):
        nl, nh = int(tab[q]), int(tab[q + 1])
        counts.append((nl, nh))
        out.append(("l", tab[q + 2:q + 2 + nl]))
        out.append(("h", tab[q + 2 + nl:q + 2 + nl + nh]))
        q += 2 + nl + nh
    nz = int(tab[q])
    out.append(("z", tab[q + 1:q + 1 + nz]))
    return out, counts


def _lane(kind, x):
    return x if kind == "z" else x >> _xs()


def _half(x):
    """uint16 values two per word, value 2k in the low half"""
    a = np.asarray(x, np.uint32)
    if len(a) % 2:
        a = np.r_[a, 0]
    return (a[0::2] | a[1::2] << 16).tolist()


def _hdir(E, i):
    """words of core i's halo directory, just below tab_halo (encode.py _tables): NH slot plans + plan offsets"""
    return NH - 1 + len(E.m.maps.plans) if E.halo_dir[i] else 0


def _by_src_tile(ent, bounds):
    """sort: within each page tile, a local section's entries by source tile (idx >> 10), so patch() walks runs of one
    cached tile (vm_ncrisc.cpp CFD_TSORT). A lane is patched at most once per map, so the order is free."""
    ent = ent.copy()
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > 1:
            seg = ent[a:b]
            ent[a:b] = seg[np.argsort((seg & _xm()) >> 10, kind="stable")]
    return ent


def stream(E, sort=False):
    m = E.m
    P, T, nm = len(E.tab), m.T, len(m.maps.tab)      # E's rows: every core, or the cores of Encoded(cores=)
    nparts = [t["nparts"] for t in m.maps.tab]
    l1, blob, halo_off, staging, loc = [], [], [], 0, []
    for i in range(P):
        tab = E.tab[i]
        L = [nm] + [0] * nm
        B = []
        for mid in range(nm):
            secs, counts = _sections(tab, mid, nparts[mid])
            tm = max([int(_lane(k, e.astype(np.int64)).max()) // 1024 + 1 for k, e in secs if len(e)] or [1])
            assert tm <= T and len(L) < 1 << 20
            L[1 + mid] = len(L) | tm << 20
            base, H = len(B), []
            per_tile = np.zeros(tm, np.int64)
            loc_tile = np.zeros(tm, np.int64)
            for kind, ent in secs:
                lanes = _lane(kind, ent.astype(np.int64))
                bounds = np.searchsorted(lanes, np.arange(tm + 1) * 1024)
                assert bounds[-1] == len(ent)
                if sort and kind == "l":
                    ent = _by_src_tile(ent, bounds)
                if RUNSTATS is not None and kind == "l" and len(ent):
                    # runs: consecutive entries with lane and idx both +1, cut at page-tile bounds
                    e64 = ent.astype(np.int64)
                    step = (np.diff(e64 >> _xs()) == 1) & (np.diff(e64 & _xm()) == 1)
                    cut = np.zeros(len(ent), bool)
                    cut[bounds[(bounds > 0) & (bounds < len(ent))]] = True
                    start = np.r_[True, ~step] | cut
                    ln = np.diff(np.r_[np.flatnonzero(start), len(ent)])
                    RUNSTATS.append(np.bincount(np.minimum(ln, 64), minlength=65))
                H += (len(B) - base + bounds[:tm]).tolist()
                B += _cut(kind, ent.astype(np.int64)).tolist()
                per_tile += np.diff(bounds)
                if kind == "l":
                    loc_tile += np.diff(bounds)
            H.append(len(B) - base)
            assert (H[-1] < 1 << 32 if _enc.WIDE else H[-1] < 1 << 16) and len(counts) <= 32
            # section offsets: two uint16 a word, or (WIDE: a map may hold more than 65,536 entries on a core) one a word
            L += [base, sum(1 << k for k, (_, nh) in enumerate(counts) if nh)] + (H if _enc.WIDE else _half(H))
            loc.append(loc_tile[loc_tile > 0])
            staging = max(staging, int(per_tile.max()) if T else 0)
        d = _hdir(E, i)    # the halo directory below tab_halo (more than one plan) moves with the plans
        halo_off.append(len(L) + d)
        L += tab[int(E.tab_halo[i]) - d:E.tab_len[i]].tolist()   # not the padding up to the longest core's table
        l1.append(np.array(L, np.uint32))
        blob.append(np.array(B, np.uint32))
    loc = np.concatenate(loc) if loc else np.zeros(0)
    # loc_mean: local residue entries per patched (map, tile), mean over cores (run_vm --bpatch auto)
    return dict(sort=sort, loc_mean=float(loc.mean()) if len(loc) else 0.0, l1=l1, blob=blob, halo_off=halo_off, staging_words=staging, nparts=nparts,
                l1_words=max(len(x) for x in l1), blob_words=max(len(x) for x in blob),
                today_words=E.tab.shape[1])


def _walk_cursor(tab, mid, nparts, T):
    """NCRISC today: per section a cursor advanced while the entry's lane is below the tile's end."""
    secs, _ = _sections(tab, mid, nparts)
    cur = [0] * len(secs)
    out = []
    for t in range(T):
        hi = (t + 1) * 1024
        got = []
        for s, (kind, ent) in enumerate(secs):
            while cur[s] < len(ent) and _lane(kind, int(ent[cur[s]])) < hi:
                got.append((kind, int(ent[cur[s]])))
                cur[s] += 1
        out.append(got)
    assert all(c == len(e) for c, (_, e) in zip(cur, secs)), "entries beyond the last tile"
    return out


def _walk_stream(L, B, mid, nparts, T):
    q, tm = int(L[1 + mid]) & 0xFFFFF, int(L[1 + mid]) >> 20
    kinds = ["l", "h"] * nparts + ["z"]
    base, nh = int(L[q]), len(kinds) * tm + 1
    if _enc.WIDE:
        H = base + np.asarray(L[q + 2:q + 2 + nh], np.int64)
    else:
        w = np.asarray(L[q + 2:q + 2 + (nh + 1) // 2], np.int64)
        H = base + np.stack([w & 0xFFFF, w >> 16], 1).ravel()[:nh]
    return [[(kind, int(x)) for k, kind in enumerate(kinds) for x in B[H[k * tm + t]:H[k * tm + t + 1]]] if t < tm
            else [] for t in range(T)]


def check(E, S, tab=None):
    """tab: the cursor tables (E.tab of a tab_stream=False encoding, or E.tab_cursor)."""
    m = E.m
    tab = E.tab if tab is None else tab
    for i in range(m.P):
        for mid, nparts in enumerate(S["nparts"]):
            a = [[(k, _cut(k, x)) for k, x in tile] for tile in _walk_cursor(tab[i], mid, nparts, m.T)]
            b = _walk_stream(S["l1"][i], S["blob"][i], mid, nparts, m.T)
            if S.get("sort"):   # same entries per tile, order free
                a, b = [sorted(x) for x in a], [sorted(x) for x in b]
            assert a == b, f"core {i} map {mid}: streamed walk differs"
        h0, h1 = int(E.tab_halo_cursor[i] if hasattr(E, "tab_halo_cursor") else E.tab_halo[i]), S["halo_off"][i]
        d = _hdir(E, i)
        n = len(S["l1"][i]) - h1
        assert (S["l1"][i][h1 - d:].astype(np.int64) == tab[i][h0 - d:h0 + n]).all() and not tab[i][h0 + n:].any()
    return True


if __name__ == "__main__":
    # python3 -m ttfluid.resident.tabstream <pkl> P [--fuse]   (from resident/, with solver/ on PYTHONPATH)
    import pickle
    import sys
    import time
    from .machine import Machine
    from .encode import Encoded
    p, mesh = pickle.load(open(sys.argv[1], "rb"))
    t0 = time.time()
    m = Machine(p, mesh, int(sys.argv[2]), layout="rcb", fuse="--fuse" in sys.argv, log=lambda *a: None)
    m.place(0)
    E = Encoded(m, tab_stream=True)
    S = E.tab_stream
    check(E, S, E.tab_cursor)
    print(f"P {m.P} T {m.T}: streamed walk == cursor walk on every core, map and tile ({time.time() - t0:.0f} s). "
          f"L1 table words today {S['today_words']} -> {S['l1_words']} ({S['l1_words'] * 4 / 1000:.0f} KB); DRAM blob "
          f"{S['blob_words']} words/core; staging per tile {S['staging_words']} words", flush=True)
