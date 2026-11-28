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

Dense maps (stream(dense=), encode.py tab_dense): where nearly every lane of a reader-patched page has a residue entry,
the local section of the map is stored in destination order instead, one uint16 source word index per lane of the
tile (512 words a tile): the residue's source where the lane has one, lane + shift (the window's lane) elsewhere.
The reader then gathers the page in one pass with no window read (vm_ncrisc.cpp gather16: 6.4 cycles a value against
12.1 for a residue entry); the map's header word 1 carries DENSE. Halo and zero sections are unchanged.

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


DENSE = 1 << 31   # header word 1 of a map (the halo mask): its local section is a dense uint16 index table


def dense_index(ent, tm, shift):
    """The source word index of every lane of tiles [0, tm): the local residue entry's source where the lane has
    one, lane + shift elsewhere (what the window read; its own lane where that falls outside the 16-bit span of the
    source: a lane no consumer reads, memu._shifted)."""
    lane = np.arange(tm * 1024, dtype=np.int64)
    idx = lane + shift
    idx = np.where((idx >= 0) & (idx < 1 << 16), idx, lane)
    e = np.asarray(ent, np.int64)
    idx[e >> 16] = e & 0xFFFF
    return idx


def _enc_seg(seg, rmin):
    """Encode one contiguous lane segment into (COPY-RUN, LITERAL-GATHER) 32-bit words."""
    n = len(seg)
    if n == 0:
        return [], 0, 0.0
    cont = np.zeros(n, dtype=bool)
    cont[1:] = seg[1:] == seg[:-1] + 1
    starts = np.flatnonzero(~cont)
    ends = np.r_[starts[1:], n]
    words = []
    lit = []
    run_lanes = 0
    cost = 0.0

    def flush_lit():
        nonlocal cost
        if not lit:
            return
        L = len(lit)
        words.append(L)
        words.extend(_half(lit))
        cost += L * 6.91 + 8.0
        lit.clear()

    for s0, e0 in zip(starts, ends):
        L = int(e0 - s0)
        if L >= rmin:
            flush_lit()
            words.append((1 << 31) | (L << 16) | int(seg[s0]))
            run_lanes += L
            cost += L * 2.10 + 8.0
        else:
            lit.extend(seg[s0:e0].tolist())
    flush_lit()
    return words, run_lanes, cost


def encode_runcopy(tile_idx, tile_dc, max_src, rmin=4, weights=(1, 1, 1), brk=0):
    """Encode one 1024-lane dense tile as a run-copy stream (< 512 words) with a 5-word thread-split header,
    or return None to fall back to a plain 512-word dense tile.

    Header words 0..4: c1|c1_off<<16, c2|c2_off<<16 (3-thread), q1|q1_off<<16, q2|q2_off<<16, q3|q3_off<<16
    (4-thread, with q2 also serving as the balanced 2-thread midpoint); 1-thread runs from word 5.
    Don't-care lanes (overwritten by halo or zero sections) bridge runs while staying within [0, max_src]."""
    v = tile_idx.copy()
    if tile_dc.any():
        for j in range(1, 1024):
            if tile_dc[j] and v[j - 1] + 1 <= max_src:
                v[j] = v[j - 1] + 1
        j = 0
        while j < 1024 and tile_dc[j]:
            j += 1
        if j < 1024:
            for q in range(j - 1, -1, -1):
                if v[q + 1] > 0:
                    v[q] = v[q + 1] - 1
    cont = np.zeros(1024, dtype=bool)
    cont[1:] = v[1:] == v[:-1] + 1
    starts = np.flatnonzero(~cont)
    ends = np.r_[starts[1:], 1024]
    cost = np.full(1024, 6.91)
    is_run = np.zeros(1024, dtype=bool)
    for s0, e0 in zip(starts, ends):
        if e0 - s0 >= rmin:
            cost[s0:e0] = 2.10
            cost[s0] += 8.0
            is_run[s0:e0] = True
    nr = ~is_run
    lit_s = np.flatnonzero(nr & np.r_[True, ~nr[:-1]])
    if len(lit_s):
        cost[lit_s] += 8.0
    cc = np.cumsum(cost)
    tot = float(cc[-1])
    wn, wt, wb = weights
    ws = float(wn + wt + wb)
    c1 = int(np.clip(np.searchsorted(cc, tot * wn / ws), 1, 1022))
    c2 = int(np.clip(np.searchsorted(cc, tot * (wn + wt) / ws), c1 + 1, 1023))
    q1 = int(np.clip(np.searchsorted(cc, tot * 0.25), 1, 1021))
    q2 = int(np.clip(np.searchsorted(cc, tot * 0.50), q1 + 1, 1022))
    q3 = int(np.clip(np.searchsorted(cc, tot * 0.75), q2 + 1, 1023))
    pts = sorted({0, 1024, c1, c2, q1, q2, q3})
    off = {0: 5}
    body = []
    tot_runs = 0
    seg_cost = {}
    for a, b in zip(pts[:-1], pts[1:]):
        w_ab, r_ab, k_ab = _enc_seg(v[a:b], rmin)
        body.extend(w_ab)
        tot_runs += r_ab
        seg_cost[(a, b)] = k_ab
        off[b] = 5 + len(body)
    words = [
        c1 | (off[c1] << 16),
        c2 | (off[c2] << 16),
        q1 | (off[q1] << 16),
        q2 | (off[q2] << 16),
        q3 | (off[q3] << 16),
    ] + body
    if len(words) >= 512:
        return None
    # Per-tile win gate: keep plain DENSE if balanced 4T cost does not beat plain DENSE 256-lane d16a16
    def _span_cost(lo, hi):
        return sum(seg_cost[(a, b)] for a, b in zip(pts[:-1], pts[1:]) if lo <= a and b <= hi)
    if not brk and max(_span_cost(0, q1), _span_cost(q1, q2), _span_cost(q2, q3), _span_cost(q3, 1024)) >= 256 * 6.91:
        return None
    if brk:
        left = brk
        p = 5
        while p < len(words) and left > 0:
            w = words[p]
            if w >> 31:
                s = w & 0xFFFF
                ns = (s + 1) % max(max_src + 1, 1)
                if ns == s:
                    ns = s ^ 1
                words[p] = (w & 0xFFFF0000) | ns
                left -= 1
                p += 1
            else:
                p += 1 + ((w + 1) >> 1)
    return words, tot_runs


def bpatch_auto(loc, bpatch_min):
    """--bpatch auto: (K, local residue entries per patched map tile). K is 4 when the lowering counted at least
    `bpatch_min` entries per patched (map, tile), else 0. `loc` is stream()'s pair of sums; None when there are no
    streamed tables or the images were lowered before they recorded it."""
    lm = loc[0] / max(loc[1], 1) if loc else 0.0
    return (4 if lm >= bpatch_min else 0), lm


def stream(E, sort=False, dense=None, brk=0, run=0, run_weights=(1, 1, 1), run_brk=0):
    """dense: [(map, shift, min local entries a tile)]: twin j is map nm + j, the same entries as its map, stored
    dense on a core when the local residue entries per tile reach the minimum (encode.py points the pages whose
    window and residue come from one L1 vector at the twin). brk: the broken arm of the dense gate: on core 0 the
    first brk pairs of residue lanes of every dense table read each other's source (the state compare must fail).
    run: RMIN > 0 encodes qualifying dense tiles (< 512 words, faster than gather16) as run-copy command streams
    with balanced 2T/3T split headers and halo/zero don't-care bridging (vm_ncrisc/vm_brisc/vm_compute CFD_RUNCOPY)."""
    m = E.m
    dense = list(dense or [])
    assert not (dense and _enc.WIDE), "dense tables index 16-bit lanes"
    ndense = 0
    nrun_tiles = ntot_dense_tiles = run_words = dense_words = run_lanes = tot_lanes = 0
    P, T, nm = len(E.tab), m.T, len(m.maps.tab)      # E's rows: every core, or the cores of Encoded(cores=)
    nparts = [t["nparts"] for t in m.maps.tab]
    nparts += [nparts[o] for o, _, _ in dense]
    assert all(nparts[nm + j] == 1 for j in range(len(dense))), "a dense twin has one part"
    l1, blob, halo_off, staging, loc = [], [], [], 0, []
    for i in range(P):
        tab = E.tab[i]
        L = [len(nparts)] + [0] * len(nparts)
        B = []
        for mid in range(len(nparts)):
            orig, dsh, dmin = (mid, 0, 0) if mid < nm else dense[mid - nm]
            secs, counts = _sections(tab, orig, nparts[mid])
            tm = max([int(_lane(k, e.astype(np.int64)).max()) // 1024 + 1 for k, e in secs if len(e)] or [1])
            assert tm <= T and len(L) < 1 << 20
            L[1 + mid] = len(L) | tm << 20
            base, H = len(B), []
            per_tile = np.zeros(tm, np.int64)
            loc_tile = np.zeros(tm, np.int64)
            dn = mid >= nm and len(secs[0][1]) >= dmin * tm
            for si, (kind, ent) in enumerate(secs):
                lanes = _lane(kind, ent.astype(np.int64))
                bounds = np.searchsorted(lanes, np.arange(tm + 1) * 1024)
                assert bounds[-1] == len(ent)
                if dn and si == 0:
                    assert nparts[mid] == 1 and kind == "l" and len(np.unique(lanes)) == len(lanes)
                    loc_tile += np.diff(bounds)
                    idx = dense_index(ent, tm, dsh)
                    if brk and i == 0:
                        ln = lanes[:2 * (min(brk, len(lanes) // 2))]
                        idx[ln[0::2]], idx[ln[1::2]] = idx[ln[1::2]].copy(), idx[ln[0::2]].copy()
                    if run:
                        dc = np.zeros(tm * 1024, dtype=bool)
                        h_lanes = _lane("h", secs[1][1].astype(np.int64))
                        z_lanes = _lane("z", secs[2][1].astype(np.int64))
                        if len(h_lanes):
                            dc[h_lanes[h_lanes < tm * 1024]] = True
                        if len(z_lanes):
                            dc[z_lanes[z_lanes < tm * 1024]] = True
                        real = ~dc
                        max_src = int(idx[real].max()) if real.any() else 0
                        for t in range(tm):
                            H.append(len(B) - base)
                            sl_t = slice(t * 1024, (t + 1) * 1024)
                            enc = encode_runcopy(
                                idx[sl_t], dc[sl_t], max_src, rmin=run, weights=run_weights,
                                brk=run_brk if i == 0 else 0,
                            )
                            ntot_dense_tiles += 1
                            tot_lanes += 1024
                            dense_words += 512
                            if enc is not None:
                                w_t, rl_t = enc
                                B += w_t
                                per_tile[t] += len(w_t)
                                nrun_tiles += 1
                                run_words += len(w_t)
                                run_lanes += rl_t
                            else:
                                B += _half(idx[sl_t])
                                per_tile[t] += 512
                                run_words += 512
                    else:
                        H += (len(B) - base + np.arange(tm) * 512).tolist()
                        B += _half(idx)
                        per_tile += 512
                    ndense += 1
                    continue
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
            L += [base, sum(1 << k for k, (_, nh) in enumerate(counts) if nh) | (DENSE if dn else 0)] + (H if _enc.WIDE else _half(H))
            if mid < nm:
                loc.append(loc_tile[loc_tile > 0])
            staging = max(staging, int(per_tile.max()) if T else 0)
        d = _hdir(E, i)    # the halo directory below tab_halo (more than one plan) moves with the plans
        halo_off.append(len(L) + d)
        L += tab[int(E.tab_halo[i]) - d:E.tab_len[i]].tolist()   # not the padding up to the longest core's table
        l1.append(np.array(L, np.uint32))
        blob.append(np.array(B, np.uint32))
    loc = np.concatenate(loc) if loc else np.zeros(0)
    # loc: local residue entries and the patched (map, tile)s they sit in, over these cores; run_mesh --bpatch auto
    # tests their ratio. Two integers, so the chips of a per-chip lowering add up to the global lowering's pair.
    return dict(sort=sort, dense=dense, ndense=ndense, loc=(int(loc.sum()), len(loc)), l1=l1, blob=blob, halo_off=halo_off, staging_words=staging, nparts=nparts,
                l1_words=max(len(x) for x in l1), blob_words=max(len(x) for x in blob),
                today_words=E.tab.shape[1],
                run=run, nrun_tiles=nrun_tiles, ntot_dense_tiles=ntot_dense_tiles,
                run_words=run_words, dense_words=dense_words, run_lanes=run_lanes, tot_lanes=tot_lanes)


def gcensus(E):
    """Static census of the gathered values by path (encode.py gops), one execution of every op (the loop counts are
    the reader's): {(phase id or None, class): [ops, local entries, of them through a dense table, halo entries]},
    entries as means over the cores. Says which gathers a dense twin reaches and which it leaves on the entry loops."""
    S, npt, P = E.tab_stream, E.tab_stream["nparts"], len(E.tab_stream["l1"])
    cache, out = {}, {}
    for cls, mid, tw, _, ph in E.gops:
        if mid not in cache:
            cnt = [_sections(E.tab_cursor[i], mid, npt[mid])[1] for i in range(P)]
            cache[mid] = np.array([[sum(nl for nl, _ in c), sum(nh for _, nh in c)] for c in cnt], np.float64)
        dn = np.zeros(P)
        if tw is not None:
            dn = np.array([float(int(L[(int(L[1 + tw]) & 0xfffff) + 1]) >> 31) for L in S["l1"]])
        r = out.setdefault((ph, cls), [0, 0.0, 0.0, 0.0])
        r[0] += 1
        r[1] += cache[mid][:, 0].mean()
        r[2] += (cache[mid][:, 0] * dn).mean()
        r[3] += cache[mid][:, 1].mean()
    return out


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
    if int(L[q + 1]) & DENSE:   # 'd': the tile's 512 index words as stored
        kinds[0] = "d"
    return [[(kind, int(x)) for k, kind in enumerate(kinds) for x in B[H[k * tm + t]:H[k * tm + t + 1]]] if t < tm
            else [] for t in range(T)]


def _unrun(w):
    """Exact Python mirror of vm_gather.h cfd_gather_run: verifies 1T, 2T, 3T and 4T splits match."""
    assert 5 <= len(w) < 512, f"bad run-copy tile length {len(w)}"
    c1, c1_off = int(w[0] & 0xFFFF), int(w[0] >> 16)
    c2, c2_off = int(w[1] & 0xFFFF), int(w[1] >> 16)
    q1, q1_off = int(w[2] & 0xFFFF), int(w[2] >> 16)
    q2, q2_off = int(w[3] & 0xFFFF), int(w[3] >> 16)
    q3, q3_off = int(w[4] & 0xFFFF), int(w[4] >> 16)

    def dec(off, nd, expect_end=None):
        out = []
        p = off
        while len(out) < nd:
            assert p < len(w), f"run-copy overrun at word {p}/{len(w)} (decoded {len(out)}/{nd})"
            x = int(w[p])
            p += 1
            if x >> 31:
                L = (x >> 16) & 0x7FFF
                s = x & 0xFFFF
                out.extend(range(s, s + L))
            else:
                L = x
                nw = (L + 1) >> 1
                assert p + nw <= len(w), "literal overrun"
                for k in range(L):
                    word = int(w[p + (k >> 1)])
                    out.append((word >> 16) if (k & 1) else (word & 0xFFFF))
                p += nw
        assert len(out) == nd, f"run-copy lane overshoot {len(out)} != {nd}"
        if expect_end is not None:
            assert p == expect_end, f"run-copy end offset {p} != {expect_end}"
        return np.array(out, np.int64)

    d1 = dec(5, 1024, len(w))
    d2 = np.concatenate([dec(5, q2, q2_off), dec(q2_off, 1024 - q2, len(w))])
    d3 = np.concatenate([dec(5, c1, c1_off), dec(c1_off, c2 - c1, c2_off), dec(c2_off, 1024 - c2, len(w))])
    d4 = np.concatenate([dec(5, q1, q1_off), dec(q1_off, q2 - q1, q2_off), dec(q2_off, q3 - q2, q3_off), dec(q3_off, 1024 - q3, len(w))])
    assert (d1 == d2).all() and (d1 == d3).all() and (d1 == d4).all(), "run-copy 1T/2T/3T/4T split mismatch"
    return d1


def _undense(tile, t, shift):
    """a dense tile as the reader applies it, checked against the residue walk: (lane, source) of every lane"""
    w = np.array([x for k, x in tile if k == "d"], np.int64)
    if len(w) == 512:
        return np.stack([w & 0xFFFF, w >> 16], 1).ravel(), False
    return _unrun(w), True


def check(E, S, tab=None):
    """tab: the cursor tables (E.tab of a tab_stream=False encoding, or E.tab_cursor)."""
    m = E.m
    tab = E.tab if tab is None else tab
    for i in range(m.P):
        nm = len(m.maps.tab)
        for mid, nparts in enumerate(S["nparts"]):
            orig = mid if mid < nm else S["dense"][mid - nm][0]    # a twin holds its map's entries
            a = [[(k, _cut(k, x)) for k, x in tile] for tile in _walk_cursor(tab[i], orig, nparts, m.T)]
            b = _walk_stream(S["l1"][i], S["blob"][i], mid, nparts, m.T)
            if not any(k == "d" for tile in b for k, _ in tile) and S.get("sort"):   # same entries per tile, order free
                a, b = [sorted(x) for x in a], [sorted(x) for x in b]
            if any(k == "d" for tile in b for k, _ in tile):
                # a dense map: every lane's index is its residue source, or the window's lane; the rest as stored
                sh = S["dense"][mid - nm][1]
                # compute max valid source index across all real lanes of this (core, map) for run-copy bounds check
                max_src = 0
                if S.get("run"):
                    for t, ta in enumerate(a):
                        if not b[t]:
                            continue
                        lane = t * 1024 + np.arange(1024)
                        want = np.where((lane + sh >= 0) & (lane + sh < 1 << 16), lane + sh, lane)
                        dc = np.zeros(1024, dtype=bool)
                        for k, x in ta:
                            if k == "l":
                                want[(x >> 16) - t * 1024] = x & 0xFFFF
                            elif k == "h":
                                dc[(x >> 16) - t * 1024] = True
                            elif k == "z":
                                dc[x - t * 1024] = True
                        if (~dc).any():
                            max_src = max(max_src, int(want[~dc].max()))
                for t, (ta, tb) in enumerate(zip(a, b)):
                    if not tb:
                        assert not ta
                        continue
                    idx, is_run = _undense(tb, t, sh)
                    lane = t * 1024 + np.arange(1024)
                    want = np.where((lane + sh >= 0) & (lane + sh < 1 << 16), lane + sh, lane)
                    dc = np.zeros(1024, dtype=bool)
                    for k, x in ta:
                        if k == "l":
                            want[(x >> 16) - t * 1024] = x & 0xFFFF
                        elif k == "h":
                            dc[(x >> 16) - t * 1024] = True
                        elif k == "z":
                            dc[x - t * 1024] = True
                    if is_run:
                        assert (idx[~dc] == want[~dc]).all(), f"core {i} map {mid} tile {t}: dense index differs"
                        assert ((idx >= 0) & (idx <= max_src)).all(), f"core {i} map {mid} tile {t}: run-copy src out of bounds"
                    else:
                        assert (idx == want).all(), f"core {i} map {mid} tile {t}: dense index differs"
                    assert sorted(e for e in ta if e[0] != "l") == sorted(e for e in tb if e[0] != "d"), \
                        f"core {i} map {mid}: streamed walk differs"
                continue
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
