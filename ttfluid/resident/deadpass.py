"""Whole-pass exact live-tile bitmap analysis and palette packing (--deadpass 1).

Analyzes lowered Machine MIR per core using bitwise-exact rules only (no algebraic zero-annihilation,
no copy/constant store-bypass to live passes). A tile pass (mir_idx, t) of a vector tile loop is marked
dead (bit 0) only when every store it executes is either:
  - a bitwise self-copy (the destination tile already holds the exact same value), or
  - unused by every live consumer (including untaken branches of where(K, a, b) where K is a known
    uniform constant tile, since SFPU where_tile ignores its unselected operand).

Per-core bitmaps are packed into a compact palette + index table stored in one (or more) reserved
arena tiles (Machine.dp_tile, define CFD_DP), read directly by compute, NCRISC, and BRISC.
"""
import math
import struct
from collections import defaultdict

import numpy as np

from .machine import DBASE, KBASE, LANES

ZERO_POS = 0x00000000
ZERO_NEG = 0x80000000
ONE_U32 = 0x3F800000


def _is_zero_u32(b):
    return (b & 0x7FFFFFFF) == 0


def _is_subnormal_u32(b):
    return (b & 0x7F800000) == 0 and (b & 0x007FFFFF) != 0


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _u32_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def _eval_un_u32(name, b):
    """Exact fp32 unary evaluation on a uniform tile; None if non-finite, subnormal, or unsafe."""
    if _is_subnormal_u32(b):
        return None
    x = _u32_f32(b)
    if math.isnan(x) or math.isinf(x):
        return None
    if name == "neg":
        return b ^ 0x80000000
    if name == "abs":
        return b & 0x7FFFFFFF
    if name == "gtz":
        return ONE_U32 if x > 0.0 else ZERO_POS
    if name == "gez":
        return ONE_U32 if x >= 0.0 else ZERO_POS
    if name == "ltz":
        return ONE_U32 if x < 0.0 else ZERO_POS
    if name == "lez":
        return ONE_U32 if x <= 0.0 else ZERO_POS
    if name == "eqz":
        return ONE_U32 if x == 0.0 else ZERO_POS
    if name == "nez":
        return ONE_U32 if x != 0.0 else ZERO_POS
    return None


def _eval_bin_u32(name, ba, bb):
    """Exact fp32 binary evaluation on uniform tiles; skips SFPU-approximated div and subnormals."""
    if _is_subnormal_u32(ba) or _is_subnormal_u32(bb):
        return None
    xa, xb = _u32_f32(ba), _u32_f32(bb)
    if math.isnan(xa) or math.isnan(xb) or math.isinf(xa) or math.isinf(xb):
        return None
    with np.errstate(all="ignore"):
        fa, fb = np.float32(xa), np.float32(xb)
        if name == "add":
            r = np.float32(fa + fb)
        elif name == "sub":
            r = np.float32(fa - fb)
        elif name == "mul":
            r = np.float32(fa * fb)
        elif name == "max":
            if _is_zero_u32(ba) and _is_zero_u32(bb):
                return ZERO_POS if (ba == ZERO_POS or bb == ZERO_POS) else ZERO_NEG
            return ba if xa >= xb else bb
        elif name == "min":
            if _is_zero_u32(ba) and _is_zero_u32(bb):
                return ZERO_NEG if (ba == ZERO_NEG or bb == ZERO_NEG) else ZERO_POS
            return ba if xa <= xb else bb
        else:
            return None
    if np.isnan(r) or np.isinf(r):
        return None
    rb = int(r.view(np.uint32))
    return None if _is_subnormal_u32(rb) else rb


def _eval_mad_u32(name, ba, bb, bc):
    """Exact fp32 mad (c + a*b) / msb (c - a*b) on uniform tiles (only when FMA and mul+add agree bit-for-bit)."""
    if _is_subnormal_u32(ba) or _is_subnormal_u32(bb) or _is_subnormal_u32(bc):
        return None
    xa, xb, xc = _u32_f32(ba), _u32_f32(bb), _u32_f32(bc)
    if any(math.isnan(v) or math.isinf(v) for v in (xa, xb, xc)):
        return None
    with np.errstate(all="ignore"):
        fa, fb, fc32 = np.float32(xa), np.float32(xb), np.float32(xc)
        pr = np.float64(fa) * np.float64(fb)
        fc = np.float64(fc32)
        r = np.float32(fc + pr if name == "mad" else fc - pr)
        pr32 = np.float32(fa * fb)
        r2 = np.float32(fc32 + pr32 if name == "mad" else fc32 - pr32)
    if np.isnan(r) or np.isinf(r) or np.isnan(r2) or np.isinf(r2):
        return None
    rb, rb2 = int(r.view(np.uint32)), int(r2.view(np.uint32))
    if rb != rb2 or _is_subnormal_u32(rb):
        return None
    return rb


def _eval_fold_u32(b):
    """Exact sum of 1024 identical fp32 lanes (exact for 0.0 and normal values without overflow/subnormal)."""
    if _is_zero_u32(b):
        return b
    if _is_subnormal_u32(b):
        return None
    x = _u32_f32(b)
    if math.isnan(x) or math.isinf(x):
        return None
    with np.errstate(all="ignore"):
        r = np.float32(np.float32(x) * np.float32(LANES))
    if np.isnan(r) or np.isinf(r):
        return None
    rb = int(r.view(np.uint32))
    return None if _is_subnormal_u32(rb) else rb


# ---- compact per-core DP table encoding / decoding (exact mirror of C++ dp_words / dp_live)

def dp_pack_core(masks, bt_max):
    """Pack a core's list of M int bitmasks into a 1-D uint32 array.

    Layout:
      word 0:       n_pal | (nw << 16) | (is_u16 << 31)
      words 1..:    palette, n_pal * nw uint32 words (little-endian word order per mask)
      remainder:    M indices, uint8 if n_pal <= 256 else uint16, zero-padded to uint32 boundary.
    """
    nw = max(1, (bt_max + 31) // 32)
    assert nw < 16, f"bt_max={bt_max} exceeds 480 tiles"
    pal_map = {}
    pal = []
    idx = []
    for m in masks:
        key = tuple((m >> (32 * w)) & 0xFFFFFFFF for w in range(nw))
        pi = pal_map.get(key)
        if pi is None:
            pi = len(pal)
            pal_map[key] = pi
            pal.append(key)
        idx.append(pi)
    n_pal = len(pal)
    assert n_pal < (1 << 16), f"too many unique live-tile masks on a core: {n_pal}"
    is_u16 = 1 if n_pal > 256 else 0
    hdr = np.uint32(n_pal | (nw << 16) | (is_u16 << 31))
    pal_arr = np.array([w for k in pal for w in k], np.uint32) if n_pal else np.zeros(0, np.uint32)
    idx_arr = np.asarray(idx, np.uint16 if is_u16 else np.uint8)
    raw = idx_arr.tobytes()
    if len(raw) % 4:
        raw += b"\x00" * (4 - len(raw) % 4)
    idx_u32 = np.frombuffer(raw, np.uint32) if raw else np.zeros(0, np.uint32)
    return np.concatenate(([hdr], pal_arr, idx_u32))


def dp_live_core(dp_row, dpid, t):
    """Scalar lookup matching vm_compute/vm_ncrisc/vm_brisc dp_live (1-based dpid)."""
    k = dpid - 1
    hdr = int(dp_row[0])
    n_pal = hdr & 0xFFFF
    nw = (hdr >> 16) & 0x7FFF
    is_u16 = hdr >> 31
    off = 1 + nw * n_pal
    if is_u16:
        w = int(dp_row[off + (k >> 1)])
        pi = (w >> (16 * (k & 1))) & 0xFFFF
    else:
        w = int(dp_row[off + (k >> 2)])
        pi = (w >> (8 * (k & 3))) & 0xFF
    word = int(dp_row[1 + nw * pi + (t >> 5)])
    return bool((word >> (t & 31)) & 1)


def dp_live_vec(dp_rows, dpid, t):
    """Vectorized over all P rows of dp_rows (P, W) uint32 -> bool array (P,)."""
    P = dp_rows.shape[0]
    out = np.empty(P, bool)
    for r in range(P):
        out[r] = dp_live_core(dp_rows[r], dpid, t)
    return out


# ---- MIR helpers

def _unwrap(ins):
    """Unwrap ('range', t0, t1, inner) -> (t0, t1, inner) or (0, None, ins)."""
    if ins[0] == "range":
        return ins[1], ins[2], ins[3]
    return 0, None, ins


def _blks_of(inner):
    if inner[0] == "blk":
        return [inner]
    if inner[0] == "seg":
        return [x for x in inner[1] if x[0] == "blk"]
    return []


def _uniform_pages(arr_f32):
    """(P, N, LANES) float32 -> (P, N) int64: uint32 bits if all lanes equal, else -1."""
    u = arr_f32.view(np.uint32)
    same = (u == u[:, :, :1]).all(axis=2)
    out = np.full(u.shape[:2], -1, np.int64)
    out[same] = u[:, :, 0][same]
    return out


def _build_map_tile_counts(m, cores):
    """Precompute per-map (nl, nh, nz) of shape (P_local, T) int32 for cores."""
    P = len(cores)
    T = m.T
    c0 = int(cores[0])
    contiguous = (P == m.P) or np.array_equal(cores, np.arange(c0, c0 + P))
    counts = {}
    for mid, tab in enumerate(m.maps.tab):
        c = tab["core"]
        if not contiguous:
            sel = np.isin(c, cores)
            loc_c = np.searchsorted(cores, c[sel])
            kind = tab["kind"][sel]
            tile = tab["lane"][sel] // LANES
        elif P == m.P:
            loc_c = c
            kind = tab["kind"]
            tile = tab["lane"] // LANES
        else:
            sel = (c >= c0) & (c < c0 + P)
            loc_c = c[sel] - c0
            kind = tab["kind"][sel]
            tile = tab["lane"][sel] // LANES
        lc = np.zeros((P, T), np.int32)
        hc = np.zeros((P, T), np.int32)
        zc = np.zeros((P, T), np.int32)
        if len(loc_c):
            m0 = (kind == 0) & (tile < T)
            if m0.any():
                lc = np.bincount(loc_c[m0] * T + tile[m0], minlength=P * T).reshape(P, T).astype(np.int32)
            m1 = (kind == 1) & (tile < T)
            if m1.any():
                hc = np.bincount(loc_c[m1] * T + tile[m1], minlength=P * T).reshape(P, T).astype(np.int32)
        z_c = tab["zc"]
        if len(z_c):
            if not contiguous:
                zsel = np.isin(z_c, cores)
                zl_c = np.searchsorted(cores, z_c[zsel])
                zt = tab["zl"][zsel] // LANES
            elif P == m.P:
                zl_c = z_c
                zt = tab["zl"] // LANES
            else:
                zsel = (z_c >= c0) & (z_c < c0 + P)
                zl_c = z_c[zsel] - c0
                zt = tab["zl"][zsel] // LANES
            mz = zt < T
            if mz.any():
                zc = np.bincount(zl_c[mz] * T + zt[mz], minlength=P * T).reshape(P, T).astype(np.int32)
        counts[mid] = (lc, hc, zc)
    return counts


def _compile_mir(m, skip_mir, skip_gtile, tab_stream):
    """Pre-compile m.mir into fast integer-loc instructions and CFG interval barriers.

    Flat location IDs:
      L1 tile t:   0 <= id < ntiles
      HOME tile t: ntiles <= id < ntiles + hpages
    """
    phys = m.phys
    ntiles = m.ntiles
    hpages = max(m.hpages, 1)
    nlocs = ntiles + hpages
    T = m.T
    narrow_c = getattr(m, "narrow", ())
    narrow_h = getattr(m, "narrow_home", ())
    fetched_from = getattr(m, "fetched_from", {})
    vphys = {**phys, **getattr(m, "ovl", {})}
    vec_fn = lambda r: T > 1 and r != "__anchor__" and (phys[r] >= m.nsmall or r in getattr(m, "ovl", ()))

    def loc_base(r):
        p = phys[r]
        if p >= KBASE:
            return None
        return (ntiles + (p - DBASE)) if p >= DBASE else p

    mir = m.mir
    labels = {ins[1]: j for j, ins in enumerate(mir) if ins[0] == "label"}

    # Identify all jumps/branches and classify back-edges (loops) vs forward skips.
    # Outer loop: the widest back-edge (largest back - hdr); all other back-edges are inner loops.
    back_edges = []
    fwd_edges = []
    for j, ins in enumerate(mir):
        if ins[0] == "jump":
            tgt = labels[ins[1]]
            (back_edges if tgt < j else fwd_edges).append((j, tgt))
        elif ins[0] == "jumpif":
            tgt = labels[ins[2]]
            (back_edges if tgt < j else fwd_edges).append((j, tgt))

    outer_edge = max(back_edges, key=lambda e: e[0] - e[1]) if back_edges else None
    outer_hdr = outer_edge[1] if outer_edge else 0
    outer_end = outer_edge[0] if outer_edge else len(mir)
    inner_loops = [(tgt, j) for j, tgt in back_edges if (j, tgt) != outer_edge]
    # Forward skips inside the program; exit jumps (to end of program) still need state_locs live at the jump
    fwd_skips = [(j, tgt) for j, tgt in fwd_edges if tgt < len(mir) - 1]
    exit_jumps = {j for j, tgt in fwd_edges if tgt >= len(mir) - 1}

    # Pre-parse every MIR entry once
    ops = []
    written_all = set()
    writes_at = {}  # j -> set of flat locs written by mir[j]
    loop_entries = {}  # j -> (t0, t1, eligible)

    for j, ins in enumerate(mir):
        if j in skip_mir:
            ops.append(("nop",))
            continue
        t0, t1, inner = _unwrap(ins)
        op = inner[0]
        if op in ("blk", "seg"):
            blks = _blks_of(inner)
            bt_full = m.tiles(blks)
            if t1 is None:
                t1 = bt_full
            uops_all = [u for x in blks for u in x[1]]
            lds_r = [u[2] for u in uops_all if u[0] in ("ld", "lds", "rvf")]
            ldr_u = [u for u in uops_all if u[0] == "ldr"]
            sts_r = [r for x in blks for _, r in x[2]]
            is_loop = any(vec_fn(r) for r in lds_r + sts_r) or bool(ldr_u) or t0 > 0
            if not is_loop:
                t0, t1 = 0, 1
            has_pat = any(u[4] is not None for u in ldr_u)
            # Only C_TBEG vector loops (t1 - t0 > 1, not gtile, and tab_stream if patched) get a bitmap
            eligible = (is_loop and (t1 - t0) > 1 and j not in skip_gtile
                        and (tab_stream or not has_pat) and outer_hdr <= j <= outer_end)
            loop_entries[j] = (t0, t1, eligible)

            c_blks = []
            wset = set()
            for x in blks:
                cu = []
                for u in x[1]:
                    uk = u[0]
                    if uk == "ld":
                        r = u[2]
                        if r == "__anchor__":
                            cu.append(("anchor", u[1]))
                        else:
                            cu.append(("ld", u[1], loc_base(r), vec_fn(r), r in narrow_h))
                    elif uk == "rvf":
                        cu.append(("rvf", u[1], loc_base(u[2])))
                    elif uk == "lds":
                        r = u[2]
                        cu.append(("lds", u[1], loc_base(r), vec_fn(r), u[3], m.size(r)))
                    elif uk == "ldr":
                        r, sh, mid, parts = u[2], u[3], u[4], u[5]
                        if r is None:
                            s_kind, s_base, s_sz, s_nar = 0, 0, T, False
                        elif phys[r] >= KBASE:
                            k = fetched_from.get(r)
                            s_kind, s_base, s_sz, s_nar = 1, phys[r] - KBASE, m.size(r), (k in narrow_c)
                        else:
                            s_kind, s_base, s_sz, s_nar = 2, loc_base(r), m.size(r), (r in narrow_h)
                        p_info = []
                        for pr, _ in parts:
                            if pr is None:
                                continue
                            if phys[pr] >= KBASE:
                                p_info.append((1, phys[pr] - KBASE, m.size(pr)))
                            else:
                                p_info.append((2, loc_base(pr), m.size(pr)))
                        cu.append(("ldr", u[1], s_kind, s_base, s_sz, s_nar, sh, mid, tuple(p_info)))
                    elif uk == "fill":
                        cu.append(("fill", u[1], _f32_bits(u[2])))
                    elif uk == "cpd":
                        cu.append(("cpd", u[1], u[2]))
                    elif uk == "un":
                        cu.append(("un", u[1], u[2]))
                    elif uk == "fold":
                        cu.append(("fold", u[1]))
                    elif uk == "op":
                        cu.append(("op", u[1], u[2], u[3:]))
                    else:
                        raise NotImplementedError(uk)
                cs = []
                for s_slot, r in x[2]:
                    lb = loc_base(r)
                    iv = vec_fn(r)
                    nar = r in narrow_h
                    cs.append((s_slot, lb, iv, nar, vphys[r]))
                    for t in range(t0, t1):
                        loc = lb + (t if iv else 0)
                        written_all.add(loc)
                        wset.add(loc)
                c_blks.append((tuple(cu), tuple(cs)))
            writes_at[j] = wset
            ops.append(("blk_run", t0, t1, eligible, tuple(c_blks)))
        elif op == "fetch":
            dst_lb = loc_base(ins[1])
            coff = m.coff[fetched_from[ins[1]]]
            sz = m.size(ins[1])
            wset = set(range(dst_lb, dst_lb + sz))
            written_all |= wset
            writes_at[j] = wset
            ops.append(("fetch", dst_lb, coff, sz, vphys[ins[1]]))
        elif op == "gather":
            dst_lb = loc_base(ins[1])
            sz = m.size(ins[1])
            mid = ins[2]
            mode = ins[4] if len(ins) > 4 else 0
            p_info = []
            for pr, _ in ins[3]:
                if phys[pr] >= KBASE:
                    p_info.append((1, phys[pr] - KBASE, m.size(pr)))
                else:
                    p_info.append((2, loc_base(pr), m.size(pr)))
            wset = set(range(dst_lb, dst_lb + sz))
            written_all |= wset
            writes_at[j] = wset
            ops.append(("gather", dst_lb, sz, mid, tuple(p_info), mode, vphys[ins[1]]))
        elif op == "halo":
            r = ins[2]
            if phys[r] < KBASE:
                ops.append(("halo", loc_base(r), m.size(r)))
            else:
                ops.append(("nop",))
        elif op == "red":
            pl = loc_base(ins[2])
            fl = loc_base(ins[1])
            written_all.add(fl)
            writes_at[j] = {fl}
            ops.append(("red", ((pl, fl),)))
        elif op == "red2":
            pairs = tuple((loc_base(p), loc_base(f)) for f, p in zip(ins[1::2], ins[2::2]))
            wset = {fl for _, fl in pairs}
            written_all |= wset
            writes_at[j] = wset
            ops.append(("red", pairs))
        elif op == "out":
            ops.append(("read1", loc_base(ins[2])))
        elif op == "jumpif":
            ops.append(("read1", loc_base(ins[1])))
        elif op == "bscal":
            reads = tuple(loc_base(u[2]) for u in ins[1] if u[0] == "ld" and u[2] != "__anchor__")
            writes = tuple(loc_base(r) for _, r in ins[2])
            wset = set(writes)
            written_all |= wset
            writes_at[j] = wset
            ops.append(("bscal", reads, writes))
        else:
            ops.append(("nop",))

    # Build CFG barrier sets:
    # - Before entering an inner loop [hdr, back], or at a forward skip [src, tgt]:
    #   flush (mark live + reset) all locations written inside that interval!
    # - After exiting an inner loop (at back + 1) or at join label tgt of a forward skip:
    #   flush (mark live + reset) all locations written inside that interval!
    flush_before = defaultdict(set)
    flush_after = defaultdict(set)
    for hdr, back in inner_loops:
        w = set()
        for k in range(hdr, back + 1):
            w.update(writes_at.get(k, ()))
        flush_before[hdr] |= w
        flush_after[back] |= w
    for src, tgt in fwd_skips:
        w = set()
        for k in range(src, tgt):
            w.update(writes_at.get(k, ()))
        flush_before[src] |= w
        flush_before[tgt] |= w
    # Also mark outer loop entry boundary
    flush_before[outer_hdr] |= written_all

    # State locations (must be live at end of outer loop)
    state_locs = set()
    for _, cr in m.prog.state:
        for r in cr.ravel():
            r2 = m.ren(r)
            if r2 in phys and phys[r2] < KBASE:
                lb = loc_base(r2)
                for t in range(m.size(r2)):
                    state_locs.add(lb + t)

    return dict(
        ops=ops,
        nlocs=nlocs,
        ntiles=ntiles,
        T=T,
        written_all=written_all,
        state_locs=state_locs,
        outer_hdr=outer_hdr,
        outer_end=outer_end,
        flush_before={k: tuple(sorted(v)) for k, v in flush_before.items() if v},
        flush_after={k: tuple(sorted(v)) for k, v in flush_after.items() if v},
        exit_jumps=exit_jumps,
        loop_entries=loop_entries,
    )


def _analyze_core(comp, const_uni_r, ro_uni_r, map_counts_r, dense_mids_r, dense_gather_j):
    """Run exact forward symbolic eval + backward pass-liveness on one core.

    Returns dict mir_idx -> live_bitmask (int) for every eligible loop in comp['loop_entries'].
    """
    ops = comp["ops"]
    nlocs = comp["nlocs"]
    T = comp["T"]
    outer_hdr = comp["outer_hdr"]
    outer_end = comp["outer_end"]
    flush_before = comp["flush_before"]
    flush_after = comp["flush_after"]
    exit_jumps = comp["exit_jumps"]
    state_locs = comp["state_locs"]
    written_all = comp["written_all"]

    # Flat memory arrays: mem_val[loc] = ("V", vid) or ("K", u32_bits, is_folded), mem_def[loc] = pid or None
    vid = 0
    mem_val = [None] * nlocs
    mem_def = [None] * nlocs
    mem_slot = [None] * nlocs
    for loc, (b, fld) in ro_uni_r.items():
        mem_val[loc] = ("K", b, fld)

    # Track written_all locations read while mem_def[loc] is None inside the outer loop (loop-carried!)
    rbw_outer = set()
    roots = set()
    pass_deps = {}  # pid = (j, t) -> set of producer pids

    def get_loc(loc, in_outer):
        nonlocal vid
        v = mem_val[loc]
        d = mem_def[loc]
        if v is None:
            vid += 1
            v = ("V", vid)
            mem_val[loc] = v
        if in_outer and d is None and loc in written_all:
            rbw_outer.add(loc)
        return v, d

    def flush_locs(locs):
        nonlocal vid
        for loc in locs:
            d = mem_def[loc]
            if d is not None:
                roots.add(d)
                mem_def[loc] = None
            vid += 1
            mem_val[loc] = ("V", vid)
            mem_slot[loc] = None

    for j, op_t in enumerate(ops):
        fb = flush_before.get(j)
        if fb:
            flush_locs(fb)
        if j in exit_jumps:
            for loc in state_locs:
                d = mem_def[loc]
                if d is not None:
                    roots.add(d)
        in_outer = outer_hdr <= j <= outer_end
        kind = op_t[0]

        if kind == "nop":
            pass
        elif kind == "fetch":
            _, dst_lb, coff, sz, dst_vs = op_t
            for t in range(sz):
                g = coff + t
                b = int(const_uni_r[g]) if g < len(const_uni_r) else -1
                if b >= 0:
                    mem_val[dst_lb + t] = ("K", b, False)
                else:
                    vid += 1
                    mem_val[dst_lb + t] = ("V", vid)
                mem_def[dst_lb + t] = None
                mem_slot[dst_lb + t] = dst_vs
        elif kind == "gather":
            _, dst_lb, sz, mid, p_info, mode, dst_vs = op_t
            lc, hc, zc = map_counts_r[mid]
            is_d = (j in dense_gather_j) or (mid in dense_mids_r)
            for t in range(sz):
                nl = int(lc[t]) if t < T else 0
                nh = int(hc[t]) if t < T else 0
                nz = int(zc[t]) if t < T else 0
                loc = dst_lb + t
                full_ov = is_d or (nl + nh + nz >= LANES and mode == 0)
                if nl + nh + nz == 0 and not full_ov:
                    continue
                pz = True
                if not full_ov:
                    pv, pd = get_loc(loc, in_outer)
                    if pd is not None:
                        roots.add(pd)
                    if not (pv[0] == "K" and not pv[2] and pv[1] == ZERO_POS and mem_slot[loc] == dst_vs):
                        pz = False
                az = pz and (nh == 0) and (mode == 0) and not is_d
                if is_d or nl > 0 or nh > 0:
                    for p_kind, p_base, p_sz in p_info:
                        if p_kind == 1:
                            az = False
                        else:
                            for st_t in range(p_sz):
                                sv, sd = get_loc(p_base + st_t, in_outer)
                                if sd is not None:
                                    roots.add(sd)
                                if not (sv[0] == "K" and not sv[2] and sv[1] == ZERO_POS):
                                    az = False
                if az:
                    mem_val[loc] = ("K", ZERO_POS, False)
                else:
                    vid += 1
                    mem_val[loc] = ("V", vid)
                mem_def[loc] = None
                mem_slot[loc] = dst_vs
        elif kind == "halo":
            _, sb, ssz = op_t
            for t in range(ssz):
                _, sd = get_loc(sb + t, in_outer)
                if sd is not None:
                    roots.add(sd)
        elif kind == "red":
            for pl, fl in op_t[1]:
                _, sd = get_loc(pl, in_outer)
                if sd is not None:
                    roots.add(sd)
                vid += 1
                mem_val[fl] = ("V", vid)
                mem_def[fl] = None
                mem_slot[fl] = None
        elif kind == "read1":
            _, loc = op_t
            _, sd = get_loc(loc, in_outer)
            if sd is not None:
                roots.add(sd)
        elif kind == "bscal":
            _, reads, writes = op_t
            for loc in reads:
                _, sd = get_loc(loc, in_outer)
                if sd is not None:
                    roots.add(sd)
            for loc in writes:
                vid += 1
                mem_val[loc] = ("V", vid)
                mem_def[loc] = None
                mem_slot[loc] = None
        elif kind == "blk_run":
            _, t0, t1, eligible, c_blks = op_t
            for t in range(t0, t1):
                pid = (j, t)
                all_store_deps = set()
                non_noop = 0
                pending_writes = []
                # Track intra-seg writes so later blocks in the same seg see earlier blocks' writes on tile t!
                for cu, cs in c_blks:
                    S = {}
                    for u in cu:
                        uk = u[0]
                        if uk == "anchor":
                            vid += 1
                            S[u[1]] = (("V", vid), ())
                        elif uk == "ld":
                            _, sl, lb, iv, nar = u
                            v, d = get_loc(lb + (t if iv else 0), in_outer)
                            ds = (d,) if d is not None else ()
                            if nar:
                                vid += 1
                                v = ("V", vid)
                            S[sl] = (v, ds)
                        elif uk == "rvf":
                            _, sl, lb = u
                            v, d = get_loc(lb, in_outer)
                            ds = (d,) if d is not None else ()
                            if v[0] == "K":
                                S[sl] = (("K", v[1], False), ds)
                            else:
                                vid += 1
                                S[sl] = (("V", vid), ds)
                        elif uk == "lds":
                            _, sl, lb, iv, sh, ssz = u
                            lo_t = (t * LANES + sh) // LANES
                            hi_t = ((t + 1) * LANES + sh - 1) // LANES
                            ds = set()
                            same_k = (lo_t >= 0 and hi_t < ssz)
                            kv = None
                            for tt in range(max(0, lo_t), min(ssz, hi_t + 1)):
                                v, d = get_loc(lb + (tt if iv else 0), in_outer)
                                if d is not None:
                                    ds.add(d)
                                if same_k and v[0] == "K" and not v[2]:
                                    if kv is None:
                                        kv = v[1]
                                    elif kv != v[1]:
                                        same_k = False
                                else:
                                    same_k = False
                            if same_k and kv is not None:
                                S[sl] = (("K", kv, False), tuple(ds))
                            else:
                                vid += 1
                                S[sl] = (("V", vid), tuple(ds))
                        elif uk == "ldr":
                            _, sl, s_kind, s_base, s_sz, s_nar, sh, mid, p_info = u
                            plain = (sh == 0 and mid is None and not p_info)
                            nl = nh = nz = 0
                            if mid is not None:
                                lc, hc, zc = map_counts_r[mid]
                                if t < T:
                                    nl, nh, nz = int(lc[t]), int(hc[t]), int(zc[t])
                            if s_kind == 0:
                                # Zero window (+ optional patch)
                                if (nl == 0 and nh == 0):
                                    S[sl] = (("K", ZERO_POS, False), ())
                                else:
                                    ds = set()
                                    apz = (nh == 0)
                                    for pk, pb, psz in p_info:
                                        if pk == 1:
                                            apz = False
                                        else:
                                            for st_t in range(psz):
                                                sv, sd = get_loc(pb + st_t, in_outer)
                                                if sd is not None:
                                                    ds.add(sd)
                                                if not (sv[0] == "K" and not sv[2] and sv[1] == ZERO_POS):
                                                    apz = False
                                    if apz:
                                        S[sl] = (("K", ZERO_POS, False), tuple(ds))
                                    else:
                                        vid += 1
                                        S[sl] = (("V", vid), tuple(ds))
                            elif s_kind == 1:
                                # DRAM constant
                                g = s_base + t
                                b = int(const_uni_r[g]) if (plain and not s_nar and g < len(const_uni_r)) else -1
                                if b >= 0:
                                    S[sl] = (("K", b, False), ())
                                else:
                                    ds = set()
                                    for pk, pb, psz in p_info:
                                        if pk == 2:
                                            for st_t in range(psz):
                                                _, sd = get_loc(pb + st_t, in_outer)
                                                if sd is not None:
                                                    ds.add(sd)
                                    vid += 1
                                    S[sl] = (("V", vid), tuple(ds))
                            else:
                                # HOME or L1 stream source
                                if plain and not s_nar and t < s_sz:
                                    v, d = get_loc(s_base + t, in_outer)
                                    S[sl] = (v, (d,) if d is not None else ())
                                else:
                                    ds = set()
                                    lo_t = (t * LANES + sh) // LANES
                                    hi_t = ((t + 1) * LANES + sh - 1) // LANES
                                    for tt in range(max(0, lo_t), min(s_sz, hi_t + 1)):
                                        _, d = get_loc(s_base + tt, in_outer)
                                        if d is not None:
                                            ds.add(d)
                                    for pk, pb, psz in p_info:
                                        if pk == 2:
                                            for st_t in range(psz):
                                                _, sd = get_loc(pb + st_t, in_outer)
                                                if sd is not None:
                                                    ds.add(sd)
                                    vid += 1
                                    S[sl] = (("V", vid), tuple(ds))
                        elif uk == "fill":
                            S[u[1]] = (("K", u[2], False), ())
                        elif uk == "cpd":
                            S[u[2]] = S[u[1]]
                        elif uk == "un":
                            v, ds = S[u[2]]
                            if v[0] == "K":
                                rb = _eval_un_u32(u[1], v[1])
                                if rb is not None:
                                    S[u[2]] = (("K", rb, v[2]), ds)
                                    continue
                            vid += 1
                            S[u[2]] = (("V", vid), ds)
                        elif uk == "fold":
                            v, ds = S[u[1]]
                            if v[0] == "K" and not v[2]:
                                rb = _eval_fold_u32(v[1])
                                if rb is not None:
                                    S[u[1]] = (("K", rb, True), ds)
                                    continue
                            vid += 1
                            S[u[1]] = (("V", vid), ds)
                        elif uk == "op":
                            nm, dst_s, arg_slots = u[1], u[2], u[3]
                            args = [S[a] for a in arg_slots]
                            vals = [a[0] for a in args]
                            folded = None
                            if nm == "where":
                                cv, av, bv = vals
                                if cv[0] == "K" and not cv[2]:
                                    chosen = args[1] if not _is_zero_u32(cv[1]) else args[2]
                                    ud = tuple(set(args[0][1]) | set(chosen[1])) if (args[0][1] or chosen[1]) else ()
                                    folded = (chosen[0], ud)
                                elif av[0] == "K" and bv[0] == "K" and not av[2] and not bv[2] and av[1] == bv[1]:
                                    ud = tuple(set(args[1][1]) | set(args[2][1]))
                                    folded = (av, ud)
                                elif av[0] == "V" and bv[0] == "V" and av[1] == bv[1]:
                                    ud = tuple(set(args[1][1]) | set(args[2][1]))
                                    folded = (av, ud)
                            elif all(v[0] == "K" and not v[2] for v in vals):
                                rb = (_eval_mad_u32(nm, vals[0][1], vals[1][1], vals[2][1])
                                      if nm in ("mad", "msb") else _eval_bin_u32(nm, vals[0][1], vals[1][1]))
                                if rb is not None:
                                    ud = tuple(set().union(*(a[1] for a in args)))
                                    folded = (("K", rb, False), ud)
                            elif nm == "mul":
                                if vals[0][0] == "K" and not vals[0][2] and vals[0][1] == ONE_U32:
                                    folded = (vals[1], tuple(set(args[0][1]) | set(args[1][1])))
                                elif vals[1][0] == "K" and not vals[1][2] and vals[1][1] == ONE_U32:
                                    folded = (vals[0], tuple(set(args[0][1]) | set(args[1][1])))
                            elif nm == "add":
                                if vals[0][0] == "K" and not vals[0][2] and vals[0][1] == ZERO_NEG:
                                    folded = (vals[1], tuple(set(args[0][1]) | set(args[1][1])))
                                elif vals[1][0] == "K" and not vals[1][2] and vals[1][1] == ZERO_NEG:
                                    folded = (vals[0], tuple(set(args[0][1]) | set(args[1][1])))
                            elif nm == "sub":
                                if vals[1][0] == "K" and not vals[1][2] and vals[1][1] == ZERO_POS:
                                    folded = (vals[0], tuple(set(args[0][1]) | set(args[1][1])))
                            elif nm in ("max", "min"):
                                if vals[0][0] == "V" and vals[1][0] == "V" and vals[0][1] == vals[1][1]:
                                    folded = (vals[0], tuple(set(args[0][1]) | set(args[1][1])))
                            if folded is not None:
                                S[dst_s] = folded
                            else:
                                ud = tuple(set().union(*(a[1] for a in args)))
                                vid += 1
                                S[dst_s] = (("V", vid), ud)

                    for s_slot, lb, iv, nar, vs in cs:
                        loc = lb + (t if iv else 0)
                        val, vds = S[s_slot]
                        all_store_deps.update(vds)
                        if nar:
                            vid += 1
                            val = ("V", vid)
                        cur_v = mem_val[loc]
                        is_noop = False
                        if cur_v is not None and not nar and mem_slot[loc] == vs:
                            if val[0] == "V" and cur_v[0] == "V" and val[1] == cur_v[1]:
                                is_noop = True
                            elif val[0] == "K" and cur_v[0] == "K" and not val[2] and not cur_v[2] and val[1] == cur_v[1]:
                                is_noop = True
                        if not is_noop:
                            non_noop += 1
                            pending_writes.append((loc, val))
                            # Update mem immediately so subsequent blocks in the same seg see it
                            mem_val[loc] = val
                            mem_def[loc] = pid
                            mem_slot[loc] = vs

                if non_noop > 0 or not eligible:
                    # Soundness: if ANY store makes pid live (or pid is ineligible for skipping),
                    # hardware executes ALL stores of pid, so keep all_store_deps (incl. noop stores' inputs!).
                    pass_deps[pid] = all_store_deps
                    if not eligible:
                        roots.add(pid)
                else:
                    # Every store of this eligible pass is a bitwise no-op -> guaranteed dead!
                    pass_deps[pid] = ()

        fa = flush_after.get(j)
        if fa:
            flush_locs(fa)
        if j == outer_end:
            for loc in comp["state_locs"] | rbw_outer:
                d = mem_def[loc]
                if d is not None:
                    roots.add(d)

    # Mark final values of state locations + any loop-carried outer locations live
    for loc in comp["state_locs"] | rbw_outer:
        d = mem_def[loc]
        if d is not None:
            roots.add(d)

    # Backward reachability from roots
    live = set()
    stk = list(roots)
    while stk:
        p = stk.pop()
        if p in live:
            continue
        live.add(p)
        deps = pass_deps.get(p)
        if deps:
            for d in deps:
                if d not in live:
                    stk.append(d)

    # Build per-eligible-loop live bitmask
    masks = {}
    for j, (t0, t1, eligible) in comp["loop_entries"].items():
        if not eligible:
            continue
        m_bits = 0
        for t in range(t0, t1):
            if (j, t) in live:
                m_bits |= 1 << t
        masks[j] = m_bits
    return masks


def analyze_deadpass(m, cores, consts, arena, home, skip_mir=(), skip_gtile=(), tab_stream=True, dense_mids=None, dense_gather_j=()):
    """Run exact whole-pass live-tile analysis for every core in `cores`.

    Returns:
      eligible_j: sorted list of MIR indices eligible for C_TBEGM
      per_core_masks: list (len P) of dicts {j: int_bitmask}
      any_dead: bool numpy array of length len(eligible_j), True where at least one core in `cores`
                has a dead tile in [t0, t1)
      ranges: dict j -> (t0, t1)
    """
    comp = _compile_mir(m, set(skip_mir), set(skip_gtile), tab_stream)
    dense_gather_j = set(dense_gather_j)
    P = len(cores)
    const_uni = _uniform_pages(consts[:, :m.cpages]) if m.cpages else np.zeros((P, 0), np.int64)
    arena_uni = _uniform_pages(arena)
    home_uni = _uniform_pages(home[:, :m.hpages]) if m.hpages else np.zeros((P, 0), np.int64)
    map_counts = _build_map_tile_counts(m, cores)
    written_all = comp["written_all"]
    ntiles = comp["ntiles"]

    # Pre-identify read-only candidate locations so per-core ro_uni build is tiny
    ro_l1 = [i for i in range(ntiles) if i not in written_all]
    ro_hm = [i for i in range(m.hpages) if (ntiles + i) not in written_all]

    eligible_j = sorted(j for j, (_, _, el) in comp["loop_entries"].items() if el)
    ranges = {j: (comp["loop_entries"][j][0], comp["loop_entries"][j][1]) for j in eligible_j}
    full_masks = {j: sum(1 << t for t in range(ranges[j][0], ranges[j][1])) for j in eligible_j}

    per_core_masks = []
    any_dead = np.zeros(len(eligible_j), bool)
    for r in range(P):
        ro_r = {}
        for i in ro_l1:
            b = int(arena_uni[r, i])
            if b >= 0:
                ro_r[i] = (b, False)
            elif i < m.nsmall:
                # Non-uniform one-tile slot (scalar: lane 0 only; lanes 1..1023 may be NaN or vary)
                ro_r[i] = (int(arena[r, i, 0].view(np.uint32)), True)
        for i in ro_hm:
            b = int(home_uni[r, i])
            if b >= 0:
                ro_r[ntiles + i] = (b, False)
        mc_r = {mid: (lc[r], hc[r], zc[r]) for mid, (lc, hc, zc) in map_counts.items()}
        dm_r = dense_mids[r] if dense_mids is not None else ()
        ms = _analyze_core(comp, const_uni[r], ro_r, mc_r, dm_r, dense_gather_j)
        per_core_masks.append(ms)
        for k, j in enumerate(eligible_j):
            if ms[j] != full_masks[j]:
                any_dead[k] = True

    return eligible_j, per_core_masks, any_dead, ranges
