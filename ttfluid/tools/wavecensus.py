"""Lever 23 wavefront window census (D40.8) + D39.7a octant run-copy census across k2/k4/k8/k16 and hood15.

    python3 -m ttfluid.tools.wavecensus --small SMALL_K2.npz --hood HOOD15.npz [--parts-cache PARTS.npz]
                                        [--jobs 8] [--json OUT.json]

Measures from the mesh cache alone (no lowering):
1. Wavefront census (Small k2/k4/k8/k16 + hood15):
   For each order (`sliced+index`, `cls+rcm`, `cls+box`, `cls+boxw`, `cls+bfs`, `rcm`, `box`, `boxw`, `bfs`,
   `ch8+rcm`, `ch4+rcm`):
   - `W1`, `W2`: exact active L1 sliding window (tiles) for 1-hop and 2-hop wavefront sweeps, both full-core and
     main-class-only (excluding the narrow wide-cell prefix tiles `0..P_w-1` under `cls+*`).
   - `w_max`, `w_p99`: tile band `|dt - st|`.
   - `halo_L1/L2/L3`: share of tiles containing a cell at graph distance 0, <=1, <=2 from inter-core faces.
   - `halo_recv`: max unique remote cells received by any core (exact CB_HALO size in floats / KB).
   - `slot_tiles`: total slot tiles `sum_k L_k` (per-order cost of giving up `--sliced`).
2. Octant run-copy census (Small k2 + hood15, `ell_order="dir"` matching `ttfluid/solver/ell.py`):
   Splits each 1024-lane tile into 8 octants of 128 lanes (matching 3-thread NCRISC/UNPACK/BRISC splits),
   encodes each octant into COPY-RUN (len >= RMIN, stride +1) and LITERAL-GATHER commands, and reports:
   - `% local entries in runs` (mean, p10, min across cores)
   - `words/tile` (vs 512 dense) and `cmds/tile`
   - Estimated 3-thread and 2-thread gather cycles/tile (BH gbench unit costs) for mean and worst (p90/max) cores.
"""
import argparse
import itertools
import json
import os
import time
from multiprocessing import Pool

import numpy as np

from ..resident.lattice import BIG, pieces as lattice, rows8
from ..resident.machine import _quarter, rcb

LANES = 1024
OCT = 128
PERMS6 = np.array(list(itertools.permutations((0, 1, 2))), dtype=np.int64)
G = {}


def _rank(key):
    r = np.empty(len(key), np.int64)
    r[np.argsort(key, kind="stable")] = np.arange(len(key))
    return r


def _box_order_perm(comp, xyz, size, tie, perm_per_piece):
    crank = np.empty(len(size), np.int64)
    crank[np.argsort(-size, kind="stable")] = np.arange(len(size))
    crank = np.where(size[comp] > 1, crank[comp], len(size))
    pc = np.take_along_axis(xyz, perm_per_piece[comp], 1)
    return np.lexsort((tie, pc[:, 0], pc[:, 1], pc[:, 2], crank)).argsort()


def _box_perms(comp, xyz, size, dims):
    """Per-piece axis permutations:
    - global_ax: (fast, mid, slow) cyclic from global rows8 argmax (current box_rank)
    - best6: (fast, mid, slow) sorted descending by per-piece rows8 (breaks ties by dims descending)
    - wave6: (fast, mid, slow) with slow = longest dim (min cross-section fast*mid), fast = longer remaining dim
    """
    nco = len(size)
    r8 = rows8(comp, xyz, nco)
    # best6: sort axes per piece by (-r8, -dims)
    key_run = -(r8 * (BIG // 16) + dims)
    p_run = np.argsort(key_run, axis=1, kind="stable")
    # wave6: slow = longest dim, then fast = longer remaining dim (so fast*mid minimal, fast >= mid)
    order_d = np.argsort(dims, axis=1, kind="stable")  # [min, mid, max]
    p_wave = np.stack([order_d[:, 1], order_d[:, 0], order_d[:, 2]], axis=1)
    # best_a: cyclic from per-piece rows8 argmax (matches lattice.box_order fast=None)
    cyc = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64)
    p_cyc = cyc[r8.argmax(1)]
    return p_cyc, p_run, p_wave, r8.sum(0)


def _active_window(dt, st, T, t_min=0):
    """Exact 1-hop and 2-hop active L1 sliding window (tiles) over tile range [t_min, T)."""
    m = (dt >= t_min) & (st >= t_min)
    dt, st = dt[m] - t_min, st[m] - t_min
    Tn = T - t_min
    if Tn <= 0:
        return 0, 0, 0, 0
    hi = np.arange(Tn, dtype=np.int64)
    src_hi = np.arange(Tn, dtype=np.int64)
    if len(dt):
        np.maximum.at(hi, dt, st)
        np.maximum.at(src_hi, st, dt)
    F1 = np.maximum.accumulate(hi)
    # W1 = max_t |{s : s <= F1[t] and src_hi[s] >= t}|
    w1 = 0
    for t in range(Tn):
        live = int(np.sum(src_hi[: F1[t] + 1] >= t))
        if live > w1:
            w1 = live
    # 2-hop: level 2 reads level 1 tile s in S(t), level 1 tile s reads level 0 tile u in S(s)
    F0 = np.array([int(hi[: F1[t] + 1].max()) for t in range(Tn)], dtype=np.int64)
    src_hi2 = src_hi.copy()
    if len(dt):
        # edge (dt, st): level-2 tile dt uses level-1 tile st; if level-1 tile s uses level-0 tile u,
        # then level-0 tile u is needed until max_{s : u in S(s)} src_hi[s]
        u_hi = np.zeros(Tn, dtype=np.int64)
        np.maximum.at(u_hi, st, src_hi[dt])
        src_hi2 = np.maximum(src_hi2, u_hi)
    w2 = 0
    for t in range(Tn):
        live = int(np.sum(src_hi2[: F0[t] + 1] >= t))
        if live > w2:
            w2 = live
    d = np.abs(dt - st) if len(dt) else np.zeros(1, dtype=np.int64)
    return w1, w2, int(d.max()), int(np.percentile(d, 99))


def _octant_runcopy(dl, sl, loc, slot, nslot, T, L_ext, rmin_list=(4, 6, 8)):
    """Exact 8-octant run-copy encoding per tile & slot, matching 3-thread octant boundaries.

    Inside each 128-lane octant, dense entries are filled (padded lanes map to 0, never read or treated as
    literal-0 if padding; actually real lanes are dense 0..n_k-1, and octants past n_k are skipped).
    Run-copy: maximal runs of length >= rmin where sl[i] == sl[i-1] + 1 and BOTH are local.
    Wait: a dense tile has 1024 lanes; halo/boundary lanes in the local dense table hold `0` (overwritten by
    sparse patch) — so for run-copy we can EITHER treat non-local lanes as breaks (exact) OR check how
    many breaks come from boundary/halo lanes! Here we treat non-local lanes as breaks (conservative, exact),
    while encoding non-local lanes as literal 0 inside the literal-gather stream (same as DENSE).
    """
    # Build per-(slot, tile) dense source array: src = sl if loc else -1 (non-local break)
    out = {}
    for rmin in rmin_list:
        out[rmin] = dict(
            loc_total=0, run_loc=0, n_runs=0, n_lits=0, lit_lanes=0,
            enc_words=0, dense_words=0, tiles_total=0, tiles_win=0,
            cyc3_run=0.0, cyc3_dense=0.0, cyc2_run=0.0, cyc2_dense=0.0,
        )
    # Sort local entries by (slot, dl)
    # Note: we only need slots that have local entries
    for k in range(nslot):
        e = slot == k
        if not e.any():
            continue
        lk = int(L_ext[k]) if k < len(L_ext) else T
        # Build dense array for tiles 0..lk-1
        nlanes = lk * LANES
        src = np.full(nlanes, -1, dtype=np.int32)
        ek = e & loc
        if ek.any():
            src[dl[ek]] = sl[ek]
        # Reshape into (lk, 8, 128)
        S = src.reshape(lk, 8, OCT)
        for t in range(lk):
            tile = S[t]
            loc_t = int((tile >= 0).sum())
            if loc_t == 0:
                continue
            for rmin in rmin_list:
                st = out[rmin]
                st["tiles_total"] += 1
                st["loc_total"] += loc_t
                st["dense_words"] += 512
                # Per octant encoding
                w_tile = 5  # 9 uint16 octant offsets -> 5 uint32 words
                # 3-thread octant partition: NCRISC [0,2), UNPACK [2,4), BRISC [4,8) (example bgts=4, tgts=2)
                # Also match Stack 3 default bgts=6, tgts=3 -> wait: 6+3=9>8! Actually in Stack 3:
                # lo = 1024*bgts/8 = 768? Let's check: bgts=6 -> BRISC takes [768,1024) = 2 octants (6..8),
                # tgts=3 -> UNPACK takes [384,768) = 3 octants (3..6), NCRISC takes [0,384) = 3 octants (0..3)!
                # So thread octants are: N=[0,3), T=[3,6), B=[6,8) (3, 3, 2 octants!).
                # For 2-thread (bgts=4, tgts=0): N=[0,4), B=[4,8).
                oct_c3 = []
                oct_c2 = []
                runs_t = 0
                lits_t = 0
                rloc_t = 0
                llan_t = 0
                for o in range(8):
                    v = tile[o]
                    # Identify runs of v >= 0 with v[i] == v[i-1] + 1
                    is_loc = v >= 0
                    cont = np.zeros(OCT, dtype=bool)
                    cont[1:] = is_loc[1:] & is_loc[:-1] & (v[1:] == v[:-1] + 1)
                    starts = np.flatnonzero(~cont)
                    ends = np.r_[starts[1:], OCT]
                    lens = ends - starts
                    is_run = is_loc[starts] & (lens >= rmin)
                    # Build command stream: merge consecutive non-run segments into single LITERAL-GATHER
                    n_run_o = int(is_run.sum())
                    rloc_o = int(lens[is_run].sum())
                    # Count literal blocks (runs of ~is_run segments)
                    lit_lanes_o = OCT - rloc_o
                    # Number of literal blocks = number of transitions into ~is_run
                    nonrun = ~is_run
                    n_lit_o = int(np.sum(nonrun & np.r_[True, ~nonrun[:-1]])) if len(nonrun) else 0
                    # Words in octant: 1 word per run + 1 header + ceil(lit_len/2) per literal block
                    # Compute exact literal block lengths for exact word count
                    w_o = n_run_o
                    if n_lit_o:
                        lit_cur = 0
                        for seg_i in range(len(starts)):
                            if is_run[seg_i]:
                                if lit_cur:
                                    w_o += 1 + (lit_cur + 1) // 2
                                    lit_cur = 0
                            else:
                                lit_cur += int(lens[seg_i])
                        if lit_cur:
                            w_o += 1 + (lit_cur + 1) // 2
                    w_tile += w_o
                    runs_t += n_run_o
                    lits_t += n_lit_o
                    rloc_t += rloc_o
                    llan_t += lit_lanes_o
                    # Cycles in octant (BH gbench: copy_run 3.82/3.41, gather16 7.47/6.71, +10 cyc/cmd)
                    cmds_o = n_run_o + n_lit_o
                    oct_c3.append(rloc_o * 3.82 + lit_lanes_o * 7.47 + cmds_o * 10.0)
                    oct_c2.append(rloc_o * 3.41 + lit_lanes_o * 6.71 + cmds_o * 10.0)
                # Thread makespan for this tile (3T: N=[0,3), T=[3,6), B=[6,8); 2T: N=[0,4), B=[4,8))
                c3_run = max(sum(oct_c3[0:3]), sum(oct_c3[3:6]), sum(oct_c3[6:8]))
                c3_den = 3 * OCT * 7.47  # max(3, 3, 2) octants
                c2_run = max(sum(oct_c2[0:4]), sum(oct_c2[4:8]))
                c2_den = 4 * OCT * 6.71
                # Per-tile fallback: only use RUNCOPY if w_tile <= 512 and c3_run < c3_den
                if w_tile <= 512 and c3_run < c3_den:
                    st["tiles_win"] += 1
                    st["enc_words"] += w_tile
                    st["run_loc"] += rloc_t
                    st["n_runs"] += runs_t
                    st["n_lits"] += lits_t
                    st["lit_lanes"] += llan_t
                    st["cyc3_run"] += c3_run
                    st["cyc2_run"] += min(c2_run, c2_den)
                else:
                    st["enc_words"] += 512
                    st["cyc3_run"] += c3_den
                    st["cyc2_run"] += c2_den
                st["cyc3_dense"] += c3_den
                st["cyc2_dense"] += c2_den
    return out


def _core_worker(c):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import reverse_cuthill_mckee

    R, O, FID, DC, L, orders, do_runcopy, global_fast = (
        G[k] for k in ("R", "O", "FID", "DC", "L", "orders", "do_runcopy", "global_fast")
    )
    lo, hi = G["start"][c], G["start"][c + 1]
    cid = np.flatnonzero(G["own"] == c)
    n = len(cid)
    row = np.searchsorted(cid, R[lo:hi])
    o = O[lo:hi].astype(np.int64)
    loc = o >= 0
    loc[loc] = G["own"][o[loc]] == c
    halo = (o >= 0) & ~loc
    oth = np.where(loc, np.searchsorted(cid, np.where(loc, o, cid[0])), -1)
    # Unique remote source cells received by this core (exact CB_HALO count!)
    halo_recv = int(len(np.unique(o[halo]))) if halo.any() else 0
    fc = np.bincount(row, minlength=n)
    first = np.zeros(n + 1, np.int64)
    np.cumsum(fc, out=first[1:])
    N = hi - lo
    slot_face = np.arange(N) - first[row]
    od = np.lexsort((FID[lo:hi], DC[lo:hi], row))
    slot_dir = np.empty(N, np.int64)
    slot_dir[od] = np.arange(N) - first[row[od]]
    nslot = int(slot_face.max()) + 1 if N else 0

    # Local adjacency graph
    A = csr_matrix((np.ones(int(loc.sum()), np.int8), (row[loc], oth[loc])), shape=(n, n))
    rcm = np.empty(n, np.int64)
    rcm[reverse_cuthill_mckee(A, symmetric_mode=True)] = np.arange(n)

    # Halo distance levels: d0 (senders), d1 (<=1 hop), d2 (<=2 hops)
    d0 = np.zeros(n, dtype=bool)
    d0[row[halo]] = True
    d1 = d0 | (A.dot(d0.astype(np.int8)) > 0)
    d2 = d1 | (A.dot(d1.astype(np.int8)) > 0)

    # BFS from halo
    lev = np.where(d0, 0, -1)
    x, lv = d0.astype(np.int8), 0
    while True:
        lv += 1
        new = (A.dot(x) > 0) & (lev < 0)
        if not new.any():
            break
        lev[new] = lv
        x = new.astype(np.int8)
    lev[lev < 0] = lv
    bfs = _rank(lev * n + rcm)

    # Octree lattice pieces + box variants
    _, comp, xyz, size, dims, _ = lattice(n, row, np.where(loc, oth, 0), loc, DC[lo:hi], fc)
    p_cyc, p_run, p_wave, _ = _box_perms(comp, xyz, size, dims)
    cyc = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]], dtype=np.int64)
    p_glob = np.broadcast_to(cyc[global_fast], (len(size), 3))

    K = {
        "index": np.arange(n, dtype=np.int64),
        "rcm": rcm,
        "bfs": bfs,
        "box": _box_order_perm(comp, xyz, size, rcm, p_glob),
        "boxa": _box_order_perm(comp, xyz, size, rcm, p_cyc),
        "box6": _box_order_perm(comp, xyz, size, rcm, p_run),
        "boxw": _box_order_perm(comp, xyz, size, rcm, p_wave),
    }

    T = -(-n // LANES)
    out = dict(core=c, n=n, T=T, local=int(loc.sum()), halo=int(halo.sum()),
               halo_recv=halo_recv, senders=int(d0.sum()), orders={})

    for how in orders:
        cls, _, key = how.rpartition("+")
        Q = 0
        if cls.startswith("clsq"):
            Q = int(cls[4:])
            cls_base = "cls"
        else:
            cls_base = cls
        cl = -fc if cls_base == "sliced" else L[fc - 1] if cls_base == "cls" else np.zeros(n, np.int64)
        lane = np.empty(n, np.int64)
        if cls_base.startswith("ch"):
            B = int(cls_base[2:])
            cl = L[fc - 1]
            ch = K[key] // (B * LANES)
            ord_idx = np.lexsort((K[key], cl, ch))
            lane[ord_idx] = np.arange(n)
            pw_tiles = 0
        else:
            ord_idx = np.lexsort((K[key], cl))
            pos = np.arange(n, dtype=np.int64)
            if Q:
                ck = cl[ord_idx]
                starts = np.flatnonzero(np.r_[True, ck[1:] != ck[:-1]])
                runs = np.diff(np.r_[starts, n])
                m = np.arange(n) - np.repeat(starts, runs)
                pos = pos - m + _quarter(m, np.repeat(runs, runs), Q)
            lane[ord_idx] = pos
            if cls_base in ("cls", "sliced"):
                # Wide prefix tiles: tiles containing any cell with fc > 6 (or cl < cl.max())
                main_mask = (fc <= 6) if cls_base == "sliced" else (cl == cl.max())
                pw_tiles = int((lane[~main_mask].max() >> 10) + 1) if (~main_mask).any() else 0
            else:
                pw_tiles = 0

        dl = lane[row]
        sl = np.where(loc, lane[np.where(loc, oth, 0)], 0)
        dt_loc, st_loc = dl[loc] >> 10, sl[loc] >> 10

        w1, w2, wmax, wp99 = _active_window(dt_loc, st_loc, T, t_min=0)
        w1m, w2m, wmaxm, wp99m = _active_window(dt_loc, st_loc, T, t_min=pw_tiles)

        # Halo-coupled tile shares at L=1, 2, 3
        ht1 = int(len(np.unique(lane[d0] >> 10))) if d0.any() else 0
        ht2 = int(len(np.unique(lane[d1] >> 10))) if d1.any() else 0
        ht3 = int(len(np.unique(lane[d2] >> 10))) if d2.any() else 0

        # Slot extents L_k (using slot_face/slot_dir — identical counts per cell, so same L_k!)
        ext = [int(dl[slot_face == k].max() >> 10) + 1 if (slot_face == k).any() else 0 for k in range(nslot)]

        rec = dict(
            T=T, pw=pw_tiles, w1=w1, w2=w2, wmax=wmax, wp99=wp99,
            w1m=w1m, w2m=w2m, wmaxm=wmaxm, wp99m=wp99m,
            ht1=ht1, ht2=ht2, ht3=ht3, ext=ext,
        )
        if do_runcopy and how in do_runcopy:
            sl_use = slot_face if how == "sliced+index" else slot_dir
            rec["rc"] = _octant_runcopy(dl, sl, loc, sl_use, nslot, T, ext)
        out["orders"][how] = rec
    return out


def _prepare(D, own, orders, do_runcopy, jobs):
    """Build global edge arrays in G and run pool over all P cores."""
    P = int(own.max()) + 1
    nC = len(own)
    o, nb = D["owner"].astype(np.int64), D["neighbour"].astype(np.int64)
    nI = len(nb)
    b = np.concatenate([np.arange(s, s + z) for t, s, z in zip(D["ptype"], D["pstart"], D["psize"]) if t != "empty"])
    ax, sg = D["ax"].astype(np.int8), D["sg"]
    R = np.concatenate([o[:nI], nb, o[b]]).astype(np.int32)
    O = np.concatenate([nb, o[:nI], np.full(len(b), -1, np.int32)]).astype(np.int32)
    FID = np.concatenate([np.arange(nI, dtype=np.int32), np.arange(nI, dtype=np.int32), b.astype(np.int32)])
    DC = np.concatenate([
        2 * ax[:nI] + sg[:nI].astype(np.int8),
        2 * ax[:nI] + (~sg[:nI]).astype(np.int8),
        2 * ax[b] + sg[b].astype(np.int8),
    ]).astype(np.int8)
    by = np.lexsort((FID, R, own[R]))
    R, O, FID, DC = R[by], O[by], FID[by], DC[by]
    start = np.searchsorted(own[R], np.arange(P + 1))

    fc = np.bincount(R, minlength=nC)
    W = int(fc.max())
    L = np.zeros(W, np.int64)
    for k in range(W):
        L[k] = int(np.bincount(own[fc > k], minlength=P).max() + LANES - 1) // LANES

    # Quick global fast axis (y=1 on Small/hood; compute exact on core 0..min(P,16) or default 1)
    G.clear()
    G.update(R=R, O=O, FID=FID, DC=DC, own=own, start=start, L=L, orders=orders,
             do_runcopy=set(do_runcopy), global_fast=1)

    t0 = time.time()
    if jobs > 1:
        with Pool(jobs) as pool:
            res = pool.map(_core_worker, range(P))
    else:
        res = [_core_worker(c) for c in range(P)]
    dt = time.time() - t0
    return res, L.tolist(), dt


def _summarize_wave(name, res, L_base, orders):
    P = len(res)
    T_max = max(r["T"] for r in res)
    halo_recv = [r["halo_recv"] for r in res]
    senders = [r["senders"] for r in res]
    base_slots = sum(L_base)
    print(f"\n=== WAVEFRONT CENSUS: {name} (P={P}, T={T_max}, base sum(L_k)={base_slots}, "
          f"halo_recv mean={np.mean(halo_recv):.0f} max={max(halo_recv)} ({max(halo_recv)*4/1024:.1f} KB), "
          f"senders mean={np.mean(senders):.0f} max={max(senders)}) ===")
    hdr = (f"{'order':<13} {'sumL':>4} {'pw':>2} | {'W1 p50/95/mx':>13} {'W1m p50/mx':>10} | "
           f"{'W2 p50/mx':>9} {'W2m p50/mx':>10} | {'w99/mx':>6} | {'haloT L1/L2/L3 %':>18}")
    print(hdr)
    print("-" * len(hdr))
    summary = dict(P=P, T=T_max, base_sumL=base_slots,
                   halo_recv_mean=float(np.mean(halo_recv)), halo_recv_max=int(max(halo_recv)),
                   halo_kb_max=float(max(halo_recv) * 4 / 1024), orders={})
    for how in orders:
        rs = [r["orders"][how] for r in res]
        W_ext = max(len(x["ext"]) for x in rs)
        L_ord = [max(x["ext"][k] if k < len(x["ext"]) else 0 for x in rs) for k in range(W_ext)]
        sumL = sum(L_ord)
        pw_mx = max(x["pw"] for x in rs)
        w1 = np.array([x["w1"] for x in rs])
        w1m = np.array([x["w1m"] for x in rs])
        w2 = np.array([x["w2"] for x in rs])
        w2m = np.array([x["w2m"] for x in rs])
        wp99 = int(np.median([x["wp99"] for x in rs]))
        wmx = int(max(x["wmax"] for x in rs))
        Ts = np.array([x["T"] for x in rs], dtype=float)
        h1 = 100.0 * np.mean([x["ht1"] for x in rs] / Ts)
        h2 = 100.0 * np.mean([x["ht2"] for x in rs] / Ts)
        h3 = 100.0 * np.mean([x["ht3"] for x in rs] / Ts)
        print(f"{how:<13} {sumL:>4} {pw_mx:>2} | "
              f"{int(np.median(w1)):>3}/{int(np.percentile(w1,95)):>3}/{int(w1.max()):>3}   "
              f"{int(np.median(w1m)):>3}/{int(w1m.max()):>3}    | "
              f"{int(np.median(w2)):>3}/{int(w2.max()):>3}   "
              f"{int(np.median(w2m)):>3}/{int(w2m.max()):>3}    | "
              f"{wp99:>2}/{wmx:<3} | {h1:>5.1f}/{h2:>5.1f}/{h3:>5.1f}%")
        summary["orders"][how] = dict(
            sumL=int(sumL), pw_max=int(pw_mx),
            W1=[int(np.median(w1)), int(np.percentile(w1, 95)), int(w1.max())],
            W1_main=[int(np.median(w1m)), int(np.percentile(w1m, 95)), int(w1m.max())],
            W2=[int(np.median(w2)), int(np.percentile(w2, 95)), int(w2.max())],
            W2_main=[int(np.median(w2m)), int(np.percentile(w2m, 95)), int(w2m.max())],
            wp99_p50=wp99, wmax=wmx,
            halo_tile_pct=[round(float(h1), 1), round(float(h2), 1), round(float(h3), 1)],
        )
    return summary


def _summarize_runcopy(name, res, rc_orders):
    print(f"\n=== OCTANT RUN-COPY CENSUS (ell_order=dir, 8x128 octants, BH gbench costs): {name} ===")
    hdr = (f"{'order':<14} {'rmin':>4} | {'run% mean/p10/min':>18} | {'w/tile':>6} {'win%':>5} | "
           f"{'3T mean':>7} {'3T p90':>6} {'3T max':>6} | {'2T mean':>7} {'2T max':>6}")
    print(hdr)
    print("-" * len(hdr))
    summary = {}
    for how in rc_orders:
        summary[how] = {}
        for rmin in (4, 6, 8):
            per_core = [r["orders"][how]["rc"][rmin] for r in res]
            run_pcts = np.array([100.0 * x["run_loc"] / max(x["loc_total"], 1) for x in per_core])
            w_tile = sum(x["enc_words"] for x in per_core) / max(sum(x["tiles_total"] for x in per_core), 1)
            win_pct = 100.0 * sum(x["tiles_win"] for x in per_core) / max(sum(x["tiles_total"] for x in per_core), 1)
            c3_r = np.array([x["cyc3_run"] for x in per_core])
            c3_d = np.array([x["cyc3_dense"] for x in per_core])
            c2_r = np.array([x["cyc2_run"] for x in per_core])
            c2_d = np.array([x["cyc2_dense"] for x in per_core])
            sp3_mean = float(c3_d.sum() / c3_r.sum())
            sp3_p90 = float(np.percentile(c3_d, 90) / np.percentile(c3_r, 90))
            sp3_max = float(c3_d.max() / c3_r.max())
            sp2_mean = float(c2_d.sum() / c2_r.sum())
            sp2_max = float(c2_d.max() / c2_r.max())
            print(f"{how:<14} {rmin:>4} | "
                  f"{run_pcts.mean():>5.1f}/{np.percentile(run_pcts,10):>5.1f}/{run_pcts.min():>5.1f} | "
                  f"{w_tile:>6.1f} {win_pct:>4.1f}% | "
                  f"{sp3_mean:>6.2f}x {sp3_p90:>5.2f}x {sp3_max:>5.2f}x | "
                  f"{sp2_mean:>6.2f}x {sp2_max:>5.2f}x")
            summary[how][str(rmin)] = dict(
                run_pct_mean=round(float(run_pcts.mean()), 2),
                run_pct_p10=round(float(np.percentile(run_pcts, 10)), 2),
                run_pct_min=round(float(run_pcts.min()), 2),
                words_per_tile=round(float(w_tile), 1),
                win_tile_pct=round(float(win_pct), 1),
                sp3_mean=round(sp3_mean, 3), sp3_p90=round(sp3_p90, 3), sp3_max=round(sp3_max, 3),
                sp2_mean=round(sp2_mean, 3), sp2_max=round(sp2_max, 3),
            )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", required=True)
    ap.add_argument("--hood", required=True)
    ap.add_argument("--parts-cache", default="/home/moritz/flx/order/work/small_parts.npz")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--json", default="/home/moritz/flx/order/work/wavecensus.json")
    a = ap.parse_args()

    out_all = {}

    # 1. Hood15 (P=15, T=37)
    Dh = np.load(a.hood)
    own_h = Dh["own_cut"].astype(np.int64)
    wave_orders_big = [
        "sliced+index", "cls+rcm", "cls+box", "cls+boxa", "cls+box6", "cls+boxw", "clsq256+box", "cls+bfs",
        "rcm", "box6", "boxw", "bfs", "ch8+rcm", "ch4+rcm",
    ]
    rc_orders = ["sliced+index", "cls+rcm", "cls+box", "cls+boxa", "cls+box6", "cls+boxw", "clsq256+box"]
    res_h, L_h, dt_h = _prepare(Dh, own_h, wave_orders_big, rc_orders, jobs=min(a.jobs, 15))
    print(f"[hood15 done in {dt_h:.1f}s]")
    out_all["hood15_wave"] = _summarize_wave("hood15 (k1-proxy)", res_h, L_h, wave_orders_big)
    out_all["hood15_runcopy"] = _summarize_runcopy("hood15", res_h, rc_orders)

    # 2. Small k2/k4/k8/k16
    Ds = np.load(a.small)
    if os.path.exists(a.parts_cache):
        P_cache = dict(np.load(a.parts_cache))
    else:
        P_cache = {}
    nI = len(Ds["neighbour"])
    faces = (Ds["owner"][:nI].astype(np.int64), Ds["neighbour"].astype(np.int64))
    C = Ds["C"]
    changed = False
    for tag, P in (("k4", 440), ("k8", 880), ("k16", 1760)):
        if tag not in P_cache:
            t0 = time.time()
            print(f"[computing rcb min-cut partition Small {tag} (P={P})...]", flush=True)
            P_cache[tag] = rcb(C, P, chips=None, faces=faces).astype(np.int16)
            print(f"  done in {time.time() - t0:.1f}s", flush=True)
            changed = True
    if changed:
        os.makedirs(os.path.dirname(a.parts_cache), exist_ok=True)
        np.savez(a.parts_cache, **P_cache)

    # Small k2 (full orders + run-copy)
    own_k2 = Ds["own_cut"].astype(np.int64)
    res_k2, L_k2, dt_k2 = _prepare(Ds, own_k2, wave_orders_big, rc_orders, jobs=a.jobs)
    print(f"[small_k2 done in {dt_k2:.1f}s]")
    out_all["small_k2_wave"] = _summarize_wave("Small k2 (220 cores, 1x2)", res_k2, L_k2, wave_orders_big)
    out_all["small_k2_runcopy"] = _summarize_runcopy("Small k2", res_k2, rc_orders)

    # Small k4, k8, k16 (wavefront orders)
    for tag, P, orders in (
        ("k4", 440, ["sliced+index", "cls+rcm", "cls+box6", "cls+boxw", "cls+bfs", "rcm", "box6", "boxw", "bfs", "ch8+rcm", "ch4+rcm"]),
        ("k8", 880, ["sliced+index", "cls+rcm", "cls+box6", "cls+boxw", "cls+bfs", "rcm", "box6", "boxw", "bfs", "ch4+rcm"]),
        ("k16", 1760, ["sliced+index", "cls+rcm", "cls+box6", "cls+boxw", "cls+bfs", "rcm", "box6", "boxw", "bfs"]),
    ):
        own_k = P_cache[tag].astype(np.int64)
        res_k, L_k, dt_k = _prepare(Ds, own_k, orders, [], jobs=a.jobs)
        print(f"[small_{tag} done in {dt_k:.1f}s]")
        out_all[f"small_{tag}_wave"] = _summarize_wave(f"Small {tag} ({P} cores)", res_k, L_k, orders)

    os.makedirs(os.path.dirname(a.json), exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out_all, f, indent=2)
    print(f"\nWrote {a.json}")


if __name__ == "__main__":
    main()
