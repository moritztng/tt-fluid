"""The chips' halo plan (gateways, shadows, block placement) and the halo-table rewrite, from the cross-chip halo pairs
alone, so the global lowering (run_mesh.py) and the per-chip one (lower/spmd.py) share it.

    pl = plan(cross, K, P, R, C, gw=1, nsh=1, nhalo=m.nhalo, hmax=m.maps.hmax, runs=runs)
    rewrite(tab, tab_halo, cores, pl)      # rows of E.tab (lists) for global cores `cores`, in place
    pl["gtab"], pl["place"], pl["SHS"], pl["NSH"], ...

cross: (sender j, receiver i) -> (recv_off, words) for every pair on two different chips. runs: the same for every pair
the caller knows (default cross), for the check that a receiver's padded run does not land on another sender's run (the
per-chip lowering knows its own receivers' same-chip runs; the receiver's own chip checks the rest).
"""
import itertools


def _place_search(K, cells, adj, budget=2_000_000):
    """Block k -> mesh cell index with every adjacent pair one hop apart (no turns), blocks assigned in breadth-first
    order from block 0, free cells tried in index order: an R x C grid partition numbered row-major lands on the
    identity at once. None when there is none (or the budget of tried assignments runs out)."""
    nb = {k: set() for k in range(K)}
    for x, y in adj:
        nb[x].add(y)
        nb[y].add(x)
    order = []
    for s0 in range(K):
        if s0 in order:
            continue
        q = [s0]
        order.append(s0)
        while q:
            x = q.pop(0)
            for y in sorted(nb[x]):
                if y not in order:
                    order.append(y)
                    q.append(y)
    hop = lambda u, v: abs(cells[u][0] - cells[v][0]) + abs(cells[u][1] - cells[v][1]) == 1
    at, used, tries = {}, set(), [0]

    def go(t):
        if t == K:
            return True
        x = order[t]
        for u in range(len(cells)):
            if u in used or not all(hop(u, at[y]) for y in nb[x] if y in at):
                continue
            tries[0] += 1
            if tries[0] > budget:
                return False
            at[x] = u
            used.add(u)
            if go(t + 1):
                return True
            del at[x]
            used.discard(u)
        return False
    return [at[k] for k in range(K)] if go(0) else None


def chip_adjacency(chip, o, n, K, layers=1):
    """Sorted chip pairs (x < y) with a cell of y within `layers` face hops of a cell of x; chip = chip of every cell,
    o / n = internal faces' owner / neighbour cells. One layer is what plan() sees: on occDrivAer Small 8x4 --block2d
    it gave the same 112 pairs as the lowering's cross-chip halo pairs."""
    import numpy as np
    assert K <= 64, K
    chip = np.asarray(chip)
    reach = np.left_shift(np.uint64(1), chip.astype(np.uint64))   # per cell: the chips within h hops, as K bits
    for _ in range(layers):
        nxt = reach.copy()
        np.bitwise_or.at(nxt, o, reach[n])
        np.bitwise_or.at(nxt, n, reach[o])
        reach = nxt
    adj = set()
    for k in range(K):
        mask = int(np.bitwise_or.reduce(reach[chip == k]))
        adj |= {(min(k, j), max(k, j)) for j in range(K) if j != k and mask >> j & 1}
    return sorted(adj)


def place_blocks(adj, K, R, C):
    """Block k -> index into the row-major R x C mesh cells, every adjacent pair of blocks (adj: sorted (x, y) pairs)
    one hop apart (FABRIC_1D, no turns). Raises when there is none; lower/placecheck.py runs this on the chip
    partition alone, in minutes, before a lowering spends an hour reaching plan()."""
    cells = [(r, c) for r in range(R) for c in range(C)]
    place = None
    if all(y == x + 1 for x, y in adj):           # strips: block k on the k-th cell of a snake through the mesh
        place = [cells.index((r, c if r % 2 == 0 else C - 1 - c)) for r in range(R) for c in range(C)][:K]
    for perm in (() if place is not None or K > 9 else itertools.permutations(range(K))):
        if all(abs(cells[perm[x]][0] - cells[perm[y]][0]) + abs(cells[perm[x]][1] - cells[perm[y]][1]) == 1 for x, y in adj):
            place = perm                          # block k sits at mesh cell cells[perm[k]]
            break
    if place is None and K > 9:      # a 2D block partition (drivaer's 8x4_y): backtracking, row-major identity first
        place = _place_search(K, cells, adj)
    assert place is not None, ("no one-hop placement of the chip blocks", adj)
    return place


def plan(cross, K, P, R, C, gw=1, nsh=1, nhalo=1, hmax=0, runs=None):
    chip = lambda g: g // P
    pairs = {}                                    # (chip a, chip b) -> [(j, i)] sender j on a, receiver i on b
    for (j, i) in sorted(cross, key=lambda x: (x[1], x[0])):
        pairs.setdefault((chip(j), chip(i)), []).append((j, i))
    nbrs = {(a_, b_) for (a_, b_) in pairs} | {(b_, a_) for (a_, b_) in pairs}
    # lane[(a, b)][i]: which of the pair's gateways carries receiver i (on b); largest receiver first into the least
    # loaded lane, so the biggest gateway's shadow (the L1 that binds) is ~1/G of the interface
    lane, G = {}, {}
    for (a_, b_) in sorted(nbrs):
        lst = pairs.get((a_, b_), [])
        words = {}
        for (j, i) in lst:
            words[i] = words.get(i, 0) + cross[(j, i)][1]
        n_in = len({i for _, i in pairs.get((b_, a_), [])})
        G[(a_, b_)] = max(1, min(gw, max(len(words), n_in)))
        load = [0] * G[(a_, b_)]
        lane[(a_, b_)] = {}
        for i in sorted(words, key=lambda i: (-words[i], i)):
            k = load.index(min(load))
            lane[(a_, b_)][i] = k
            load[k] += -(-words[i] // 4) * 4
    gws = {}                                      # (a, b, k) -> global core of a's gateway k towards b
    for (a_, b_) in sorted(nbrs):
        for k in range(G[(a_, b_)]):
            cnt = {}
            for (j, i) in pairs.get((a_, b_), []):
                if lane[(a_, b_)][i] == k:
                    cnt[j] = cnt.get(j, 0) + 1
            for (j, i) in pairs.get((b_, a_), []):
                if lane[(b_, a_)][i] == k:
                    cnt[i] = cnt.get(i, 0) + 1
            taken = set(gws.values()) | {a_ * P}
            cand = sorted((c for c in range(a_ * P, a_ * P + P) if c not in taken), key=lambda c: (-cnt.get(c, 0), c))
            gws[(a_, b_, k)] = cand[0]
    gw_to = lambda x, y, i: gws[(x, y, lane[(x, y)][i])]     # x's gateway carrying x -> y traffic into receiver i on y
    cells = [(r, c) for r in range(R) for c in range(C)]
    place = place_blocks(sorted({tuple(sorted(x)) for x in nbrs}), K, R, C)
    assert hmax % 4 == 0 or K == 1, hmax
    # A gateway's shadow holds, per halo slot, only the runs [lo, lo + n4) the senders behind it fill (16-B units),
    # packed receiver after receiver: SHS words per region (the largest gateway's total). NSH regions serve the
    # exchanges round-robin in program order (every core runs the same B stream), whatever their slot. A member's
    # offset for a remote receiver is rebased to its run (lo % 4 == 0 keeps the 16-B phase the NoC write needs).
    span = {}                                     # (gateway, receiver) -> (lo, n4)
    for (a_, b_), lst in sorted(pairs.items()):
        for i in sorted({i for _, i in lst}):
            snd = [j for j, ii in lst if ii == i]
            lo = min(cross[(j, i)][0] for j in snd)
            hi = max(cross[(j, i)][0] + cross[(j, i)][1] for j in snd)
            assert lo % 4 == 0 and hi - lo == sum(cross[(j, i)][1] for j in snd), (i, lo, hi)
            # whole 16-B units: an ethernet hop moves 16-B words (a 4-B payload arrived as nothing while its fused
            # increment did). The pad lands in the gap before the next sender chip's run (16-B aligned) or past the end
            n4 = -(-(hi - lo) // 4) * 4
            assert lo + n4 <= hmax, (i, lo, n4, hmax)
            # the pad must not land on another sender's run into i (a 2D corner receiver has runs from two chips)
            assert not any(j not in snd and ii == i and off < lo + n4 and off + n > hi
                           for (j, ii), (off, n) in (runs or cross).items()), (i, lo, hi, n4)
            span[(gw_to(a_, b_, i), i)] = (lo, n4)
    base, used = {}, {}                           # (gateway, receiver) -> run start in the slot (words)
    for (g, i), (lo, n4) in sorted(span.items()):
        base[(g, i)] = used.get(g, 0)
        used[g] = base[(g, i)] + n4
    SHS = max(list(used.values()) + [4])          # words per shadow region
    NSH = max(1, min(nsh, nhalo))
    assert SHS // 4 < 1 << 15, SHS                # a run start rides in 15 bits of the receiver word, in 16-B units
    # gateway tables, appended: [gR, gP, remote gateway xy (filled on the device side), nrcv, (rxy, q, lo, n, cnt)...,
    # nmem, (member, pairs)...]
    gtab = {}
    for (a_, b_, k) in sorted(gws):
        g = gws[(a_, b_, k)]
        out = [(j, i) for (j, i) in pairs.get((a_, b_), []) if lane[(a_, b_)][i] == k]
        inn = [(j, i) for (j, i) in pairs.get((b_, a_), []) if lane[(b_, a_)][i] == k]
        rec = []
        for i in sorted({i for _, i in out}):
            lo, n4 = span[(g, i)]
            rec.append((i, base[(g, i)], lo, n4, len([1 for _, ii in out if ii == i])))
        mem = {}
        for j, _ in out:
            mem[j] = mem.get(j, 0) + 1
        gtab[g] = dict(peer=gws[(b_, a_, k)], lane=k, gR=len(inn), gP=len(out), rec=rec, mem=sorted(mem.items()))
    return dict(P=P, K=K, pairs=pairs, nbrs=nbrs, gw=gws, lane=lane, place=place, cells=cells, span=span, base=base,
                SHS=SHS, NSH=NSH, gtab=gtab, cross=cross)


def rewrite(tab, tab_halo, cores, pl, nplans=1):
    """Halo sections of the table rows (lists) of global cores `cores`: receivers / senders in local ids, remote ones
    through their gateway. nplans > 1 (multigrid spaces, encode's halo directory below tab_halo): plans 1.. are
    chip-local (the gateway carries plan 0 only), their ids made local too."""
    P, gw, lane, span, base, cross = pl["P"], pl["gw"], pl["lane"], pl["span"], pl["base"], pl["cross"]
    chip = lambda g: g // P
    gw_to = lambda x, y, i: gw[(x, y, lane[(x, y)][i])]
    from .machine import NH
    for t, h0, g in zip(tab, tab_halo, cores):
        h0 = int(h0)
        p = h0
        nrecv = t[p]
        p += 1
        for _ in range(nrecv):
            rc, n = t[p], t[p + 2]
            if chip(rc) == chip(g):
                t[p] = rc - chip(g) * P
            else:
                gg = gw_to(chip(g), chip(rc), rc)
                t[p] = 1 << 31 | base[(gg, rc)] // 4 << 16 | (gg - chip(g) * P)
                assert t[p + 1] == cross[(g, rc)][0] and t[p + 1] >= span[(gg, rc)][0]
                t[p + 1] -= span[(gg, rc)][0]
                assert t[p + 1] + n <= span[(gg, rc)][1]
            p += 3 + n
        nsend = t[p]
        for q in range(p + 1, p + 1 + nsend):
            j = t[q]
            t[q] = j - chip(g) * P if chip(j) == chip(g) else 0x8000 | (gw[(chip(g), chip(j), lane[(chip(j), chip(g))][g])] - chip(g) * P)
        for qn in range(1, nplans):
            p = h0 + t[h0 - NH - qn]
            nrecv = t[p]
            p += 1
            for _ in range(nrecv):
                assert chip(t[p]) == chip(g), ("a multigrid halo plan crosses chips", qn, g, t[p])
                t[p] -= chip(g) * P
                p += 3 + t[p + 2]
            nsend = t[p]
            for q in range(p + 1, p + 1 + nsend):
                assert chip(t[q]) == chip(g), ("a multigrid halo plan crosses chips", qn, g, t[q])
                t[q] -= chip(g) * P
