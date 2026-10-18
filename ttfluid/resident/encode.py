"""Encode a lowered Machine program for the device: three uint32 instruction streams (compute, BRISC, NCRISC) that
share one control flow, the per-core tables NCRISC and BRISC read, the initial arena of every core and the DRAM
constant store. The kernels in resident/device/ decode exactly this.

Word: op | f0 << 8 | f1 << 12 | f2 << 16 | f3 << 20 (4-bit Dest slot fields), some ops followed by operand words.
Compute LD / ST / RVF / LDS carry their tile in the top 16 bits (op | slot << 8 | tile << 16), C_TBEG its count
there. Branch targets are word offsets in the agent's own stream.

Multi-tile vectors (T > 1): a block touching a cell vector runs once per tile between C_TBEG T and C_TEND; LD, LDS
and ST carrying the TILE bit add the loop counter to their tile operand (a vector's T tiles are contiguous).

DRAM homes (Machine l1_vec): compute loads a DRAM vector's tile with C_LDR from the input ring, whose pages the
reader builds in the same order (N_STREAM: per tile, per ldr: the source read at its shift, or zeros, then the map's
residue patched in), and stores one with C_STW to the output ring, which the writer drains (B_SINK). A source is a
descriptor word kind << 28 | value: 0 an L1 tile, 1 a DRAM home at tile offset value (page core * hpages + value + t),
2 a DRAM constant at tile offset value (page core * cpages + value + t), 3 none. Every slot and constant has its own
tile count, so a coarse multigrid level takes its own pages, not a fine vector's T. A page is written at the 32-B phase of its DRAM source (a DRAM
read must keep the address mod 32), so C_LDR skips `page_phase` 16-B units.
"""
import numpy as np

from .machine import LANES, DBASE, KBASE, NH, widen_bf16
from .trace import SCALAR

# compute
C_END, C_ACQ, C_COMMIT, C_REL, C_LD, C_ST, C_FILL, C_RVF = range(8)
C_BIN = {"add": 8, "sub": 9, "mul": 10, "div": 11, "max": 12, "min": 13}
C_WHERE, C_CPD = 14, 15
C_UN = {k: 16 + i for i, k in enumerate(("neg", "abs", "sqrt", "tanh", "log", "exp", "gtz", "gez", "ltz", "lez",
                                         "eqz", "nez"))}
C_FOLD, C_TOK, C_SIGB, C_SIGN, C_WAITB, C_WAITN, C_JMP, C_BRT, C_LDS, C_TBEG, C_TEND, C_LDR, C_STW = range(28, 41)
C_MAD, C_MSB = 41, 42   # a = d + b c, a = d - b c
C_LDRH = 43             # C_LDR of a bf16 page (Machine narrow), widened to fp32 in Dest
C_STWH = 44             # C_STW packed to bf16 (Machine narrow 2)
C_PH = 45               # phase marker (Encoded(phase=...)): phase id in the top 16 bits; vm_compute CFD_PHASE
NARROW = 1 << 27        # desc bit: a bf16 constant page (vm_ncrisc window)
TILE = 0x80     # op bit: tile operand += loop counter
# BRISC
B_END, B_SIGC, B_WAITC, B_SIGN, B_WAITN, B_HALO, B_RED, B_OUT, B_FETCH, B_JMP, B_BRT, B_SINK = range(12)
B_SCAL = 16     # bscal: a scalar block on BRISC (lane 0, soft-float; uops in the compute encoding)
B_PH = 17       # phase marker, phase id in bits 8-11 (vm_brisc CFD_PHASE)
B_GATHER = 15   # bgather: a local-only gather on BRISC (prof slots 12..14 are HALO's parts)
# NCRISC
N_END, N_SIGC, N_WAITC, N_SIGB, N_WAITB, N_GATHER, N_HALO, N_JMP, N_BRT, N_STREAM, N_CODE, N_CODEAT = range(12)
PW = 256        # paged compute code: words per page (1 KB)
NONE16 = 0xFFFF


def W(op, a=0, b=0, c=0, d=0):
    return op | a << 8 | b << 12 | c << 16 | d << 20


# Wide gather entries (run_mesh sets WIDE when a core holds more than 65,536 lanes, T > 64): an entry is the low 32 bits
# of lane << 22 | idx, i.e. its destination lane within its tile (10 b) over a 22-bit source lane / halo position, and
# a halo send word is lane | run length << 20. Only with the streamed tables (tabstream.py), whose walk is per tile.
WIDE = False
DA = 32   # DRAM read alignment in bytes (NOC_DRAM_READ_ALIGNMENT_BYTES): 32 Wormhole, 64 Blackhole (run_mesh --arch)


def page_phase(d):
    """16-B units a page built from a source read at lane offset d starts after the page base."""
    return ((4 * d) & (DA - 1)) >> 4


def phase_h(d):
    """page_phase() of a bf16 page: 2 B a lane."""
    return ((2 * d) & (DA - 1)) >> 4


def f32bits(v):
    return int(np.float32(v).view(np.uint32))



class Stream:
    def __init__(self, paged=False):
        self.w, self.fix, self.lab, self.paged = [], [], {}, paged

    def __call__(self, *words):
        if self.paged and len(self.w) // PW != (len(self.w) + len(words) - 1) // PW:
            while len(self.w) % PW:     # an op never straddles two pages (C_JMP: on to the next page)
                self.w.append(C_JMP)
        self.w.extend(int(x) & 0xFFFFFFFF for x in words)

    def target(self, L):
        if self.paged:              # paged compute ignores jump targets; the word keeps the op's length
            self.w.append(0)
            return
        self.fix.append((len(self.w), L))
        self.w.append(0)

    def label(self, L):
        self.lab[L] = len(self.w)

    def insert(self, pos, words):
        """Insert words at pos (before anything already there); fixups after it move, labels stay."""
        k = len(words)
        self.w[pos:pos] = [int(x) & 0xFFFFFFFF for x in words]
        self.fix = [(q + k if q >= pos else q, L) for q, L in self.fix]

    def done(self):
        for pos, L in self.fix:
            self.w[pos] = self.lab[L]
        return np.array(self.w, np.uint32)


SYNC = {  # MIR ('sync', X, Y) -> (stream, op) pairs: X signals, Y waits
    ("C", "B"): (("c", C_SIGB), ("b", B_WAITC)),
    ("B", "C"): (("b", B_SIGC), ("c", C_WAITB)),
    ("C", "N"): (("c", C_SIGN), ("n", N_WAITC)),
    ("N", "C"): (("n", N_SIGC), ("c", C_WAITN)),
    ("B", "N"): (("b", B_SIGN), ("n", N_WAITB)),
    ("N", "B"): (("n", N_SIGB), ("b", B_WAITN)),
}


def _pad(rows):
    out = np.zeros((len(rows), max(max(len(r) for r in rows), 1)), np.uint32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def runs(lanes):
    """A halo send list with its runs marked: a run of consecutive lanes keeps every lane, its first one carries the
    run's length << 16 (1 for a lone value), the others 0 there; readers mask with 0xffff (vm_brisc B_HALO copies runs)."""
    w = lanes.astype(np.int64).copy()
    rs, rmax = (20, 4095) if WIDE else (16, 65535)
    if len(w):
        assert w.max() < 1 << rs
        head = np.r_[True, np.diff(lanes) != 1]
        if WIDE:   # a run longer than the length field splits: every rmax-th lane of a run heads a new one
            k = np.arange(len(w)) - np.maximum.accumulate(np.where(head, np.arange(len(w)), 0))
            head |= (k % rmax) == 0
        h = np.nonzero(head)[0]
        w[h] |= np.diff(np.r_[h, len(w)]) << rs
    return w.tolist()


def hrun_words(lane, idx, rmin=8):
    """hrun: a gather part's halo entries (lane << 16 | position, by lane) with every run of >= rmin entries whose lane
    and position both step by one and agree mod 4 replaced by its head (bit 15 set) and its length."""
    out, k, n = [], 0, len(lane)
    while k < n:
        e = k + 1
        while e < n and lane[e] == lane[e - 1] + 1 and idx[e] == idx[e - 1] + 1:
            e += 1
        if e - k >= rmin and lane[k] % 4 == idx[k] % 4:
            out += [int(lane[k]) << 16 | int(idx[k]) | 0x8000, e - k]
        else:
            out += (lane[k:e].astype(np.int64) << 16 | idx[k:e]).tolist()
        k = e
    return out


class Encoded:
    def __init__(self, m, page=False, pdev=None, tab_stream=False, look=False, cores=None, tab_sort=False, phase=None,
                 gtile=False):
        """page: the compute stream is paged from DRAM through a ring NCRISC fills (vm_ncrisc N_CODE): every basic
        block starts on a PW-word page, the padding is C_JMP (in paged mode JMP and BRT end a block and compute moves
        on to the next delivered page), and NCRISC pushes the pages of a MIR instruction before any of its own ops
        for it (N_CODE hi want: pages up to hi; N_CODEAT p: the next page to push is p, at every block start).
        pdev: cores per chip on a mesh of chips (constant pages are per chip), default all of m.P.
        look: an N_CODE also names want, the end of its straight-line compute code (the next label, jump or branch),
        and NCRISC pushes on up to it while the ring has free slots, without waiting for them (want = hi: off).
        tab_sort: with tab_stream, local residue entries sorted by source tile per page tile (tabstream.py sort).
        cores: the global cores whose tables, arena and constants to build (row r is core cores[r]), default all; the
        per-chip lowering (lower/spmd.py) builds one chip's.
        phase: {label: phase id} (Program.phase_at, build(phase=True)): a C_PH and a B_PH right after each such label.
        None or {}: no marker, the streams are the ones without the option.
        gtile: with tab_stream, a gather followed by its N -> C sync and a tile loop that reads the gathered vector
        plainly hands it over per tile: N_GATHER carries the loop's tile count Tb in bits 16-23 and NCRISC posts a
        token after each of its first Tb - 1 tiles (the N_SIGC is the last), compute waits once per pass right after
        C_TBEG. Paged: that gather's N_CODE also looks ahead over the loop's pages. Same values, same order."""
        phase = phase or {}
        self.m, self.page, self.look = m, page, look
        self.cores = np.arange(m.P) if cores is None else np.asarray(cores)
        assert not (tab_stream and getattr(m, "hrun", False)), "hrun marks the L1 gather tables"
        assert tab_stream or not WIDE, "wide gather entries carry the lane within its tile: only the streamed walk"
        self.loop_pages = 1     # paged: the most code pages the ring holds at once (a tile loop's body, or an N_CODE's
        #   pages past compute's release point); the ring needs more than this
        P = pdev or m.P
        phys, hphys = m.phys, m.hphys
        c, b, n = Stream(page), Stream(), Stream()
        S = {"c": c, "b": b, "n": n}
        self.out_names = []
        halo_slot = lambda h: hphys[("H", h)]
        def desc(r):
            if r is None:
                return 3 << 28
            t = phys[r]
            if t >= KBASE:
                k = m.fetched_from.get(r)
                return 2 << 28 | (NARROW | m.narrow_slot[k] if k in m.narrow else t - KBASE)
            if t >= DBASE:
                return 1 << 28 | (NARROW if r in m.narrow_home else 0) | t - DBASE
            return t
        vec = lambda r: m.T > 1 and r != "__anchor__" and (phys[r] >= m.nsmall or r in getattr(m, "ovl", ()))
        tf = lambda r: TILE if vec(r) and phys[r] < DBASE else 0

        def blk_rw(x):
            uops = x[1]
            return uops, [u[2] for u in uops if u[0] in ("ld", "lds", "rvf")], [u for u in uops if u[0] == "ldr"], [r for _, r in x[2]]

        gt = {}     # gtile: MIR index of a gather -> its consumer loop's tile count
        if gtile:
            assert tab_stream, "--gtile hands over tabstream gather tiles"
            mir = m.mir
            for j in range(len(mir) - 2):
                g, sy, x = mir[j], mir[j + 1], mir[j + 2]
                if g[0] != "gather" or (len(g) > 4 and g[4] in (3, 4)) or sy[:3] != ("sync", "N", "C") or x[0] != "blk":
                    continue
                uops, lds, ldr, sts = blk_rw(x)
                d = g[1]
                if not (vec(d) and m.tiles([x]) > 1 and any(u[0] == "ld" and u[2] == d for u in uops)):
                    continue
                if d in sts or any(u[0] in ("lds", "rvf") and u[2] == d for u in uops) or \
                        any(d == u[2] or d in [y for y, _ in u[5]] for u in ldr):
                    continue
                gt[j] = m.tiles([x])
            self.gtile_n = len(gt)
        waitn = [False, None]   # gtile: a C_WAITN owed after the next C_TBEG; where it went

        def head(blks):
            """Open the tile loop of one block or of a fused segment's blocks: TBEG, the reader's one stream of every
            ldr page in program order, the writer's one sink of every DRAM store."""
            uops = [u for x in blks for u in x[1]]
            lds = [u[2] for u in uops if u[0] in ("ld", "lds", "rvf")]
            ldr = [u for u in uops if u[0] == "ldr"]
            sts = [r for x in blks for _, r in x[2]]
            loop = any(vec(r) for r in lds + sts) or bool(ldr)
            if loop:
                assert not any(u[0] == "rvf" and vec(u[2]) for u in uops)
                # a shifted load reads the neighbouring tiles: never of a vector this loop overwrites
                assert not {u[2] for u in uops if u[0] == "lds"} & set(sts)
                assert not {x for u in ldr if u[3] or u[4] is not None
                            for x in [u[2]] + [y for y, _ in u[5]]} & set(sts)
                c(W(C_TBEG) | m.tiles(blks) << 16)
                if waitn[0]:
                    waitn[:] = [False, len(c.w)]
                    c(W(C_WAITN))
            # pages per stream / sink: a vector's tile count, 0 = T (the multigrid levels have fewer)
            tn = 0 if m.tiles(blks) == m.T else m.tiles(blks)
            if ldr:
                # one patched entry per stream, or any number with pseg >= 3 (vm_ncrisc: CFD_TABSTREAM without CFD_TRID)
                assert sum(u[4] is not None for u in ldr) <= 1 or getattr(m, "pseg", 0) >= 3
                n(W(N_STREAM), len(ldr) | tn << 16)
                for u in ldr:
                    assert len(u[5]) <= 8      # the reader keeps 8 part cursors (vm_ncrisc.cpp)
                    ss = 14 if WIDE else 16     # WIDE: 18-bit shift, 10-bit map (vm_ncrisc SS, SM)
                    assert -(1 << 31 - ss) <= u[3] < 1 << 31 - ss and (u[4] or 0) < (1 << 28 - ss) - 1, (u[3], u[4])
                    n(desc(u[2]), (u[3] & (1 << 32 - ss) - 1) << ss | (0 if u[4] is None else u[4] + 1) << 4 | len(u[5]))
                    for x, h in u[5]:
                        n(desc(x), NONE16 if h is None else halo_slot(h))
            sink = [r for r in sts if phys[r] >= DBASE]
            if sink:
                assert all(phys[r] < KBASE for r in sink)
                b(W(B_SINK), len(sink) | tn << 16)
                for r in sink:
                    b(desc(r))
            return loop, lds, sts

        def body(ins):
            c(W(C_ACQ))
            for u in ins[1]:
                k = u[0]
                if k == "ld":
                    c(W(C_LD, u[1]) | tf(u[2]) | phys[u[2]] << 16)
                elif k == "lds":      # offset load: lane l reads lane l + u[3] (a multiple of 4 = 16 B)
                    assert u[3] % 4 == 0
                    c(W(C_LDS, u[1]) | tf(u[2]) | phys[u[2]] << 16, u[3] // 4)
                elif k == "ldr":
                    if u[2] is not None and (m.fetched_from.get(u[2]) in m.narrow or u[2] in m.narrow_home):
                        assert u[3] % 8 == 0 and u[4] is None
                        c(W(C_LDRH, u[1], phase_h(u[3])))
                    else:
                        c(W(C_LDR, u[1], page_phase(u[3]) if u[2] is not None else 0))
                elif k == "fill":
                    c(W(C_FILL, u[1]), f32bits(u[2]))
                elif k == "rvf":
                    c(W(C_RVF, u[1]) | phys[u[2]] << 16)
                elif k == "cpd":
                    c(W(C_CPD, u[1], u[2]))
                elif k == "op":
                    if u[1] == "where":
                        c(W(C_WHERE, u[2], u[3], u[4], u[5]))
                    elif u[1] in ("mad", "msb"):
                        c(W(C_MAD if u[1] == "mad" else C_MSB, u[2], u[3], u[4], u[5]))
                    else:
                        c(W(C_BIN[u[1]], u[2], u[3], u[4]))
                elif k == "un":
                    c(W(C_UN[u[1]], u[2]))
                elif k == "fold":
                    c(W(C_FOLD, u[1]))
                else:
                    raise NotImplementedError(k)
            c(W(C_COMMIT))
            for s, r in ins[2]:
                if phys[r] >= DBASE:
                    c(W(C_STWH if r in m.narrow_home else C_STW, s))
                else:
                    c(W(C_ST, s) | tf(r) | phys[r] << 16)
            c(W(C_REL))

        def pad():
            if page:
                while len(c.w) % PW:
                    c(W(C_JMP))

        ncur = 0    # paged: NCRISC's code cursor in straight-line n code (an N_CODE up to it pushes nothing)
        wants = []  # paged: n positions of the want words of the N_CODEs in the current straight-line stretch

        def seal(pg):
            for q in wants:
                if look:
                    n.w[q] = max(n.w[q], pg)
            wants.clear()
        # paged: compute's release point (vm_compute W_CREL) as it runs the code in order. It moves only on entering a
        # page, and inside a tile loop it stays at the body's first page, so a loop ending mid-page keeps its body held
        # while NCRISC pushes the next instruction's pages: the ring must hold both
        rel, tpc = 0, None
        gwant = {}
        mir = m.mir
        skip = set()
        for j, ins in enumerate(mir):
            if j in skip:
                continue
            op = ins[0]
            neg = 0
            if op == "jumpif" and j + 2 < len(mir) and mir[j + 1][0] == "jump" and mir[j + 2][:2] == ("label", ins[2]):
                # `jumpif f exit; jump head; label exit`: one branch on f == 0 back to the head. A loop pass then ends
                # on its own page instead of a page that holds only the jump (in paged mode NCRISC fetches every page
                # compute enters, so the back edge cost a DRAM page per iteration)
                ins, neg = ("jumpif", ins[1], mir[j + 1][1]), 1
                skip.add(j + 1)
            n0, c0 = len(n.w), len(c.w)
            if page and op == "label":
                if ncur is None or -(-len(c.w) // PW) > ncur:
                    n(W(N_CODE), -(-len(c.w) // PW), -(-len(c.w) // PW))     # falling into the label: the code before it is pushed first
                seal(-(-len(c.w) // PW))    # N_CODEAT at the label rewinds the cursor: no push may pass it
                pad()
            if op in ("blk", "seg"):
                items = [ins] if op == "blk" else ins[1]
                loop, lds, sts = head([x for x in items if x[0] == "blk"])
                tpc = (len(c.w) if waitn[1] is None else waitn[1]) if loop else None
                assert not waitn[0], "gtile: the consumer loop did not open"
                waitn[1] = None
                for x in items:
                    if x[0] == "blk":
                        body(x)
                    else:
                        c(W(C_TOK))
                if loop:
                    # one tile carried to the next pass (a partial); a segment's sync pass places its own tokens
                    if op == "blk" and {r for r in lds if not vec(r)} & set(sts):
                        c(W(C_TOK))
                    c(W(C_TEND))
            elif op == "tok":
                c(W(C_TOK))
            elif op == "sync":
                for st, o in SYNC[(ins[1], ins[2])]:
                    if st == "c" and j - 1 in gt:
                        waitn[0] = True     # compute waits in the consumer loop instead
                    else:
                        S[st](W(o))
            elif op == "gather" and len(ins) > 4 and ins[4] in (3, 4):
                assert not tab_stream, "bgather reads the L1 gather tables"
                b(W(B_GATHER, len(ins[3])), phys[ins[1]], ins[2], *[phys[s] for s, _ in ins[3]])
            elif op == "gather":
                assert phys[ins[1]] < DBASE and len(ins[3]) < 16, "an N_GATHER carries at most 15 parts"
                n(W(N_GATHER, len(ins[3]), ins[4] if len(ins) > 4 else 0) | gt.get(j, 0) << 16, phys[ins[1]], ins[2])
                for src, h in ins[3]:
                    n(desc(src), NONE16 if h is None else halo_slot(h))
            elif op == "bscal":
                ws = []
                for u in ins[1]:
                    k = u[0]
                    if k == "ld":
                        ws += [W(C_LD, u[1]), phys[u[2]]]
                    elif k == "fill":
                        ws += [W(C_FILL, u[1]), f32bits(u[2])]
                    elif k == "cpd":
                        ws.append(W(C_CPD, u[1], u[2]))
                    elif k == "un":
                        ws.append(W(C_UN[u[1]], u[2]))
                    elif u[1] == "where":
                        ws.append(W(C_WHERE, u[2], u[3], u[4], u[5]))
                    elif u[1] in ("mad", "msb"):
                        ws.append(W(C_MAD if u[1] == "mad" else C_MSB, u[2], u[3], u[4], u[5]))
                    else:
                        ws.append(W(C_BIN[u[1]], u[2], u[3], u[4]))
                for s, r in ins[2]:
                    ws += [W(C_ST, s), phys[r]]
                b(W(B_SCAL), len(ws), *ws)
            elif op == "halo":
                b(W(B_HALO), halo_slot(ins[1]), desc(ins[2]))
                n(W(N_HALO), halo_slot(ins[1]), desc(ins[2]))
            elif op == "red":
                b(W(B_RED, 0, ins[3] if len(ins) > 3 else 0), phys[ins[2]], phys[ins[1]])   # mesh scope in bits 12-15
            elif op == "red2":   # two or three partials, one cross-core reduction (vm_brisc B_RED, count - 1)
                b(W(B_RED, len(ins) // 2 - 1), *[phys[x] for f, p in zip(ins[1::2], ins[2::2]) for x in (p, f)])
            elif op == "fetch":
                assert phys[ins[1]] < DBASE, "a fetch into a DRAM home"
                b(W(B_FETCH), phys[ins[1]], m.coff[m.fetched_from[ins[1]]], m.size(ins[1]))
            elif op == "out":
                if ins[1] not in self.out_names:
                    self.out_names.append(ins[1])
                b(W(B_OUT, 1 if ins[1] == "outer" else 0), phys[ins[2]], self.out_names.index(ins[1]))
            elif op == "label":
                for s in S.values():
                    s.label(ins[1])
                if ins[1] in phase:
                    if page:
                        pad()   # the marker opens the label's page (the paged block below starts there)
                    assert 0 <= phase[ins[1]] < 15
                    c(W(C_PH) | phase[ins[1]] << 16)
                    b(W(B_PH, phase[ins[1]]))
            elif op == "jump":
                for s, o in ((c, C_JMP), (b, B_JMP), (n, N_JMP)):
                    if s.paged:
                        s(W(o), 0)
                    else:
                        s(W(o))
                        s.target(ins[1])
            elif op == "jumpif":   # W(.., 1): taken when the flag is zero
                for s, o in ((c, C_BRT), (b, B_BRT), (n, N_BRT)):
                    if s.paged:
                        s(W(o, neg), phys[ins[1]], 0)
                    else:
                        s(W(o, neg), phys[ins[1]])
                        s.target(ins[2])
            elif op == "keep":
                pass
            else:
                raise NotImplementedError(op)
            if page:
                if c0 % PW == 0 and len(c.w) > c0:
                    rel = max(rel, c0 // PW)        # entering the page this instruction starts
                if len(n.w) > n0:   # an N_CODE ahead of this instruction's NCRISC ops
                    self.loop_pages = max(self.loop_pages, -(-len(c.w) // PW) - rel)
                if tpc is not None:     # the body alone stays held through every pass
                    self.loop_pages = max(self.loop_pages, (len(c.w) - 1) // PW - tpc // PW + 1)
                for g in range(c0 // PW + 1, (len(c.w) - 1) // PW + 1):      # pages entered later in it
                    rel = max(rel, tpc // PW if tpc is not None and g * PW > tpc else g)
                tpc = None
                if op == "label":
                    n(W(N_CODEAT), len(c.w) // PW)
                    ncur = len(c.w) // PW
                elif len(n.w) > n0 and (ncur is None or -(-len(c.w) // PW) > ncur or j in gt):
                    n.insert(n0, [W(N_CODE), -(-len(c.w) // PW), -(-len(c.w) // PW)])
                    wants.append(n0 + 2)    # inserts only happen at n0 of the current op, after every earlier want
                    ncur = -(-len(c.w) // PW)
                    if j in gt:
                        gwant[j + 2] = n0 + 2
                if j in gwant:      # gtile: the gather's N_CODE looks ahead over its consumer loop's pages
                    q = gwant.pop(j)
                    n.w[q] = max(n.w[q], -(-len(c.w) // PW))
                if op == "jumpif":
                    pad()
                    seal(len(c.w) // PW)
                    n(W(N_CODEAT), len(c.w) // PW)
                    ncur = len(c.w) // PW
                elif op == "jump":
                    pad()
                    seal(len(c.w) // PW)
                    ncur = None     # only a label follows
        c(W(C_END))
        if page:
            n(W(N_CODE), -(-len(c.w) // PW), -(-len(c.w) // PW))
            seal(-(-len(c.w) // PW))
            pad()
        b(W(B_END)), n(W(N_END))
        assert m.ntiles < 1 << 16
        self.c, self.b, self.n = c.done(), b.done(), n.done()
        self._tables()
        if tab_stream:      # residue tables in DRAM, a tile-offset index in L1 (tabstream.py)
            from .tabstream import stream
            self.tab_cursor, self.tab_halo_cursor = self.tab, self.tab_halo
            self.tab_stream = S = stream(self, sort=tab_sort)
            self.tab = _pad(S["l1"])
            self.tab_blob = _pad(S["blob"])
            self.tab_halo = np.array(S["halo_off"], np.uint32)
        else:
            self.tab_stream = None
        self._arena()

    # ---- per-core tables (uint32 words), same layout on every core, padded to one length
    def _tables(self):
        m = self.m
        maps, cores = m.maps, self.cores
        P = len(cores)
        tabs = [[] for _ in range(P)]
        nm = len(maps.tab)
        for i in range(P):
            tabs[i] += [nm] + [0] * nm
        # hrun: halo entries of maps no stream patches carry runs (lane and position consecutive, agreeing mod 4) as a
        # head word with bit 15 set and the run's length after it (vm_ncrisc hscatter moves each with one NoC copy)
        blks = lambda x: x[1] if x[0] == "seg" else [x] if x[0] == "blk" else []
        smaps = {u[4] for x in m.mir for bk in blks(x) if bk[0] == "blk" for u in bk[1] if u[0] == "ldr"}
        hrun = getattr(m, "hrun", False)
        self.hrun_stat = [0, 0]     # halo entries in marked maps, entries a run copy replaces (minus its two words)
        if hrun:
            assert maps.hmax < 0x8000 and not WIDE
        XS = 22 if WIDE else 16   # full-lane words; tabstream keeps their low 32 bits (WIDE: the lane within its tile)
        for mid, t in enumerate(maps.tab):
            # each core's entries, in entry order (a stable sort by core)
            order = np.argsort(t["core"], kind="stable")
            lo = np.searchsorted(t["core"][order], cores, "left")
            hi = np.searchsorted(t["core"][order], cores, "right")
            zo = np.argsort(t["zc"], kind="stable")
            zlo, zhi = np.searchsorted(t["zc"][zo], cores, "left"), np.searchsorted(t["zc"][zo], cores, "right")
            for r in range(P):
                tw = tabs[r]
                tw[1 + mid] = len(tw)
                on = order[lo[r]:hi[r]]
                part, kind = t["part"][on], t["kind"][on]
                lane, idx = t["lane"][on].astype(np.int64), t["idx"][on].astype(np.int64)
                for k in range(t["nparts"]):
                    sel = part == k
                    loc = sel & (kind == 0)
                    hal = sel & (kind == 1)
                    o = np.argsort(lane[loc], kind="stable")
                    lw = (lane[loc][o] << XS | idx[loc][o]).tolist()
                    o = np.argsort(lane[hal], kind="stable")
                    hl, hx = lane[hal][o], idx[hal][o]
                    hw = (hl << XS | hx).tolist()
                    if hrun and mid not in smaps:
                        hw = hrun_words(hl, hx)
                        self.hrun_stat[0] += len(hl)
                        self.hrun_stat[1] += len(hl) - len(hw)
                    tw += [len(lw), len(hw)] + lw + hw     # by destination lane: the reader walks them tile by tile
                z = np.sort(t["zl"][zo[zlo[r]:zhi[r]]].astype(np.int64))
                tw += [len(z)] + z.tolist()
        # halo: senders' lists to their receivers, receivers' senders (for credits), per plan. Plan 0 (the cells)
        # sits at tab_halo as in a one-space program; with more plans a directory sits just BELOW tab_halo, read
        # downwards: word tab_halo - 1 - s is halo slot s's plan | its receive buffer's word offset << 8 (slots sized
        # per plan, Machine.hoff), word tab_halo - NH - q plan q's offset from tab_halo (q > 0). RT arg 1 = tab_halo,
        # bit 31 set when the directory exists (tab_halo keeps its full range). One plan: slot s at s * hmax.
        nq = len(maps.plans)
        self.halo_off, self.halo_dir = [], [0] * P
        for r, i in enumerate(cores.tolist()):
            tw = tabs[r]
            if nq > 1:
                tw += [0] * (NH - 1 + nq)
                self.halo_dir[r] = 1
            h = len(tw)
            self.halo_off.append(h)
            for q, pl in enumerate(maps.plans):
                if q:
                    tw[h - NH - q] = len(tw) - h
                recv = sorted(pl["receivers"].get(i, []))
                tw.append(len(recv))
                for x in recv:
                    lanes = pl["send"][(i, x)]
                    tw += [x, pl["recv_off"][(i, x)], len(lanes)] + runs(np.asarray(lanes))
                snd = sorted(pl["senders"].get(i, []))
                tw.append(len(snd))
                tw += snd
            if nq > 1:
                assert nq < 256 and m.hwords < 1 << 24
                for s in range(NH):
                    tw[h - 1 - s] = m.halo_plan.get(s, 0) | (m.hoff[s] if s < m.nhalo else 0) << 8
        for mt in maps.tab:
            if WIDE:
                assert len(mt["idx"]) == 0 or mt["idx"].max() < 1 << 22
            else:
                assert len(mt["lane"]) == 0 or max(mt["lane"].max(), mt["idx"].max()) < 1 << 16   # zl is uint16
        self.tab_len = [len(t) for t in tabs]     # each core's own length (tabstream keeps no padding)
        L = max(len(t) for t in tabs)
        self.tab = np.zeros((P, L), np.int64 if WIDE else np.uint32)   # WIDE: full lanes until tabstream cuts them
        for i, t in enumerate(tabs):
            self.tab[i, :len(t)] = t
        self.tab_halo = np.array(self.halo_off, np.uint32)

    def _halo_word(self):
        """RT arg 1 per core: tab_halo, bit 31 set when a halo directory sits below it (more than one plan)."""
        assert self.tab_halo.max() < 1 << 31
        return self.tab_halo | np.array(self.halo_dir, np.uint32) << 31

    # ---- initial arena and the DRAM constants
    def _scatter(self, sp, values, n):
        """Global vector -> (rows, n, LANES) over self.cores (Layout.scatter for a core subset)."""
        m, cores = self.m, self.cores
        own, lane = m.lay.owner[sp.name], m.lay.lane[sp.name]
        if len(cores) == m.P:
            return m.lay.scatter(sp, values, n).reshape(m.P, n, LANES)
        out = np.zeros((len(cores), n * LANES), values.dtype)
        sel = (own >= cores[0]) & (own <= cores[-1])
        out[own[sel] - cores[0], lane[sel]] = values[sel]
        return out.reshape(len(cores), n, LANES)

    def _arena(self):
        m = self.m
        P = len(self.cores)
        assert P == m.P or np.array_equal(self.cores, np.arange(self.cores[0], self.cores[0] + P))
        A = np.zeros((P, m.ntiles, LANES), np.float32)
        Hm = np.zeros((P, max(m.hpages, 1), LANES), np.float32)   # DRAM homes: page core * hpages + offset + t
        for r, v in m.pre.items():
            t = m.phys.get(r)
            if t is None or t >= KBASE:
                continue
            n = m.size(r)
            X, t = (Hm, t - DBASE) if t >= DBASE else (A, t)
            if v is None:
                X[:, t:t + n] = 0
            else:
                X[:, t:t + n] = self._scatter(m.space[r], np.asarray(v, np.float32), n)
        self.arena, self.home = A, Hm
        # DRAM constants: page core * cpages + coff + tile, each constant its own tile count
        C = np.zeros((P, max(m.cpages, 1), LANES), np.float32)
        for r, o in m.coff.items():
            n = m.size(r)
            C[:, o:o + n] = self._scatter(m.space[r], np.asarray(m.pre[r], np.float32), n)
        for r, o in m.narrow_slot.items():   # bf16 shadow: lane 2k in the low half of word k, words 512.. unused
            n = m.size(r)
            h = self._scatter(m.space[r], widen_bf16(m.pre_h[r]), n).view(np.uint32) >> 16
            C[:, o:o + n].view(np.uint32)[..., :LANES // 2] = np.ascontiguousarray(h.astype(np.uint16)).view(np.uint32)
        self.consts = C

    def state_from_arena(self, arena, home=None):
        """arena (P, ntiles, LANES) and DRAM homes (P, hpages, LANES) -> state name -> global arrays."""
        m = self.m
        out = {}
        for name, cr in m.prog.state:
            arrs = []
            for r in cr.ravel():
                r2 = m.ren(r)
                sp = m.space[r2]
                if r2 in m.phys:
                    t = m.phys[r2]
                    X, t = (home, t - DBASE) if t >= DBASE else (arena, t)
                    arrs.append(m.lay.collect(sp, X[:, t:t + m.size(r2)].reshape(m.P, -1)))
                else:
                    arrs.append(np.asarray(m.prog_pre[r2], np.float32))
            out[name] = np.stack(arrs, -1).reshape(arrs[0].shape + cr.shape) if cr.ndim else arrs[0]
        return out

    def n_maxop(self):
        """words of the longest N op (vm_ncrisc.cpp reads an op's words from pc on): an N code window must hold it"""
        n, pc, mx = self.n, 0, 0
        while pc < len(n):
            op = int(n[pc]) & 0xff
            if op == N_GATHER:
                ln = 3 + 2 * ((int(n[pc]) >> 8) & 0xf)
            elif op == N_STREAM:
                p = pc + 2
                for _ in range(int(n[pc + 1]) & 0xFFFF):
                    p += 2 + 2 * (int(n[p + 1]) & 0xf)
                ln = p - pc
            elif op == N_BRT:
                ln = 3
            elif op in (N_HALO, N_CODE):
                ln = 3
            elif op in (N_JMP, N_CODEAT):
                ln = 2
            else:
                assert op <= N_WAITB, op
                ln = 1
            mx, pc = max(mx, ln), pc + ln
        assert pc == len(n)
        return mx

    def b_maxop(self):
        """words of the longest B op (vm_brisc.cpp reads an op's words from pc on): a B code window must hold it"""
        b, pc, mx = self.b, 0, 0
        while pc < len(b):
            w = int(b[pc])
            op, a = w & 0xff, (w >> 8) & 0xf
            ln = {B_HALO: 3, B_OUT: 3, B_FETCH: 4, B_JMP: 2, B_BRT: 3, B_PH: 1}.get(op)
            if op == B_RED:
                ln = 3 + 2 * a
            elif op == B_GATHER:
                ln = 3 + a
            elif op == B_SINK:
                ln = 2 + (int(b[pc + 1]) & 0xffff)
            elif op == B_SCAL:
                ln = 2 + int(b[pc + 1])
            elif ln is None:
                assert op <= B_WAITN, op
                ln = 1
            mx, pc = max(mx, ln), pc + ln
        assert pc == len(b)
        return mx

    def sizes(self):
        m = self.m
        return dict(page=self.page, loop_pages=self.loop_pages, c_words=len(self.c), b_words=len(self.b), n_words=len(self.n), tab_words=self.tab.shape[1],
                    arena_kb=m.ntiles * 4, home_mb=m.hpages * 4 * m.P / 1024, const_mb=m.cpages * 4 * m.P / 1024, halo=(m.nhalo, m.maps.hmax, m.hwords), consts=len(m.streamed))
