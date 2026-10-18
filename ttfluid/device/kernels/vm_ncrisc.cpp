// Resident VM, NCRISC side. Waits for BRISC's start barrier, then walks the N stream: gathers from the per-core map
// tables (own lanes, halo slots, zeros), halo credits to the senders, signals and waits, branches.
//
// Table (cb_tab, see ttres/encode.py): [nmaps, map_off...]; per map and part: n_loc, n_halo, (lane << 16 | idx)...;
// then n_zero, lanes... . Halo section at tab_halo: [nrecv, (core, off, n, lanes (runs marked)...)..., nsend, senders...].
// A gather part that reads halo slot s waits until data_sem[s] counts every sender's push of the current round;
// at each HALO for slot s this core's earlier reads of s are done, so it credits every sender (the sender pushes
// round k only after all its receivers' k-th credit).
// Waits are bounded and abort like BRISC's (see vm_brisc.cpp for the sync region and header words).
//
// DRAM homes: N_STREAM builds the pages compute's C_LDR loads (ttres/encode.py): per tile, per entry, the source
// read at its lane shift (an L1 window, or the DRAM pages it spans, at the source's 32-B phase), or zeros, then the
// entry's map residue patched in (cursors walk the lane-sorted table one tile at a time). A source is a descriptor:
// kind << 28 | value, kind 0 L1 tile, 1 DRAM home slot, 2 DRAM constant, 3 none. Single DRAM words are read
// 16 at a time into 32-B scratch slots at their phase.
#include "api/dataflow/dataflow_api.h"

// gather entries dest << XS | src (ttres/encode.py): 16-bit lanes, or CFD_WIDE (T > 64 tiles a core) 22-bit sources
// and the destination lane within its tile (tabstream only: the stream walks one destination tile at a time, WT)
#ifdef CFD_WIDE
#if !defined(CFD_TABSTREAM) || defined(CFD_HRUN)
#error "CFD_WIDE needs CFD_TABSTREAM and no CFD_HRUN"
#endif
constexpr uint32_t XS = 22, XI = 0x3fffff, XT = 0x3ffc00, HI = 0xfffff, WT = 1024;
#else
constexpr uint32_t XS = 16, XI = 0xffff, XT = 0xfc00, HI = 0xffff, WT = 0;
#endif
// N_STREAM entry word: shift << SS | (map + 1) << 4 | nparts; CFD_WIDE shifts reach past 32,767 lanes (18 bits, 10-bit map)
#ifdef CFD_WIDE
constexpr uint32_t SS = 14, SM = 0x3ff;
#else
constexpr uint32_t SS = 16, SM = 0xfff;
#endif

#if defined(CFD_OS) && CFD_OS >= 2   // run_mesh --os 2: the reader built for size too (see vm_brisc.cpp)
#pragma GCC optimize("Os")
#endif

enum : uint32_t { N_END = 0, N_SIGC, N_WAITC, N_SIGB, N_WAITB, N_GATHER, N_HALO, N_JMP, N_BRT, N_STREAM, N_CODE, N_CODEAT };
enum : uint32_t { W_NDONE = 1000, W_ABORT = 1004, W_CDONE = 1008, NH = 32 };   // vm_brisc.cpp sync layout

// hsplit: BRISC sends a halo to receivers [0, h), NCRISC to [h, nrecv); h splits the values about in half. t = the
// halo section [nrecv, (core, off, n, lanes...)...]
static inline uint32_t hsplit_at(const uint32_t* t) {
    const uint32_t nr = t[0];
    uint32_t total = 0, p = 1;
    for (uint32_t r = 0; r < nr; ++r) {
        total += t[p + 2];
        p += 3 + t[p + 2];
    }
    uint32_t acc = 0, h = 0;
    p = 1;
    while (h < nr && 2 * acc < total) {
        acc += t[p + 2];
        p += 3 + t[p + 2];
        ++h;
    }
    return h;
}

#ifdef CFD_TRACE
// run_vm --trace N: agent g (0 unpack, 1 math, 2 pack, 3 BRISC, 4 NCRISC) records its first N ops issued at least
// CFD_TRACE_AT cycles after its own start, at hdr word 1024 + g * (2N + 2): [count, start, (pc | op << 24, issue tick)...]
#define TR_REC(pc_, op_, t_)                                                         \
    if (!trs_ && hdr[1025] - tre_ < 0x40000000u) trs_ = hdr[1025]; /* UNPACK's start, after ours */ \
    if (trs_ && trn_ < CFD_TRACE && (t_) - trs_ >= CFD_TRACE_AT) {                   \
        tr_[2 + 2 * trn_] = (pc_) | ((op_) << 24);                                   \
        tr_[3 + 2 * trn_] = (t_);                                                    \
        tr_[0] = ++trn_;                                                             \
    }
#endif

void kernel_main() {
#ifdef CFD_TRACE
    const uint32_t tre_ = get_timestamp_32b();   // kernel entry: compute starts after it, a stale start from the last run before it
#endif
    constexpr uint32_t xy0 = 14;   // same common args as BRISC: limit at 13, P words x | y << 16 from 14
    const uint32_t limit = get_common_arg_val<uint32_t>(13);
    if (limit == 0) return;   // run_mesh --l1-check
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t tab_halo = get_arg_val<uint32_t>(1) & 0x7fffffff, hdir = get_arg_val<uint32_t>(1) >> 31;
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ncode = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(3);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(4);
    constexpr uint32_t cb_nc = get_compile_time_arg_val(5);
    constexpr uint32_t cb_cn = get_compile_time_arg_val(6);
    constexpr uint32_t cb_nb = get_compile_time_arg_val(7);
    constexpr uint32_t cb_bn = get_compile_time_arg_val(8);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(9);
    constexpr uint32_t hmax = get_compile_time_arg_val(10);
    constexpr uint32_t P = get_compile_time_arg_val(11);
    constexpr uint32_t cb_in = get_compile_time_arg_val(12);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(13);   // 4096 zero bytes, then 16 x 32-B word slots
    constexpr uint32_t NT = get_compile_time_arg_val(14);      // tiles per cell vector
    constexpr uint32_t HP = get_compile_time_arg_val(15);      // DRAM home pages per core (encode.py: packed slots)
    constexpr uint32_t CP = CFD_CP;                             // DRAM constant pages per core
    constexpr uint32_t RING = get_compile_time_arg_val(16);    // cb_in pages (4096 + 32 bytes each)
    // paged compute code (encode.py page=True): N_CODE pushes the 1-KB DRAM pages of the compute stream into the CR
    // pages of cb_ccode, a slot only once compute released it (sync word W_CREL, written by compute's PACK)
    constexpr uint32_t CR = get_compile_time_arg_val(17);
    constexpr uint32_t cb_ccode = get_compile_time_arg_val(18);
    constexpr uint32_t W_CREL = 1012;
#if defined(CFD_BPATCH) || defined(CFD_BGTS)
    // B-patch mailbox: the last 256 B of cb_sync (vm_brisc.cpp bsvc; a single chip's cb_sync is 8 KB under CFD_BPATCH)
#ifdef CFD_SYNCB
    constexpr uint32_t MB = CFD_SYNCB / 4 - 64;
#else
    constexpr uint32_t MB = 2048 - 64;
#endif
    enum : uint32_t { M_SEQ = MB, M_DONE = MB + 4, M_N = MB + 8, M_JOB = MB + 12 };
    uint32_t bseq = 0;
#endif
#ifdef CFD_TGTS
    // T-gather mailbox: the 256 B of cb_sync before the B one, same layout; compute's UNPACK copies its jobs from
    // inside its own waits (vm_compute.cpp tsvc) and answers DONE = SEQ
    constexpr uint32_t TMB = CFD_SYNCB / 4 - 128;
    enum : uint32_t { T_SEQ = TMB, T_DONE = TMB + 4, T_N = TMB + 8, T_JOB = TMB + 12 };
    uint32_t tseq = 0;
#endif
    constexpr auto a_const = TensorAccessorArgs<19>();
    constexpr auto a_home = TensorAccessorArgs<a_const.next_compile_time_args_offset()>();
    constexpr auto a_code = TensorAccessorArgs<a_home.next_compile_time_args_offset()>();
    const auto t_code = TensorAccessor(a_code, get_common_arg_val<uint32_t>(1), get_compile_time_arg_val(a_code.next_compile_time_args_offset()));
    uint32_t ccur = 0, cpushed = 0;
    // N code window (run_vm --nwin): cb_ncode holds NWIN words of the n stream from word wb on; an op whose NMAX
    // words could leave it refills it from DRAM at pc rounded down to 1 KB (NWIN = 0: the whole stream is in L1)
    constexpr uint32_t xn = a_code.next_compile_time_args_offset() + 1;
    constexpr uint32_t NWIN = get_compile_time_arg_val(xn), NMAX = get_compile_time_arg_val(xn + 1);
    constexpr auto a_ncode = TensorAccessorArgs<xn + 2>();
    const auto t_ncode = TensorAccessor(a_ncode, get_common_arg_val<uint32_t>(3), get_compile_time_arg_val(a_ncode.next_compile_time_args_offset()));
    uint32_t wb = 0, nrefill = 0;   // BRISC loaded the first window
    // gather tile cache (run_vm --gcache K): a DRAM-homed source value is read as its whole 4-KB tile into one of K L1
    // slots (round robin), later values of that tile come from L1. Valid within one N op (reset at N_GATHER/N_STREAM),
    // the span over which the per-value reads it replaces already assumed the source unchanged
    constexpr uint32_t xg = a_ncode.next_compile_time_args_offset() + 1;
    constexpr uint32_t cb_gc = get_compile_time_arg_val(xg), NGC = get_compile_time_arg_val(xg + 1);
    uint32_t gcd[NGC ? NGC : 1], gct[NGC ? NGC : 1], gcn = 0, gmiss = 0;
    // the last tile hit: consecutive values of a part mostly read one source tile, so most skip the tag search
    uint32_t gld = 0xffffffffu, glt = 0;
    const volatile uint32_t* glp = nullptr;
    auto ginv = [&]() { for (uint32_t j = 0; j < NGC; ++j) gcd[j] = 0xffffffffu; gld = 0xffffffffu; };
    ginv();
    const auto t_const = TensorAccessor(a_const, get_common_arg_val<uint32_t>(5), 4096);
    const auto t_home = TensorAccessor(a_home, get_common_arg_val<uint32_t>(xy0 + P), 4096);
    const uint32_t zero = get_write_ptr(cb_rs), scr = zero + 4096;
    constexpr uint32_t DA = NOC_DRAM_READ_ALIGNMENT_BYTES;   // a DRAM read keeps its address mod DA: 32 B WH, 64 B BH
    constexpr uint32_t PS = 4096 + DA;
    const uint32_t in1 = get_write_ptr(cb_in) + RING * PS;   // cb_in's fifo, nothing written yet
#ifdef CFD_TRID
    const uint32_t in0 = in1 - RING * PS;
#endif
    if (HP)
        for (uint32_t i = 0; i < 1024; ++i) reinterpret_cast<volatile uint32_t*>(zero)[i] = 0;

    const uint32_t sync = get_write_ptr(cb_sync);
    volatile tt_l1_ptr uint32_t* S = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    uint32_t* A = reinterpret_cast<uint32_t*>(get_write_ptr(cb_v));
    const uint32_t* H = reinterpret_cast<const uint32_t*>(get_write_ptr(cb_halo));
    const uint32_t* T = reinterpret_cast<const uint32_t*>(get_write_ptr(cb_tab));
    const uint32_t nwl1 = get_write_ptr(cb_ncode);
    const uint32_t* N = reinterpret_cast<const uint32_t*>(nwl1);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_hdr));
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    uint32_t pc = 0;
    const uint32_t arena = get_write_ptr(cb_v);
#ifdef CFD_TRID
    // run_vm --trid 1: STREAM tags ring slot k's reads with NOC transaction id 1 + k and pushes each slot, in order, as
    // soon as its own reads landed (the id's outstanding count is 0) while later slots stay in flight; without it a
    // batch waits for one full read barrier, i.e. for its last read. Residue-table reads carry TID_TAB + buffer,
    // gather-cache and single-value reads TID_GC, so the patched page and a cache miss wait for their own reads only.
    // Id 0 carries every untagged read and is never waited on: its count is an NIU register no program resets, and a
    // count an earlier program left never drains (da35ea2 hung on it). Any id can be left so, hence a core whose ids
    // 1..TID_GC do not all read 0 at start uses full read barriers for the run (hdr 48: bitmask of those ids).
    // Every wait is bounded: past `limit` it records the id, its count, RD_RESP_RECEIVED and reads issued (hdr 44..47,
    // the first time; hdr 49 counts them), then waits for the full read counters, bounded too: flushed means the id's
    // count lies and the core keeps going on full barriers; not flushed is a lost read and the run aborts (N code 21)
    constexpr uint32_t TID_TAB = RING + 1, TID_GC = RING + 3;
    static_assert(TID_GC <= NOC_MAX_TRANSACTION_ID, "one id per ring slot, two table buffers and the gather cache: --ring <= 12");
    noc_async_read_set_trid(0);   // the read tag is a cmd buffer register: a killed run may have left one set
    uint32_t tdirty = 0;
    for (uint32_t i = 1; i <= TID_GC; ++i)
        if (NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(i)) != 0) tdirty |= 1u << i;
    bool tclean = tdirty == 0, tdead = false;
    auto slot_id = [&](uint32_t a) { return 1 + (a - in0) / PS; };
    auto landed = [&](uint32_t id) {
        return tdead || (tclean ? ncrisc_noc_read_with_transaction_id_flushed(noc_index, id) : ncrisc_noc_reads_flushed(noc_index));
    };
    auto tbar = [&](uint32_t id) {
        uint32_t s_ = get_timestamp_32b();
        while (!landed(id)) {
            if (get_timestamp_32b() - s_ > limit) {
                if (hdr[49]++ == 0) {
                    hdr[44] = id;
                    hdr[45] = NOC_STATUS_READ_REG(noc_index, NIU_MST_REQS_OUTSTANDING_ID(id));
                    hdr[46] = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
                    hdr[47] = noc_reads_num_issued[noc_index];
                }
                tclean = false;
                s_ = get_timestamp_32b();
                while (!ncrisc_noc_reads_flushed(noc_index))
                    if (get_timestamp_32b() - s_ > limit) {
                        tdead = true;   // the main loop aborts; every later wait returns at once
                        break;
                    }
                break;
            }
        }
        invalidate_l1_cache();
    };
#endif
#ifdef CFD_TABSTREAM
    // residue tables streamed from DRAM (ttres/tabstream.py): per map [base, halo part mask, uint16 tile bounds...],
    // Tm = T[1 + map] >> 20 (tiles past Tm have no entries); section k tile t is blob[base + H[k * Tm + t],
    // base + H[k * Tm + t + 1]) (pages core * TBP + k), read into cb_ts at its 32-B phase
    constexpr auto a_blob = TensorAccessorArgs<xg + 2>();   // after the gather cache args (cb_gc, NGC)
    const auto t_blob = TensorAccessor(a_blob, get_common_arg_val<uint32_t>(xy0 + P + 1), 4096);
    const uint32_t TBP = get_arg_val<uint32_t>(2);
    const uint32_t ts0 = get_write_ptr(CFD_TABSTREAM);
    // two staging buffers of 4096 + 64 * 32 bytes (run_vm.py CB_TS): GATHER reads tile t + 1 while it applies tile t
    constexpr uint32_t TSB = 4096 + 64 * 32;
    const uint32_t* sp_[2][17];
    uint32_t sn_[2][17];
    const uint32_t** sp = sp_[0];
    uint32_t* sn = sn_[0];
    // issue the reads of tile t of sections [qi, qi + ns * (tm + 1)) of the L1 index into buffer b (pointers
    // sp_[b], sizes sn_[b]); noc_async_read_barrier() (CFD_TRID: or tbar(TID_TAB + b)) before use. t >= tm: no entries
    auto load_tile = [&](uint32_t qi, uint32_t ns, uint32_t t, uint32_t b, uint32_t tm) {
        uint32_t st = ts0 + b * TSB;
#ifdef CFD_TRID
        noc_async_read_set_trid(TID_TAB + b);
#endif
        const uint32_t** sp = sp_[b];
        uint32_t* sn = sn_[b];
        const uint32_t base = T[qi];
#ifdef CFD_WIDE
        const uint32_t* H = T + qi + 2;   // CFD_WIDE: whole-word offsets
#else
        const uint16_t* H = reinterpret_cast<const uint16_t*>(T + qi + 2);   // section k tile t at H[k * tm + t]
#endif
        for (uint32_t k = 0; k < ns; ++k) {
            const uint32_t j = k * tm + t;
            const uint32_t a = t < tm ? base + H[j] : 0, n = t < tm ? H[j + 1] - H[j] : 0;
            uint32_t o = (core * TBP * 1024 + a) * 4;
            st += o & (DA - 1);
            sp[k] = reinterpret_cast<const uint32_t*>(st);
            sn[k] = n;
            for (uint32_t r = n * 4, d = st; r;) {
                const uint32_t w = o & 4095, c = r < 4096 - w ? r : 4096 - w;
                noc_async_read(t_blob.get_noc_addr(o >> 12, 0) + w, d, c);
                o += c, d += c, r -= c;
            }
            st = (st + n * 4 + DA - 1) & ~(DA - 1);
        }
#ifdef CFD_TRID
        noc_async_read_set_trid(0);
#endif
    };
#endif

    // DRAM page of tile t of a kind-1/2 source
    auto dpage = [&](uint32_t d, uint32_t t) -> uint64_t {
        const uint32_t v = d & 0x7ffffff;   // bit 27: a bf16 page (encode.NARROW)
        return (d >> 28) == 1 ? t_home.get_noc_addr(core * HP + v + t, 0) : t_const.get_noc_addr(core * CP + v + t, 0);
    };
    // one value of a source at lane i: L1 now, DRAM through the batch (flush() before using the destination)
    // gpin: gather-cache slots patch() holds pointers to, never evicted (a bit per slot: NGC >= NT may exceed 32)
    constexpr uint32_t NGW = NGC ? (NGC + 31) / 32 : 1;
    uint32_t nq = 0, gpin[NGW] = {};
    volatile uint32_t* qdst[16];
    uint32_t qsrc[16];
    auto flush = [&]() {
        if (nq == 0) return;
#ifdef CFD_TRID
        tbar(TID_GC);
#else
        noc_async_read_barrier();
#endif
        for (uint32_t j = 0; j < nq; ++j) *qdst[j] = *reinterpret_cast<volatile uint32_t*>(qsrc[j]);
        nq = 0;
    };
    auto value = [&](uint32_t d, uint32_t i, volatile uint32_t* dst) {
        if ((d >> 28) == 0) {
            *dst = A[d * 1024 + i];
            return;
        }
        if constexpr (NGC != 0) {
            const uint32_t t = i >> 10;
            if (d == gld && t == glt) {
                *dst = glp[i & 1023];
                return;
            }
            uint32_t j = 0;
            while (j < NGC && !(gcd[j] == d && gct[j] == t)) ++j;
            if (j == NGC) {
                while ((gpin[gcn >> 5] >> (gcn & 31)) & 1) gcn = gcn + 1 == NGC ? 0 : gcn + 1;   // patch()'s live tile pointers
                j = gcn;
                gcn = gcn + 1 == NGC ? 0 : gcn + 1;
#ifdef CFD_TRID
                noc_async_read_set_trid(TID_GC);
                noc_async_read(dpage(d, t), get_write_ptr(cb_gc) + j * 4096, 4096);
                noc_async_read_set_trid(0);
                tbar(TID_GC);
#else
                noc_async_read(dpage(d, t), get_write_ptr(cb_gc) + j * 4096, 4096);
                noc_async_read_barrier();
#endif
                gcd[j] = d;
                gct[j] = t;
                ++gmiss;
            }
            gld = d, glt = t;
            glp = reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_gc) + j * 4096);
            *dst = glp[i & 1023];
            return;
        }
        const uint32_t off = (i & 1023) * 4;
        const uint32_t l1 = scr + DA * nq + (off & (DA - 1));
#ifdef CFD_TRID
        noc_async_read_set_trid(TID_GC);
        noc_async_read(dpage(d, i >> 10) + off, l1, 4);
        noc_async_read_set_trid(0);
#else
        noc_async_read(dpage(d, i >> 10) + off, l1, 4);
#endif
        qdst[nq] = dst;
        qsrc[nq] = l1;
        if (++nq == 16) flush();
    };
    // out[x >> XS] = src[x & XI] for n entries x, four at a time with all loads before the stores, so the in-order
    // core overlaps the L1 load latencies
    auto scatter = [](const uint32_t* src, const uint32_t* xs, uint32_t n, uint32_t* out) {
        uint32_t e = 0;
        for (; e + 4 <= n; e += 4) {
            const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
            const uint32_t v0 = src[x0 & XI], v1 = src[x1 & XI], v2 = src[x2 & XI], v3 = src[x3 & XI];
            out[x0 >> XS] = v0;
            out[x1 >> XS] = v1;
            out[x2 >> XS] = v2;
            out[x3 >> XS] = v3;
        }
        for (; e < n; ++e) {
            const uint32_t x = xs[e];
            out[x >> XS] = src[x & XI];
        }
        asm volatile("" ::: "memory");   // the page is pushed (or the tile used) after this
    };
#ifdef CFD_HRUN
    // hrun: a halo entry with bit 15 set heads a run (its length in the next word) that agrees with its destination
    // mod 16 B, moved by one NoC copy out of the halo slot instead of word by word (encode.py hrun_words)
    // the copies of a stretch of gathers stay in flight together: the barrier is at the next op that is not a gather
    bool hpend = false;
    auto hscatter = [&](const uint32_t* src, const uint32_t* xs, uint32_t n, uint32_t* out) {
        for (uint32_t e = 0; e < n;) {
            const uint32_t x = xs[e];
            if (x & 0x8000) {
                noc_async_read(get_noc_addr(reinterpret_cast<uint32_t>(src + (x & 0x7fff))),
                               reinterpret_cast<uint32_t>(out + (x >> 16)), 4 * xs[e + 1]);
                hpend = true;
                e += 2;
            } else {
                out[x >> 16] = src[x & 0xffff];
                ++e;
            }
        }
        asm volatile("" ::: "memory");
    };
#else
#define hscatter scatter
#endif
    // one part's residue entries (lane << 16 | source lane) into out, a tight loop per source kind: an L1 source
    // directly, a DRAM-homed one a tile at a time through the gather cache (consecutive entries mostly share a tile).
    // Without the cache the values go through value()'s batch; the caller flushes
    auto patch = [&](uint32_t d, const uint32_t* xs, uint32_t n, uint32_t* out) {
        if ((d >> 28) == 0) {
            scatter(A + d * 1024, xs, n, out);
            return;
        }
#ifdef CFD_TSORT
        // entries sorted by source tile (tabstream.py sort): one cache lookup per run; xs[e + 3] still in the run means
        // xs[e..e + 3] all are
        if constexpr (NGC != 0) {
            uint32_t e = 0, dummy;
            while (e < n) {
                const uint32_t tb = xs[e] & XT;
                value(d, xs[e] & XI, reinterpret_cast<volatile uint32_t*>(&dummy));
                const uint32_t* cp = const_cast<const uint32_t*>(glp);
                asm volatile("" ::: "memory");
                for (; e + 4 <= n && (xs[e + 3] & XT) == tb; e += 4) {
                    const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
                    const uint32_t v0 = cp[x0 & 1023], v1 = cp[x1 & 1023], v2 = cp[x2 & 1023], v3 = cp[x3 & 1023];
                    out[x0 >> XS] = v0;
                    out[x1 >> XS] = v1;
                    out[x2 >> XS] = v2;
                    out[x3 >> XS] = v3;
                }
                for (; e < n && (xs[e] & XT) == tb; ++e) out[xs[e] >> XS] = cp[xs[e] & 1023];
            }
            asm volatile("" ::: "memory");
            return;
        }
#endif
        if constexpr (NGC >= NT) {
            // every tile of d fits in the gather cache at once, so keep a pointer per tile (its slot pinned against
            // eviction until the part ends) and stay in the unrolled loop when consecutive entries change tile
            const uint32_t* tp[NT];
            for (uint32_t t = 0; t < NT; ++t) tp[t] = nullptr;
            auto tile = [&](uint32_t i) -> const uint32_t* {
                const uint32_t t = i >> 10;
                if (tp[t] == nullptr) {
                    uint32_t dummy;
                    value(d, i, reinterpret_cast<volatile uint32_t*>(&dummy));
                    tp[t] = const_cast<const uint32_t*>(glp);
                    const uint32_t sl = (reinterpret_cast<uint32_t>(glp) - get_write_ptr(cb_gc)) >> 12;
                    gpin[sl >> 5] |= 1u << (sl & 31);
                    asm volatile("" ::: "memory");
                }
                return tp[t];
            };
            uint32_t e = 0;
            for (; e + 4 <= n; e += 4) {
                const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
                const uint32_t i0 = x0 & XI, i1 = x1 & XI, i2 = x2 & XI, i3 = x3 & XI;
                const uint32_t *p0 = tile(i0), *p1 = tile(i1), *p2 = tile(i2), *p3 = tile(i3);
                const uint32_t v0 = p0[i0 & 1023], v1 = p1[i1 & 1023], v2 = p2[i2 & 1023], v3 = p3[i3 & 1023];
                out[x0 >> XS] = v0;
                out[x1 >> XS] = v1;
                out[x2 >> XS] = v2;
                out[x3 >> XS] = v3;
            }
            for (; e < n; ++e) {
                const uint32_t x = xs[e], i = x & XI;
                out[x >> XS] = tile(i)[i & 1023];
            }
            for (uint32_t w = 0; w < NGW; ++w) gpin[w] = 0;
            asm volatile("" ::: "memory");
        } else if constexpr (NGC != 0) {
            uint32_t ct = 0xffffffffu, dummy;
            const uint32_t* cp = nullptr;
            uint32_t e = 0;
            while (e < n) {
                for (; e + 4 <= n; e += 4) {   // the common case: four entries in the cached tile, unrolled as in scatter
                    const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
                    if (((x0 | x1 | x2 | x3) & XT) != (ct << 10) || (x0 & x1 & x2 & x3 & XT) != (ct << 10)) break;
                    const uint32_t v0 = cp[x0 & 1023], v1 = cp[x1 & 1023], v2 = cp[x2 & 1023], v3 = cp[x3 & 1023];
                    out[x0 >> XS] = v0;
                    out[x1 >> XS] = v1;
                    out[x2 >> XS] = v2;
                    out[x3 >> XS] = v3;
                }
                if (e == n) break;
                const uint32_t x = xs[e++], i = x & XI;   // one entry, through the cache when its tile changes
                if ((i >> 10) != ct) {
                    ct = i >> 10;
                    value(d, i, reinterpret_cast<volatile uint32_t*>(&dummy));
                    cp = const_cast<const uint32_t*>(glp);
                    asm volatile("" ::: "memory");
                }
                out[x >> XS] = cp[i & 1023];
            }
            asm volatile("" ::: "memory");
        } else {
            for (uint32_t e = 0; e < n; ++e) {
                const uint32_t x = xs[e];
                value(d, x & XI, reinterpret_cast<volatile uint32_t*>(out + (x >> XS)));
            }
        }
    };
    // tile t of source d read at lane shift sh into dst (dst + phase: see encode.phase)
    auto window = [&](uint32_t d, uint32_t t, int32_t sh, uint32_t dst) {
        const uint32_t k = d >> 28;
        if ((k == 1 || k == 2) && (d & (1u << 27))) {   // bf16 constant or home page: 2 B a lane, the first 2048 B of its 4096-B page
            const int32_t o = static_cast<int32_t>(t * 2048) + 2 * sh;
            const int32_t pi = o >> 11;
            const uint32_t w = static_cast<uint32_t>(o) & 2047;
            dst += (2 * sh) & (DA - 1);
            if (pi >= 0 && pi < static_cast<int32_t>(NT)) noc_async_read(dpage(d, pi) + w, dst, 2048 - w);
            if (w && pi + 1 >= 0 && pi + 1 < static_cast<int32_t>(NT))
                noc_async_read(dpage(d, pi + 1), dst + 2048 - w, w);
            return;
        }
        const int32_t o = static_cast<int32_t>(t * 4096) + 4 * sh;
        dst += (4 * sh) & (DA - 1);
        if (k == 3) {
            noc_async_read(get_noc_addr(zero), dst, 4096);
        } else if (k == 0) {
            noc_async_read(get_noc_addr(arena + d * 4096 + o), dst, 4096);
        } else {
            const int32_t pi = o >> 12;
            const uint32_t w = static_cast<uint32_t>(o) & 4095;
            if (pi >= 0 && pi < static_cast<int32_t>(NT)) noc_async_read(dpage(d, pi) + w, dst, 4096 - w);
            if (w && pi + 1 >= 0 && pi + 1 < static_cast<int32_t>(NT))
                noc_async_read(dpage(d, pi + 1), dst + 4096 - w, w);
        }
    };

    auto fail = [&](uint32_t why) {
        if (hdr[32] == 0) {
            hdr[32] = why;
            hdr[33] = pc;
        }
        if (S[W_ABORT] == 0) {
            S[W_ABORT] = 1;
            for (uint32_t j = 0; j < P; ++j) {
                if (j == core) continue;
                const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + j);
                noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, sync + 4 * W_ABORT), 1);
            }
        }
    };
#define WAIT_UNTIL(cond, why)                                                    \
    ({                                                                           \
        bool ok_ = true;                                                         \
        const uint32_t s_ = get_timestamp_32b();                                 \
        while (!(cond)) {                                                        \
            invalidate_l1_cache();                                               \
            if (S[W_ABORT] || get_timestamp_32b() - s_ > limit) {                \
                fail(why);                                                       \
                ok_ = false;                                                     \
                break;                                                           \
            }                                                                    \
        }                                                                        \
        ok_;                                                                     \
    })

    // BRISC zeroes the sync region and loads the tables before it releases `go` (1 = run, 2 = its barrier failed)
    bool run = true;
    {
        const uint32_t s0 = get_timestamp_32b();
        while (*go == 0 && get_timestamp_32b() - s0 < 2 * limit) {
            invalidate_l1_cache();
        }
        if (*go != 1) {
            if (hdr[32] == 0) hdr[32] = 11;
            run = false;
        }
    }
#ifdef CFD_TRID
    hdr[48] = tdirty;   // BRISC zeroed the header before `go`
#endif
    // per halo slot its plan's senders (ttres/encode.py: plan 0 at tab_halo, a directory below tab_halo if hdir)
    // and its receive buffer's word offset in the halo CB (slots sized per plan; one plan: s * hmax)
    uint32_t hcnt[NH], nsend[NH], hoff[NH];
    const uint32_t* senders[NH];
    for (uint32_t s = 0; s < NH; ++s) {
        hcnt[s] = 0;
        const uint32_t dw = hdir ? T[tab_halo - 1 - s] : 0, q = dw & 0xff;
        hoff[s] = hdir ? dw >> 8 : s * hmax;
        uint32_t p = tab_halo + (q ? T[tab_halo - NH - q] : 0);
        const uint32_t nrecv = T[p++];
        for (uint32_t r = 0; r < nrecv; ++r) p += 3 + T[p + 2];   // one word per lane (encode.py runs())
        nsend[s] = T[p];
        senders[s] = T + p + 1;
    }
    // a part's halo buffer (NONE16: the part reads no halo, the pointer is never read)
    auto hbuf = [&](uint32_t s) -> const uint32_t* { return H + (s < NH ? hoff[s] : 0); };
    uint32_t nops = 0, ngather = 0, nvals = 0, npages = 0;
#ifdef CFD_PROF
    uint64_t cyc[24];   // stored in 16-cycle units (>> 4), as BRISC
    uint32_t cnt[24];   // census: cycles and counts per op, to hdr 96 + op / 112 + op; parts 16.. to 288 / 296
    for (uint32_t i = 0; i < 24; ++i) cyc[i] = cnt[i] = 0;
    uint32_t tprev = get_timestamp_32b();
    // inside STREAM (hdr 96 + 10..12): waiting for a free cb_in slot, read barriers, the patch; parts 16..17: the
    // patch's barrier and residue entries; inside GATHER (parts 18..19): waiting for the halo, the local and halo scatters
#define PSUB(i, x)                                     \
    {                                                  \
        const uint32_t t0_ = get_timestamp_32b();      \
        x;                                             \
        cyc[i] += get_timestamp_32b() - t0_;           \
        ++cnt[i];                                      \
    }
#else
#define PSUB(i, x) x
#endif
#ifdef CFD_TRACE
    volatile tt_l1_ptr uint32_t* tr_ = hdr + 1024 + 4 * (2 * CFD_TRACE + 2);
    uint32_t trn_ = 0;
    const uint32_t trt0_ = get_timestamp_32b();
    tr_[0] = 0;
    tr_[1] = trt0_;
    uint32_t trs_ = 0;
#endif
    while (run) {
        if constexpr (NWIN != 0) {
            if (pc < wb || pc + NMAX > wb + NWIN) {
                wb = pc & ~255u;
                noc_async_read(t_ncode.get_noc_addr(0, wb * 4), nwl1, NWIN * 4);
                noc_async_read_barrier();
                N = reinterpret_cast<const uint32_t*>(nwl1 - wb * 4);
                ++nrefill;
            }
        }
        asm volatile("" ::: "memory");   // L1 written by the other agents between ops
#ifdef CFD_TRID
        if (tdead) {
            fail(21);
            break;
        }
#endif
        const uint32_t w = N[pc++];
        const uint32_t op = w & 0xff;
#ifdef CFD_HRUN
        if (hpend && op != N_GATHER) {
            noc_async_read_barrier();
            hpend = false;
        }
#endif
        ++nops;
#ifdef CFD_TRACE
        {
            const uint32_t t_ = get_timestamp_32b();
            TR_REC(pc - 1, op, t_);
        }
#endif
        switch (op) {
            case N_END: run = false; break;
            case N_SIGC:
                if (!WAIT_UNTIL(cb_pages_reservable_at_back(cb_nc, 1), 12)) { run = false; break; }
                cb_reserve_back(cb_nc, 1);
                cb_push_back(cb_nc, 1);
                break;
            case N_WAITC:
                if (!WAIT_UNTIL(cb_pages_available_at_front(cb_cn, 1), 13)) { run = false; break; }
                cb_wait_front(cb_cn, 1);
                cb_pop_front(cb_cn, 1);
                break;
            case N_SIGB:
                if (!WAIT_UNTIL(cb_pages_reservable_at_back(cb_nb, 1), 14)) { run = false; break; }
                cb_reserve_back(cb_nb, 1);
                cb_push_back(cb_nb, 1);
                break;
            case N_WAITB:
                if (!WAIT_UNTIL(cb_pages_available_at_front(cb_bn, 1), 15)) { run = false; break; }
                cb_wait_front(cb_bn, 1);
                cb_pop_front(cb_bn, 1);
                break;
            case N_GATHER: {
                ginv();
                const uint32_t nparts = (w >> 8) & 0xf;
#ifdef CFD_TABSTREAM
                {
                    uint32_t* dst = A + N[pc] * 1024;
                    const uint32_t q0 = T[1 + N[pc + 1]] & 0xfffff, tm = T[1 + N[pc + 1]] >> 20;
                    const uint32_t* parts = N + pc + 2;
                    pc += 2 + 2 * nparts;
                    ++ngather;
                    // tile 0's table reads go out before the halo wait, tile t + 1's before tile t is applied
                    load_tile(q0, 2 * nparts + 1, 0, 0, tm);
                    for (uint32_t k = 0; k < nparts && run; ++k)
                        if (T[q0 + 1] >> k & 1) {
                            const uint32_t s = parts[2 * k + 1];
                            if (!WAIT_UNTIL(S[4 * s] >= hcnt[s] * nsend[s], 16)) { run = false; break; }
                        }
                    if (!run) { noc_async_read_barrier(); break; }
                    asm volatile("" ::: "memory");
#ifdef CFD_GTILE
                    // gtile (encode.py): compute waits once per pass of the consumer loop, gt passes; this gather posts
                    // a token after each of its first gt - 1 tiles (the N_SIGC that follows is the last), into a free
                    // slot only, the ones it could not post at the end. Built only with --gtile: 704 B of NCRISC
                    // pushed the BH k4 Small program past the kernel config buffer
                    const uint32_t gt = (w >> 16) & 0xff;
                    uint32_t gowe = gt ? gt - 1 : 0;
#endif
                    for (uint32_t t = 0; t < tm; ++t) {
                        noc_async_read_barrier();   // tile t's tables (and the last flush of tile t - 1)
                        sp = sp_[t & 1], sn = sn_[t & 1];
                        if (t + 1 < tm) load_tile(q0, 2 * nparts + 1, t + 1, (t + 1) & 1, tm);
                        uint32_t* const dt = dst + WT * t;   // CFD_WIDE: entries carry the lane within tile t
#ifdef CFD_BGTS
                        // BRISC scatters the upper (8 - CFD_BGTS) / 8 of every L1-source local and every halo section
                        // of this tile (vm_brisc.cpp bsvc, from inside its waits); this core does the rest and the
                        // DRAM-source and zero entries, then waits for its DONE. Disjoint lanes: the same values
                        uint32_t nj = 0, nk[16];
                        for (uint32_t k = 0; k < 2 * nparts; ++k) {
                            const uint32_t n = sn[k];
                            const uint32_t* src = k & 1 ? hbuf(parts[k]) : A + parts[k] * 1024;
                            nk[k] = n;
                            if (n >= 32 && nj < 15 && ((k & 1) || (parts[k] >> 28) == 0)) {
                                const uint32_t h = n * CFD_BGTS / 8;
                                S[M_JOB + 4 * nj] = reinterpret_cast<uint32_t>(src);
                                S[M_JOB + 4 * nj + 1] = reinterpret_cast<uint32_t>(sp[k] + h);
                                S[M_JOB + 4 * nj + 2] = n - h;
                                S[M_JOB + 4 * nj + 3] = reinterpret_cast<uint32_t>(dt);
                                nk[k] = h;
                                ++nj;
                            }
                        }
                        if (nj) {
                            S[M_N] = nj;
                            asm volatile("" ::: "memory");
                            S[M_SEQ] = ++bseq;
                        }
#elif defined(CFD_TGTS)
                        uint32_t nk[16];
                        for (uint32_t k = 0; k < 2 * nparts; ++k) nk[k] = sn[k];
#else
                        const uint32_t* nk = sn;
#endif
#ifdef CFD_TGTS
                        // compute's UNPACK copies the top CFD_TGTS / 8 of each section (of what BRISC left this core),
                        // from inside its waits; disjoint lanes again
                        uint32_t nt = 0;
                        for (uint32_t k = 0; k < 2 * nparts; ++k) {
                            const uint32_t n = sn[k], h = nk[k], d = n * CFD_TGTS / 8;
                            if (n >= 32 && nt < 12 && d && d <= h && ((k & 1) || (parts[k] >> 28) == 0)) {
                                S[T_JOB + 4 * nt] = reinterpret_cast<uint32_t>(k & 1 ? hbuf(parts[k]) : A + parts[k] * 1024);
                                S[T_JOB + 4 * nt + 1] = reinterpret_cast<uint32_t>(sp[k] + h - d);
                                S[T_JOB + 4 * nt + 2] = d;
                                S[T_JOB + 4 * nt + 3] = reinterpret_cast<uint32_t>(dst);
                                nk[k] = h - d;
                                ++nt;
                            }
                        }
                        if (nt) {
                            S[T_N] = nt;
                            asm volatile("" ::: "memory");
                            S[T_SEQ] = ++tseq;
                        }
#endif
                        for (uint32_t k = 0; k < nparts; ++k) {
                            const uint32_t sd = parts[2 * k];
                            patch(sd, sp[2 * k], nk[2 * k], dt);
                            flush();
                            const uint32_t* h = hbuf(parts[2 * k + 1]);
                            scatter(h, sp[2 * k + 1], nk[2 * k + 1], dt);
                        }
                        for (uint32_t e = 0; e < sn[2 * nparts]; ++e) dt[sp[2 * nparts][e]] = 0;
#ifdef CFD_BGTS
                        if (nj && !WAIT_UNTIL(S[M_DONE] == bseq, 21)) { run = false; break; }
#endif
#ifdef CFD_TGTS
                        if (nt && !WAIT_UNTIL(S[T_DONE] == tseq, 22)) { run = false; break; }
#endif
#ifdef CFD_GTILE
                        if (gowe + t + 1 >= gt && gowe && cb_pages_reservable_at_back(cb_nc, 1)) {
                            cb_reserve_back(cb_nc, 1);
                            cb_push_back(cb_nc, 1);
                            --gowe;
                        }
#endif
                    }
#ifdef CFD_GTILE
                    for (; gowe && run; --gowe) {
                        if (!WAIT_UNTIL(cb_pages_reservable_at_back(cb_nc, 1), 12)) { run = false; break; }
                        cb_reserve_back(cb_nc, 1);
                        cb_push_back(cb_nc, 1);
                    }
#endif
                    nvals += 1;
                    break;
                }
#endif
                uint32_t* dst = A + N[pc] * 1024;
                uint32_t q = T[1 + N[pc + 1]];
                const uint32_t* parts = N + pc + 2;   // per part: source descriptor, halo slot
                pc += 2 + 2 * nparts;
                ++ngather;
                const uint32_t mode = (w >> 12) & 0xf;   // gsplit: 1 local entries + zeros, 2 halo entries only
                for (uint32_t k = 0; k < nparts && run; ++k) {
                    const uint32_t nl = T[q], nh = T[q + 1];
                    q += 2;
                    const uint32_t sd = parts[2 * k];
                    if (mode == 2) {
                    } else if ((sd >> 28) == 0) {
                        PSUB(19, scatter(A + sd * 1024, T + q, nl, dst));
                    } else {
                        for (uint32_t e = 0; e < nl; ++e) {
                            const uint32_t x = T[q + e];
                            value(sd, x & XI, reinterpret_cast<volatile uint32_t*>(dst + (x >> XS)));
                        }
                        flush();
                    }
                    q += nl;
                    if (nh && mode == 1) {
                        q += nh;
                    } else if (nh) {
                        const uint32_t s = parts[2 * k + 1];
                        bool okw;
                        PSUB(18, okw = WAIT_UNTIL(S[4 * s] >= hcnt[s] * nsend[s], 16));
                        if (!okw) { run = false; break; }
                        asm volatile("" ::: "memory");
                        const uint32_t* h = hbuf(s);
                        PSUB(19, hscatter(h, T + q, nh, dst));
                        q += nh;
                    }
                    nvals += nl + nh;
                }
                if (!run) break;
                const uint32_t nz = T[q++];
                if (mode != 2)
                    for (uint32_t e = 0; e < nz; ++e) dst[T[q + e]] = 0;
                break;
            }
            case N_STREAM: {
                ginv();
                // entries: src, shift << SS | (map + 1) << 4 | nparts (map + 1 = 0: no patch), then (part src, halo
                // slot) per part
                const uint32_t ne = N[pc] & 0xffff, tn = N[pc] >> 16 ? N[pc] >> 16 : NT;   // tn: pages per entry
                const uint32_t e0 = pc + 1;
                // pe: the last patched entry's word offset. CFD_TABSTREAM without CFD_TRID patches any number of
                // entries per stream (--pseg 3), each from its own tables; the other builds keep one (encode.py)
                uint32_t p = e0, pe = 0;
                for (uint32_t e = 0; e < ne; ++e) {
                    if ((N[p + 1] >> 4) & SM) pe = p;
                    p += 2 + 2 * (N[p + 1] & 0xf);
                }
                pc = p;
                // the patched entry: table sections and halo readiness, once per stream
                uint32_t np_ = 0, qz = 0, nz = 0;
                uint32_t ql[8], nl[8], qh[8], nh[8], cl[8], ch[8], cz = 0;   // up to 8 gather parts (encode.py)
#ifdef CFD_TABSTREAM
                uint32_t qi = 0, tm = 0;
                for (uint32_t e = 0, pp = e0; e < ne && run; ++e, pp += 2 + 2 * (N[pp + 1] & 0xf)) {
                    if (!((N[pp + 1] >> 4) & SM)) continue;
                    np_ = N[pp + 1] & 0xf;
                    const uint32_t q = T[(N[pp + 1] >> 4) & SM] & 0xfffff;   // T[1 + map]
                    tm = T[(N[pp + 1] >> 4) & SM] >> 20;
                    qi = q;
                    for (uint32_t k = 0; k < np_; ++k)
                        if (T[q + 1] >> k & 1) {
                            const uint32_t s = N[pp + 3 + 2 * k];
                            if (!WAIT_UNTIL(S[4 * s] >= hcnt[s] * nsend[s], 16)) { run = false; break; }
                        }
                }
                (void)pe, (void)qz, (void)nz, (void)ql, (void)nl, (void)qh, (void)nh, (void)cl, (void)ch, (void)cz;
#ifdef CFD_TRID
                // tile 0's tables go out now, tile t + 1's once tile t is patched (two staging buffers, as in GATHER),
                // so the patched page rarely waits for its tables
                if (pe && run) load_tile(qi, 2 * np_ + 1, 0, 0, tm);
#endif
#else
                if (pe) {
                    np_ = N[pe + 1] & 0xf;
                    uint32_t q = T[(N[pe + 1] >> 4) & SM];   // T[1 + map]
                    for (uint32_t k = 0; k < np_; ++k) {
                        nl[k] = T[q];
                        nh[k] = T[q + 1];
                        ql[k] = q + 2;
                        qh[k] = q + 2 + nl[k];
                        q = qh[k] + nh[k];
                        cl[k] = ch[k] = 0;
                        if (nh[k]) {
                            const uint32_t s = N[pe + 3 + 2 * k];
                            if (!WAIT_UNTIL(S[4 * s] >= hcnt[s] * nsend[s], 16)) { run = false; break; }
                        }
                    }
                    nz = T[q];
                    qz = q + 1;
                }
#endif
                if (!run) break;
                uint32_t pend = 0;
#ifdef CFD_TRID
                // up to RING pages in flight, slot k's reads under id 1 + k: after every issue the slots whose reads
                // landed are pushed in order (retire), a full ring waits for its slots one at a time oldest first, the
                // patched page waits for its own id. A slot is addressed modulo the ring, so the fifo's end is no stop.
                // Pushing does not free a reservation (pages pushed, not popped, still count), so a full ring is drained
                auto retire = [&](bool wait) {
                    while (pend) {
                        const uint32_t id = slot_id(get_write_ptr(cb_in));
                        if (!landed(id)) {
                            if (!wait) return;
                            PSUB(11, tbar(id));
                        }
                        cb_push_back(cb_in, 1);
                        ++npages;
                        --pend;
                    }
                };
#else
                // up to RING pages in flight: reads are issued into reserved slots and pushed together after one
                // barrier, when the ring is full, when compute would otherwise wait, at the fifo's end (a push may land on
                // it but not cross it), at the patched entry and at the end
#if defined(CFD_DEFER) && defined(CFD_TABSTREAM)
                // CFD_DEFER: a patched page whose reads landed is held (oldest reserved slot, not pushed) and patched
                // after the next entry's reads are issued, so its residue work overlaps their DRAM latency
                volatile uint32_t* hpl = nullptr;
                uint32_t hp = 0, hnp = 0;
                auto unhold = [&]() {
                    if (hpl == nullptr) return;
#ifdef CFD_PROF
                    const uint32_t tp_ = get_timestamp_32b();
#endif
                    for (uint32_t k = 0; k < hnp; ++k) {
                        patch(N[hp + 2 + 2 * k], sp[2 * k], sn[2 * k], const_cast<uint32_t*>(hpl));
                        scatter(H + N[hp + 3 + 2 * k] * hmax, sp[2 * k + 1], sn[2 * k + 1], const_cast<uint32_t*>(hpl));
                    }
                    for (uint32_t e = 0; e < sn[2 * hnp]; ++e) hpl[sp[2 * hnp][e]] = 0;
                    flush();
                    nvals += 1;
                    cb_push_back(cb_in, 1);
                    ++npages;
                    --pend;
                    hpl = nullptr;
#ifdef CFD_PROF
                    cyc[12] += get_timestamp_32b() - tp_;
                    ++cnt[12];
#endif
                };
#endif
                auto drain = [&]() {
#if defined(CFD_DEFER) && defined(CFD_TABSTREAM)
                    unhold();
#endif
                    if (pend == 0) return;
                    PSUB(11, noc_async_read_barrier());
                    cb_push_back(cb_in, pend);
                    npages += pend;
                    pend = 0;
                };
#endif
                for (uint32_t t = 0; t < tn && run; ++t) {
                    const uint32_t lo = t * 1024, hi = lo + 1024;
                    p = e0;
                    for (uint32_t e = 0; e < ne && run; ++e) {
                        const uint32_t d = N[p];
                        const int32_t sh = static_cast<int32_t>(N[p + 1]) >> SS;
#ifdef CFD_TRID
                        if (pend && !cb_pages_reservable_at_back(cb_in, pend + 1)) retire(true);
#else
                        if (pend && !cb_pages_reservable_at_back(cb_in, pend + 1)) drain();
#endif
                        bool ok = true;
                        PSUB(10, ok = WAIT_UNTIL(cb_pages_reservable_at_back(cb_in, pend + 1), 17));
                        if (!ok) { run = false; break; }
                        cb_reserve_back(cb_in, pend + 1);
#ifdef CFD_TRID
                        uint32_t pg = get_write_ptr(cb_in) + pend * PS;
                        if (pg >= in1) pg -= RING * PS;
                        const uint32_t id = slot_id(pg);
                        noc_async_read_set_trid(id);
                        window(d, t, sh, pg);
                        noc_async_read_set_trid(0);
#else
                        const uint32_t pg = get_write_ptr(cb_in) + pend * PS;
                        window(d, t, sh, pg);
#endif
                        ++pend;
#if defined(CFD_DEFER) && defined(CFD_TABSTREAM)
                        unhold();   // this entry's reads are in flight now
#endif
                        const bool pat = (N[p + 1] >> 4) & SM;
                        if (pat) {
#ifdef CFD_PROF
                            const uint32_t tp_ = get_timestamp_32b();
#endif
#ifdef CFD_TRID
                            // the page's own reads and tile t's tables (prefetched): earlier slots stay in flight
#ifdef CFD_TABSTREAM
                            PSUB(16, tbar(id); tbar(TID_TAB + (t & 1)));
                            sp = sp_[t & 1], sn = sn_[t & 1];
#else
                            PSUB(16, tbar(id));
#endif
#else
#ifdef CFD_TABSTREAM
                            np_ = N[p + 1] & 0xf;
                            qi = T[(N[p + 1] >> 4) & SM] & 0xfffff;
                            load_tile(qi, 2 * np_ + 1, t, 0, T[(N[p + 1] >> 4) & SM] >> 20);
                            sp = sp_[0], sn = sn_[0];
#endif
                            PSUB(16, noc_async_read_barrier());
#endif
                            volatile uint32_t* pl =
                                reinterpret_cast<volatile uint32_t*>(pg + ((4 * sh) & (DA - 1))) - (WT ? 0 : lo);   // CFD_WIDE: lanes within the tile
#if defined(CFD_DEFER) && defined(CFD_TABSTREAM)
                            if (pg + PS != in1) {
#ifdef CFD_PROF
                                for (uint32_t k = 0; k <= 2 * np_; ++k) cyc[17] += sn[k];
                                cyc[12] += get_timestamp_32b() - tp_;
#endif
                                // the earlier slots landed with this page: push them, keep this one reserved
                                if (pend > 1) {
                                    cb_push_back(cb_in, pend - 1);
                                    npages += pend - 1;
                                    pend = 1;
                                }
                                hpl = pl, hp = p, hnp = np_;
                                p += 2 + 2 * (N[p + 1] & 0xf);
                                continue;
                            }
#endif
#ifdef CFD_TABSTREAM
#ifdef CFD_PROF
                            for (uint32_t k = 0; k <= 2 * np_; ++k) cyc[17] += sn[k];
#endif
#ifdef CFD_BPATCH
                            // BRISC scatters the upper (8 - CFD_BPATCH) / 8 of every L1-source local section (vm_brisc.cpp
                            // bsvc); this core does the rest, then waits for its DONE before the page is pushed
                            uint32_t nj = 0, nkeep[8];
                            for (uint32_t k = 0; k < np_; ++k) {
                                const uint32_t sd = N[p + 2 + 2 * k], n = sn[2 * k];
                                nkeep[k] = n;
                                if ((sd >> 28) == 0 && n >= 32) {
                                    const uint32_t h = n * CFD_BPATCH / 8;
                                    S[M_JOB + 4 * nj] = reinterpret_cast<uint32_t>(A + sd * 1024);
                                    S[M_JOB + 4 * nj + 1] = reinterpret_cast<uint32_t>(sp[2 * k] + h);
                                    S[M_JOB + 4 * nj + 2] = n - h;
                                    S[M_JOB + 4 * nj + 3] = reinterpret_cast<uint32_t>(pl);
                                    nkeep[k] = h;
                                    ++nj;
                                }
                            }
                            if (nj) {
                                S[M_N] = nj;
                                asm volatile("" ::: "memory");
                                S[M_SEQ] = ++bseq;
                            }
#endif
                            for (uint32_t k = 0; k < np_; ++k) {
                                const uint32_t sd = N[p + 2 + 2 * k];
#ifdef CFD_PROF
                                // parts 20..23: local L1-source, local DRAM-source, halo, zero entries (cycles, entries)
                                uint32_t t0_ = get_timestamp_32b();
                                const uint32_t kk_ = (sd >> 28) == 0 ? 20 : 21;
#endif
#ifdef CFD_BPATCH
                                patch(sd, sp[2 * k], nkeep[k], const_cast<uint32_t*>(pl));
#else
                                patch(sd, sp[2 * k], sn[2 * k], const_cast<uint32_t*>(pl));
#endif
#ifdef CFD_PROF
                                uint32_t t1_ = get_timestamp_32b();
                                cyc[kk_] += t1_ - t0_, cnt[kk_] += sn[2 * k];
#endif
                                const uint32_t* h = hbuf(N[p + 3 + 2 * k]);
                                scatter(h, sp[2 * k + 1], sn[2 * k + 1], const_cast<uint32_t*>(pl));
#ifdef CFD_PROF
                                cyc[22] += get_timestamp_32b() - t1_, cnt[22] += sn[2 * k + 1];
#endif
                            }
#ifdef CFD_PROF
                            const uint32_t tz_ = get_timestamp_32b();
#endif
                            for (uint32_t e = 0; e < sn[2 * np_]; ++e) pl[sp[2 * np_][e]] = 0;
#ifdef CFD_PROF
                            cyc[23] += get_timestamp_32b() - tz_, cnt[23] += sn[2 * np_];
#endif
#ifdef CFD_BPATCH
                            if (nj) {
                                bool okb = true;
                                PSUB(18, okb = WAIT_UNTIL(S[M_DONE] == bseq, 21));   // part 18 (GATHER halo wait, 0 in STREAM runs)
                                if (!okb) { run = false; break; }
                            }
#endif
#ifdef CFD_TRID
                            if (t + 1 < tn) load_tile(qi, 2 * np_ + 1, t + 1, (t + 1) & 1, tm);
#endif
#else
                            for (uint32_t k = 0; k < np_; ++k) {
                                const uint32_t sd = N[p + 2 + 2 * k];
                                for (; cl[k] < nl[k]; ++cl[k]) {
                                    const uint32_t x = T[ql[k] + cl[k]];
                                    if ((x >> 16) >= hi) break;
                                    value(sd, x & 0xffff, pl + (x >> 16));
                                }
                                const uint32_t* h = hbuf(N[p + 3 + 2 * k]);
                                for (; ch[k] < nh[k]; ++ch[k]) {
                                    const uint32_t x = T[qh[k] + ch[k]];
                                    if ((x >> 16) >= hi) break;
                                    pl[x >> 16] = h[x & 0xffff];
                                }
                            }
                            for (; cz < nz && T[qz + cz] < hi; ++cz) pl[T[qz + cz]] = 0;
#endif
                            flush();
                            nvals += 1;
#ifdef CFD_PROF
                            cyc[12] += get_timestamp_32b() - tp_;
                            ++cnt[12];
#endif
                        }
#ifdef CFD_TRID
                        retire(false);
#else
                        if (pat || pg + PS == in1) drain();
#endif
                        p += 2 + 2 * (N[p + 1] & 0xf);
                    }
                }
#ifdef CFD_TRID
                if (run) retire(true);
#else
                if (run) drain();
#endif
                break;
            }
            case N_HALO: {
                const uint32_t s = N[pc], tile = N[pc + 1];
                pc += 2;
                ++hcnt[s];
                for (uint32_t j = 0; j < nsend[s]; ++j) {
                    const uint32_t sj = senders[s][j];
                    const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + (sj & 0x7fff));
#ifdef CFD_MESH
                    // a sender on another chip: credit the gateway that relays it (cons[s], vm_brisc.cpp)
                    const uint32_t at = sj & 0x8000 ? CFD_CONS : 512;
#else
                    constexpr uint32_t at = 512;
#endif
                    noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, sync + at + 16 * s), 1);
                }
#ifdef CFD_HSPLIT
                // hsplit: the receivers from hsplit_at() on are ours (L1 sources only; BRISC keeps DRAM-sourced halos):
                // copy into the stage after BRISC's part, write, then signal those receivers
                const uint32_t nrecv = T[tab_halo];   // plan 0 only: hsplit builds have one plan (Machine asserts it)
                if (nrecv && (tile >> 28) == 0) {
                    if (!WAIT_UNTIL(S[(512 + 16 * s) / 4] >= hcnt[s] * nrecv, 30)) { run = false; break; }
                    const uint32_t* x = A + tile * 1024;
                    const uint32_t stage = get_write_ptr(CFD_HSPLIT);
                    uint32_t* ST = reinterpret_cast<uint32_t*>(stage);
                    const uint32_t h = hsplit_at(T + tab_halo);
                    // stage words 2048.. (BRISC's halo part starts at 0, its reductions use words < 3 P and >= 4088)
                    uint32_t p = tab_halo + 1, so = 2048;
                    bool fits = true;
                    for (uint32_t r = 0; r < nrecv; ++r) {
                        const uint32_t rc = T[p], off = T[p + 1], n = T[p + 2];
                        p += 3;
                        const uint32_t dw = hoff[s] + off;
                        so = ((so + 3) & ~3u) + (dw & 3);
                        if (so + n > 4088) fits = false;
                        if (r >= h && fits) {
                            const uint32_t* ix = T + p;
                            uint32_t* sd = ST + so;
                            uint32_t q = 0;
                            for (; q + 4 <= n; q += 4) {
                                const uint32_t v0 = x[ix[q] & HI], v1 = x[ix[q + 1] & HI], v2 = x[ix[q + 2] & HI], v3 = x[ix[q + 3] & HI];
                                sd[q] = v0;
                                sd[q + 1] = v1;
                                sd[q + 2] = v2;
                                sd[q + 3] = v3;
                            }
                            for (; q < n; ++q) sd[q] = x[ix[q] & HI];
                            asm volatile("" ::: "memory");
                            const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + rc);
                            noc_async_write(stage + so * 4, get_noc_addr(xy & 0xffff, xy >> 16, get_write_ptr(cb_halo) + dw * 4), n * 4);
                        }
                        p += n;
                        so += n;
                    }
                    if (!fits) {   // the halo outgrew the stage's second half
                        fail(32);
                        run = false;
                        break;
                    }
                    if (!WAIT_UNTIL(ncrisc_noc_nonposted_writes_flushed(noc_index), 31)) { run = false; break; }
                    p = tab_halo + 1;
                    for (uint32_t r = 0; r < nrecv; ++r) {
                        const uint32_t rc = T[p], n = T[p + 2];
                        p += 3 + n;
                        if (r < h) continue;
                        const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + rc);
                        noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, sync + 16 * s), 1);
                    }
                }
#else
                (void)tile;
#endif
                break;
            }
            case N_CODE: {
                const uint32_t chi = N[pc], want = N[pc + 1];
                pc += 2;
                while (ccur < chi) {
                    if (!WAIT_UNTIL(cpushed - S[W_CREL] < CR && cb_pages_reservable_at_back(cb_ccode, 1), 20)) {
                        run = false;
                        break;
                    }
                    cb_reserve_back(cb_ccode, 1);
                    noc_async_read(t_code.get_noc_addr(0, ccur * 1024), get_write_ptr(cb_ccode), 1024);
                    noc_async_read_barrier();
                    cb_push_back(cb_ccode, 1);
                    ++ccur;
                    ++cpushed;
                }
                while (run && ccur < want && cpushed - S[W_CREL] < CR && cb_pages_reservable_at_back(cb_ccode, 1)) {
                    cb_reserve_back(cb_ccode, 1);   // look-ahead (encode look): only into free slots, never a wait
                    noc_async_read(t_code.get_noc_addr(0, ccur * 1024), get_write_ptr(cb_ccode), 1024);
                    noc_async_read_barrier();
                    cb_push_back(cb_ccode, 1);
                    ++ccur;
                    ++cpushed;
                }
                break;
            }
            case N_CODEAT: ccur = N[pc++]; break;
            case N_JMP: pc = N[pc]; break;
            case N_BRT: {
                const uint32_t v = reinterpret_cast<volatile uint32_t*>(A)[N[pc] * 1024], target = N[pc + 1];
                pc += 2;
                if (((v & 0x7fffffffu) != 0) != ((w >> 8) & 1)) pc = target;   // bit 8: taken on a zero flag
                break;
            }
            default:
                hdr[21] = 0xBAD00000u | op;
                fail(19);
                run = false;
                break;
        }
#ifdef CFD_PROF
        {
            const uint32_t t = get_timestamp_32b();
            cyc[op & 15] += t - tprev;
            ++cnt[op & 15];
            tprev = t;
        }
#endif
    }
#ifdef CFD_PROF
    for (uint32_t i = 0; i < 16; ++i) {
        hdr[96 + i] = static_cast<uint32_t>(cyc[i] >> 4);
        hdr[112 + i] = cnt[i];
    }
    for (uint32_t i = 0; i < 8; ++i) {
        hdr[288 + i] = static_cast<uint32_t>(cyc[16 + i] >> 4);
        hdr[296 + i] = cnt[16 + i];
    }
#endif
    // drain until BRISC reports compute done: feed compute's and BRISC's waits, swallow their signals
    if (S[W_ABORT] || hdr[32]) {
        const uint32_t s0 = get_timestamp_32b();
        while (S[W_CDONE] == 0 && get_timestamp_32b() - s0 < 4 * limit) {
            invalidate_l1_cache();
            if (cb_pages_available_at_front(cb_cn, 1)) { cb_wait_front(cb_cn, 1); cb_pop_front(cb_cn, 1); }
            if (cb_pages_reservable_at_back(cb_nc, 1)) { cb_reserve_back(cb_nc, 1); cb_push_back(cb_nc, 1); }
            if (cb_pages_available_at_front(cb_bn, 1)) { cb_wait_front(cb_bn, 1); cb_pop_front(cb_bn, 1); }
            if (cb_pages_reservable_at_back(cb_nb, 1)) { cb_reserve_back(cb_nb, 1); cb_push_back(cb_nb, 1); }
            if (cb_pages_reservable_at_back(cb_in, 1)) { cb_reserve_back(cb_in, 1); cb_push_back(cb_in, 1); }
            if (CR && cb_pages_reservable_at_back(cb_ccode, 1)) {   // a page of C_END for a compute waiting on code
                cb_reserve_back(cb_ccode, 1);
                for (uint32_t i = 0; i < 256; ++i) reinterpret_cast<volatile uint32_t*>(get_write_ptr(cb_ccode))[i] = 0;
                cb_push_back(cb_ccode, 1);
            }
        }
    }
    {
        const uint32_t s_ = get_timestamp_32b();
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index) && get_timestamp_32b() - s_ < limit) {
        }
    }
    hdr[22] = nops;
    hdr[23] = ngather;
    hdr[24] = nvals;
    hdr[25] = pc;
    hdr[26] = npages;
    hdr[27] = nrefill;
    hdr[28] = gmiss;
    S[W_NDONE] = 1;
}
