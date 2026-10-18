// Resident VM, BRISC side. Loads this core's arena, the three code streams and the tables, joins the start barrier,
// then walks the B stream: signals and waits (token CBs), halo pushes with receiver credits, the all-to-all
// reduction of per-core partials, DRAM constant fetches, the output history (core 0) and branches. At END it waits
// for compute and NCRISC and writes [header | arena] back.
//
// Nothing waits forever: every wait gives up after `limit` cycles, records why and where, and raises ABORT on every
// core. An aborting BRISC/NCRISC then drains (supplies every token compute waits for, swallows every token it
// sends) until compute, which checks ABORT at each branch, reaches END. So a bad program still ends and the host
// reads the diagnosis instead of a wedged chip.
//
// Sync region (cb_sync, zeroed before the barrier, same address on every core), byte offsets:
//   16 * s          data_sem[s]   receivers count halo pushes into slot s (NCRISC waits on it)
//   512 + 16 * s    credit[s]     senders count receiver credits for slot s (BRISC waits on it), s < NH = 32
//   4000 NDONE (NCRISC finished)   4016 ABORT (any core gave up)   4032 CDONE (compute finished, for NCRISC's drain)
//   4048 / 4064 / 4080 compute's paged-code words CREL / CUNP / CMATH (vm_compute.cpp)
//   GB + 16 * (par * P + j)       core 0's reduction gather slots (GB = 1024; a mesh build at P > 92: 4096) {value, seq} written by core j
//   RB + par * BW                 reduction broadcast {P partials, seq} from core 0 (RB = 1024 + 32 * P, BW = 4 * KW + 16;
//                                 a mesh build RB = 4096, BW = 8 * KW + 16, seq last; further sums of a joint
//                                 reduction go to their fold tiles). KW = K * P and PP = P rounded up to 4 words: every
//                                 block of partials starts 16-B aligned, as NoC source and destination must agree mod
//                                 16 B (Blackhole's P = 110 is not a multiple of 4; Wormhole's 72 is, its layout is unchanged)
// Header (cb_hdr words): 0-2 compute magic, 4-6 compute ops, 8-9 compute bad op; 10 run ticks, 11 incl. drain,
//   12 B ops, 13 reductions, 14 rows, 15 fetches, 16 B pc, 17 halos, 18 B magic, 20 B bad op, 21 N bad op,
//   22-24 N ops/gathers/values, 30 B abort reason, 31 B abort pc, 32 N abort reason, 33 N abort pc, 34 ABORT seen,
//   40-43 at a write barrier timeout: acks expected, acks received, writes issued, writes sent (NoC registers)
#include "api/dataflow/dataflow_api.h"

// gather entries dest << XS | src, halo send words runlen << HS | src (ttres/encode.py; CFD_WIDE: T > 64 tiles a core)
#ifdef CFD_WIDE
constexpr uint32_t XS = 22, XI = 0x3fffff, HS = 20, HI = 0xfffff;
#else
constexpr uint32_t XS = 16, XI = 0xffff, HS = 16, HI = 0xffff;
#endif
#ifdef CFD_MESH
// Mesh of chips (resident/mesh/run_mesh.py): the program was lowered for K * P cores, chip blocks first; this chip
// runs cores pid * P .. pid * P + P - 1. Three additions, all over the fabric (FABRIC_1D):
//  * start barrier: core 0 of every chip, after its own chip's barrier, exchanges a run tag along its mesh row, then
//    the column, so no fabric write lands in a sync region that is still to be zeroed;
//  * B_RED: core 0 all-gathers the K chips' blocks of P partials (row multicast, then column multicast of the row's
//    blocks, each a fused write + count) and broadcasts all K * P in partition order, so every core of every chip
//    folds the same K * P numbers in the same order and takes the same branches;
//  * B_HALO to a core on another chip: the member writes into the gateway's shadow of the receiver's halo region and
//    counts it there; the gateway (one core per neighbour chip, one fabric connection) relays its own chip's
//    receiver credits to the remote gateway, waits for its members and the remote credit, sends one packet per
//    remote receiver (fused write + data_sem increment by the senders it bundles) and credits its members.
// Mesh sync words follow the reduction region (E0 = RB + 2 BW): memb[s] E0 + 16 s, cons[s] E0 + 512 + 16 s, rcred[s]
// E0 + 1024 + 16 s (s < NH), chip-block counts E0 + 1536 + 16 * (2 * par + stage), a member's shadow-free count E0 + 1600.
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;
#endif

#ifdef CFD_OS   // run_mesh --os: size over speed, when the five kernels overflow the 70,656-B kernel config buffer
#pragma GCC optimize("Os")
#endif

#ifdef CFD_MESH
#define MESH_T(v) const uint32_t v = get_timestamp_32b()
// 64-bit accumulators: the low word in hdr[w], the carry in hdr[w + 400] (64-127 and 288-303 hold the --prof counters). A 32-bit sum wraps after 4.3 s
// at 1 GHz, so every run past that read its comm time modulo 2^32 cycles.
#define MESH_ACC(w, v) do { const uint32_t o_ = hdr[w]; hdr[w] = o_ + (get_timestamp_32b() - v); if (hdr[w] < o_) ++hdr[(w) + 400]; } while (0)
#else
#define MESH_T(v)
#define MESH_ACC(w, v)
#endif

// bscal uop codes: the compute kernel's (vm_compute.cpp C_*)
enum : uint32_t { S_LD = 4, S_ST = 5, S_FILL = 6, S_ADD = 8, S_SUB, S_MUL, S_DIV, S_MAX, S_MIN, S_WHERE, S_CPD, S_NEG, S_ABS,
                  S_GTZ = 22, S_GEZ, S_LTZ, S_LEZ, S_EQZ, S_NEZ, S_MAD = 41, S_MSB = 42 };
static inline float fbits(uint32_t b) { union { uint32_t u; float v; } c{b}; return c.v; }
static inline uint32_t ubits(float v) { union { float v; uint32_t u; } c{v}; return c.u; }
enum : uint32_t { B_END = 0, B_SIGC, B_WAITC, B_SIGN, B_WAITN, B_HALO, B_RED, B_OUT, B_FETCH, B_JMP, B_BRT, B_SINK, B_GATHER = 15, B_SCAL = 16, B_PH = 17 };
enum : uint32_t { W_NDONE = 1000, W_ABORT = 1004, W_CDONE = 1008, NH = 32 };

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
    uint32_t ai = 0;
    const uint32_t arena_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t ccode_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t bcode_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t ncode_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t tab_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t const_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t hist_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(ai++);
    const uint32_t mx0 = get_common_arg_val<uint32_t>(ai++), my0 = get_common_arg_val<uint32_t>(ai++);
    const uint32_t mx1 = get_common_arg_val<uint32_t>(ai++), my1 = get_common_arg_val<uint32_t>(ai++);
    const uint32_t hist_rows = get_common_arg_val<uint32_t>(ai++);
    const uint32_t limit = get_common_arg_val<uint32_t>(ai++);
    if (limit == 0) return;   // run_mesh --l1-check: the program is enqueued to size it, nothing runs
    const uint32_t xy0 = ai;   // then P words x | y << 16
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t tab_halo = get_arg_val<uint32_t>(1) & 0x7fffffff, hdir = get_arg_val<uint32_t>(1) >> 31;

    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ccode = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bcode = get_compile_time_arg_val(2);
    constexpr uint32_t cb_ncode = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(4);
    constexpr uint32_t cb_halo = get_compile_time_arg_val(5);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(6);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(7);
    constexpr uint32_t cb_cb = get_compile_time_arg_val(8);
    constexpr uint32_t cb_bc = get_compile_time_arg_val(9);
    constexpr uint32_t cb_bn = get_compile_time_arg_val(10);
    constexpr uint32_t cb_nb = get_compile_time_arg_val(11);
    constexpr uint32_t cb_done = get_compile_time_arg_val(12);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(13);
    constexpr uint32_t nv = get_compile_time_arg_val(14);
    constexpr uint32_t P = get_compile_time_arg_val(15);
    constexpr uint32_t cbytes = get_compile_time_arg_val(16);
    constexpr uint32_t bbytes = get_compile_time_arg_val(17);
    constexpr uint32_t nbytes = get_compile_time_arg_val(18);
    constexpr uint32_t tbytes = get_compile_time_arg_val(19);
    constexpr uint32_t hmax = get_compile_time_arg_val(20);
#ifdef CFD_MESH
    constexpr uint32_t K = CFD_MESH;              // chips
    constexpr uint32_t SYNCB = CFD_SYNCB;         // cb_sync bytes
#else
    constexpr uint32_t K = 1;
#if defined(CFD_BPATCH) || defined(CFD_BGTS)
    constexpr uint32_t SYNCB = 8192;              // + the B-patch mailbox in the last 256 B
#else
    constexpr uint32_t SYNCB = 4096;
#endif
#endif
    constexpr uint32_t KP = K * P;                // partials every core folds
    constexpr uint32_t KW = (KP + 3) & ~3u;       // their stride in words, so every block starts 16-B aligned
    constexpr uint32_t PP = (P + 3) & ~3u;        // one chip's P partials, padded the same way
#if defined(CFD_MESH) && defined(CFD_RED2)
    // two-stage reductions only (K * P > 1024): a fold tile gets this chip's P partials or the K chip sums, no pairs
    constexpr uint32_t KF = K > P ? K : P;
    constexpr uint32_t BW = 4 * ((KF + 3) & ~3u) + 16;
    constexpr uint32_t XB = 16;                   // all-gather block per chip: its chip sum
#elif defined(CFD_MESH)
    constexpr uint32_t KF = KP;
    constexpr uint32_t BW = 8 * KW + 16;          // {K * P partials, seq last} (the sync layout run_mesh.py assumes)
    constexpr uint32_t XB = 12 * PP;              // all-gather block per chip: P partials per sum, up to 3 sums
#else
    constexpr uint32_t KF = KP;
    constexpr uint32_t BW = 4 * KW + 16;
#endif
    // reduction broadcast base: after core 0's gather slots, or on a mesh (BW grows with K * P) past the fixed words
    // W_NDONE.. at 4000, so the mesh words at E0 never meet them
#ifdef CFD_MESH
    // any K, one chip included: run_mesh.py sizes SYNCB and CONS from them. Gather slots at GB: 1024 while 2P of them
    // end before the fixed words at 4000 (Wormhole's 72), else from 4096 with the broadcast after them (Blackhole's 110)
    constexpr uint32_t GB = CFD_GB, RB = CFD_RB;
#else
    constexpr uint32_t GB = 1024;
    constexpr uint32_t RB = 1024 + 32 * P;
#endif
    constexpr uint32_t E0 = RB + 2 * BW;
#ifdef CFD_MESH
    static_assert(KF <= 1024 && GB + 32 * P <= (GB == 1024 ? 4 * W_NDONE : RB) && E0 + 1616 <= SYNCB, "sync region");
#else
    static_assert(KF <= 1024 && 1024 + 32 * P <= 4 * W_NDONE && E0 <= 4 * W_NDONE, "sync region");
#endif
    constexpr auto a_arena = TensorAccessorArgs<21>();
    constexpr auto a_c = TensorAccessorArgs<a_arena.next_compile_time_args_offset()>();
    constexpr auto a_b = TensorAccessorArgs<a_c.next_compile_time_args_offset()>();
    constexpr auto a_n = TensorAccessorArgs<a_b.next_compile_time_args_offset()>();
    constexpr auto a_tab = TensorAccessorArgs<a_n.next_compile_time_args_offset()>();
    constexpr auto a_const = TensorAccessorArgs<a_tab.next_compile_time_args_offset()>();
    constexpr auto a_hist = TensorAccessorArgs<a_const.next_compile_time_args_offset()>();
    constexpr auto a_out = TensorAccessorArgs<a_hist.next_compile_time_args_offset()>();
    constexpr auto a_home = TensorAccessorArgs<a_out.next_compile_time_args_offset()>();
    constexpr uint32_t xa = a_home.next_compile_time_args_offset();
    constexpr uint32_t cb_out = get_compile_time_arg_val(xa);       // compute -> writer ring (DRAM-homed stores)
    constexpr uint32_t cb_bs = get_compile_time_arg_val(xa + 1);    // 16 x DA-byte slots for single DRAM words
    constexpr uint32_t NT = get_compile_time_arg_val(xa + 2);
    constexpr uint32_t HP = get_compile_time_arg_val(xa + 3);   // DRAM home pages per core (encode.py: packed slots)
    constexpr uint32_t CP = CFD_CP;                               // DRAM constant pages per core
    constexpr uint32_t RING = get_compile_time_arg_val(xa + 4);   // cb_out pages
    constexpr uint32_t CR = get_compile_time_arg_val(xa + 5);     // paged compute code: NCRISC fills cb_ccode
    // B code window (run_vm --bwin): cb_bcode holds BWIN words of the b stream from word bwb on; an op whose BMAX
    // words could leave it refills it from DRAM at pc rounded down to 1 KB (BWIN = 0: the whole stream, bbytes)
#ifdef CFD_MESH
    constexpr uint32_t xw = xa + 10;              // after the mesh args cb_xr, cb_sh, SHS, cb_fh (xa + 6..9)
#else
    constexpr uint32_t xw = xa + 6;
#endif
    constexpr uint32_t BWIN = get_compile_time_arg_val(xw), BMAX = get_compile_time_arg_val(xw + 1);
    constexpr uint32_t bfull = get_compile_time_arg_val(xw + 2);
    constexpr uint32_t abytes = nv * 4096;
    const auto t_arena = TensorAccessor(a_arena, arena_addr, abytes);
    const auto t_c = TensorAccessor(a_c, ccode_addr, cbytes);
    const auto t_b = TensorAccessor(a_b, bcode_addr, bfull);
    const auto t_n = TensorAccessor(a_n, ncode_addr, nbytes);
    const auto t_tab = TensorAccessor(a_tab, tab_addr, tbytes);
    const auto t_const = TensorAccessor(a_const, const_addr, 4096);
    const auto t_hist = TensorAccessor(a_hist, hist_addr, 64);
#ifndef CFD_HDRB
#define CFD_HDRB 4096   // --trace appends the agents' event records after the 4 KB header
#endif
    const auto t_out = TensorAccessor(a_out, out_addr, CFD_HDRB + abytes);
    const auto t_home = TensorAccessor(a_home, get_common_arg_val<uint32_t>(xy0 + P), 4096);
    const uint32_t bscr = get_write_ptr(cb_bs);
    constexpr uint32_t DA = NOC_DRAM_READ_ALIGNMENT_BYTES;   // a DRAM read keeps its address mod DA: 32 B WH, 64 B BH
    const uint32_t out1 = get_read_ptr(cb_out) + RING * 4096;   // end of cb_out's fifo, nothing read yet

    const uint32_t arena = get_write_ptr(cb_v);
    const uint32_t sync = get_write_ptr(cb_sync);
    const uint32_t halo = get_write_ptr(cb_halo);
    const uint32_t stage = get_write_ptr(cb_stage);
    volatile tt_l1_ptr uint32_t* S = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sync);
    volatile tt_l1_ptr uint32_t* hdr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_hdr));
#ifdef CFD_TRACE
    hdr[1025] = 0;   // UNPACK writes its start here once compute runs (it waits for the arena BRISC loads)
#endif
    volatile tt_l1_ptr uint32_t* A = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arena);
    volatile tt_l1_ptr uint32_t* ST = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    const uint32_t* T = reinterpret_cast<const uint32_t*>(get_write_ptr(cb_tab));
    const uint32_t bwl1 = get_write_ptr(cb_bcode);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(bwl1);
    uint32_t pc = 0, bwb = 0;   // B window: words [bwb, bwb + BWIN) of the b stream (the first one loaded below)

    // give up on a wait: record why and where, raise ABORT here and on every other core
    auto fail = [&](uint32_t why) {
        if (hdr[30] == 0) {
            hdr[30] = why;
            hdr[31] = pc;
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
#if defined(CFD_BPATCH) || defined(CFD_BGTS)
    // B-patch mailbox (the last 256 B of cb_sync, past every sync word): NCRISC posts the upper parts of a patched page's L1-source residue sections
    // (src, entries, n, page) and bumps SEQ; BRISC scatters them from inside its waits and answers DONE = SEQ
    constexpr uint32_t MB = SYNCB / 4 - 64;
    static_assert((K > 1 ? E0 + 1616 : 4096u) <= 4 * MB, "B-patch mailbox past every sync word");
    enum : uint32_t { M_SEQ = MB, M_DONE = MB + 4, M_N = MB + 8, M_JOB = MB + 12 };
    uint32_t bseen = 0;
    auto bsvc = [&]() {
        const uint32_t q = S[M_SEQ];
        if (q == bseen) return;
        for (uint32_t j = 0, nj = S[M_N]; j < nj; ++j) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(S[M_JOB + 4 * j]);
            const uint32_t* xs = reinterpret_cast<const uint32_t*>(S[M_JOB + 4 * j + 1]);
            const uint32_t n = S[M_JOB + 4 * j + 2];
            uint32_t* out = reinterpret_cast<uint32_t*>(S[M_JOB + 4 * j + 3]);
            uint32_t e = 0;
            for (; e + 4 <= n; e += 4) {
                const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
                const uint32_t v0 = src[x0 & XI], v1 = src[x1 & XI], v2 = src[x2 & XI], v3 = src[x3 & XI];
                out[x0 >> XS] = v0;
                out[x1 >> XS] = v1;
                out[x2 >> XS] = v2;
                out[x3 >> XS] = v3;
            }
            for (; e < n; ++e) out[xs[e] >> XS] = src[xs[e] & XI];
        }
        asm volatile("" ::: "memory");
        S[M_DONE] = q;
        bseen = q;
    };
#define BSVC() bsvc()
#else
#define BSVC()
#endif
// true when the wait succeeded; on timeout or a raised ABORT it fails with `why` and is false
#define WAIT_UNTIL(cond, why)                                                    \
    ({                                                                           \
        bool ok_ = true;                                                         \
        const uint32_t s_ = get_timestamp_32b();                                 \
        while (!(cond)) {                                                        \
            invalidate_l1_cache();                                               \
            BSVC();                                                              \
            if (S[W_ABORT] || get_timestamp_32b() - s_ > limit) {                \
                fail(why);                                                       \
                ok_ = false;                                                     \
                break;                                                           \
            }                                                                    \
        }                                                                        \
        ok_;                                                                     \
    })

    // NoC barriers are bounded too: a transaction that never completes is reported, not waited on forever
    auto wbar = [&](uint32_t why) {
        const uint32_t s_ = get_timestamp_32b();
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            BSVC();
            if (get_timestamp_32b() - s_ > limit) {
                hdr[40] = noc_nonposted_writes_acked[noc_index];
                hdr[41] = NOC_STATUS_READ_REG(noc_index, NIU_MST_WR_ACK_RECEIVED);
                hdr[42] = noc_nonposted_writes_num_issued[noc_index];
                hdr[43] = NOC_STATUS_READ_REG(noc_index, NIU_MST_NONPOSTED_WR_REQ_SENT);
                fail(why);
                return false;
            }
        }
        return true;
    };
    auto rbar = [&](uint32_t why) {
        const uint32_t s_ = get_timestamp_32b();
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            BSVC();
            if (get_timestamp_32b() - s_ > limit) { fail(why); return false; }
        }
        return true;
    };

    for (uint32_t i = 0; i < SYNCB / 4; ++i) S[i] = 0;
    for (uint32_t i = 0; i < 1024; ++i) hdr[i] = 0;
    cb_reserve_back(cb_v, nv);
    if constexpr (CR == 0) cb_reserve_back(cb_ccode, 1);
    noc_async_read(t_arena.get_noc_addr(core), arena, abytes);
    if constexpr (CR == 0) noc_async_read(t_c.get_noc_addr(0), get_write_ptr(cb_ccode), cbytes);
    noc_async_read(t_b.get_noc_addr(0), get_write_ptr(cb_bcode), bbytes);
    noc_async_read(t_n.get_noc_addr(0), get_write_ptr(cb_ncode), nbytes);
    noc_async_read(t_tab.get_noc_addr(core), get_write_ptr(cb_tab), tbytes);
    rbar(20);
    if constexpr (CR == 0) cb_push_back(cb_ccode, 1);
    cb_push_back(cb_v, nv);

#ifdef CFD_MESH
    // mesh: compile args after CR (cb_xr, cb_sh, SHS, cb_fh), runtime args after tab_halo (tab_gw, run, pid,
    // mx, mw, my, mh, the partition id at each mesh position, then fabric connection args: core 0 its W E S N
    // reduce links, a gateway its one link)
    constexpr uint32_t cb_xr = get_compile_time_arg_val(xa + 6);   // core 0: barrier tags, then 2 x K blocks
    constexpr uint32_t cb_sh = get_compile_time_arg_val(xa + 7);   // gateway: shadow halo regions
    constexpr uint32_t SHS = get_compile_time_arg_val(xa + 8);     // words per shadow slot ([slot][run] layout)
    constexpr uint32_t cb_fh = get_compile_time_arg_val(xa + 9);   // packet headers
    constexpr uint32_t HDR = sizeof(PACKET_HEADER_TYPE);
    constexpr uint32_t MAXPAY = 2048;
    constexpr uint32_t CBS = 4 * PP;                               // one chip block of partials, 16-B aligned
    const uint32_t tab_gw = get_arg_val<uint32_t>(2), mrun = get_arg_val<uint32_t>(3), pid = get_arg_val<uint32_t>(4);
    const uint32_t mx = get_arg_val<uint32_t>(5), mw = get_arg_val<uint32_t>(6);
    const uint32_t my = get_arg_val<uint32_t>(7), mh = get_arg_val<uint32_t>(8);
    const uint32_t xr = get_write_ptr(cb_xr), shadow = get_write_ptr(cb_sh), fh = get_write_ptr(cb_fh);
    auto pkt = [&](uint32_t i) { return reinterpret_cast<volatile PACKET_HEADER_TYPE*>(fh + 64 * i); };
    const bool gw = tab_gw != 0;
    const uint32_t gR = gw ? T[tab_gw] : 0, gP = gw ? T[tab_gw + 1] : 0, rgw = gw ? T[tab_gw + 2] : 0;
    const uint32_t nrcv = gw ? T[tab_gw + 3] : 0;
    // a member's remote pairs: the gateways' shadows are shared by every exchange in program order, CFD_NSH regions
    // round-robin on the exchange count, so the shadow-free credit is one counter (E0 + 1600), preset for the first NSH
    uint32_t nrem = 0;
    {
        uint32_t p = tab_halo + 1;
        for (uint32_t r = 0, n = T[tab_halo]; r < n; ++r) {
            nrem += T[p] >> 31;
            p += 3 + T[p + 2];
        }
        for (uint32_t q = 0; q < NH; ++q) S[(512 + 16 * q) / 4] = 0;
#ifdef CFD_MESH
        S[(E0 + 1600) / 4] = CFD_NSH * nrem;
#endif
    }
    size_t cidx = 9 + K;
    const bool fab = core == 0;
    const bool hasd[4] = {fab && mx > 0, fab && mx + 1 < mw, fab && my > 0, fab && my + 1 < mh};
    WorkerToFabricEdmSender conn[4], gconn;
    for (uint32_t d = 0; d < 4; ++d) if (hasd[d]) conn[d] = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(cidx);
    if (gw) gconn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(cidx);
    // the fabric API's own waits spin forever; these are bounded and fail with a code instead
    auto slot = [&](WorkerToFabricEdmSender& c, uint32_t why) {
        const uint32_t s_ = get_timestamp_32b();
        while (!c.edm_has_space_for_packet<1>()) {
            if (S[W_ABORT] || get_timestamp_32b() - s_ > limit) { fail(why); return false; }
        }
        return true;
    };
    auto shut = [&](WorkerToFabricEdmSender& c, uint32_t why) {
        c.close_start();
        const uint32_t s_ = get_timestamp_32b();
        while (*c.worker_teardown_addr != 1) {
            invalidate_l1_cache();
            if (get_timestamp_32b() - s_ > limit) { fail(why); return; }
        }
        noc_async_write_barrier(get_fabric_worker_noc());
        *c.worker_teardown_addr = 0;
    };
    for (uint32_t d = 0; d < 4; ++d) if (hasd[d]) conn[d].open_start();
    if (gw) gconn.open_start();
    for (uint32_t d = 0; d < 4; ++d) if (hasd[d]) conn[d].open_finish();
    if (gw) gconn.open_finish();
    hdr[57] = 1;
    // core 0: send `bytes` at L1 `src` to the same address on every chip along the row (axis 0) or column, fused
    // with an increment of the count at `cnt` (0: no count)
    auto mcast = [&](uint32_t axis, uint32_t src, uint32_t bytes, uint32_t cnt) {
        const uint32_t pos = axis ? my : mx, len = axis ? mh : mw;
        for (uint32_t side = 0; side < 2; ++side) {
            const uint32_t d = 2 * axis + side, range = side ? len - 1 - pos : pos;
            if (!hasd[d] || range == 0) continue;
            auto* ph = pkt(1 + d);
            ph->to_chip_multicast(MulticastRoutingCommandHeader{1, (uint8_t)range});
            if (cnt)
                ph->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(src), get_noc_addr(cnt), 1, true}, bytes);
            else
                ph->to_noc_unicast_write(NocUnicastCommandHeader{get_noc_addr(src)}, bytes);
            if (!slot(conn[d], 40 + d)) return;
            conn[d].send_payload_without_header_non_blocking_from_address(src, bytes);
            conn[d].send_payload_flush_blocking_from_address((uint32_t)ph, HDR);
        }
    };
#else
    constexpr uint32_t nrem = 0;   // one chip: no remote halo pairs
#endif
    // start barrier: every core's sync region is zero before anyone's remote write lands in it
    const uint32_t sem_in = get_semaphore(0), sem_go = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_go);
    volatile tt_l1_ptr uint32_t* in = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_in);
    const uint32_t x00 = get_common_arg_val<uint32_t>(xy0);
    noc_semaphore_inc(get_noc_addr(x00 & 0xffff, x00 >> 16, sem_in), 1);
    // a multicast rectangle is given start -> end in the direction its NoC travels: NoC 1 runs the other way
    const uint64_t mc_base = noc_index == 0 ? get_noc_multicast_addr(mx0, my0, mx1, my1, 0)
                                            : get_noc_multicast_addr(mx1, my1, mx0, my0, 0);
    bool run = true;
    if (core == 0) {
        bool ok = WAIT_UNTIL(*in >= P, 1);
#ifdef CFD_MESH
        // then every chip: run tags along the row, then the column (a chip's column tag says its row is ready)
        if (ok) {
            MESH_T(tb);
            volatile tt_l1_ptr uint32_t* X = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xr);
            const uint32_t tag = mrun << 8 | 0x5a;
            X[4 * mx] = tag;
            mcast(0, xr + 16 * mx, 16, 0);
            for (uint32_t c = 0; c < mw && ok; ++c) ok = WAIT_UNTIL(X[4 * c] == tag, 33);
            if (ok) {
                X[64 + 4 * my] = tag;
                mcast(1, xr + 256 + 16 * my, 16, 0);
                for (uint32_t r = 0; r < mh && ok; ++r) ok = WAIT_UNTIL(X[64 + 4 * r] == tag, 34);
            }
            MESH_ACC(56, tb);
        }
#endif
        if (ok) {   // then release everyone by unicast increments of their `go`
#ifdef CFD_MESH
            hdr[57] |= 2;
#endif
            *go = 1;
            for (uint32_t j = 1; j < P; ++j) {
                const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + j);
                noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, sem_go), 1);
            }
        } else {
            *go = 2;   // releases this core's NCRISC into its drain
            run = false;
        }
    }
    if (run && !WAIT_UNTIL(*go != 0, 2)) {
        *go = 2;
        run = false;
    }
#if defined(CFD_STOP) && CFD_STOP == 1
    // bisect (run_mesh --stop 1): every chip's core 0 ends the run right after the start barrier and the fabric opens
    if (run && core == 0) fail(90);
#endif
    const uint32_t t0 = get_timestamp_32b();
    const uint64_t T0 = get_timestamp();   // hdr[10] wraps at 2^32 cycles (4.3 s at 1 GHz): hdr[58] = high word

    // halo plan of each slot (ttres/encode.py): plan 0 at tab_halo; with more plans (hdir, RT arg 1 bit 31) a
    // directory below tab_halo gives each slot's plan and each plan q > 0's offset from tab_halo
    // and its receive buffer's word offset (slots sized per plan; one plan: s * hmax). Only plan 0 (the cells) crosses
    // chips: a mesh build's coarse plans are chip-local (run_mesh asserts it), so the gateway path and the shadow
    // round-robin count plan-0 exchanges only (nh0)
    uint32_t hcnt[NH], hb[NH], hoff[NH];
    bool hp0[NH];
    for (uint32_t s = 0; s < NH; ++s) {
        hcnt[s] = 0;
        const uint32_t dw = hdir ? T[tab_halo - 1 - s] : 0, q = dw & 0xff;
        hb[s] = tab_halo + (q ? T[tab_halo - NH - q] : 0);
        hoff[s] = hdir ? dw >> 8 : s * hmax;
        hp0[s] = q == 0;
    }
    uint32_t nh0 = 0;   // plan-0 halo exchanges (mesh shadow rounds)
    uint32_t nxg = 0;   // chip all-gathers (mesh)
    uint32_t nred = 0, nrow = 0, nfetch = 0, nhalo = 0, nops = 0, nsink = 0;
    volatile tt_l1_ptr uint32_t* row = hdr + 992;   // history row staging, 16 words
#ifdef CFD_PHASE
    // run_mesh --phase: per SIMPLE phase (B_PH opens phase id = w bits 8-11; id 15 = before the first marker), 64-bit
    // (low word at the even hdr word, high at the next): BRISC wall cycles at hdr 544 + 2 id, inside B_RED at 576 + 2 id,
    // inside B_HALO at 608 + 2 id, waiting on compute (B_WAITC) at 640 + 2 id
    uint64_t pht = get_timestamp(), phc[16], phr[16], phh[16], phw[16];
    for (uint32_t i = 0; i < 16; ++i) phc[i] = phr[i] = phh[i] = phw[i] = 0;
    uint32_t phid = 15;
#endif
#ifdef CFD_PROF
    uint64_t cyc[16];            // census: cycles (64-bit, stored in 16-cycle units so a 100-it run fits a word)
    uint32_t cnt[16];            // and counts per op, to hdr 64 + op / 80 + op
    for (uint32_t i = 0; i < 16; ++i) cyc[i] = cnt[i] = 0;
    uint32_t tprev = get_timestamp_32b();
    // inside HALO (hdr 64 + 12..14): waiting for the receivers' credits, the copy into the stage, the write barrier
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
    volatile tt_l1_ptr uint32_t* tr_ = hdr + 1024 + 3 * (2 * CFD_TRACE + 2);
    uint32_t trn_ = 0;
    const uint32_t trt0_ = get_timestamp_32b();
    tr_[0] = 0;
    tr_[1] = trt0_;
    uint32_t trs_ = 0;
#endif
    while (run) {
        if constexpr (BWIN != 0) {
            if (pc < bwb || pc + BMAX > bwb + BWIN) {
                bwb = pc & ~255u;
                noc_async_read(t_b.get_noc_addr(0, bwb * 4), bwl1, BWIN * 4);
                noc_async_read_barrier();
                B = reinterpret_cast<const uint32_t*>(bwl1 - bwb * 4);
            }
        }
        asm volatile("" ::: "memory");   // L1 written by the other agents between ops
        const uint32_t w = B[pc++];
        const uint32_t op = w & 0xff;
        ++nops;
#ifdef CFD_TRACE
        {
            const uint32_t t_ = get_timestamp_32b();
            TR_REC(pc - 1, op, t_);
        }
#endif
        hdr[16] = pc;
#ifdef CFD_PHASE
        const uint64_t pho = (op == B_RED || op == B_HALO || op == B_WAITC) ? get_timestamp() : 0;
#endif
        switch (op) {
            case B_END: run = false; break;
            case B_SIGC:
                if (!WAIT_UNTIL(cb_pages_reservable_at_back(cb_bc, 1), 3)) { run = false; break; }
                cb_reserve_back(cb_bc, 1);
                cb_push_back(cb_bc, 1);
                break;
            case B_WAITC:
                if (!WAIT_UNTIL(cb_pages_available_at_front(cb_cb, 1), 4)) { run = false; break; }
                cb_wait_front(cb_cb, 1);
                cb_pop_front(cb_cb, 1);
                break;
            case B_SIGN:
                if (!WAIT_UNTIL(cb_pages_reservable_at_back(cb_bn, 1), 5)) { run = false; break; }
                cb_reserve_back(cb_bn, 1);
                cb_push_back(cb_bn, 1);
                break;
            case B_WAITN:
                if (!WAIT_UNTIL(cb_pages_available_at_front(cb_nb, 1), 6)) { run = false; break; }
                cb_wait_front(cb_nb, 1);
                cb_pop_front(cb_nb, 1);
                break;
            case B_HALO: {
                const uint32_t s = B[pc], tile = B[pc + 1];
                pc += 2;
                const uint32_t k = ++hcnt[s];
                ++nhalo;
                MESH_T(th0);
#ifdef CFD_MESH
                if (hp0[s]) ++nh0;
                // gateway, receiving side: this chip's receivers have credited round k for the senders behind
                // this gateway; pass it on to the remote gateway as one increment of its rcred[s]
                if (gw && gR && hp0[s]) {
                    if (!WAIT_UNTIL(S[(E0 + 512 + 16 * s) / 4] >= k * gR, 30)) { run = false; break; }
                    auto* ph = pkt(0);
                    fabric_set_unicast_route<false>((LowLatencyPacketHeader*)ph, 1);
                    ph->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
                        get_noc_addr(rgw & 0xffff, rgw >> 16, sync + E0 + 1024 + 16 * s), 1, true});
                    if (!slot(gconn, 44)) { run = false; break; }
                    gconn.send_payload_flush_blocking_from_address((uint32_t)ph, HDR);
                }
                MESH_ACC(50, th0);
                const uint32_t nr = hp0[s] ? nrem : 0;
#else
                constexpr uint32_t nr = 0;
#endif
                const uint32_t nrecv = T[hb[s]];
                if (nrecv) {
                    bool okc;
                    PSUB(12, okc = WAIT_UNTIL(S[(512 + 16 * s) / 4] >= k * (nrecv - nr), 7));
                    if (!okc) { run = false; break; }
#ifdef CFD_MESH
                    // the gateways have sent exchange nh0 - NSH, so this exchange's shadow region is free
                    if (nr && !WAIT_UNTIL(S[(E0 + 1600) / 4] >= nh0 * nr, 33)) { run = false; break; }
#endif
                    // the source: an L1 tile, or (descriptor kind 1 / 2) a DRAM home / constant read word by word
                    const volatile tt_l1_ptr uint32_t* x = A + (tile & 0xfffffff) * 1024;
                    const uint32_t kd = tile >> 28, v = tile & 0xfffffff;
                    // the stage holds HCAP words (its last 8 carry RED's seq/semaphore): a send wider than what is
                    // left waits for the writes already issued and refills from word 0 (an SA coarse level's halo
                    // reaches 36,780 words on motorBike 354k and 6,007 on r13; unchecked, it ran through the token
                    // CBs into the header)
                    constexpr uint32_t HCAP = 4096 - 8;
                    uint32_t p = hb[s] + 1, so = 0;
                    for (uint32_t r = 0; r < nrecv && run; ++r) {
                        const uint32_t rc = T[p], off = T[p + 1], n = T[p + 2];
                        p += 3;
                        const uint32_t* ix = T + p;   // one word per lane, runs marked (encode.py runs())
                        const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + (rc & 0xffff));
                        for (uint32_t c0 = 0; c0 < n;) {
                            const uint32_t dw = hoff[s] + off + c0;
                            so = ((so + 3) & ~3u) + (dw & 3);   // NoC: source and destination agree mod 16 B
                            if (so + 1 > HCAP) {
                                bool okf;
                                PSUB(14, okf = wbar(21));
                                if (!okf) { run = false; break; }
                                so = dw & 3;
                            }
                            const uint32_t cn = n - c0 < HCAP - so ? n - c0 : HCAP - so;
                            if (kd == 0) {
                                // runs (a run's first lane carries its length << 16, the rest 0 there) copy word by
                                // word, clipped at the chunk end (the rest of a cut run reads as lone values); four
                                // lone values go in flight together (volatile pointers serialise: ~29 cyc each)
                                auto copy = [&]() {
                                    const uint32_t* xs = const_cast<const uint32_t*>(x);
                                    uint32_t* sd = const_cast<uint32_t*>(ST) + so;
                                    const uint32_t* iq = ix + c0;
                                    uint32_t q = 0;
                                    while (q < cn) {
                                        const uint32_t e = iq[q];
                                        uint32_t L = e >> HS;
                                        if (L > 1) {
                                            if (L > cn - q) L = cn - q;
                                            const uint32_t* s0 = xs + (e & HI);
                                            uint32_t* d0 = sd + q;
                                            uint32_t j = 0;
                                            for (; j + 4 <= L; j += 4) {
                                                const uint32_t v0 = s0[j], v1 = s0[j + 1], v2 = s0[j + 2], v3 = s0[j + 3];
                                                d0[j] = v0;
                                                d0[j + 1] = v1;
                                                d0[j + 2] = v2;
                                                d0[j + 3] = v3;
                                            }
                                            for (; j < L; ++j) d0[j] = s0[j];
                                            q += L;
                                        } else if (q + 4 <= cn && (iq[q + 1] >> HS) == 1 && (iq[q + 2] >> HS) == 1 && (iq[q + 3] >> HS) == 1) {
                                            const uint32_t v0 = xs[e & HI], v1 = xs[iq[q + 1] & HI], v2 = xs[iq[q + 2] & HI],
                                                           v3 = xs[iq[q + 3] & HI];
                                            sd[q] = v0;
                                            sd[q + 1] = v1;
                                            sd[q + 2] = v2;
                                            sd[q + 3] = v3;
                                            q += 4;
                                        } else {
                                            sd[q] = xs[e & HI];
                                            ++q;
                                        }
                                    }
                                    asm volatile("" ::: "memory");   // the stage is pushed by the NoC after this
                                };
                                PSUB(13, copy());
                            } else {
                                const uint32_t* L = ix + c0;
                                for (uint32_t q0 = 0; q0 < cn; q0 += 16) {
                                    const uint32_t m = cn - q0 < 16 ? cn - q0 : 16;
                                    for (uint32_t j = 0; j < m; ++j) {
                                        const uint32_t i = L[q0 + j] & HI, o = (i & 1023) * 4;
                                        const uint32_t pg = core * (kd == 1 ? HP : CP) + v + (i >> 10);
                                        noc_async_read(kd == 1 ? t_home.get_noc_addr(pg, o) : t_const.get_noc_addr(pg, o),
                                                       bscr + DA * j + (o & (DA - 1)), 4);
                                    }
                                    if (!rbar(27)) { run = false; break; }
                                    for (uint32_t j = 0; j < m; ++j)
                                        ST[so + q0 + j] = *reinterpret_cast<volatile uint32_t*>(bscr + DA * j + ((L[q0 + j] & 1023) * 4 & (DA - 1)));
                                }
                                if (!run) break;
                            }
                            uint32_t dst = halo + dw * 4;
#ifdef CFD_MESH
                            // via the gateway (run start, off rebased)
                            if (rc >> 31) dst = shadow + (nh0 % CFD_NSH * SHS + ((rc >> 16) & 0x7fff) * 4 + off + c0) * 4;
#endif
                            noc_async_write(stage + so * 4, get_noc_addr(xy & 0xffff, xy >> 16, dst), cn * 4);
                            so += cn, c0 += cn;
                        }
                        p += n;
                    }
                    if (!run) break;
                    bool okb;
                    PSUB(14, okb = wbar(21));
                    if (!okb) { run = false; break; }
                    p = hb[s] + 1;
                    for (uint32_t r = 0; r < nrecv; ++r) {
                        const uint32_t rc = T[p], n = T[p + 2];
                        p += 3 + n;
                        const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + (rc & 0xffff));
                        uint32_t sem = sync + 16 * s;
#ifdef CFD_MESH
                        if (rc >> 31) sem = sync + E0 + 16 * s;   // the gateway's member count
#endif
                        noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, sem), 1);
                    }
                }
#ifdef CFD_MESH
                MESH_T(th1);
                // gateway, sending side: every member's values for the remote receivers are in the shadow and the
                // remote side has freed round k; one packet per receiver, then the members may refill
                if (gw && gP && hp0[s]) {
                    if (!WAIT_UNTIL(S[(E0 + 16 * s) / 4] >= k * gP, 31)) { run = false; break; }
                    MESH_ACC(51, th1);
                    if (!WAIT_UNTIL(S[(E0 + 1024 + 16 * s) / 4] >= k, 32)) { run = false; break; }
                    MESH_ACC(52, th1);
                    uint32_t q = tab_gw + 4;
                    for (uint32_t r = 0; r < nrcv; ++r, q += 5) {
                        const uint32_t rxy = T[q], sq = T[q + 1], lo = T[q + 2], n = T[q + 3], cnt = T[q + 4];
                        const uint32_t w0 = hoff[s] + lo;
                        uint32_t src = shadow + (nh0 % CFD_NSH * SHS + sq) * 4, dst = halo + w0 * 4, left = n * 4;
                        while (left) {
                            const uint32_t b = left > MAXPAY ? MAXPAY : left;
                            left -= b;
                            auto* ph = pkt(0);
                            fabric_set_unicast_route<false>((LowLatencyPacketHeader*)ph, 1);
                            const uint64_t na = get_noc_addr(rxy & 0xffff, rxy >> 16, dst);
                            if (left)
                                ph->to_noc_unicast_write(NocUnicastCommandHeader{na}, b);
                            else
                                ph->to_noc_fused_unicast_write_atomic_inc(NocUnicastAtomicIncFusedCommandHeader{
                                    na, get_noc_addr(rxy & 0xffff, rxy >> 16, sync + 16 * s), cnt, true}, b);
                            if (!slot(gconn, 45)) { run = false; break; }
                            gconn.send_payload_without_header_non_blocking_from_address(src, b);
                            gconn.send_payload_flush_blocking_from_address((uint32_t)ph, HDR);
                            src += b;
                            dst += b;
                        }
                    }
                    noc_async_writes_flushed();
                    for (uint32_t r = 0, nm = T[q++]; r < nm; ++r, q += 2) {
                        const uint32_t xy = get_common_arg_val<uint32_t>(xy0 + T[q]);
                        noc_semaphore_inc(get_noc_addr(xy & 0xffff, xy >> 16, sync + E0 + 1600), T[q + 1]);
                    }
                    MESH_ACC(53, th1);
                }
                MESH_ACC(54, th0);
#endif
#if defined(CFD_STOP) && CFD_STOP == 3
                if (nhalo == 1) { fail(92); run = false; break; }   // bisect (--stop 3): after the first halo exchange
#endif
                break;
            }
            case B_RED: {
                // core j writes {partial, seq} into core 0's gather slot; core 0, once it holds all P, multicasts
                // the P partials and then the seq word (one source: concurrent multicasts from every core into the
                // same rectangle lose acks on this NoC); every core copies the first sum's partials into its fold tile;
                // a joint reduction (w bits 8-11 = count - 1, joint_red/redm) carries further partials in words 2
                // and 3 of the same slot, and core 0 multicasts them straight into their fold tiles (the arena has
                // one address on every core) before the seq word.
                // Mesh scopes (w bits 12-15): 0 all K chips' P partials (chip all-gather of P-word blocks, one per
                // sum); 1 this chip's P partials only; 2 one value per chip (every core of a chip holds the chip
                // sum, core 0 sends its own): the K chip sums. 1 then 2 is the two-stage reduction for K * P > 1024.
                const uint32_t more = (w >> 8) & 0xf;
                const uint32_t part = B[pc], fold = B[pc + 1];
                const uint32_t part2 = more ? B[pc + 2] : 0, fold2 = more ? B[pc + 3] : 0;
                const uint32_t part3 = more > 1 ? B[pc + 4] : 0, fold3 = more > 1 ? B[pc + 5] : 0;
                pc += 2 + 2 * more;
                ++nred;
                const uint32_t par = nred & 1;
#ifdef CFD_MESH
                const uint32_t scope = (w >> 12) & 0xf;
                const uint32_t nb = scope == 1 ? P : scope == 2 ? K : KP;   // words that reach each fold tile
#else
                const uint32_t scope = 1, nb = P;
#endif
                volatile tt_l1_ptr uint32_t* land = ST + 4096 - 4;
                if (scope != 2) {
                    land[0] = A[part * 1024];
                    land[1] = nred;
                    land[2] = more ? A[part2 * 1024] : 0;
                    land[3] = more > 1 ? A[part3 * 1024] : 0;
                    noc_async_write(stage + (4096 - 4) * 4,
                                    get_noc_addr(x00 & 0xffff, x00 >> 16, sync + GB + 16 * (par * P + core)), 16);
                    if (!wbar(22)) { run = false; break; }
                }
                const uint32_t bc = sync + RB + par * BW;
                if (core == 0) {
                    if (scope != 2) {
                        volatile tt_l1_ptr uint32_t* R = S + (GB + 16 * par * P) / 4;
                        for (uint32_t k = 0; k < P && run; ++k) {
                            if (!WAIT_UNTIL(R[4 * k + 1] == nred, 8)) run = false;
                            ST[k] = R[4 * k];
                            ST[KW + k] = R[4 * k + 2];
                            ST[2 * KW + k] = R[4 * k + 3];
                        }
                        if (!run) break;
                    }
#ifdef CFD_MESH
                    if (scope != 1) {
                        // chip blocks: own block into slot pid, the row's by multicast, then the row's blocks down
                        // and up the column; then all K blocks in partition order
                        MESH_T(tr);
                        ++nxg;
                        const uint32_t bs = scope == 2 ? 16 : CBS * (1 + more);   // block bytes
                        const uint32_t blk = xr + 512 + (nxg & 1) * K * XB, u = (nxg + 1) >> 1;
                        const uint32_t crow = E0 + 1536 + 16 * (2 * (nxg & 1)), ccol = crow + 16;   // sync offsets
                        volatile tt_l1_ptr uint32_t* own = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blk + pid * bs);
                        if (scope == 2)
                            own[0] = A[part * 1024];
                        else
                            for (uint32_t j = 0; j <= more; ++j)
                                for (uint32_t k = 0; k < P; ++k) own[j * PP + k] = ST[j * KW + k];
                        mcast(0, blk + pid * bs, bs, sync + crow);
                        if (!WAIT_UNTIL(S[crow / 4] >= u * (mw - 1), 35)) { run = false; break; }
                        for (uint32_t c = 0; c < mw; ++c)
                            mcast(1, blk + get_arg_val<uint32_t>(9 + my * mw + c) * bs, bs, sync + ccol);
                        if (!WAIT_UNTIL(S[ccol / 4] >= u * (mh - 1) * mw, 36)) { run = false; break; }
                        const volatile tt_l1_ptr uint32_t* all = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(blk);
                        if (scope == 2)
                            for (uint32_t k = 0; k < K; ++k) ST[k] = all[4 * k];
                        else
                            for (uint32_t c = 0; c < K; ++c)
                                for (uint32_t j = 0; j <= more; ++j)
                                    for (uint32_t k = 0; k < P; ++k) ST[j * KW + c * P + k] = all[((1 + more) * c + j) * PP + k];
                        MESH_ACC(48, tr);
                        ++hdr[49];
#if defined(CFD_STOP) && CFD_STOP == 2
                        fail(91);   // bisect (--stop 2): after the first chip all-gather
                        run = false;
                        break;
#endif
                    }
#endif
                    volatile tt_l1_ptr uint32_t* seqw = ST + 4096 - 8;
                    *seqw = nred;
                    noc_async_write_multicast_loopback_src(stage, mc_base | bc, 4 * nb, P, true);
                    if (more)
                        noc_async_write_multicast_loopback_src(stage + 4 * KW, mc_base | (arena + fold2 * 4096), 4 * nb, P, true);
                    if (more > 1)
                        noc_async_write_multicast_loopback_src(stage + 8 * KW, mc_base | (arena + fold3 * 4096), 4 * nb, P, true);
                    noc_semaphore_set_multicast_loopback_src(stage + (4096 - 8) * 4, mc_base | (bc + BW - 16), P);
                    if (!wbar(26)) { run = false; break; }
                }
                volatile tt_l1_ptr uint32_t* Bc = S + (RB + par * BW) / 4;
                if (!WAIT_UNTIL(Bc[(BW - 16) / 4] == nred, 8)) { run = false; break; }
                volatile tt_l1_ptr uint32_t* F = A + fold * 1024;
                for (uint32_t k = 0; k < nb; ++k) F[k] = Bc[k];
                break;
            }
            case B_GATHER: {
                // bgather: a gather with no halo entries (local L1 sources), off NCRISC; per part n_loc, n_halo = 0,
                // (lane << 16 | idx) entries, then the zero lanes (ttres/encode.py table layout)
                const uint32_t nparts = (w >> 8) & 0xf;
                uint32_t* dst = const_cast<uint32_t*>(A + B[pc] * 1024);
                uint32_t q = T[1 + B[pc + 1]];
                const uint32_t* srcs = B + pc + 2;
                pc += 2 + nparts;
                for (uint32_t k = 0; k < nparts; ++k) {
                    const uint32_t nl = T[q], nh = T[q + 1];   // bloc: the halo entries stay with NCRISC (its mode 2)
                    q += 2;
                    const uint32_t* src = const_cast<const uint32_t*>(A + srcs[k] * 1024);
                    const uint32_t* xs = T + q;
                    uint32_t e = 0;
                    for (; e + 4 <= nl; e += 4) {
                        const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
                        const uint32_t v0 = src[x0 & XI], v1 = src[x1 & XI], v2 = src[x2 & XI], v3 = src[x3 & XI];
                        dst[x0 >> XS] = v0;
                        dst[x1 >> XS] = v1;
                        dst[x2 >> XS] = v2;
                        dst[x3 >> XS] = v3;
                    }
                    for (; e < nl; ++e) dst[xs[e] >> XS] = src[xs[e] & XI];
                    q += nl + nh;
                }
                const uint32_t nz = T[q++];
                for (uint32_t e = 0; e < nz; ++e) dst[T[q + e]] = 0;
                asm volatile("" ::: "memory");   // compute reads the tile after the next signal
                break;
            }
            case B_OUT: {
                const uint32_t tile = B[pc], idx = B[pc + 1];
                pc += 2;
                const bool outer = (w >> 8) & 0xf;
                if (core == 0) {
                    row[idx] = A[tile * 1024];
                    if (outer) {
                        row[15] = nrow;
                        const uint32_t page = nrow < hist_rows ? nrow : hist_rows - 1;
                        noc_async_write(get_write_ptr(cb_hdr) + 992 * 4, t_hist.get_noc_addr(page), 64);
                        if (!wbar(23)) { run = false; break; }
                    }
                }
                if (outer) ++nrow;
                break;
            }
            case B_FETCH: {
                // n tiles from the constant's pages: core * CP + its offset + t
                const uint32_t tile = B[pc], base = core * CP + B[pc + 1], n = B[pc + 2] & 0xffff;
                pc += 3;
                ++nfetch;
                for (uint32_t t = 0; t < n; ++t)
                    noc_async_read(t_const.get_noc_addr(base + t), arena + (tile + t) * 4096, 4096);
                if (!rbar(24)) { run = false; break; }
                break;
            }
            case B_SINK: {
                // drain compute's stores of this block into their DRAM homes: per tile, per store
                const uint32_t n = B[pc] & 0xffff, tn = B[pc] >> 16 ? B[pc] >> 16 : NT;   // tn: tiles per store
                const uint32_t* dsc = B + pc + 1;
                pc += 1 + n;
                // every page compute has ready, up to the fifo's end (a pop may land on it but not cross it), goes out
                // before one flush (store i is tile i / n, store i % n)
                for (uint32_t i = 0, ns = tn * n; i < ns && run;) {
                    if (!WAIT_UNTIL(cb_pages_available_at_front(cb_out, 1), 28)) { run = false; break; }
                    uint32_t rp = get_read_ptr(cb_out), k = 1;
                    const uint32_t km = (out1 - rp) / 4096;
                    while (k < km && i + k < ns && cb_pages_available_at_front(cb_out, k + 1)) ++k;
                    cb_wait_front(cb_out, k);
                    for (uint32_t j = 0; j < k; ++j, ++i) {
                        // bit 27 (encode.NARROW): compute packed a bf16 page (C_STWH), its first 2048 B are the page
                        const uint32_t dd = dsc[i % n];
                        noc_async_write(rp, t_home.get_noc_addr(core * HP + (dd & 0x7ffffff) + i / n, 0),
                                        (dd >> 27) & 1 ? 2048 : 4096);
                        rp += 4096;
                    }
                    noc_async_writes_flushed();
                    cb_pop_front(cb_out, k);
                    nsink += k;
                }
                if (run && !wbar(29)) run = false;
                break;
            }
#if !defined(CFD_MESH) || defined(CFD_BSCAL)   // a mesh build carries the soft-float only with --bscal 1 (kernel config buffer)
            case B_SCAL: {   // bscal: a scalar block on lane 0 in soft-float, the compute ops' encoding (encode.py)
                const uint32_t end = pc + 1 + B[pc];
                ++pc;
                float D[8];
                while (pc < end) {
                    const uint32_t u = B[pc++];
                    const uint32_t a = (u >> 8) & 0xf, b = (u >> 12) & 0xf, c = (u >> 16) & 0xf, d = (u >> 20) & 0xf;
                    switch (u & 0x7f) {
                        case S_LD: D[a] = fbits(A[B[pc++] * 1024]); break;
                        case S_FILL: D[a] = fbits(B[pc++]); break;
                        case S_ST: A[B[pc++] * 1024] = ubits(D[a]); break;
                        case S_ADD: D[a] = D[b] + D[c]; break;
                        case S_SUB: D[a] = D[b] - D[c]; break;
                        case S_MUL: D[a] = D[b] * D[c]; break;
                        case S_DIV: D[a] = D[b] / D[c]; break;
                        case S_MAX: D[a] = D[b] > D[c] ? D[b] : D[c]; break;
                        case S_MIN: D[a] = D[b] < D[c] ? D[b] : D[c]; break;
                        case S_WHERE: D[a] = D[b] != 0.f ? D[c] : D[d]; break;
                        case S_CPD: D[b] = D[a]; break;
                        case S_NEG: D[a] = -D[a]; break;
                        case S_ABS: D[a] = D[a] < 0.f ? -D[a] : D[a]; break;
                        case S_GTZ: D[a] = D[a] > 0.f ? 1.f : 0.f; break;
                        case S_GEZ: D[a] = D[a] >= 0.f ? 1.f : 0.f; break;
                        case S_LTZ: D[a] = D[a] < 0.f ? 1.f : 0.f; break;
                        case S_LEZ: D[a] = D[a] <= 0.f ? 1.f : 0.f; break;
                        case S_EQZ: D[a] = D[a] == 0.f ? 1.f : 0.f; break;
                        case S_NEZ: D[a] = D[a] != 0.f ? 1.f : 0.f; break;
                        case S_MAD: D[a] = D[d] + D[b] * D[c]; break;
                        case S_MSB: D[a] = D[d] - D[b] * D[c]; break;
                        default: hdr[20] = 0xBAD10000u | u; fail(9); run = false; pc = end; break;
                    }
                }
                break;
            }
#endif
            case B_PH:
#ifdef CFD_PHASE
            {
                const uint64_t t_ = get_timestamp();
                phc[phid] += t_ - pht;
                pht = t_;
                phid = (w >> 8) & 15;
            }
#endif
                break;
            case B_JMP: pc = B[pc]; break;
            case B_BRT: {
                const uint32_t v = A[B[pc] * 1024], target = B[pc + 1];
                pc += 2;
                if (((v & 0x7fffffffu) != 0) != ((w >> 8) & 1)) pc = target;   // bit 8: taken on a zero flag
                break;
            }
            default:
                hdr[20] = 0xBAD00000u | op;
                fail(9);
                run = false;
                break;
        }
#ifdef CFD_PHASE
        if (pho) {
            const uint64_t d_ = get_timestamp() - pho;
            if (op == B_RED) phr[phid] += d_;
            else if (op == B_HALO) phh[phid] += d_;
            else phw[phid] += d_;
        }
#endif
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
        hdr[64 + i] = static_cast<uint32_t>(cyc[i] >> 4);
        hdr[80 + i] = cnt[i];
    }
#endif
#ifdef CFD_PHASE
    phc[phid] += get_timestamp() - pht;
    for (uint32_t i = 0; i < 16; ++i) {
        hdr[544 + 2 * i] = static_cast<uint32_t>(phc[i]);
        hdr[545 + 2 * i] = static_cast<uint32_t>(phc[i] >> 32);
        hdr[576 + 2 * i] = static_cast<uint32_t>(phr[i]);
        hdr[577 + 2 * i] = static_cast<uint32_t>(phr[i] >> 32);
        hdr[608 + 2 * i] = static_cast<uint32_t>(phh[i]);
        hdr[609 + 2 * i] = static_cast<uint32_t>(phh[i] >> 32);
        hdr[640 + 2 * i] = static_cast<uint32_t>(phw[i]);
        hdr[641 + 2 * i] = static_cast<uint32_t>(phw[i] >> 32);
    }
#endif
    const uint32_t t1 = get_timestamp_32b();
    hdr[58] = (get_timestamp() - T0) >> 32;
    // drain until compute is done: feed compute's and NCRISC's waits, swallow their signals
    if (S[W_ABORT] || hdr[30]) {
        const uint32_t s0 = get_timestamp_32b();
        while (!cb_pages_available_at_front(cb_done, 1) && get_timestamp_32b() - s0 < 4 * limit) {
            BSVC();
            if (cb_pages_available_at_front(cb_cb, 1)) { cb_wait_front(cb_cb, 1); cb_pop_front(cb_cb, 1); }
            if (cb_pages_reservable_at_back(cb_bc, 1)) { cb_reserve_back(cb_bc, 1); cb_push_back(cb_bc, 1); }
            if (cb_pages_available_at_front(cb_nb, 1)) { cb_wait_front(cb_nb, 1); cb_pop_front(cb_nb, 1); }
            if (cb_pages_reservable_at_back(cb_bn, 1)) { cb_reserve_back(cb_bn, 1); cb_push_back(cb_bn, 1); }
            if (cb_pages_available_at_front(cb_out, 1)) { cb_wait_front(cb_out, 1); cb_pop_front(cb_out, 1); }
        }
    }
    cb_wait_front(cb_done, 1);
    S[W_CDONE] = 1;
    {
        const uint32_t s0 = get_timestamp_32b();
        while (S[W_NDONE] == 0 && get_timestamp_32b() - s0 < 4 * limit) {
            invalidate_l1_cache();
            if (cb_pages_available_at_front(cb_nb, 1)) { cb_wait_front(cb_nb, 1); cb_pop_front(cb_nb, 1); }
            if (cb_pages_reservable_at_back(cb_bn, 1)) { cb_reserve_back(cb_bn, 1); cb_push_back(cb_bn, 1); }
        }
    }
    const uint32_t t2 = get_timestamp_32b();
    {
        const uint32_t s_ = get_timestamp_32b();
        while (!ncrisc_noc_nonposted_atomics_flushed(noc_index) && get_timestamp_32b() - s_ < limit) {
        }
    }
#ifdef CFD_MESH
    hdr[57] |= 4;
    for (uint32_t d = 0; d < 4; ++d) if (hasd[d]) shut(conn[d], 46 + d);
    if (gw) shut(gconn, 50);
#endif
    hdr[10] = t1 - t0;
    hdr[11] = t2 - t0;
    hdr[12] = nops;
    hdr[13] = nred;
    hdr[14] = nrow;
    hdr[15] = nfetch;
    hdr[17] = nhalo;
    hdr[19] = nsink;
    hdr[18] = 0xB0DE0000u;
    hdr[34] = S[W_ABORT];
    const uint64_t dst = t_out.get_noc_addr(core);
    noc_async_write(get_write_ptr(cb_hdr), dst, CFD_HDRB);
    noc_async_write(arena, dst + CFD_HDRB, abytes);
    wbar(25);
}
