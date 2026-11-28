// Resident VM, compute side (the three TRISCs). BRISC loads the encoded compute stream into cb_code and the arena
// into cb_v; UNPACK takes the stream's address with get_tile_address (mailboxed to MATH and PACK) and all three
// threads decode the same stream, each doing its own part of every op. Opcodes and word format: ttres/encode.py.
//
// Cross-agent order goes through one-page token CBs: a signal is a push on PACK (after the block's packs have
// landed, llk_push_to_brisc stalls on the packer), a wait is a pop on UNPACK. TOK orders PACK before UNPACK inside
// compute (a packed tile is read back). Values that steer control flow (RVF, BRT) come from read_tile_value, read on
// UNPACK and mailboxed, so the three threads always agree. Each branch also reads the sync region's ABORT word
// the same way and stops if a data mover gave up (they then feed every token compute still waits for).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/addcmul.h"
#ifdef CFD_BLK
#include "blk_exec.h"   // generated SFPU bodies (resident/blockgen.py): blk_init, blk_run
#endif
#if defined(CFD_TGTS) || defined(CFD_DU) || defined(CFD_PU)
#define CFD_TMBOX 1
#endif
#if defined(CFD_DENSE) && ((defined(CFD_TMBOX) && defined(TRISC_UNPACK)) || \
    (defined(CFD_DMP_M) && defined(TRISC_MATH)) || (defined(CFD_DMP_P) && defined(TRISC_PACK)))
#include "vm_gather.h"
#endif

#if defined(CFD_TMBOX) && defined(TRISC_UNPACK)
// Out-of-line UNPACK gather helper at -O3: TWAIT only checks the sequence word inline.
static __attribute__((noinline, optimize("O3"))) void cfd_tsvc_run(volatile uint32_t* mb, uint32_t q) {
    for (uint32_t j = 0, nj = mb[8]; j < nj; ++j) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(mb[12 + 4 * j]);
        const uint32_t* xs = reinterpret_cast<const uint32_t*>(mb[12 + 4 * j + 1]);
        const uint32_t n = mb[12 + 4 * j + 2];
        uint32_t* out = reinterpret_cast<uint32_t*>(mb[12 + 4 * j + 3]);
#ifdef CFD_DENSE
        if (n >> 31) {   // lanes of a dense gather tile (vm_ncrisc.cpp): out[e] = src[ix[e]]
            cfd_gather16(src, reinterpret_cast<const uint16_t*>(xs), n & 0x7fffffffu, out);
            continue;
        }
#endif
        uint32_t e = 0;
        for (; e + 4 <= n; e += 4) {
            const uint32_t x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];
            const uint32_t v0 = src[x0 & 0xffff], v1 = src[x1 & 0xffff], v2 = src[x2 & 0xffff], v3 = src[x3 & 0xffff];
            out[x0 >> 16] = v0;
            out[x1 >> 16] = v1;
            out[x2 >> 16] = v2;
            out[x3 >> 16] = v3;
        }
        for (; e < n; ++e) out[xs[e] >> 16] = src[xs[e] & 0xffff];
    }
    asm volatile("" ::: "memory");
    mb[4] = q;
}
#endif

#if defined(CFD_DENSE) && ((defined(CFD_DMP_M) && defined(TRISC_MATH)) || (defined(CFD_DMP_P) && defined(TRISC_PACK)))
// Dense-only helper for MATH (trisc1) and PACK (trisc2) on the 64-B DMB mailbox below TMB.
static __attribute__((noinline, optimize("O3"))) void cfd_dsvc_run(
    volatile uint32_t* dmb, uint32_t off_idx, uint32_t done_idx, uint32_t q) {
    const uint32_t spec = dmb[off_idx];
    const uint32_t len = spec >> 16;
    if (len) {
        const uint32_t off = spec & 0xffffu;
        cfd_gather16(reinterpret_cast<const uint32_t*>(dmb[0]),
                     reinterpret_cast<const uint16_t*>(dmb[1]) + off,
                     len,
                     reinterpret_cast<uint32_t*>(dmb[2]) + off);
    }
    asm volatile("" ::: "memory");
    dmb[done_idx] = q;
}
#endif

enum : uint32_t {
    C_END = 0, C_ACQ, C_COMMIT, C_REL, C_LD, C_ST, C_FILL, C_RVF,
    C_ADD, C_SUB, C_MUL, C_DIV, C_MAX, C_MIN, C_WHERE, C_CPD,
    C_NEG, C_ABS, C_SQRT, C_TANH, C_LOG, C_EXP, C_GTZ, C_GEZ, C_LTZ, C_LEZ, C_EQZ, C_NEZ,
    C_FOLD, C_TOK, C_SIGB, C_SIGN, C_WAITB, C_WAITN, C_JMP, C_BRT, C_LDS, C_TBEG, C_TEND, C_LDR, C_STW, C_MAD, C_MSB, C_LDRH, C_STWH, C_PH, C_BLK
};
constexpr uint32_t TILE = 0x80;   // op bit: the tile operand is offset by the tile loop counter

void kernel_main() {
    if (get_common_arg_val<uint32_t>(0) == 0) return;   // limit 0: run_mesh --l1-check, nothing runs
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_code = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tok = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cb = get_compile_time_arg_val(3);    // C -> B
    constexpr uint32_t cb_bc = get_compile_time_arg_val(4);    // B -> C
    constexpr uint32_t cb_cn = get_compile_time_arg_val(5);    // C -> N
    constexpr uint32_t cb_nc = get_compile_time_arg_val(6);    // N -> C
    constexpr uint32_t cb_done = get_compile_time_arg_val(7);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(8);
    constexpr uint32_t nv = get_compile_time_arg_val(9);
    constexpr uint32_t cb_sync = get_compile_time_arg_val(10);
    constexpr uint32_t cb_in = get_compile_time_arg_val(11);     // reader -> compute ring (DRAM-homed loads)
    constexpr uint32_t cb_out = get_compile_time_arg_val(12);    // compute -> writer ring (DRAM-homed stores)
    constexpr uint32_t W_ABORT = 1004;   // word of the sync region a giving-up BRISC/NCRISC raises (vm_brisc.cpp)
    // paged code (encode.py page=True): cb_code is a ring of CR pages of PW words NCRISC fills in the order the code
    // runs; pc is virtual (page k of it = the k-th delivered page). Entering a new page: UNPACK pops the previous one,
    // waits for it and publishes the count in sync word W_CUNP; MATH publishes its own entries in W_CMATH; pages
    // behind pc (and behind an open tile loop) are released by PACK once MATH is past them, through W_CREL, which
    // NCRISC reads. L1 words, not mailboxes: the mailboxes carry RVF/BRT values and must not fill up. JMP and BRT end
    // a basic block: the next one starts on the next page. CR = 0: BRISC loads the whole stream (one page).
    constexpr uint32_t CR = get_compile_time_arg_val(13), PW = 256, W_CREL = 1012, W_CUNP = 1016, W_CMATH = 1020;

    init_sfpu(cb_v, cb_v);
    if constexpr (CR == 0) cb_wait_front(cb_code, 1);
    cb_wait_front(cb_v, nv);
    const volatile uint32_t* P = reinterpret_cast<const volatile uint32_t*>(get_tile_address(cb_code, 0));
    volatile uint32_t* sw = reinterpret_cast<volatile uint32_t*>(get_tile_address(cb_sync, 0));
    uint32_t hi = 0, rel = 0;   // paged: pages acquired (virtual pc below hi * PW is readable), pages released
    volatile uint32_t* hw = reinterpret_cast<volatile uint32_t*>(get_tile_address(cb_hdr, 0));
    uint32_t pc = 0, nops = 0, taken = 0;
#ifdef CFD_TMBOX
    // T-gather mailbox (vm_ncrisc.cpp, the 256 B of cb_sync before the B-patch one): NCRISC posts UNPACK's
    // share of each GATHER/STREAM tile (src, entries, n, page) and bumps SEQ; UNPACK copies them from
    // inside every wait on a data mover or PACK and answers DONE = SEQ. Disjoint lanes: the same values, bit-exact
    constexpr uint32_t TMB = CFD_SYNCB / 4 - 128, T_SEQ = TMB;
    UNPACK(uint32_t tseen = 0;)
#define TSVC() UNPACK(({ const uint32_t q_ = sw[T_SEQ]; if (q_ != tseen) { cfd_tsvc_run(sw + TMB, q_); tseen = q_; } }))
    // UNPACK spins on TSVC until cb has a page; the cb_wait_front after it then returns at once. Every CB wait of UNPACK
    // is one of these; a TRISC stuck behind a PACK that waits on a full cb_cn would not be, and NCRISC then aborts (22)
#define TWAIT(cb) UNPACK(({ while (static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(cb))) - get_local_cb_interface(cb).tiles_acked) == 0) TSVC(); }))
#else
#define TWAIT(cb)
#endif
#ifdef CFD_DMP
    // 64-B dense mailbox below TMB: NCRISC posts (src, ix, out) and bumps D_MSEQ/D_PSEQ; MATH and PACK spin on
    // D_USYNC at every UNPACK external wait (C_WAITB, C_WAITN, C_LDR, C_LDRH) and during paged-code/CB-free waits
    // so they service dense gathers in parallel without advancing into hardware stalls.
    constexpr uint32_t DMB = CFD_SYNCB / 4 - 144;
    enum : uint32_t { D_MSEQ = DMB + 4, D_PSEQ = DMB + 8, D_USYNC = DMB + 12 };
    UNPACK(uint32_t usync = 0;)
#define USYNC() UNPACK(({ asm volatile("" ::: "memory"); sw[D_USYNC] = ++usync; }))
#ifdef CFD_DMP_M
    MATH(uint32_t usync = 0;)
    MATH(uint32_t mseen = 0;)
#define MSVC() MATH(({ const uint32_t q_ = sw[D_MSEQ]; if (q_ != mseen) { cfd_dsvc_run(sw + DMB, 5, 6, q_); mseen = q_; } }))
#define MWAIT_U() MATH(({ ++usync; while (static_cast<int32_t>(sw[D_USYNC] - usync) < 0) MSVC(); }))
#else
#define MSVC()
#define MWAIT_U()
#endif
#ifdef CFD_DMP_P
    PACK(uint32_t usync = 0;)
    PACK(uint32_t pseen = 0;)
#define PSVC() PACK(({ const uint32_t q_ = sw[D_PSEQ]; if (q_ != pseen) { cfd_dsvc_run(sw + DMB, 9, 10, q_); pseen = q_; } }))
#define PWAIT_U() PACK(({ ++usync; while (static_cast<int32_t>(sw[D_USYNC] - usync) < 0) PSVC(); }))
#define PWAIT(cb) PACK(({ \
    const uint16_t rec_ = get_local_cb_interface(cb).tiles_received; \
    const uint16_t cap_ = get_local_cb_interface(cb).fifo_num_pages; \
    while (static_cast<uint16_t>(cap_ - (rec_ - static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_acked_ptr(cb)))))) == 0) PSVC(); \
}))
#else
#define PSVC()
#define PWAIT_U()
#define PWAIT(cb)
#endif
#else
#define USYNC()
#define MSVC()
#define MWAIT_U()
#define PSVC()
#define PWAIT_U()
#define PWAIT(cb)
#endif
    uint32_t tl = 0, tn = 0, tpc = 0;   // tile loop: counter, count, first op of the body
    bool run = true;
#ifdef CFD_PROF
    // census: per-op wall-clock cycles of this thread (time from one op's issue to the next's) and op counts
    uint64_t cyc[40];   // stored in 16-cycle units (>> 4): a 100-it mesh run is ~2^35 cycles
    uint32_t cnt[40];
#pragma GCC unroll 1
    for (uint32_t i = 0; i < 40; ++i) cyc[i] = cnt[i] = 0;
    uint32_t tprev = static_cast<uint32_t>(ckernel::read_wall_clock());
#endif
#ifdef CFD_PROFW
    // run_mesh --prof 2: who MATH waits for. UNPACK and PACK publish what they wait in (hdr 304 / 305: 0 at work; UNPACK
    // 1 a reader page, 2 NCRISC's token, 3 BRISC's token, 4 PACK's token, 5 a code page; PACK 1 a free writer page, 2 a
    // code page, 3 / 4 a free token page to BRISC / NCRISC), NCRISC its op and part (hdr 306, vm_ncrisc.cpp). MATH drains
    // its Tensix pipeline before a load or an acquire (its own queued work), issues it, and drains again: the post of
    // that drain lands when the unpacker (the packer) has let the instruction through, and each poll's cycles go to the
    // state published at that poll: loads to hdr 700 + 2 (16 unpack state + reader class), acquires to 960 + 2 pack
    // state, own work to 976, the after-drains to 978 (load) and 952 (acquire), the last poll of each drain kind to
    // 980 + 2 kind, of the load's poll to 986, of the acquire's to 954 (two words each: low, high); counts at 988..990. Nothing from 992 on: BRISC stages its history row there (16 words)
#define USTATE(s) UNPACK((hw[304] = (s)))
#define PSTATE(s) PACK((hw[305] = (s)))
#ifdef TRISC_MATH
#pragma GCC unroll 1
    for (uint32_t i = 700; i < 992; ++i) hw[i] = 0;
    auto wadd = [&](uint32_t i, uint32_t dt) __attribute__((noinline, noclone, optimize("Os"))) {   // out of line, as the three below: one copy each, the kernel config buffer is 69 KB
        const uint32_t lo = hw[i], nl = lo + dt;
        hw[i] = nl;
        if (nl < lo) hw[i + 1] = hw[i + 1] + 1;
    };
    auto ncls = [&]() -> uint32_t {   // the reader's class: the part it is in, else its op (32 + op)
        const uint32_t q = hw[306];
        switch (q) {
            case 10: return 9;        // STREAM waits for a free cb_in slot
            case 11: return 8;        // STREAM's DRAM read barrier
            case 16: return 4;        // the patch's barrier (entries read from DRAM)
            case 18: return 5;        // GATHER waits for the halo
            case 19: return 6;        // GATHER's scatters
            case 32 + 2: return 1;    // WAITC
            case 32 + 4: return 2;    // WAITB
            case 32 + 5: return 3;    // GATHER outside its parts
            case 32 + 9: return 7;    // STREAM outside its parts
            case 32 + 6: return 10;   // HALO
            default: return q ? (q < 32 ? 12 : 11) : 0;
        }
    };
    // kind 0: after a load (cross-check of mload's poll, to 978: only the get and the four clears are left), 1: after an
    // acquire (cross-check of macq's poll, to 952: nothing is left), 2: own work (976)
    auto mdrain = [&](uint32_t kind) __attribute__((noinline, noclone, optimize("Os"))) {
        uint32_t tp = static_cast<uint32_t>(ckernel::read_wall_clock());
        ckernel::t6_semaphore_wait_on_max<ckernel::p_stall::STALL_SYNC>(ckernel::semaphore::MATH_DONE);
        ckernel::t6_semaphore_post<ckernel::p_stall::MATH | ckernel::p_stall::WAIT_SFPU>(ckernel::semaphore::MATH_DONE);
        while (ckernel::semaphore_read(ckernel::semaphore::MATH_DONE) == 0) {
            const uint32_t i = kind == 0 ? 978 : kind == 1 ? 952 : 976;
            const uint32_t t = static_cast<uint32_t>(ckernel::read_wall_clock());
            wadd(i, t - tp);
            tp = t;
        }
        ckernel::semaphore_get(ckernel::semaphore::MATH_DONE);
        wadd(980 + 2 * kind, static_cast<uint32_t>(ckernel::read_wall_clock()) - tp);
        hw[988 + kind] = hw[988 + kind] + 1;
    };
    // an fp32 load's MATH side, as the unpack-to-dest branch of _llk_math_eltwise_unary_datacopy_, with the wait for the
    // unpacker polled on the RISC-V between the mailbox write UNPACK blocks on and the semaphore get: nothing is queued
    // then, so nothing blocks ahead of the poll, and each poll's cycles go to 700 + 2 (16 unpack state + reader class);
    // the last poll to 986
    auto mload = [&](uint32_t dst) __attribute__((noinline, noclone, optimize("Os"))) {
        mdrain(2);
        ckernel::math::math_unpack_to_dest_math_ready();
        ckernel::math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::DestReg>(dst);
        uint32_t tp = static_cast<uint32_t>(ckernel::read_wall_clock());
        while (ckernel::semaphore_read(ckernel::semaphore::UNPACK_TO_DEST) == 0) {
            const uint32_t u = hw[304] & 7;   // at BRISC's token (3) the class is the writer's op (1 + op, word 956; 13..15 = GATHER, SCAL, PH), else the reader's
            const uint32_t qb = hw[956];
            const uint32_t i = 700 + 2 * (16 * u + (u == 3 ? (qb <= 12 ? qb : qb == 16 ? 13 : qb == 17 ? 14 : 15) : ncls()));
            const uint32_t t = static_cast<uint32_t>(ckernel::read_wall_clock());
            wadd(i, t - tp);
            tp = t;
        }
        wadd(986, static_cast<uint32_t>(ckernel::read_wall_clock()) - tp);
        ckernel::math::math_unpack_to_dest_tile_ready();
        const uint32_t lt = dst & 3;   // fp32 dest: four tiles a bank
        for (uint32_t f = 0; f < 4; ++f) TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, ckernel::math::get_dest_index_in_faces(lt, f));
        mdrain(0);
    };
    // an acquire's wait for the packer, polled on the RISC-V before the acquire is issued: MATH_PACK counts the dest
    // sections committed and not yet released, the acquire stalls while it is at its maximum (2 with SyncHalf, 1 with
    // SyncFull). Each poll's cycles go to 960 + 2 pack state, the last poll to 954
    auto macq = [&]() __attribute__((noinline, noclone, optimize("Os"))) {
        constexpr uint32_t mpmax = (DST_SYNC_MODE == DstSync::SyncFull) ? 1 : 2;
        mdrain(2);
        uint32_t tp = static_cast<uint32_t>(ckernel::read_wall_clock());
        while (ckernel::semaphore_read(ckernel::semaphore::MATH_PACK) >= mpmax) {
            const uint32_t i = 960 + 2 * (hw[305] & 7);
            const uint32_t t = static_cast<uint32_t>(ckernel::read_wall_clock());
            wadd(i, t - tp);
            tp = t;
        }
        wadd(954, static_cast<uint32_t>(ckernel::read_wall_clock()) - tp);
    };
#define MDRAIN(k) MATH((mdrain(k)))
#define MACQ() MATH((macq()))
#else
#define MDRAIN(k)
#define MACQ()
#endif
#else
#define USTATE(s)
#define PSTATE(s)
#define MDRAIN(k)
#define MACQ()
#endif
#ifdef CFD_PROFW
#define COPY_TILE(cb, i, d) do { UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(cb, i))); MATH((mload(d))); } while (0)
#else
#define COPY_TILE(cb, i, d) copy_tile(cb, i, d)
#endif
#define COPY_TILE_N(cb, i, d) do { MDRAIN(2); copy_tile(cb, i, d); MDRAIN(0); } while (0)   // a bf16 source: not unpack-to-dest
#ifdef CFD_PHASE
    // run_mesh --phase: wall-clock cycles of this thread per SIMPLE phase (C_PH opens phase id = w >> 16; id 15 = before
    // the first marker), 64-bit, kept by MATH only and written to hdr 512 + 2 id (low word) / 513 + 2 id (high word)
    uint64_t phc[16];
    for (uint32_t i = 0; i < 16; ++i) phc[i] = 0;
    uint32_t phid = 15;
    uint64_t pht = ckernel::read_wall_clock();
#endif
#ifdef CFD_TRACE
// run_vm --trace N: agent g (0 unpack, 1 math, 2 pack, 3 BRISC, 4 NCRISC) records its first N ops issued at least
// CFD_TRACE_AT cycles after its own start, at hdr word 1024 + g * (2N + 2): [count, start, (pc | op << 24, issue tick)...]
#define TR_REC(pc_, op_, t_)                                                         \
    if (trn_ < CFD_TRACE && (t_) - trt0_ >= CFD_TRACE_AT) {                          \
        tr_[2 + 2 * trn_] = (pc_) | ((op_) << 24);                                   \
        tr_[3 + 2 * trn_] = (t_);                                                    \
        tr_[0] = ++trn_;                                                             \
    }
#endif
#define CW(i) P[CR ? ((i) & (CR * PW - 1)) : (i)]
#ifdef CFD_ICACHE
    // init cache (run_vm --icache): an op's _init only when the last init of its kind was a different one. Loads key on
    // the source CB; SFPU inits on the op (exp, div, ... program shared SFPU constants, so any other op re-inits)
    uint32_t ci = 0xffffffffu, si = 0xffffffffu;
#define LDI(cb) do { if (ci != (cb)) { copy_tile_to_dst_init_short(cb); ci = (cb); } } while (0)
#define LDX() (ci = 0xffffffffu)
#define SFI(k, init) do { if (si != (k)) { init; si = (k); } } while (0)
#else
#define LDI(cb) copy_tile_to_dst_init_short(cb)
#define LDX()
#define SFI(k, init) init
#endif
#ifdef CFD_TRACE
    uint32_t trg_ = 0;
    UNPACK(trg_ = 0);
    MATH(trg_ = 1);
    PACK(trg_ = 2);
    volatile uint32_t* tr_ = hw + 1024 + trg_ * (2 * CFD_TRACE + 2);
    uint32_t trn_ = 0;
    const uint32_t trt0_ = static_cast<uint32_t>(ckernel::read_wall_clock());
    tr_[0] = 0;
    tr_[1] = trt0_;
#endif
    while (run) {
        if constexpr (CR != 0) {
            if (pc >= hi * PW) {   // entering page hi (pc is at its start: ops never straddle pages)
                UNPACK(({
                    if (hi) cb_pop_front(cb_code, 1);
                    USTATE(5);
                    TWAIT(cb_code);
                    cb_wait_front(cb_code, 1);
                    USTATE(0);
                    sw[W_CUNP] = hi + 1;
                }));
                MATH(({
                    while (sw[W_CUNP] <= hi) {
                        MSVC();
                    }
                    sw[W_CMATH] = hi + 1;
                }));
                PACK(({
                    PSTATE(2);
                    while (sw[W_CUNP] <= hi) {
                        PSVC();
                    }
                    PSTATE(0);
                }));
                ++hi;
                const uint32_t r = (tl < tn) ? tpc / PW : pc / PW;   // an open tile loop keeps its body
                if (r > rel) {
                    rel = r;
                    PACK(({
                        while (sw[W_CMATH] < hi) {   // MATH has entered this page, so it left the released ones
                            PSVC();
                        }
                        sw[W_CREL] = r;
                    }));
                }
            }
        }
        const uint32_t w = CW(pc);
        ++pc;
        const uint32_t op = w & 0x7f, to = (w & TILE) ? tl : 0, a = (w >> 8) & 0xf, b = (w >> 12) & 0xf, c = (w >> 16) & 0xf, d = (w >> 20) & 0xf;
        ++nops;
#ifdef CFD_TRACE
        {
            const uint32_t t_ = static_cast<uint32_t>(ckernel::read_wall_clock());
            TR_REC(pc - 1, op, t_);
        }
#endif
        switch (op) {
            case C_END: run = false; break;
            case C_ACQ: MACQ(); tile_regs_acquire(); MDRAIN(1); break;
            case C_COMMIT: tile_regs_commit(); tile_regs_wait(); break;
            case C_REL: tile_regs_release(); break;
            case C_LD: LDI(cb_v); COPY_TILE(cb_v, (w >> 16) + to, a); break;
            case C_LDS: {   // offset load: the unpacker reads the tile from a 16-byte-unit shifted L1 address
                const int32_t units = static_cast<int32_t>(CW(pc));
                ++pc;
                LDI(cb_v);
                UNPACK((get_local_cb_interface(cb_v).fifo_rd_ptr += units));
                COPY_TILE(cb_v, (w >> 16) + to, a);
                UNPACK((get_local_cb_interface(cb_v).fifo_rd_ptr -= units));
                break;
            }
            case C_ST: pack_tile<true>(a, cb_v, (w >> 16) + to); break;
            case C_LDR:   // next reader page; the tile starts b 16-B units in (the DRAM source's 32-B phase)
                USTATE(1);
                TWAIT(cb_in);
                cb_wait_front(cb_in, 1);
                USTATE(0);
                USYNC();
                MWAIT_U();
                PWAIT_U();
                LDI(cb_in);
                UNPACK((get_local_cb_interface(cb_in).fifo_rd_ptr += b));
                COPY_TILE(cb_in, 0, a);
                UNPACK((get_local_cb_interface(cb_in).fifo_rd_ptr -= b));
                cb_pop_front(cb_in, 1);
                break;
#ifdef CFD_NARROW
            case C_LDRH:   // C_LDR of a bf16 page: CB CFD_NARROW (bfloat16) views cb_in's front page, srcA widens it to fp32
                USTATE(1);
                TWAIT(cb_in);
                cb_wait_front(cb_in, 1);
                USTATE(0);
                USYNC();
                MWAIT_U();
                PWAIT_U();
                copy_tile_to_dst_init_short_with_dt(cb_in, CFD_NARROW);
                UNPACK((get_local_cb_interface(CFD_NARROW).fifo_rd_ptr = get_local_cb_interface(cb_in).fifo_rd_ptr + b));
                COPY_TILE_N(CFD_NARROW, 0, a);
                copy_tile_to_dst_init_short_with_dt(CFD_NARROW, cb_in);
                LDX();
                cb_pop_front(cb_in, 1);
                break;
#endif
#ifdef CFD_NARROW_OUT
            case C_STWH:   // C_STW packed to bf16: CB CFD_NARROW_OUT (bfloat16) views cb_out's back page
                PSTATE(1);
                PWAIT(cb_out);
                cb_reserve_back(cb_out, 1);
                PSTATE(0);
                PACK((get_local_cb_interface(CFD_NARROW_OUT).fifo_wr_ptr = get_local_cb_interface(cb_out).fifo_wr_ptr));
                pack_reconfig_data_format(cb_out, CFD_NARROW_OUT);
                pack_tile<true>(a, CFD_NARROW_OUT, 0);
                pack_reconfig_data_format(CFD_NARROW_OUT, cb_out);
                cb_push_back(cb_out, 1);
                break;
#endif
            case C_STW:
                PSTATE(1);
                PWAIT(cb_out);
                cb_reserve_back(cb_out, 1);
                PSTATE(0);
                pack_tile(a, cb_out);
                cb_push_back(cb_out, 1);
                break;
            case C_PH:
#ifdef CFD_PHASE
                MATH(({
                    const uint64_t t_ = ckernel::read_wall_clock();
                    phc[phid] += t_ - pht;
                    pht = t_;
                    phid = (w >> 16) & 15;
                }));
#endif
                break;
#ifdef CFD_BLK
#ifdef CFD_BLKTAB
            case C_BLK:     // a generated body, table form: w >> 16 is the word offset of its program (header, config
                // words, row program; blk_exec.h) in BLK_TAB, the table of this lowering's distinct programs that
                // run_mesh --blktab compiles into the MATH kernel (blk_tab.h, included ahead of this file)
                MATH(({
                    SFI(C_BLK, blk_init());
                    blk_run(&BLK_TAB[w >> 16]);
                }));
                break;
#else
            case C_BLK: {   // a generated body: the n words after this one are its program (header, config words, row
                // program; blk_exec.h). MATH pushes them, UNPACK and PACK step over them. An op never straddles a page
                const uint32_t n = w >> 16;
                MATH(({
                    SFI(C_BLK, blk_init());
                    blk_run(const_cast<const uint32_t*>(&P[CR ? (pc & (CR * PW - 1)) : pc]));
                }));
                pc += n;
                break;
            }
#endif
#endif
            case C_TBEG: tn = w >> 16; tl = 0; tpc = pc; break;
            case C_TEND: if (++tl < tn) pc = tpc; break;
            case C_FILL: SFI(C_FILL, fill_tile_init()); fill_tile_bitcast(a, CW(pc)); ++pc; break;
            case C_RVF: {
                const uint32_t v = read_tile_value(cb_v, w >> 16, 0);
                SFI(C_FILL, fill_tile_init());
                fill_tile_bitcast(a, v);
                break;
            }
            // binary: a = b op c
            case C_ADD: SFI(C_ADD, add_binary_tile_init()); add_binary_tile(b, c, a); break;
            case C_SUB: SFI(C_SUB, sub_binary_tile_init()); sub_binary_tile(b, c, a); break;
            case C_MUL: SFI(C_MUL, mul_binary_tile_init()); mul_binary_tile(b, c, a); break;
            case C_DIV: SFI(C_DIV, div_binary_tile_init()); div_binary_tile(b, c, a); break;
            case C_MAX: SFI(C_MAX, binary_max_tile_init()); binary_max_tile(b, c, a); break;
            case C_MIN: SFI(C_MIN, binary_min_tile_init()); binary_min_tile(b, c, a); break;
            // a = b != 0 ? c : d
            case C_WHERE: SFI(C_WHERE, where_tile_init()); where_tile<DataFormat::Float32>(b, c, d, a); break;
            // a = d +- b c in one SFPU pass (addcmul: d + value b c, value = +-1 is exact)
            case C_MAD: SFI(C_MAD, addcmul_tile_init()); addcmul_tile<DataFormat::Float32>(d, b, c, a, 0x3f800000u); break;
            case C_MSB: SFI(C_MAD, addcmul_tile_init()); addcmul_tile<DataFormat::Float32>(d, b, c, a, 0xbf800000u); break;
            case C_CPD: SFI(C_CPD, copy_dest_values_init()); copy_dest_values<DataFormat::Float32>(a, b); break;
            case C_NEG: SFI(C_NEG, negative_tile_init()); negative_tile(a); break;
            case C_ABS: SFI(C_ABS, abs_tile_init()); abs_tile(a); break;
            case C_SQRT: SFI(C_SQRT, sqrt_tile_init()); sqrt_tile<false>(a); break;
            case C_TANH: SFI(C_TANH, tanh_tile_init<false>()); tanh_tile<false>(a); break;
            case C_LOG: SFI(C_LOG, log_tile_init<false>()); log_tile<false>(a); break;
            case C_EXP: SFI(C_EXP, (exp_tile_init<false, false>())); exp_tile<false, false>(a); break;
            case C_GTZ: SFI(C_GTZ, gtz_tile_init()); gtz_tile(a); break;
            case C_GEZ: SFI(C_GEZ, gez_tile_init()); gez_tile(a); break;
            case C_LTZ: SFI(C_LTZ, ltz_tile_init()); ltz_tile(a); break;
            case C_LEZ: SFI(C_LEZ, lez_tile_init()); lez_tile(a); break;
            case C_EQZ: SFI(C_EQZ, eqz_tile_init()); eqz_tile(a); break;
            case C_NEZ: SFI(C_NEZ, nez_tile_init()); nez_tile(a); break;
            case C_FOLD:
                SFI(C_FOLD, (sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>()));
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(a);
                sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(a);
                break;
            case C_TOK:
                PWAIT(cb_tok);
                cb_reserve_back(cb_tok, 1);   // PACK
                cb_push_back(cb_tok, 1);
                USTATE(4);
                TWAIT(cb_tok);
                cb_wait_front(cb_tok, 1);     // UNPACK
                USTATE(0);
                cb_pop_front(cb_tok, 1);
                break;
            case C_SIGB: PSTATE(3); PWAIT(cb_cb); cb_reserve_back(cb_cb, 1); PSTATE(0); cb_push_back(cb_cb, 1); break;
            case C_SIGN: PSTATE(4); PWAIT(cb_cn); cb_reserve_back(cb_cn, 1); PSTATE(0); cb_push_back(cb_cn, 1); break;
            case C_WAITB: USTATE(3); TWAIT(cb_bc); cb_wait_front(cb_bc, 1); USTATE(0); USYNC(); MWAIT_U(); PWAIT_U(); cb_pop_front(cb_bc, 1); break;
            case C_WAITN: USTATE(2); TWAIT(cb_nc); cb_wait_front(cb_nc, 1); USTATE(0); USYNC(); MWAIT_U(); PWAIT_U(); cb_pop_front(cb_nc, 1); break;
            case C_JMP: pc = CR ? (pc + PW - 1) / PW * PW : CW(pc); break;
            case C_BRT: {
                const uint32_t v = read_tile_value(cb_v, CW(pc), 0);
                const uint32_t target = CR ? (pc + 2 + PW - 1) / PW * PW : CW(pc + 1);
                pc += 2;
                if (read_tile_value(cb_sync, 0, W_ABORT)) {   // a data mover gave up: run out to END
                    run = false;
                    break;
                }
                if (((v & 0x7fffffffu) != 0) != (a & 1)) {   // a = 1: taken on a zero flag (encoder's fused back edge)
                    pc = target;
                    ++taken;
                } else if (CR) {
                    pc = target;   // paged: the fall-through block starts on the next page too
                }
                break;
            }
            default:
                UNPACK((hw[8] = 0xBAD00000u | op, hw[9] = pc));
                run = false;
                break;
        }
#ifdef CFD_PROF
        {
            const uint32_t t = static_cast<uint32_t>(ckernel::read_wall_clock());
            const uint32_t ps = op == C_BLK ? 0 : op < 40 ? op : 39;   // bodies in C_END's slot (one END per run)
            cyc[ps] += t - tprev;
            ++cnt[ps];
            tprev = t;
        }
#endif
    }
#ifdef CFD_PROF
    // hdr words 128 + 40 * thread + op (UNPACK, MATH, PACK), op counts at 248 + op
    UNPACK(({ _Pragma("GCC unroll 1") for (uint32_t i = 0; i < 40; ++i) hw[128 + i] = static_cast<uint32_t>(cyc[i] >> 4); }));
    MATH(({ _Pragma("GCC unroll 1") for (uint32_t i = 0; i < 40; ++i) hw[168 + i] = static_cast<uint32_t>(cyc[i] >> 4); }));
    PACK(({ _Pragma("GCC unroll 1") for (uint32_t i = 0; i < 40; ++i) hw[208 + i] = static_cast<uint32_t>(cyc[i] >> 4); }));
    UNPACK(({ _Pragma("GCC unroll 1") for (uint32_t i = 0; i < 40; ++i) hw[248 + i] = cnt[i]; }));
#endif
#ifdef CFD_PHASE
    MATH(({
        phc[phid] += ckernel::read_wall_clock() - pht;
        for (uint32_t i = 0; i < 16; ++i) {
            hw[512 + 2 * i] = static_cast<uint32_t>(phc[i]);
            hw[513 + 2 * i] = static_cast<uint32_t>(phc[i] >> 32);
        }
    }));
#endif
    UNPACK((hw[0] = 0xC0DE0000u | (taken & 0xffff), hw[4] = nops));
    MATH((hw[1] = 0xC0DE0000u | (taken & 0xffff), hw[5] = nops));
    PACK((hw[2] = 0xC0DE0000u | (taken & 0xffff), hw[6] = nops));
    cb_reserve_back(cb_done, 1);
    cb_push_back(cb_done, 1);
}
