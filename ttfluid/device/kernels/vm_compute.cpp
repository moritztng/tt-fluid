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

enum : uint32_t {
    C_END = 0, C_ACQ, C_COMMIT, C_REL, C_LD, C_ST, C_FILL, C_RVF,
    C_ADD, C_SUB, C_MUL, C_DIV, C_MAX, C_MIN, C_WHERE, C_CPD,
    C_NEG, C_ABS, C_SQRT, C_TANH, C_LOG, C_EXP, C_GTZ, C_GEZ, C_LTZ, C_LEZ, C_EQZ, C_NEZ,
    C_FOLD, C_TOK, C_SIGB, C_SIGN, C_WAITB, C_WAITN, C_JMP, C_BRT, C_LDS, C_TBEG, C_TEND, C_LDR, C_STW, C_MAD, C_MSB, C_LDRH, C_STWH, C_PH
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
#ifdef CFD_TGTS
    // T-gather mailbox (vm_ncrisc.cpp, the 256 B of cb_sync before the B-patch one): NCRISC posts the top CFD_TGTS / 8
    // of each L1-source and halo section of a GATHER tile (src, entries, n, page) and bumps SEQ; UNPACK copies them from
    // inside every wait on a data mover or PACK and answers DONE = SEQ. Disjoint lanes: the same values, bit-exact
    constexpr uint32_t TMB = CFD_SYNCB / 4 - 128, T_SEQ = TMB, T_DONE = TMB + 4, T_N = TMB + 8, T_JOB = TMB + 12;
    uint32_t tseen = 0;
    auto tsvc = [&]() {
        const uint32_t q = sw[T_SEQ];
        if (q == tseen) return;
        for (uint32_t j = 0, nj = sw[T_N]; j < nj; ++j) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(sw[T_JOB + 4 * j]);
            const uint32_t* xs = reinterpret_cast<const uint32_t*>(sw[T_JOB + 4 * j + 1]);
            const uint32_t n = sw[T_JOB + 4 * j + 2];
            uint32_t* out = reinterpret_cast<uint32_t*>(sw[T_JOB + 4 * j + 3]);
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
        sw[T_DONE] = q;
        tseen = q;
    };
    // UNPACK spins on tsvc until cb has a page; the cb_wait_front after it then returns at once. Every CB wait of UNPACK
    // is one of these; a TRISC stuck behind a PACK that waits on a full cb_cn would not be, and NCRISC then aborts (22)
#define TWAIT(cb) UNPACK(({ while (static_cast<uint16_t>(reg_read(reinterpret_cast<uint32_t>(get_cb_tiles_received_ptr(cb))) - get_local_cb_interface(cb).tiles_acked) == 0) tsvc(); }))
#else
#define TWAIT(cb)
#endif
    uint32_t tl = 0, tn = 0, tpc = 0;   // tile loop: counter, count, first op of the body
    bool run = true;
#ifdef CFD_PROF
    // census: per-op wall-clock cycles of this thread (time from one op's issue to the next's) and op counts
    uint64_t cyc[40];   // stored in 16-cycle units (>> 4): a 100-it mesh run is ~2^35 cycles
    uint32_t cnt[40];
    for (uint32_t i = 0; i < 40; ++i) cyc[i] = cnt[i] = 0;
    uint32_t tprev = static_cast<uint32_t>(ckernel::read_wall_clock());
#endif
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
                    TWAIT(cb_code);
                    cb_wait_front(cb_code, 1);
                    sw[W_CUNP] = hi + 1;
                }));
                MATH(({
                    while (sw[W_CUNP] <= hi) {
                    }
                    sw[W_CMATH] = hi + 1;
                }));
                PACK(({
                    while (sw[W_CUNP] <= hi) {
                    }
                }));
                ++hi;
                const uint32_t r = (tl < tn) ? tpc / PW : pc / PW;   // an open tile loop keeps its body
                if (r > rel) {
                    rel = r;
                    PACK(({
                        while (sw[W_CMATH] < hi) {   // MATH has entered this page, so it left the released ones
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
            case C_ACQ: tile_regs_acquire(); break;
            case C_COMMIT: tile_regs_commit(); tile_regs_wait(); break;
            case C_REL: tile_regs_release(); break;
            case C_LD: LDI(cb_v); copy_tile(cb_v, (w >> 16) + to, a); break;
            case C_LDS: {   // offset load: the unpacker reads the tile from a 16-byte-unit shifted L1 address
                const int32_t units = static_cast<int32_t>(CW(pc));
                ++pc;
                LDI(cb_v);
                UNPACK((get_local_cb_interface(cb_v).fifo_rd_ptr += units));
                copy_tile(cb_v, (w >> 16) + to, a);
                UNPACK((get_local_cb_interface(cb_v).fifo_rd_ptr -= units));
                break;
            }
            case C_ST: pack_tile<true>(a, cb_v, (w >> 16) + to); break;
            case C_LDR:   // next reader page; the tile starts b 16-B units in (the DRAM source's 32-B phase)
                TWAIT(cb_in);
                cb_wait_front(cb_in, 1);
                LDI(cb_in);
                UNPACK((get_local_cb_interface(cb_in).fifo_rd_ptr += b));
                copy_tile(cb_in, 0, a);
                UNPACK((get_local_cb_interface(cb_in).fifo_rd_ptr -= b));
                cb_pop_front(cb_in, 1);
                break;
#ifdef CFD_NARROW
            case C_LDRH:   // C_LDR of a bf16 page: CB CFD_NARROW (bfloat16) views cb_in's front page, srcA widens it to fp32
                TWAIT(cb_in);
                cb_wait_front(cb_in, 1);
                copy_tile_to_dst_init_short_with_dt(cb_in, CFD_NARROW);
                UNPACK((get_local_cb_interface(CFD_NARROW).fifo_rd_ptr = get_local_cb_interface(cb_in).fifo_rd_ptr + b));
                copy_tile(CFD_NARROW, 0, a);
                copy_tile_to_dst_init_short_with_dt(CFD_NARROW, cb_in);
                LDX();
                cb_pop_front(cb_in, 1);
                break;
#endif
#ifdef CFD_NARROW_OUT
            case C_STWH:   // C_STW packed to bf16: CB CFD_NARROW_OUT (bfloat16) views cb_out's back page
                cb_reserve_back(cb_out, 1);
                PACK((get_local_cb_interface(CFD_NARROW_OUT).fifo_wr_ptr = get_local_cb_interface(cb_out).fifo_wr_ptr));
                pack_reconfig_data_format(cb_out, CFD_NARROW_OUT);
                pack_tile<true>(a, CFD_NARROW_OUT, 0);
                pack_reconfig_data_format(CFD_NARROW_OUT, cb_out);
                cb_push_back(cb_out, 1);
                break;
#endif
            case C_STW:
                cb_reserve_back(cb_out, 1);
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
                cb_reserve_back(cb_tok, 1);   // PACK
                cb_push_back(cb_tok, 1);
                TWAIT(cb_tok);
                cb_wait_front(cb_tok, 1);     // UNPACK
                cb_pop_front(cb_tok, 1);
                break;
            case C_SIGB: cb_reserve_back(cb_cb, 1); cb_push_back(cb_cb, 1); break;
            case C_SIGN: cb_reserve_back(cb_cn, 1); cb_push_back(cb_cn, 1); break;
            case C_WAITB: TWAIT(cb_bc); cb_wait_front(cb_bc, 1); cb_pop_front(cb_bc, 1); break;
            case C_WAITN: TWAIT(cb_nc); cb_wait_front(cb_nc, 1); cb_pop_front(cb_nc, 1); break;
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
            cyc[op < 40 ? op : 39] += t - tprev;
            ++cnt[op < 40 ? op : 39];
            tprev = t;
        }
#endif
    }
#ifdef CFD_PROF
    // hdr words 128 + 40 * thread + op (UNPACK, MATH, PACK), op counts at 248 + op
    UNPACK(for (uint32_t i = 0; i < 40; ++i) hw[128 + i] = static_cast<uint32_t>(cyc[i] >> 4));
    MATH(for (uint32_t i = 0; i < 40; ++i) hw[168 + i] = static_cast<uint32_t>(cyc[i] >> 4));
    PACK(for (uint32_t i = 0; i < 40; ++i) hw[208 + i] = static_cast<uint32_t>(cyc[i] >> 4));
    UNPACK(for (uint32_t i = 0; i < 40; ++i) hw[248 + i] = cnt[i]);
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
