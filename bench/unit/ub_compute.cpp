// Unit costs of the compute pipeline on one core: cycles per fp32 tile load, pack and SFPU tile op, stock LLK calls
// against lean forms. Each cell runs NS acquire/commit/release sections of NB units; the three TRISCs meet at a
// barrier before a cell (each after tensix_sync, so its Tensix queue is empty) and stamp the wall clock when their
// own part of the cell has drained. Header words: 32 + 4 id + {0 unpack, 1 math, 2 pack} = cycles, + 3 = units.
// Tiles of cb_v: 0-7 input, 8-11 stock-load check, 12-15 lean-load check, 16-25 op results, 20-23 pack scratch
// (the pack cell runs before the checks).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/addcmul.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/copy_dest_values.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/comp.h"
#include "api/compute/eltwise_unary/fill.h"

constexpr uint32_t NB = 4;   // fp32 tiles of one DEST half
// UB_BREAK (unit_costs.py --break-arm): 1 = the lean-load check moves one tile where it claims four, 2 = the fused
// blocks store the first multiply-add instead of the second. Either must turn its check red
#ifndef UB_BREAK
#define UB_BREAK 0
#endif
#define FUSED_OUT (UB_BREAK == 2 ? p_sfpu::LREG4 : p_sfpu::LREG6)

// one 32-lane step of r = (s0 s1 + s2) s1 + s0 into slot 3: five loads, two multiply-adds, one store
#define FUSED_STEP()                                                             \
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);          \
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 64);         \
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 128);        \
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG4, 0);   \
    TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::FP32, ADDR_MOD_7, 64);         \
    TTI_SFPLOAD(p_sfpu::LREG5, InstrModLoadStore::FP32, ADDR_MOD_7, 0);          \
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG3, p_sfpu::LREG5, p_sfpu::LREG6, 0);   \
    TTI_SFPNOP;                                                                  \
    TTI_SFPSTORE(FUSED_OUT, InstrModLoadStore::FP32, ADDR_MOD_7, 192);           \
    TTI_INCRWC(0, 2, 0, 0)
constexpr uint32_t FUSED_LEN = 10;
// the same block as a lowering would schedule it: each operand loaded once, the pointer step folded into the store
// (ADDR_MOD_6 advances DEST by one step). GAP is what sits between a multiply-add and the instruction that reads it
#define SCHED_STEP(GAP)                                                          \
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);          \
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 64);         \
    TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 128);        \
    TTI_SFPMAD(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG4, 0);   \
    GAP;                                                                         \
    TTI_SFPMAD(p_sfpu::LREG4, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpu::LREG6, 0);   \
    GAP;                                                                         \
    TTI_SFPSTORE(FUSED_OUT, InstrModLoadStore::FP32, ADDR_MOD_6, 192)
// r = s0 + s1 into slot 2: the smallest body, for the cost per SFPU instruction
#define ADD_STEP()                                                                   \
    TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::FP32, ADDR_MOD_7, 0);              \
    TTI_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::FP32, ADDR_MOD_7, 64);             \
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0);    \
    TTI_SFPNOP;                                                                      \
    TTI_SFPSTORE(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 128);           \
    TTI_INCRWC(0, 2, 0, 0)
constexpr uint32_t ADD_LEN = 6;
#define NEXT_FACE()                                                              \
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);    \
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D)

#ifdef TRISC_MATH
#define SFPU_BEGIN() _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0)
#define SFPU_END() _llk_math_eltwise_binary_sfpu_done_()
// one tile of the fused block, every instruction issued by the RISC-V
[[gnu::always_inline]] inline void fused_issued() {
    SFPU_BEGIN();
    for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll 8
        for (uint32_t d = 0; d < 8; ++d) {
            FUSED_STEP();
        }
        NEXT_FACE();
    }
    SFPU_END();
}
// the same block from sfpi C++: the compiler picks the registers and drops the repeated loads
[[gnu::always_inline]] inline void fused_sfpi() {
    SFPU_BEGIN();
    for (uint32_t f = 0; f < 4; ++f) {
#pragma GCC unroll 8
        for (uint32_t d = 0; d < 8; ++d) {
            sfpi::vFloat a = sfpi::dst_reg[0], b = sfpi::dst_reg[32], c = sfpi::dst_reg[64];
            sfpi::vFloat x = a * b + c;
            sfpi::dst_reg[96] = UB_BREAK == 2 ? x : x * b + a;
            sfpi::dst_reg++;
        }
        NEXT_FACE();
    }
    SFPU_END();
}
// body in the replay buffer, the tile loop in the MOP expander: one run() per tile
template <typename F>
[[gnu::always_inline]] inline void mop_body(uint32_t len, F&& body) {
    load_replay_buf(0, len, body);
    ckernel::ckernel_template tmp_(4, 8, lltt::replay_insn(0, len));
    tmp_.set_end_ops(TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D),
                     TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D));
    tmp_.program();
}
[[gnu::always_inline]] inline void mop_tile() {
    SFPU_BEGIN();
    ckernel::ckernel_template::run();
    SFPU_END();
}
// r = s0 + s1 written back to slot 1 at two cycles per row (SFPLOADMACRO): the load of s1 schedules the add on the MAD
// unit for the next cycle and the store of its result, to the address it was loaded from, three cycles after the
// load. The add's result is readable two cycles after it executes, so a store delay of 0 (UB_BREAK 2) stores s1 itself.
// The loaded register alternates so that a row's store still finds its result after the next row has loaded.
[[gnu::always_inline]] inline void pipe_add_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG2, 0, 12, 0);                  // InstructionTemplate[0]: VD = 1 * L2 + VC
    constexpr uint32_t mad_bits = (0 << 3) | 4, store_bits = ((UB_BREAK == 2 ? 0 : 2) << 3) | 3;
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | 0);
    TTI_SFPCONFIG(0, 4 + 0, 0);                                             // Sequence[0] from LREG0
    TTI_SFPCONFIG(0x010, 8, 1);                                             // Misc: store with the load's Mod0, delays in cycles
    load_replay_buf(0, 4, [] {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::FP32, ADDR_MOD_6, 64);
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPLOADMACRO((0 << 2) | 1, InstrModLoadStore::FP32, ADDR_MOD_6, 64);
    });
    ckernel::ckernel_template tmp_(4, 4, lltt::replay_insn(0, 4));
    tmp_.set_end_ops(TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D),
                     TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D));
    tmp_.program();
}
// r = s0 * s1 + s2 as a mul and an add (two roundings, as the two stock ops), written back to slot 2 at three cycles
// per row. The load of s1 schedules the mul for the next cycle (result in the loaded register, readable two cycles
// on); the load of s2 schedules the add two cycles after itself (result in LReg[16]) and the store of LReg[16] to
// s2's address four cycles after itself. UB_BREAK 2 runs the add one cycle early, before the product is readable.
[[gnu::always_inline]] inline void pipe_chain_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
    TTI_SFPMUL(p_sfpu::LREG2, 0, p_sfpu::LCONST_0, 12, 0);                  // InstructionTemplate[0]: VD = L2 * VB + 0
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, 0, 13, 0);                  // InstructionTemplate[1]: VD = 1 * L0 + VC
    constexpr uint32_t mul_bits = 0x80 | (0 << 3) | 4;                      // loaded register is VB and the result
    constexpr uint32_t add_bits = 0x40 | ((UB_BREAK == 2 ? 0 : 1) << 3) | 5, store_bits = 0x40 | (3 << 3) | 3;   // result and store through LReg[16]
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mul_bits << 8) | 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, 0);
    TTI_SFPCONFIG(0, 4 + 0, 0);                                             // Sequence[0] from LREG0: MAD unit only
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (add_bits << 8) | 0);
    TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | 0);
    TTI_SFPCONFIG(0, 4 + 1, 0);                                             // Sequence[1] from LREG0
    TTI_SFPCONFIG(0x020, 8, 1);                                             // Misc: macro 1's store takes the load's Mod0, delays in cycles
    load_replay_buf(0, 3, [] {
        TTI_SFPLOAD(p_sfpu::LREG2, InstrModLoadStore::FP32, ADDR_MOD_7, 0);
        TTI_SFPLOADMACRO((0 << 2) | 0, InstrModLoadStore::FP32, ADDR_MOD_7, 64);
        TTI_SFPLOADMACRO((1 << 2) | 1, InstrModLoadStore::FP32, ADDR_MOD_6, 128);
    });
    ckernel::ckernel_template tmp_(4, 8, lltt::replay_insn(0, 3));
    tmp_.set_end_ops(TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D),
                     TT_OP_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D));
    tmp_.program();
}
[[gnu::always_inline]] inline void pipe_chain_tile() {
    SFPU_BEGIN();
    ckernel::ckernel_template::run();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    SFPU_END();
}
[[gnu::always_inline]] inline void pipe_tile() {
    SFPU_BEGIN();
    ckernel::ckernel_template::run();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    SFPU_END();
}
#endif

void kernel_main() {
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_done = get_compile_time_arg_val(2);
    constexpr uint32_t nv = get_compile_time_arg_val(3);
    constexpr uint32_t cb_h = get_compile_time_arg_val(4);   // bfloat16 view of cb_v's memory (CFD_NARROW)
    const uint32_t mask = get_common_arg_val<uint32_t>(0), NS = get_common_arg_val<uint32_t>(1);

    init_sfpu(cb_v, cb_v);
    cb_wait_front(cb_v, nv);
    volatile uint32_t* hw = reinterpret_cast<volatile uint32_t*>(get_tile_address(cb_hdr, 0));
    uint32_t self = 0, bk = 0;
    MATH(self = 1);
    PACK(self = 2);
    auto clk = []() { return static_cast<uint32_t>(ckernel::read_wall_clock()); };
    auto barrier = [&]() {
        ckernel::tensix_sync();
        hw[16 + self] = ++bk;
        while (hw[16] < bk || hw[17] < bk || hw[18] < bk) {
        }
    };
#define SEC_BEGIN() tile_regs_acquire()
#define SEC_END() tile_regs_commit(); tile_regs_wait(); tile_regs_release()
#define CELL(id, units, ...)                                     \
    if (mask & (1u << (id))) {                                   \
        barrier();                                               \
        const uint32_t t0_ = clk();                              \
        __VA_ARGS__;                                             \
        ckernel::tensix_sync();                                  \
        hw[32 + 4 * (id) + self] = clk() - t0_;                  \
        UNPACK((hw[32 + 4 * (id) + 3] = (units)));               \
    }
    // the unpacker's MOP: 4 n faces to DEST in one run (the stock init programs n = 1)
#define LOAD_MOP(n) UNPACK(({ ckernel::ckernel_template tmp_(4 * (n), 1, TT_OP_UNPACR(SrcA, 0b00010001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1)); tmp_.program(); }))
    // n consecutive L1 tiles from tile t into DEST slots 0..n-1: one handshake, then the zero-flag clears of the rest
#define LEAN_LOAD(t, n)                                                                                                        \
    UNPACK((llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(cb_v, (t))));           \
    MATH(({                                                                                                                    \
        llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(0, cb_v);                    \
        for (uint32_t f_ = 4; f_ < 4 * (n); ++f_) TT_ZEROACC(p_zeroacc::CLR_16, 1, 1, ADDR_MOD_3, f_);                         \
    }))

    // 0: an empty section
    CELL(0, NS, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); SEC_END(); });
    // 1: stock load, init once (run_mesh --icache); 2: init before every load (the default build)
    copy_tile_to_dst_init_short(cb_v);
    CELL(1, NS * NB, for (uint32_t s = 0; s < NS; ++s) {
        SEC_BEGIN();
        for (uint32_t j = 0; j < NB; ++j) copy_tile(cb_v, (s + j) & 7, j);
        SEC_END();
    });
    CELL(2, NS * NB, for (uint32_t s = 0; s < NS; ++s) {
        SEC_BEGIN();
        for (uint32_t j = 0; j < NB; ++j) { copy_tile_to_dst_init_short(cb_v); copy_tile(cb_v, (s + j) & 7, j); }
        SEC_END();
    });
    // 3: stock pack of a DEST slot to an L1 tile (C_ST)
    CELL(3, NS * NB, for (uint32_t s = 0; s < NS; ++s) {
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < NB; ++j) pack_tile<true>(j, cb_v, 20 + j);
        tile_regs_release();
    });
    // 4-7: stock SFPU tile ops on DEST slots, init once
    add_binary_tile_init();
    CELL(4, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) add_binary_tile(0, 1, 2); SEC_END(); });
    mul_binary_tile_init();
    CELL(5, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) mul_binary_tile(0, 1, 2); SEC_END(); });
    where_tile_init();
    CELL(7, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) where_tile<DataFormat::Float32>(0, 1, 2, 3); SEC_END(); });
    addcmul_tile_init();
    CELL(6, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) addcmul_tile<DataFormat::Float32>(0, 1, 2, 3, 0x3f800000u); SEC_END(); });
    // 8: bf16 page widened to fp32 in DEST, reconfigured both ways around every load as C_LDRH does
    CELL(8, NS * NB, for (uint32_t s = 0; s < NS; ++s) {
        SEC_BEGIN();
        for (uint32_t j = 0; j < NB; ++j) {
            copy_tile_to_dst_init_short_with_dt(cb_v, cb_h);
            UNPACK((get_local_cb_interface(cb_h).fifo_rd_ptr = get_local_cb_interface(cb_v).fifo_rd_ptr + ((s + j) & 7) * 256));
            copy_tile(cb_h, 0, j);
            copy_tile_to_dst_init_short_with_dt(cb_h, cb_v);
        }
        SEC_END();
    });
    // 9, 10, 11: lean load of 1, 2, 4 consecutive tiles per handshake
    copy_tile_to_dst_init_short(cb_v);
    LOAD_MOP(1);
    CELL(9, NS, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); LEAN_LOAD(s & 7, 1); SEC_END(); });
    LOAD_MOP(2);
    CELL(10, NS * 2, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); LEAN_LOAD(s & 3, 2); SEC_END(); });
    LOAD_MOP(4);
    CELL(11, NS * 4, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); LEAN_LOAD(s & 3, 4); SEC_END(); });
    // 12: fused block, every instruction issued by the RISC-V; 13: the same from sfpi C++; 14: body in the replay
    // buffer, one MOP run per tile; 15: the add body the same way
    addcmul_tile_init();
    CELL(12, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) fused_issued(); })); SEC_END(); });
    CELL(13, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) fused_sfpi(); })); SEC_END(); });
    MATH((mop_body(FUSED_LEN, [] { FUSED_STEP(); })));
    CELL(14, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) mop_tile(); })); SEC_END(); });
    MATH((mop_body(ADD_LEN, [] { ADD_STEP(); })));
    CELL(15, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) mop_tile(); })); SEC_END(); });
    // 16: scheduled block, a NOP in each gap (8 instructions); 17: the gaps left empty (6), to see what the SFPU does
    MATH((addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6)));
    MATH((mop_body(8, [] { SCHED_STEP(TTI_SFPNOP); })));
    CELL(16, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) mop_tile(); })); SEC_END(); });
    MATH((mop_body(6, [] { SCHED_STEP((void)0); })));
    CELL(17, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) mop_tile(); })); SEC_END(); });
    // 18: one add per tile through the load macro, in place, two cycles per row
    MATH((pipe_add_init()));
    CELL(18, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) pipe_tile(); })); SEC_END(); });
    // 19: mul then add fused through the load macro, three cycles per row
    MATH((pipe_chain_init()));
    CELL(19, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); MATH(({ for (uint32_t j = 0; j < NB; ++j) pipe_chain_tile(); })); SEC_END(); });
    // 20-27: the stock tile ops the VM calls and the table did not have yet
    div_binary_tile_init();
    CELL(20, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) div_binary_tile(0, 1, 2); SEC_END(); });
    binary_max_tile_init();
    CELL(21, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) binary_max_tile(0, 1, 2); SEC_END(); });
    binary_min_tile_init();
    CELL(22, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) binary_min_tile(0, 1, 2); SEC_END(); });
    negative_tile_init();
    CELL(23, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) negative_tile(0); SEC_END(); });
    abs_tile_init();
    CELL(24, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) abs_tile(0); SEC_END(); });
    gtz_tile_init();
    CELL(25, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) gtz_tile(0); SEC_END(); });
    fill_tile_init();
    CELL(26, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) fill_tile_bitcast(0, 0x3f800000u); SEC_END(); });
    copy_dest_values_init();
    CELL(27, NS * NB, for (uint32_t s = 0; s < NS; ++s) { SEC_BEGIN(); for (uint32_t j = 0; j < NB; ++j) copy_dest_values<DataFormat::Float32>(0, 1); SEC_END(); });

    // checks (mask bit 31). Stock load of tiles 0-3 packed to 8-11; lean load of tiles 4-7 packed to 12-15
    if (mask & 0x80000000u) {
        barrier();
        copy_tile_to_dst_init_short(cb_v);
        tile_regs_acquire();
        for (uint32_t j = 0; j < NB; ++j) copy_tile(cb_v, j, j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < NB; ++j) pack_tile<true>(j, cb_v, 8 + j);
        tile_regs_release();
        LOAD_MOP(UB_BREAK == 1 ? 1 : 4);
        tile_regs_acquire();
        LEAN_LOAD(4, 4);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < NB; ++j) pack_tile<true>(j, cb_v, 12 + j);
        tile_regs_release();
        // r = (t0 t1 + t2) t1 + t0 four ways: two stock addcmul calls -> 16, issued block -> 17, sfpi -> 18, MOP -> 19
        copy_tile_to_dst_init_short(cb_v);
        addcmul_tile_init();
        MATH((addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6)));
        for (uint32_t v = 0; v < 6; ++v) {   // 4 -> tile 20: scheduled with NOPs, 5 -> tile 21: gaps empty
            MATH(({
                if (v == 3) mop_body(FUSED_LEN, [] { FUSED_STEP(); });
                if (v == 4) mop_body(8, [] { SCHED_STEP(TTI_SFPNOP); });
                if (v == 5) mop_body(6, [] { SCHED_STEP((void)0); });
            }));
            tile_regs_acquire();
            for (uint32_t j = 0; j < 3; ++j) copy_tile(cb_v, j, j);
            if (v == 0) {
                addcmul_tile<DataFormat::Float32>(2, 0, 1, 3, 0x3f800000u);   // 3 = t2 + t0 t1
                addcmul_tile<DataFormat::Float32>(0, 3, 1, 3, 0x3f800000u);   // 3 = t0 + (3) t1
            }
            MATH(({
                if (v == 1) fused_issued();
                if (v == 2) fused_sfpi();
                if (v >= 3) mop_tile();
            }));
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(3, cb_v, 16 + v);
            tile_regs_release();
        }
        // r = t0 + t1: the stock add -> 22, the pipelined in-place add -> 23
        add_binary_tile_init();
        tile_regs_acquire();
        for (uint32_t j = 0; j < 2; ++j) copy_tile(cb_v, j, j);
        add_binary_tile(0, 1, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(2, cb_v, 22);
        tile_regs_release();
        MATH((pipe_add_init()));
        tile_regs_acquire();
        for (uint32_t j = 0; j < 2; ++j) copy_tile(cb_v, j, j);
        MATH((pipe_tile()));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(1, cb_v, 23);
        tile_regs_release();
        // r = t0 * t1 + t2: stock mul then stock add -> 24, the pipelined chain -> 25
        mul_binary_tile_init();
        tile_regs_acquire();
        for (uint32_t j = 0; j < 3; ++j) copy_tile(cb_v, j, j);
        mul_binary_tile(0, 1, 3);
        add_binary_tile_init();
        add_binary_tile(3, 2, 3);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(3, cb_v, 24);
        tile_regs_release();
        MATH((pipe_chain_init()));
        tile_regs_acquire();
        for (uint32_t j = 0; j < 3; ++j) copy_tile(cb_v, j, j);
        MATH((pipe_chain_tile()));
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(2, cb_v, 25);
        tile_regs_release();
    }
    barrier();
    UNPACK((hw[0] = 0xC0DE0000u | bk));
    MATH((hw[1] = 0xC0DE0000u | bk));
    PACK((hw[2] = 0xC0DE0000u | bk));
    cb_reserve_back(cb_done, 1);
    cb_push_back(cb_done, 1);
}
