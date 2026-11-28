// Generated SFPU blocks (C_BLK, LEVERS 13): the body executor of the MATH thread, shared by vm_compute.cpp and
// bench/unit/ub_compute.cpp. A body is an inline program in the compute code (ttfluid/resident/blockgen.py):
//
//     h          ncfg | L << 8 | nrep << 16 | nfl << 24
//     cfg[ncfg]  Tensix instruction words pushed once: LREG constants (SFPLOADI), load-macro templates, sequences, Misc
//     row[L]     Tensix instruction words recorded into the replay buffer at index 0 (L <= 32)
//
// The row program covers 32 / nrep SFPU rows (32 lanes each) per pass and steps the Dest row counter itself: its last
// Dest access of a row uses ADDR_MOD_6 (dest += 2), every other one ADDR_MOD_7. Dest slot s is address 64 s in
// SFPLOAD / SFPSTORE / SFPLOADMACRO. The pass loop is a RISC-V loop pushing the replay instruction, not the MOP
// expander: the loads' datacopy MOP stays programmed. A body leaves lane predication as it found it.
//
// Only on MATH (callers: MATH((blk_run(q)))). The caller runs blk_init() when the last SFPU user was not a body
// (vm_compute: SFI(C_BLK, blk_init())), and every stock op after a body runs its own _init again.
#pragma once
#ifdef TRISC_MATH
#include <cstdint>

[[gnu::always_inline]] inline void blk_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 0}}.set(ADDR_MOD_7);
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2}}.set(ADDR_MOD_6);
}

// One tile of the body at q (q -> h). load = false: the body's cfg and row words are still in place from the last
// call (same program, no other SFPU user in between), only the passes and the flush run.
[[gnu::always_inline]] inline void blk_run(const uint32_t* q, const bool load = true) {
    const uint32_t h = q[0];
    const uint32_t ncfg = h & 0xff, len = (h >> 8) & 0xff, nfl = h >> 24;
    uint32_t nrep = (h >> 16) & 0xff;
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0);
    if (load) {
        const uint32_t* p = q + 1;
        for (const uint32_t* const e = p + ncfg; p != e; ++p) ckernel::instrn_buffer[0] = *p;
        ckernel::instrn_buffer[0] = TT_OP_REPLAY(0, len, 0, 1);   // record the next len instructions at index 0, without executing them
        for (const uint32_t* const e = p + len; p != e; ++p) ckernel::instrn_buffer[0] = *p;
    }
    // the pushes must outrun the SFPU (a pass is len cycles): a row of a load-macro body overlaps the next one, and a
    // gap in the instruction stream would move the next row against the ops the last one scheduled
    const uint32_t rp = TT_OP_REPLAY(0, len, 0, 0);
    for (; nrep >= 4; nrep -= 4) {
        ckernel::instrn_buffer[0] = rp;
        ckernel::instrn_buffer[0] = rp;
        ckernel::instrn_buffer[0] = rp;
        ckernel::instrn_buffer[0] = rp;
    }
    for (; nrep; --nrep) ckernel::instrn_buffer[0] = rp;
    for (uint32_t i = nfl; i; --i) ckernel::instrn_buffer[0] = TT_OP_SFPNOP;
    _llk_math_eltwise_binary_sfpu_done_();
}
#endif
