// Roof probe, one data-movement RISC-V (role 0 = BRISC, 1 = NCRISC; the same file builds both). What the scalar
// loops, the NoC and the DRAM read path cost with nothing else in the way.
// L1 arena (cb_a): X 16 tiles | IDX 8 tiles (16,384 uint16, a permutation of 0..16383) | R 16 tiles.
// Each RISC works on its own half of the 16,384 words (scalar ops) or its own 32 KB half of R (NoC and DRAM ops).
//   1 copy     r[k] = x[k]                       2 gather   r[k] = x[idx[k]]          3 scatter  r[idx[k]] = x[k]
//   4 NoC read of BYTES from its own L1 into its own L1, a barrier after each (latency)
//   5 the same, REPS reads issued and one barrier at the end (rate)
//   6 NoC write of BYTES to its own L1, a barrier after each
//   7 DRAM read stream: pages of BYTES from this core's DRAM row into an NB-slot ring, one transaction id per slot
//   8 ping-pong with a peer core: write BYTES into the peer's R, raise its semaphore, wait for the answer
// Header page (cb_h), words role * 8 + {0: ticks of the timed loop, 1: 0xC0DE0000 | op, 2: REPS, 3: BYTES as the kernel
// saw them}. BRISC writes header + R back. ROOF_BREAK 1 makes every op do less than it reports: the host check must fail.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1), cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t SW = get_compile_time_arg_val(3);       // words of this core's DRAM stream row
    constexpr auto in_args = TensorAccessorArgs<4>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto str_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 4096, NX = 16, NI = 8, NR = 16, M = 16384;
    const uint32_t OP = get_common_arg_val<uint32_t>(3), REPS = get_common_arg_val<uint32_t>(4),
                   BYTES = get_common_arg_val<uint32_t>(5), NB = get_common_arg_val<uint32_t>(6),
                   MASK = get_common_arg_val<uint32_t>(7);
    const uint32_t core = get_arg_val<uint32_t>(0), peer_x = get_arg_val<uint32_t>(1), peer_y = get_arg_val<uint32_t>(2),
                   ping = get_arg_val<uint32_t>(3);
    const auto in = TensorAccessor(in_args, get_common_arg_val<uint32_t>(0), (NX + NI) * TB);
    const auto out = TensorAccessor(out_args, get_common_arg_val<uint32_t>(1), (1 + NR) * TB);
    const auto str = TensorAccessor(str_args, get_common_arg_val<uint32_t>(2), SW * 4);
    const uint32_t base = get_write_ptr(cb_a);
    const uint32_t* __restrict x = reinterpret_cast<const uint32_t*>(base);
    const uint16_t* __restrict id = reinterpret_cast<const uint16_t*>(base + NX * TB);
    uint32_t* __restrict r = reinterpret_cast<uint32_t*>(base + (NX + NI) * TB);
    volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_h));
    const uint32_t sem_a = get_semaphore(0), go_a = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_a);   // BRISC <-> NCRISC
    volatile tt_l1_ptr uint32_t* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go_a);     // ping <-> pong
    if constexpr (role == 0) {
        for (uint32_t i = 0; i < 1024; ++i) h[i] = 0;
        for (uint32_t i = 0; i < NR * 1024; ++i) r[i] = 0;
        noc_async_read(in.get_noc_addr(core), base, (NX + NI) * TB);
        noc_async_read_barrier();
        noc_semaphore_set(sem, 1);
    } else {
        noc_semaphore_wait(sem, 1);
    }
    const bool on = MASK >> role & 1;
    const uint32_t k0 = role * (M / 2), k1 = k0 + M / 2;
    const uint32_t mine = reinterpret_cast<uint32_t>(r) + role * 8 * TB;   // this RISC's 32 KB of R
    uint32_t t0 = 0, t1 = 0;
    if (on && OP == 1) {
        t0 = get_timestamp_32b();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t k = k0; k < k1 - 8 * ROOF_BREAK; k += 8) {
                const uint32_t a0 = x[k], a1 = x[k + 1], a2 = x[k + 2], a3 = x[k + 3];
                const uint32_t a4 = x[k + 4], a5 = x[k + 5], a6 = x[k + 6], a7 = x[k + 7];
                r[k] = a0; r[k + 1] = a1; r[k + 2] = a2; r[k + 3] = a3;
                r[k + 4] = a4; r[k + 5] = a5; r[k + 6] = a6; r[k + 7] = a7;
            }
            asm volatile("" ::: "memory");
        }
        t1 = get_timestamp_32b();
    } else if (on && OP == 2) {
        t0 = get_timestamp_32b();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t k = k0; k < k1 - 8 * ROOF_BREAK; k += 8) {
                const uint32_t a0 = x[id[k]], a1 = x[id[k + 1]], a2 = x[id[k + 2]], a3 = x[id[k + 3]];
                const uint32_t a4 = x[id[k + 4]], a5 = x[id[k + 5]], a6 = x[id[k + 6]], a7 = x[id[k + 7]];
                r[k] = a0; r[k + 1] = a1; r[k + 2] = a2; r[k + 3] = a3;
                r[k + 4] = a4; r[k + 5] = a5; r[k + 6] = a6; r[k + 7] = a7;
            }
            asm volatile("" ::: "memory");
        }
        t1 = get_timestamp_32b();
    } else if (on && OP == 3) {
        uint32_t* rr = r;   // not restrict: the stores are indexed
        t0 = get_timestamp_32b();
        for (uint32_t rep = 0; rep < REPS; ++rep) {
            for (uint32_t k = k0; k < k1 - 8 * ROOF_BREAK; k += 8) {
                const uint32_t a0 = x[k], a1 = x[k + 1], a2 = x[k + 2], a3 = x[k + 3];
                const uint32_t a4 = x[k + 4], a5 = x[k + 5], a6 = x[k + 6], a7 = x[k + 7];
                rr[id[k]] = a0; rr[id[k + 1]] = a1; rr[id[k + 2]] = a2; rr[id[k + 3]] = a3;
                rr[id[k + 4]] = a4; rr[id[k + 5]] = a5; rr[id[k + 6]] = a6; rr[id[k + 7]] = a7;
            }
            asm volatile("" ::: "memory");
        }
        t1 = get_timestamp_32b();
    } else if (on && (OP == 4 || OP == 5 || OP == 6)) {
        const uint32_t src = base + role * 8 * TB;
        const uint64_t self_src = get_noc_addr(src), self_dst = get_noc_addr(mine);
        t0 = get_timestamp_32b();
        for (uint32_t i = 0; i < REPS - ROOF_BREAK * REPS; ++i) {
            if (OP == 6) { noc_async_write(src, self_dst, BYTES); noc_async_write_barrier(); }
            else { noc_async_read(self_src, mine, BYTES); if (OP == 4) noc_async_read_barrier(); }
        }
        if (OP == 5) noc_async_read_barrier();
        t1 = get_timestamp_32b();
    } else if (on && OP == 7) {
        const uint64_t row = str.get_noc_addr(core);
        const uint32_t npg = SW * 4 / BYTES;
        t0 = get_timestamp_32b();
        for (uint32_t i = 0; i < REPS; ++i) {
            const uint32_t slot = i % NB, trid = slot + 1;
            if (i >= NB) noc_async_read_barrier_with_trid(trid);
            noc_async_read_set_trid(trid);
            noc_async_read(row + ((i + ROOF_BREAK) % npg) * BYTES, mine + slot * BYTES, BYTES);
        }
        noc_async_read_barrier();
        t1 = get_timestamp_32b();
        noc_async_read_set_trid(0);
    } else if (on && OP == 8) {
        const uint64_t peer_r = get_noc_addr(peer_x, peer_y, mine), peer_go = get_noc_addr(peer_x, peer_y, go_a);
        if (!ping) noc_semaphore_inc(peer_go, 1);   // pong announces itself
        else noc_semaphore_wait_min(go, 1);
        t0 = get_timestamp_32b();
        for (uint32_t i = 0; i < REPS; ++i) {
            if (!ping) noc_semaphore_wait_min(go, i + 1);
            noc_async_write(base + role * 8 * TB, peer_r, BYTES - ROOF_BREAK * 4);
            noc_async_write_barrier();
            noc_semaphore_inc(peer_go, 1);
            if (ping) noc_semaphore_wait_min(go, i + 2);
        }
        t1 = get_timestamp_32b();
        noc_async_atomic_barrier();
    }
    h[role * 8] = t1 - t0;
    h[role * 8 + 1] = 0xC0DE0000 | OP;
    h[role * 8 + 2] = REPS;
    h[role * 8 + 3] = BYTES;
    if constexpr (role == 1) {
        noc_semaphore_set(sem, 2);
    } else {
        noc_semaphore_wait(sem, 2);
        const uint64_t dst = out.get_noc_addr(core);
        noc_async_write(get_write_ptr(cb_h), dst, TB);
        noc_async_write(reinterpret_cast<uint32_t>(r), dst + TB, NR * TB);
        noc_async_write_barrier();
    }
}
