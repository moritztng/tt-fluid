"""Cycles per value of the gather loop forms on one Tensix core, measured on the device.

Every RISC-V of a core (BRISC, NCRISC, the three TRISCs) runs the same loops over L1: a 40-tile source vector, 1,024-lane
tables and its own 1,024-lane destination page. A run takes a mask of the RISC-Vs that take part, so each loop is timed
alone and with the others working on L1 at the same time. BRISC fills the tables, starts each test for everybody at once
and writes the result page to DRAM; every wait is bounded.

    TT_VISIBLE_DEVICES=26 python3 -m ttfluid.tools.gbench --gen /tmp/gbench --masks 1,2,4,8,16,3,7,31 --json out.json

Tests (cycles are per value = per fp32 lane, over --reps passes of a page):
  load      sum of the page, 8 loads per turn (one L1 load and one ALU op per value)
  store     one register stored to every lane
  copy      out[i] = src[i + 1024], 8 loads then 8 stores per turn
  memcpy    __builtin_memcpy of the page
  scatter   today's residue patch: out[x >> 16] = src[x & 0xffff], entries in destination order, 4 per turn
  scatter_d the same source at the kernel's default optimisation level
  d16x4     dense, destination order: out[i] = src[ix[i]], 16-bit ix, 4 per turn
  d16x8     the same, 8 per turn
  pair16    two 16-bit indices per loaded word
  off32     32-bit byte offsets: out[i] = *(base + off[i])
  abs32     32-bit addresses: out[i] = *tab[i]
  noc1      (BRISC, NCRISC) one 4-KB NoC read L1 to L1 of the own core, barrier after each
  nocq      the same, all reads issued before one barrier
  noci      the issue of one read alone (the barrier is outside the clock)
"""
import argparse
import json
import pathlib
import time

import numpy as np

TESTS = ["load", "store", "copy", "memcpy", "scatter", "scatter_d", "d16x4", "d16x8", "pair16", "off32", "abs32",
         "noc1", "nocq", "noci", "d16a16", "d16a32", "abs32a16", "abs32a32"]
WHO = ["brisc", "ncrisc", "unpack", "math", "pack"]
NSRC = 40 * 1024

BODY = r"""
typedef uint32_t u32;
typedef uint16_t u16;
#define GB_O3 __attribute__((noinline, optimize("O3")))
#define GB_DEF __attribute__((noinline))
constexpr u32 GB_NT = 18, GB_LIMIT = 2000000000u, GB_END = 0xff;

GB_O3 static u32 gb_load(const u32* src, u32 n) {
    u32 s = 0;
    for (u32 e = 0; e + 8 <= n; e += 8) {
        const u32 v0 = src[e], v1 = src[e + 1], v2 = src[e + 2], v3 = src[e + 3];
        const u32 v4 = src[e + 4], v5 = src[e + 5], v6 = src[e + 6], v7 = src[e + 7];
        s += v0 ^ v1 ^ v2 ^ v3 ^ v4 ^ v5 ^ v6 ^ v7;
    }
    return s;
}
GB_O3 static void gb_store(u32 v, u32 n, u32* out) {
    for (u32 e = 0; e + 8 <= n; e += 8) {
        out[e] = v, out[e + 1] = v, out[e + 2] = v, out[e + 3] = v;
        out[e + 4] = v, out[e + 5] = v, out[e + 6] = v, out[e + 7] = v;
    }
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_copy(const u32* src, u32 n, u32* out) {
    for (u32 e = 0; e + 8 <= n; e += 8) {
        const u32 v0 = src[e], v1 = src[e + 1], v2 = src[e + 2], v3 = src[e + 3];
        const u32 v4 = src[e + 4], v5 = src[e + 5], v6 = src[e + 6], v7 = src[e + 7];
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
        out[e + 4] = v4, out[e + 5] = v5, out[e + 6] = v6, out[e + 7] = v7;
    }
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_memcpy(const u32* src, u32 n, u32* out) {
    __builtin_memcpy(out, src, 4 * n);
    asm volatile("" ::: "memory");
}
#define GB_SCATTER                                                                                          \
    for (u32 e = 0; e + 4 <= n; e += 4) {                                                                   \
        const u32 x0 = xs[e], x1 = xs[e + 1], x2 = xs[e + 2], x3 = xs[e + 3];                               \
        const u32 v0 = src[x0 & 0xffff], v1 = src[x1 & 0xffff], v2 = src[x2 & 0xffff], v3 = src[x3 & 0xffff]; \
        out[x0 >> 16] = v0;                                                                                 \
        out[x1 >> 16] = v1;                                                                                 \
        out[x2 >> 16] = v2;                                                                                 \
        out[x3 >> 16] = v3;                                                                                 \
    }                                                                                                       \
    asm volatile("" ::: "memory");
GB_O3 static void gb_scatter(const u32* src, const u32* xs, u32 n, u32* out) { GB_SCATTER }
GB_DEF static void gb_scatter_d(const u32* src, const u32* xs, u32 n, u32* out) { GB_SCATTER }
GB_O3 static void gb_d16x4(const u32* src, const u16* ix, u32 n, u32* out) {
    for (u32 e = 0; e + 4 <= n; e += 4) {
        const u32 i0 = ix[e], i1 = ix[e + 1], i2 = ix[e + 2], i3 = ix[e + 3];
        const u32 v0 = src[i0], v1 = src[i1], v2 = src[i2], v3 = src[i3];
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
    }
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_d16x8(const u32* src, const u16* ix, u32 n, u32* out) {
    for (u32 e = 0; e + 8 <= n; e += 8) {
        const u32 i0 = ix[e], i1 = ix[e + 1], i2 = ix[e + 2], i3 = ix[e + 3];
        const u32 i4 = ix[e + 4], i5 = ix[e + 5], i6 = ix[e + 6], i7 = ix[e + 7];
        const u32 v0 = src[i0], v1 = src[i1], v2 = src[i2], v3 = src[i3];
        const u32 v4 = src[i4], v5 = src[i5], v6 = src[i6], v7 = src[i7];
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
        out[e + 4] = v4, out[e + 5] = v5, out[e + 6] = v6, out[e + 7] = v7;
    }
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_pair16(const u32* src, const u32* ix2, u32 n, u32* out) {
    for (u32 e = 0; e + 4 <= n; e += 4) {
        const u32 w0 = ix2[e >> 1], w1 = ix2[(e >> 1) + 1];
        const u32 v0 = src[w0 & 0xffff], v1 = src[w0 >> 16], v2 = src[w1 & 0xffff], v3 = src[w1 >> 16];
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
    }
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_off32(const u32* src, const u32* off, u32 n, u32* out) {
    const char* b = reinterpret_cast<const char*>(src);
    for (u32 e = 0; e + 4 <= n; e += 4) {
        const u32 o0 = off[e], o1 = off[e + 1], o2 = off[e + 2], o3 = off[e + 3];
        const u32 v0 = *reinterpret_cast<const u32*>(b + o0), v1 = *reinterpret_cast<const u32*>(b + o1);
        const u32 v2 = *reinterpret_cast<const u32*>(b + o2), v3 = *reinterpret_cast<const u32*>(b + o3);
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
    }
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_abs32(const u32* tab, u32 n, u32* out) {
    for (u32 e = 0; e + 4 <= n; e += 4) {
        const u32* p0 = reinterpret_cast<const u32*>(tab[e]);
        const u32* p1 = reinterpret_cast<const u32*>(tab[e + 1]);
        const u32* p2 = reinterpret_cast<const u32*>(tab[e + 2]);
        const u32* p3 = reinterpret_cast<const u32*>(tab[e + 3]);
        const u32 v0 = *p0, v1 = *p1, v2 = *p2, v3 = *p3;
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
    }
    asm volatile("" ::: "memory");
}

// Hand-scheduled forms. The compiled loops above put a load 3 to 5 instructions in front of its use, and this core
// then loses 0.65 to 2.7 cycles per load (flx-kernel's fit of the gb2 disassembly); a use 8 or more instructions
// later is free (copy: 19 instructions, 25.0 cycles a turn). Here a turn holds 16 values in 16 registers: 16 index
// loads, 16 address adds, 16 value loads, 16 stores, so every use is 16 instructions behind its load. The a32
// forms run two such batches a turn (the turn itself costs 3 instructions and about 6 cycles).
#define GB_LHU(r, k, b) "lhu " #r ", 2*" #k "+" #b "(%[ix])\n"
#define GB_LDP(r, k, b) "lw " #r ", 4*" #k "+" #b "(%[ix])\n"
#define GB_ADR(r, k, b) "sh2add " #r ", " #r ", %[src]\n"
#define GB_LDV(r, k, b) "lw " #r ", 0(" #r ")\n"
#define GB_STV(r, k, b) "sw " #r ", 4*" #k "+" #b "(%[out])\n"
#define GB_R16(F, b)                                                                                              \
    F(t0, 0, b) F(t1, 1, b) F(t2, 2, b) F(t3, 3, b) F(t4, 4, b) F(t5, 5, b) F(t6, 6, b) F(s2, 7, b) F(s3, 8, b) \
    F(s4, 9, b) F(s5, 10, b) F(s6, 11, b) F(s7, 12, b) F(s8, 13, b) F(s9, 14, b) F(s10, 15, b)
#define GB_CLOB "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "memory"
GB_O3 static void gb_d16a16(const u32* src, const u16* ix, u32 n, u32* out) {
    const u16* const end = ix + (n & ~15u);
    if (ix != end)
        asm volatile("1:\n" GB_R16(GB_LHU, 0) GB_R16(GB_ADR, 0) GB_R16(GB_LDV, 0) GB_R16(GB_STV, 0)
                     "addi %[ix], %[ix], 32\naddi %[out], %[out], 64\nbne %[ix], %[end], 1b\n"
                     : [ix] "+r"(ix), [out] "+r"(out) : [src] "r"(src), [end] "r"(end) : GB_CLOB);
    for (u32 e = n & ~15u; e < n; ++e) *out++ = src[*ix++];
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_d16a32(const u32* src, const u16* ix, u32 n, u32* out) {
    const u16* const end = ix + (n & ~31u);
    if (ix != end)
        asm volatile("1:\n" GB_R16(GB_LHU, 0) GB_R16(GB_ADR, 0) GB_R16(GB_LDV, 0) GB_R16(GB_STV, 0)
                     GB_R16(GB_LHU, 32) GB_R16(GB_ADR, 32) GB_R16(GB_LDV, 32) GB_R16(GB_STV, 64)
                     "addi %[ix], %[ix], 64\naddi %[out], %[out], 128\nbne %[ix], %[end], 1b\n"
                     : [ix] "+r"(ix), [out] "+r"(out) : [src] "r"(src), [end] "r"(end) : GB_CLOB);
    for (u32 e = n & ~31u; e < n; ++e) *out++ = src[*ix++];
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_abs32a16(const u32* ix, u32 n, u32* out) {
    const u32* const end = ix + (n & ~15u);
    if (ix != end)
        asm volatile("1:\n" GB_R16(GB_LDP, 0) GB_R16(GB_LDV, 0) GB_R16(GB_STV, 0)
                     "addi %[ix], %[ix], 64\naddi %[out], %[out], 64\nbne %[ix], %[end], 1b\n"
                     : [ix] "+r"(ix), [out] "+r"(out) : [end] "r"(end) : GB_CLOB);
    for (u32 e = n & ~15u; e < n; ++e) *out++ = *reinterpret_cast<const u32*>(*ix++);
    asm volatile("" ::: "memory");
}
GB_O3 static void gb_abs32a32(const u32* ix, u32 n, u32* out) {
    const u32* const end = ix + (n & ~31u);
    if (ix != end)
        asm volatile("1:\n" GB_R16(GB_LDP, 0) GB_R16(GB_LDV, 0) GB_R16(GB_STV, 0)
                     GB_R16(GB_LDP, 64) GB_R16(GB_LDV, 64) GB_R16(GB_STV, 64)
                     "addi %[ix], %[ix], 128\naddi %[out], %[out], 128\nbne %[ix], %[end], 1b\n"
                     : [ix] "+r"(ix), [out] "+r"(out) : [end] "r"(end) : GB_CLOB);
    for (u32 e = n & ~31u; e < n; ++e) *out++ = *reinterpret_cast<const u32*>(*ix++);
    asm volatile("" ::: "memory");
}

// test k on RISC-V `me`: cycles of R passes over one 1,024-lane page into sw[64 + 64 me + k], a checksum of the page
// it left into sw[96 + 64 me + k]
static void gb_test(u32 k, volatile u32* sw, u32 me) {
    const u32 R = sw[2];
    const u32* src = reinterpret_cast<const u32*>(sw[3]);
    const u32* t_sc = reinterpret_cast<const u32*>(sw[4]);
    const u16* t_16 = reinterpret_cast<const u16*>(sw[5]);
    const u32* t_off = reinterpret_cast<const u32*>(sw[6]);
    const u32* t_abs = reinterpret_cast<const u32*>(sw[7]);
    u32* out = reinterpret_cast<u32*>(sw[8]) + 1024 * me;
    for (u32 i = 0; i < 1024; ++i) out[i] = 0;
    asm volatile("" ::: "memory");
    u32 chk = 0, dt = 0;
    bool page = true, own = false;
#ifdef GB_DM
    // a NoC read keeps its address mod 64 B: the source is the second tile of the vector, at the destination's phase
    const u32 nsrc = sw[3] + 4096 + ((reinterpret_cast<u32>(out) - sw[3]) & 63);
#endif
    const u32 t0 = GB_NOW();
    switch (k) {
        case 0: for (u32 r = 0; r < R; ++r) chk += gb_load(src + 1024 * (r & 31), 1024); page = false; break;
        case 1: for (u32 r = 0; r < R; ++r) gb_store(0xA5A50000u + r, 1024, out); break;
        case 2: for (u32 r = 0; r < R; ++r) gb_copy(src + 1024, 1024, out); break;
        case 3: for (u32 r = 0; r < R; ++r) gb_memcpy(src + 1024, 1024, out); break;
        case 4: for (u32 r = 0; r < R; ++r) gb_scatter(src, t_sc, 1024, out); break;
        case 5: for (u32 r = 0; r < R; ++r) gb_scatter_d(src, t_sc, 1024, out); break;
        case 6: for (u32 r = 0; r < R; ++r) gb_d16x4(src, t_16, 1024, out); break;
        case 7: for (u32 r = 0; r < R; ++r) gb_d16x8(src, t_16, 1024, out); break;
        case 8: for (u32 r = 0; r < R; ++r) gb_pair16(src, reinterpret_cast<const u32*>(t_16), 1024, out); break;
        case 9: for (u32 r = 0; r < R; ++r) gb_off32(src, t_off, 1024, out); break;
        case 10: for (u32 r = 0; r < R; ++r) gb_abs32(t_abs, 1024, out); break;
#ifdef GB_DM
        case 11:
            for (u32 r = 0; r < R; ++r) {
                noc_async_read(get_noc_addr(nsrc), reinterpret_cast<u32>(out), 4096);
                noc_async_read_barrier();
            }
            break;
        case 12:
            for (u32 r = 0; r < R; ++r) noc_async_read(get_noc_addr(nsrc), reinterpret_cast<u32>(out), 4096);
            noc_async_read_barrier();
            break;
        case 13:
            for (u32 r = 0; r < R; ++r) {
                const u32 a = GB_NOW();
                noc_async_read(get_noc_addr(nsrc), reinterpret_cast<u32>(out), 4096);
                dt += GB_NOW() - a;
                noc_async_read_barrier();
            }
            own = true;
            break;
#endif
        case 14: for (u32 r = 0; r < R; ++r) gb_d16a16(src, t_16, 1024, out); break;
        case 15: for (u32 r = 0; r < R; ++r) gb_d16a32(src, t_16, 1024, out); break;
        case 16: for (u32 r = 0; r < R; ++r) gb_abs32a16(t_abs, 1024, out); break;
        case 17: for (u32 r = 0; r < R; ++r) gb_abs32a32(t_abs, 1024, out); break;
        default: page = false; break;
    }
    const u32 t1 = GB_NOW();
    if (page) for (u32 i = 0; i < 1024; ++i) chk = chk * 31 + out[i];
    sw[64 + 64 * me + k] = own ? dt : t1 - t0;
    sw[96 + 64 * me + k] = chk;
}

// a RISC-V other than BRISC: every test when BRISC starts it, then its done word
[[maybe_unused]] static void gb_worker(volatile u32* sw, u32 me, u32 epoch, u32 mask) {
    if (!((mask >> me) & 1)) return;
    for (u32 k = 0; k < GB_NT; ++k) {
        const u32 want = epoch + k + 1, t0 = GB_NOW();
        for (;;) {
            const u32 g = sw[0];
            if (g == want) break;
            if (g == epoch + GB_END || GB_NOW() - t0 > GB_LIMIT) return;
            for (u32 d = 0; d < 16; ++d) asm volatile("nop");
        }
        gb_test(k, sw, me);
        asm volatile("" ::: "memory");
        sw[16 + me] = want;
    }
}
"""

DM = r"""// generated by ttfluid/tools/gbench.py: BRISC (GB_ROLE 0, fills the tables and runs the tests) and NCRISC (GB_ROLE 1)
#include "api/dataflow/dataflow_api.h"
#define GB_DM 1
#define GB_NOW() get_timestamp_32b()
%(body)s
void kernel_main() {
    constexpr u32 cb_sync = get_compile_time_arg_val(0);
    const u32 epoch = get_common_arg_val<u32>(1), mask = get_common_arg_val<u32>(2);
    volatile u32* sw = reinterpret_cast<volatile u32*>(get_write_ptr(cb_sync));
#if GB_ROLE == 1
    gb_worker(sw, 1, epoch, mask);
#else
    constexpr u32 cb_src = get_compile_time_arg_val(1), cb_tab = get_compile_time_arg_val(2), cb_dst = get_compile_time_arg_val(3);
    constexpr u32 NSRC = %(nsrc)d;
    constexpr auto a_out = TensorAccessorArgs<4>();
    const auto t_out = TensorAccessor(a_out, get_common_arg_val<u32>(0), 4096);
    const u32 core = get_arg_val<u32>(0), R = get_common_arg_val<u32>(3);
    u32* src = reinterpret_cast<u32*>(get_write_ptr(cb_src));
    u32* tab = reinterpret_cast<u32*>(get_write_ptr(cb_tab));
    u16* t16 = reinterpret_cast<u16*>(tab + 1024);
    for (u32 j = 0; j < NSRC; ++j) src[j] = j * 2654435761u + 12345u;
    u32 x = 12345u + core;
    for (u32 i = 0; i < 1024; ++i) {
        x = x * 1664525u + 1013904223u;
        const u32 ix = (x >> 8) %% NSRC;
        tab[i] = (i << 16) | ix;                                  // today's entry: destination lane << 16 | source
        t16[i] = static_cast<u16>(ix);
        tab[2048 + i] = 4 * ix;
        tab[3072 + i] = reinterpret_cast<u32>(src) + 4 * ix;
    }
    for (u32 i = 1; i < 1024; ++i) sw[i] = 0;
    sw[1] = mask, sw[2] = R, sw[3] = reinterpret_cast<u32>(src), sw[4] = reinterpret_cast<u32>(tab);
    sw[5] = reinterpret_cast<u32>(t16), sw[6] = reinterpret_cast<u32>(tab + 2048), sw[7] = reinterpret_cast<u32>(tab + 3072);
    sw[8] = get_write_ptr(cb_dst), sw[9] = NSRC, sw[10] = core, sw[11] = epoch;
    u32 bad = 0;
    for (u32 k = 0; k < GB_NT && !bad; ++k) {
        const u32 want = epoch + k + 1;
        asm volatile("" ::: "memory");
        sw[0] = want;
        if (mask & 1) gb_test(k, sw, 0);
        const u32 t0 = get_timestamp_32b();
        for (u32 m = 1; m < 5 && !bad; ++m) {
            if (!((mask >> m) & 1)) continue;
            while (sw[16 + m] != want) {
                if (get_timestamp_32b() - t0 > GB_LIMIT) {
                    bad = 0xdead0000u | (m << 8) | k;
                    break;
                }
                for (u32 d = 0; d < 256; ++d) asm volatile("nop");
            }
        }
    }
    sw[15] = bad;
    sw[12] = 0xC0DE0000u | GB_NT;
    asm volatile("" ::: "memory");
    sw[0] = epoch + GB_END;
    noc_async_write(reinterpret_cast<u32>(sw), t_out.get_noc_addr(core), 4096);
    noc_async_write_barrier();
#endif
}
"""

COMP = r"""// generated by ttfluid/tools/gbench.py: the three TRISCs (unpack 2, math 3, pack 4)
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#define GB_NOW() static_cast<uint32_t>(ckernel::read_wall_clock())
%(body)s
void kernel_main() {
    constexpr u32 cb_sync = get_compile_time_arg_val(0);
    const u32 epoch = get_common_arg_val<u32>(0), mask = get_common_arg_val<u32>(1);
    volatile u32* sw = reinterpret_cast<volatile u32*>(get_tile_address(cb_sync, 0));
    u32 me = 2;
    MATH(me = 3);
    PACK(me = 4);
    gb_worker(sw, me, epoch, mask);
}
"""


def model(core, src_addr, dst_addr, me, reps):
    """The page checksum every test must leave (None: no page)."""
    j = np.arange(NSRC, dtype=np.uint64)
    src = ((j * 2654435761 + 12345) & 0xffffffff).astype(np.uint32)
    x, ix = 12345 + core, np.zeros(1024, np.int64)
    for i in range(1024):
        x = (x * 1664525 + 1013904223) & 0xffffffff
        ix[i] = (x >> 8) % NSRC

    def chk(page):
        c = 0
        for v in page.tolist():
            c = (c * 31 + v) & 0xffffffff
        return c

    out = dst_addr + 4096 * me
    off = ((out - src_addr) & 63) // 4
    g, cp = chk(src[ix]), chk(src[1024:2048])
    ld = int(sum(int(np.bitwise_xor.reduce(src[1024 * (r & 31):1024 * (r & 31) + 1024].reshape(-1, 8), axis=1).astype(np.uint64).sum()) for r in range(reps)) & 0xffffffff)
    noc = chk(src[1024 + off:2048 + off])
    st = chk(np.full(1024, 0xA5A50000 + reps - 1, np.uint32))
    return [ld, st, cp, cp, g, g, g, g, g, g, g, noc, noc, noc, g, g, g, g]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen", required=True, help="directory the two kernel sources are written to")
    ap.add_argument("--masks", default="1,2,4,8,16,3,7,31", help="RISC-V sets to run, bit 0 BRISC, 1 NCRISC, 2 unpack, 3 math, 4 pack")
    ap.add_argument("--cores", type=int, default=4, help="cores (0..n-1, 0) that run the bench at once")
    ap.add_argument("--reps", type=int, default=16, help="passes over the page per test")
    ap.add_argument("--rounds", type=int, default=3, help="runs per mask; the first is dropped (cold program cache)")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    gen = pathlib.Path(a.gen)
    gen.mkdir(parents=True, exist_ok=True)
    (gen / "gbench_dm.cpp").write_text(DM % dict(body=BODY, nsrc=NSRC))
    (gen / "gbench_compute.cpp").write_text(COMP % dict(body=BODY))

    import torch
    import ttnn
    md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        n = a.cores
        grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n - 1, 0))])
        shard = ttnn.ShardTensorToMesh(md, dim=0)
        t_in, t_out = (ttnn.from_torch(torch.from_numpy(np.zeros((n, 1024), np.int32)), dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT, device=md, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       mesh_mapper=shard) for _ in range(2))
        CB_SYNC, CB_SRC, CB_TAB, CB_DST = 0, 1, 2, 3

        def cbd(i, nbytes):
            return ttnn.CBDescriptor(total_size=nbytes, core_ranges=grid, format_descriptors=[
                ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.uint32, page_size=4096)])

        cbs = [cbd(CB_SYNC, 4096), cbd(CB_SRC, 4 * NSRC), cbd(CB_TAB, 4 * 4096), cbd(CB_DST, 6 * 4096)]
        acc = list(ttnn.TensorAccessorArgs(t_out).get_compile_time_args())
        res, run = [], 0
        for mask in [int(m, 0) for m in a.masks.split(",")]:
            for rnd in range(a.rounds):
                run += 1
                epoch = ((int(time.time()) & 0xffff) << 16) | ((run & 0xff) << 8)
                prog = ttnn.ProgramDescriptor()
                prog.cbs = cbs
                rt = ttnn.RuntimeArgs()
                for i in range(n):
                    rt[i][0] = [i]

                def dm(role, proc, noc):
                    k = ttnn.KernelDescriptor(kernel_source=str(gen / "gbench_dm.cpp"),
                                              source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
                                              compile_time_args=[CB_SYNC, CB_SRC, CB_TAB, CB_DST] + acc,
                                              config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=noc))
                    k.runtime_args = rt
                    k.common_runtime_args = [t_out.buffer_address(), epoch, mask, a.reps]
                    k.defines = [("GB_ROLE", str(role))]
                    return k

                comp = ttnn.KernelDescriptor(kernel_source=str(gen / "gbench_compute.cpp"),
                                             source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, core_ranges=grid,
                                             compile_time_args=[CB_SYNC],
                                             config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                                                 fp32_dest_acc_en=True, math_approx_mode=False))
                comp.common_runtime_args = [epoch, mask]
                prog.kernels = [dm(0, ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_0),
                                dm(1, ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_1), comp]
                mpd = ttnn.MeshProgramDescriptor()
                mc = ttnn.MeshCoordinate(0, 0)
                mpd[ttnn.MeshCoordinateRange(mc, mc)] = prog
                t0 = time.time()
                ttnn.generic_op([t_in, t_out], mpd)
                o = ttnn.to_torch(t_out, mesh_composer=ttnn.ConcatMeshToTensor(md, dim=0)).numpy().view(np.uint32).reshape(n, 1024)
                wall = time.time() - t0
                for c in range(n):
                    w = o[c]
                    ok = int(w[12]) == (0xC0DE0000 | len(TESTS)) and int(w[11]) == epoch and int(w[15]) == 0
                    row = dict(mask=mask, round=rnd, core=c, ok=bool(ok), bad=int(w[15]), wall=round(wall, 3), cyc={}, chk_ok={})
                    for m in range(5):
                        if not (mask >> m) & 1:
                            continue
                        want = model(c, int(w[3]), int(w[8]), m, a.reps) if ok else None
                        for k, name in enumerate(TESTS):
                            if name.startswith("noc") and m >= 2:
                                continue
                            row["cyc"][f"{WHO[m]}.{name}"] = int(w[64 + 64 * m + k]) / (a.reps * 1024)
                            if ok:
                                row["chk_ok"][f"{WHO[m]}.{name}"] = int(w[96 + 64 * m + k]) == want[k]
                    res.append(row)
                bad = [r for r in res[-n:] if not r["ok"] or not all(r["chk_ok"].values())]
                print("mask %2d round %d: %.2f s%s" % (mask, rnd, wall, "  BAD " + json.dumps(bad[0])[:300] if bad else ""), flush=True)
        # table: mean over cores and rounds >= 1
        print("\ncycles per value (mean over %d cores, rounds 1..%d; min-max over cores and rounds)" % (n, a.rounds - 1))
        summ = {}
        for mask in sorted({r["mask"] for r in res}, key=lambda m: (bin(m).count("1"), m)):
            rows = [r for r in res if r["mask"] == mask and r["round"] >= 1 and r["ok"]]
            if not rows:
                print("mask %d: no good run" % mask)
                continue
            who = [WHO[m] for m in range(5) if (mask >> m) & 1]
            print("mask %2d = %s" % (mask, "+".join(who)))
            for name in TESTS:
                cells = []
                for w in who:
                    v = [r["cyc"][f"{w}.{name}"] for r in rows if f"{w}.{name}" in r["cyc"]]
                    if v:
                        okc = all(r["chk_ok"].get(f"{w}.{name}", False) for r in rows)
                        summ[f"{mask}.{w}.{name}"] = dict(mean=float(np.mean(v)), min=float(np.min(v)), max=float(np.max(v)), chk=bool(okc))
                        cells.append("%s %7.2f (%.2f-%.2f)%s" % (w, np.mean(v), np.min(v), np.max(v), "" if okc else " CHK-FAIL"))
                if cells:
                    print("  %-10s %s" % (name, " | ".join(cells)))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(dict(reps=a.reps, cores=n, rounds=a.rounds, tests=TESTS, summary=summ, rows=res), f)
    finally:
        ttnn.close_mesh_device(md)


if __name__ == "__main__":
    main()
