#pragma once
// Hand-scheduled 16-register dense gather (lane e = src[ix[e]]): 16 index loads, 16 sh2add address adds,
// 16 value loads, 16 sequential stores per turn so every load-use sits 16 instructions apart on BH (6.91 cyc/lane
// on 3 threads, 1.34 cyc/lane wall across 5 threads; tools/gbench.py d16a16). Compiled noinline at -O3 so -Os /
// outer register pressure (notably vm_brisc.cpp's WAIT_UNTIL call sites) cannot collapse it to a single scratch reg.
#include <cstdint>

#ifdef CFD_DENSE
#if defined(__riscv_zba)
#define CFD_LHU(r, k) "lhu " #r ", 2*" #k "(%[ix])\n"
#define CFD_ADR(r, k) "sh2add " #r ", " #r ", %[src]\n"
#define CFD_LDV(r, k) "lw " #r ", 0(" #r ")\n"
#define CFD_STV(r, k) "sw " #r ", 4*" #k "(%[out])\n"
#define CFD_R16(F) \
    F(t0, 0) F(t1, 1) F(t2, 2) F(t3, 3) F(t4, 4) F(t5, 5) F(t6, 6) F(s2, 7) F(s3, 8) \
    F(s4, 9) F(s5, 10) F(s6, 11) F(s7, 12) F(s8, 13) F(s9, 14) F(s10, 15)
#define CFD_CLOB "t0", "t1", "t2", "t3", "t4", "t5", "t6", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "memory"
#endif

static __attribute__((noinline, optimize("O3"))) void cfd_gather16(
    const uint32_t* src, const uint16_t* ix, uint32_t n, uint32_t* out) {
#if defined(__riscv_zba)
    const uint16_t* const end = ix + (n & ~15u);
    if (ix != end)
        asm volatile("1:\n" CFD_R16(CFD_LHU) CFD_R16(CFD_ADR) CFD_R16(CFD_LDV) CFD_R16(CFD_STV)
                     "addi %[ix], %[ix], 32\naddi %[out], %[out], 64\nbne %[ix], %[end], 1b\n"
                     : [ix] "+r"(ix), [out] "+r"(out) : [src] "r"(src), [end] "r"(end) : CFD_CLOB);
    for (uint32_t e = n & ~15u; e < n; ++e) *out++ = src[*ix++];
#else
    uint32_t e = 0;
    for (; e + 8 <= n; e += 8) {
        const uint32_t i0 = ix[e], i1 = ix[e + 1], i2 = ix[e + 2], i3 = ix[e + 3];
        const uint32_t i4 = ix[e + 4], i5 = ix[e + 5], i6 = ix[e + 6], i7 = ix[e + 7];
        const uint32_t v0 = src[i0], v1 = src[i1], v2 = src[i2], v3 = src[i3];
        const uint32_t v4 = src[i4], v5 = src[i5], v6 = src[i6], v7 = src[i7];
        out[e] = v0, out[e + 1] = v1, out[e + 2] = v2, out[e + 3] = v3;
        out[e + 4] = v4, out[e + 5] = v5, out[e + 6] = v6, out[e + 7] = v7;
    }
    for (; e < n; ++e) out[e] = src[ix[e]];
#endif
    asm volatile("" ::: "memory");
}

#ifdef CFD_G3
#if defined(__riscv_zba)
#define CFD_LHU8(r, k) "lhu " #r ", 2*" #k "(%[ix])\n"
#define CFD_ADR8(v, i, src) "sh2add " #v ", " #i ", %[" #src "]\n"
#define CFD_LDV8(v, k) "lw " #v ", 0(" #v ")\n"
#define CFD_STV8(v, k, out) "sw " #v ", 4*" #k "(%[" #out "])\n"
#define CFD_R8_I(F) F(s2, 0) F(s3, 1) F(s4, 2) F(s5, 3) F(s6, 4) F(s7, 5) F(s8, 6) F(s9, 7)
#define CFD_R8_A(src) \
    CFD_ADR8(t0, s2, src) CFD_ADR8(t1, s3, src) CFD_ADR8(t2, s4, src) CFD_ADR8(t3, s5, src) \
    CFD_ADR8(t4, s6, src) CFD_ADR8(t5, s7, src) CFD_ADR8(t6, s8, src) CFD_ADR8(s10, s9, src)
#define CFD_R8_L \
    CFD_LDV8(t0, 0) CFD_LDV8(t1, 1) CFD_LDV8(t2, 2) CFD_LDV8(t3, 3) \
    CFD_LDV8(t4, 4) CFD_LDV8(t5, 5) CFD_LDV8(t6, 6) CFD_LDV8(s10, 7)
#define CFD_R8_S(out) \
    CFD_STV8(t0, 0, out) CFD_STV8(t1, 1, out) CFD_STV8(t2, 2, out) CFD_STV8(t3, 3, out) \
    CFD_STV8(t4, 4, out) CFD_STV8(t5, 5, out) CFD_STV8(t6, 6, out) CFD_STV8(s10, 7, out)
#endif

// 3-component dense gather sharing one 16-bit index stream: o{0,1,2}[e] = s{0,1,2}[ix[e]].
// Loads 8 indices once into s2..s9, then runs 8 sh2add + 8 lw + 8 sw per component (8 insns load-to-use).
static __attribute__((noinline, optimize("O3"))) void cfd_gather16_3(
    const uint32_t* s0, const uint32_t* s1, const uint32_t* s2,
    const uint16_t* ix, uint32_t n,
    uint32_t* o0, uint32_t* o1, uint32_t* o2) {
#if defined(__riscv_zba)
    const uint16_t* const end = ix + (n & ~7u);
    if (ix != end)
        asm volatile("1:\n"
                     CFD_R8_I(CFD_LHU8)
                     CFD_R8_A(s0) CFD_R8_L CFD_R8_S(o0)
                     CFD_R8_A(s1) CFD_R8_L CFD_R8_S(o1)
                     CFD_R8_A(s2) CFD_R8_L CFD_R8_S(o2)
                     "addi %[ix], %[ix], 16\n"
                     "addi %[o0], %[o0], 32\naddi %[o1], %[o1], 32\naddi %[o2], %[o2], 32\n"
                     "bne %[ix], %[end], 1b\n"
                     : [ix] "+r"(ix), [o0] "+r"(o0), [o1] "+r"(o1), [o2] "+r"(o2)
                     : [s0] "r"(s0), [s1] "r"(s1), [s2] "r"(s2), [end] "r"(end)
                     : CFD_CLOB);
    for (uint32_t e = n & ~7u; e < n; ++e) {
        const uint32_t i = *ix++;
        *o0++ = s0[i]; *o1++ = s1[i]; *o2++ = s2[i];
    }
#else
    uint32_t e = 0;
    for (; e + 4 <= n; e += 4) {
        const uint32_t i0 = ix[e], i1 = ix[e + 1], i2 = ix[e + 2], i3 = ix[e + 3];
        o0[e] = s0[i0]; o0[e + 1] = s0[i1]; o0[e + 2] = s0[i2]; o0[e + 3] = s0[i3];
        o1[e] = s1[i0]; o1[e + 1] = s1[i1]; o1[e + 2] = s1[i2]; o1[e + 3] = s1[i3];
        o2[e] = s2[i0]; o2[e + 1] = s2[i1]; o2[e + 2] = s2[i2]; o2[e + 3] = s2[i3];
    }
    for (; e < n; ++e) {
        const uint32_t i = ix[e];
        o0[e] = s0[i]; o1[e] = s1[i]; o2[e] = s2[i];
    }
#endif
    asm volatile("" ::: "memory");
}
#endif
#endif
