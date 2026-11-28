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

#ifdef CFD_RUNCOPY
#if defined(__riscv_zba)
#define CFD_LW(r, k) "lw " #r ", 4*" #k "(%[p])\n"
#define CFD_SW(r, k) "sw " #r ", 4*" #k "(%[o])\n"
#endif

// Hand-scheduled 16-register sequential run copy + Zba cfd_gather16 for literals (tabstream.encode_runcopy).
// Out-of-line at -O3 so every RISC (N, B, U, M, P) runs the same fast loop without -Os register serialization.
static __attribute__((noinline, optimize("O3"))) void cfd_gather_run(
    const uint32_t* src, const uint32_t* cmd, uint32_t nd, uint32_t* out) {
    uint32_t* o = out;
    uint32_t* const end = o + nd;
    while (o < end) {
        const uint32_t w = *cmd++;
        if (w >> 31) {
            const uint32_t len = (w >> 16) & 0x7fffu;
            const uint32_t* p = src + (w & 0xffffu);
#if defined(__riscv_zba)
            const uint32_t* const pe = p + (len & ~15u);
            if (p != pe)
                asm volatile("1:\n" CFD_R16(CFD_LW) CFD_R16(CFD_SW)
                             "addi %[p], %[p], 64\naddi %[o], %[o], 64\nbne %[p], %[pe], 1b\n"
                             : [p] "+r"(p), [o] "+r"(o) : [pe] "r"(pe) : CFD_CLOB);
            uint32_t r = len & 15u;
            while (r >= 4) {
                const uint32_t v0 = p[0], v1 = p[1], v2 = p[2], v3 = p[3];
                o[0] = v0, o[1] = v1, o[2] = v2, o[3] = v3;
                p += 4, o += 4, r -= 4;
            }
            while (r--) *o++ = *p++;
#else
            uint32_t e = 0;
            for (; e + 8 <= len; e += 8) {
                const uint32_t v0 = p[e], v1 = p[e + 1], v2 = p[e + 2], v3 = p[e + 3];
                const uint32_t v4 = p[e + 4], v5 = p[e + 5], v6 = p[e + 6], v7 = p[e + 7];
                o[e] = v0, o[e + 1] = v1, o[e + 2] = v2, o[e + 3] = v3;
                o[e + 4] = v4, o[e + 5] = v5, o[e + 6] = v6, o[e + 7] = v7;
            }
            for (; e < len; ++e) o[e] = p[e];
            o += len;
#endif
        } else {
            const uint32_t len = w;
            cfd_gather16(src, reinterpret_cast<const uint16_t*>(cmd), len, o);
            o += len;
            cmd += (len + 1) >> 1;
        }
    }
    asm volatile("" ::: "memory");
}
#endif
#endif
