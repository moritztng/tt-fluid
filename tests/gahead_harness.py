"""Host harness for --gahead: the kernel's own gprep and TSORT patch text against a direct gather, with a mock NoC whose
reads land only at a barrier (a page used before its barrier is poison)."""
import sys
K = open(sys.argv[1]).read()
def cut(a, b):
    i = K.index(a)
    j = K.index(b, i) + len(b)
    return K[i:j]
gprep = cut("    constexpr uint32_t GAR = CFD_GAHEAD;", "        ga_ok |= 1u << k;\n    };\n")
tsort = cut("        if constexpr (NGC != 0) {\n            uint32_t e = 0, dummy;\n#ifdef CFD_GAHEAD", "            asm volatile(\"\" ::: \"memory\");\n            return;\n        }\n#endif")
tsort = tsort[:tsort.rindex("#endif")]
src = r'''
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <unistd.h>
#include <sys/mman.h>
constexpr uint32_t XS = 16, XI = 0xffff, XT = 0xfc00, NT = 39;
constexpr uint32_t NGC = NGCV, cb_gc = 0;
#define CFD_GAHEAD GARV
#define CFD_TSORT 1
static uint32_t gc_base;
static uint32_t get_write_ptr(uint32_t) { return gc_base; }
static std::vector<std::vector<uint32_t>> dram;
struct Pend { uint32_t d, t, dst; };
static std::vector<Pend> pendv;
static long nreads = 0, nwaits = 0;
static uint64_t dpage(uint32_t d, uint32_t t) { return (uint64_t(d & 0xffff) << 32) | t; }
static void noc_async_read(uint64_t a, uint32_t dst, uint32_t n) {
    uint32_t* p = reinterpret_cast<uint32_t*>(uintptr_t(dst));
    for (uint32_t i = 0; i < n / 4; ++i) p[i] = 0xDEADBEEF;      // not landed yet
    pendv.push_back({uint32_t(a >> 32), uint32_t(a), dst});
    ++nreads;
}
static void noc_async_read_barrier() {
    if (!pendv.empty()) ++nwaits;
    for (auto& q : pendv) memcpy(reinterpret_cast<void*>(uintptr_t(q.dst)), &dram[q.d][q.t * 1024], 4096);
    pendv.clear();
}
int main(int argc, char** argv) {
    const int use = atoi(argv[1]);   // 0 plain, 1 read-ahead, 2 read-ahead WITHOUT its barrier (must fail)
    const uint32_t seed = atoi(argv[2]), pages = atoi(argv[3]);
    alarm(60);
    void* m = mmap(reinterpret_cast<void*>(0x20000000), (NGC + 1) * 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (m == MAP_FAILED) return 9;
    gc_base = 0x20000000;
    std::mt19937 rng(seed);
    const uint32_t D = 5;
    dram.assign(D, std::vector<uint32_t>(NT * 1024));
    for (auto& v : dram) for (auto& x : v) x = rng();
    uint32_t gcd[NGC ? NGC : 1], gct[NGC ? NGC : 1], gcn = 0, gmiss = 0;
    uint32_t gld = 0xffffffffu, glt = 0;
    volatile uint32_t* glp = nullptr;
    auto ginv = [&]() { for (uint32_t j = 0; j < NGC; ++j) gcd[j] = 0xffffffffu; gld = 0xffffffffu; };
    ginv();
    constexpr uint32_t NGW = NGC ? (NGC + 31) / 32 : 1;
    uint32_t gpin[NGW] = {};
    auto value = [&](uint32_t d, uint32_t i, volatile uint32_t* dst) {
        const uint32_t t = i >> 10;
        if (d == gld && t == glt) {
            *dst = glp[i & 1023];
            return;
        }
        uint32_t j = 0;
        while (j < NGC && !(gcd[j] == d && gct[j] == t)) ++j;
        if (j == NGC) {
            while ((gpin[gcn >> 5] >> (gcn & 31)) & 1) gcn = gcn + 1 == NGC ? 0 : gcn + 1;
            j = gcn;
            gcn = gcn + 1 == NGC ? 0 : gcn + 1;
            noc_async_read(dpage(d, t), get_write_ptr(cb_gc) + j * 4096, 4096);
            noc_async_read_barrier();
            gcd[j] = d;
            gct[j] = t;
            ++gmiss;
        }
        gld = d, glt = t;
        glp = reinterpret_cast<volatile uint32_t*>(uintptr_t(get_write_ptr(cb_gc) + j * 4096));
        *dst = glp[i & 1023];
    };
@GPREP@
    auto patch = [&](uint32_t d, const uint32_t* xs, uint32_t n, uint32_t* out) {
@TSORT@
    };
    long bad = 0, runs_tot = 0, parts_tot = 0, prepared = 0;
    for (uint32_t pg = 0; pg < pages; ++pg) {
        if (rng() % 7 == 0) {            // a new stream: homes may have changed
            ginv();
            auto& v = dram[rng() % D];
            for (uint32_t i = 0; i < 4096; ++i) v[rng() % v.size()] = rng();
        }
        const uint32_t np_ = 1 + rng() % 8;
        std::vector<std::vector<uint32_t>> X(np_);
        std::vector<uint32_t> src(np_), lanes(1024);
        for (uint32_t i = 0; i < 1024; ++i) lanes[i] = i;
        std::shuffle(lanes.begin(), lanes.end(), rng);
        uint32_t nl = 0;
        for (uint32_t k = 0; k < np_; ++k) {
            src[k] = (1u << 28) | (rng() % D);
            const uint32_t nr = rng() % 7;           // 0..6 runs, ascending distinct tiles
            std::vector<uint32_t> tiles;
            for (uint32_t t = 0; t < NT; ++t) if (rng() % NT < nr) tiles.push_back(t);
            for (uint32_t t : tiles) {
                uint32_t len = 1 + rng() % (rng() % 3 ? 9 : 150);
                for (; len && nl < 1024; --len) X[k].push_back(lanes[nl++] << XS | t << 10 | (rng() & 1023));
            }
            runs_tot += tiles.size(), ++parts_tot;
        }
        uint32_t out[1024], ref[1024];
        memset(out, 0x55, sizeof out), memset(ref, 0x55, sizeof ref);
        for (uint32_t k = 0; k < np_; ++k) for (uint32_t x : X[k]) ref[x >> XS] = dram[src[k] & 0xffff][x & XI];
        const uint32_t** sp = new const uint32_t*[2 * np_];
        std::vector<uint32_t> sn(2 * np_);
        for (uint32_t k = 0; k < np_; ++k) sp[2 * k] = X[k].data(), sn[2 * k] = X[k].size();
        const uint32_t p = 0;
        std::vector<uint32_t> N(4 + 2 * np_);
        for (uint32_t k = 0; k < np_; ++k) N[p + 2 + 2 * k] = src[k];
        if (use) {
            if constexpr (NGC > 1) {   // as in the handler
                ga_n = 0, ga_pin = 0, ga_ok = 0, ga_rd = false;
                for (uint32_t k = 0; k < np_; ++k) {
                    const uint32_t sd = N[p + 2 + 2 * k];
                    if ((sd >> 28) != 0) gprep(k, sd, sp[2 * k], sn[2 * k]);
                }
                if (ga_rd && use != 2) noc_async_read_barrier();
            }
            for (uint32_t k = 0; k < np_; ++k) prepared += (ga_ok >> k) & 1;
        }
        for (uint32_t k = 0; k < np_; ++k) {
            const uint32_t sd = N[p + 2 + 2 * k];
            ga_k = k;
            patch(sd, sp[2 * k], sn[2 * k], out);
        }
        for (uint32_t w = 0; w < NGW; ++w) gpin[w] = 0;
        ga_ok = 0;
        if (memcmp(out, ref, sizeof out)) ++bad;
        delete[] sp;
    }
    printf("use=%d NGC=%u GAR=%u pages=%u parts=%ld runs=%ld prepared_parts=%ld misses=%u reads=%ld waits=%ld bad_pages=%ld\n",
           int(use), NGC, unsigned(GAR), pages, parts_tot, runs_tot, prepared, gmiss, nreads, nwaits, bad);
    return bad != 0;
}
'''
src = src.replace("@GPREP@", gprep).replace("@TSORT@", tsort)
# the kernel builds for a 32-bit core: pointers there are uint32_t
src = src.replace("reinterpret_cast<const uint32_t*>(ga_pg[r])", "reinterpret_cast<const uint32_t*>(uintptr_t(ga_pg[r]))")
open(sys.argv[2], "w").write(src)
