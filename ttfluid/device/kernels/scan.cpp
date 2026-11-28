// State download, first pass (lower/image_dev.py download_state): one processor reads the state pages of its share
// of the homes and marks those that are not all zero in its row of a page map, which the host reads to size the
// staging tensor kernels/place.cpp then collects them into. Bit j of the row is page first + j.
//
// Runtime args: the worker's row of the map, its first page of the homes, its page count. Common: the addresses of
// the homes and the map, then NR page ranges (lo, hi) inside a core's HP pages that hold state.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);   // L1: the map row, NB pages in flight
    constexpr uint32_t MB = get_compile_time_arg_val(1);   // bytes of a map row
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t HP = get_compile_time_arg_val(3);
    constexpr uint32_t NR = get_compile_time_arg_val(4);
    constexpr auto a_arr = TensorAccessorArgs<5>();
    constexpr auto a_map = TensorAccessorArgs<a_arr.next_compile_time_args_offset()>();
    const auto t_arr = TensorAccessor(a_arr, get_common_arg_val<uint32_t>(0), 4096);
    const auto t_map = TensorAccessor(a_map, get_common_arg_val<uint32_t>(1), MB);
    const uint32_t row = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(1), end = first + get_arg_val<uint32_t>(2);
    const uint32_t l1 = get_write_ptr(cb), buf = l1 + MB;
    uint32_t* map = reinterpret_cast<uint32_t*>(l1);
    for (uint32_t i = 0; i < MB / 4; i++) map[i] = 0;
    uint32_t at[NB], n = 0;
    auto test = [&]() {
        noc_async_read_barrier();
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t* p = reinterpret_cast<const uint32_t*>(buf + i * 4096);
            uint32_t any = 0;
            for (uint32_t k = 0; k < 1024; k++) any |= p[k];
            if (any) map[at[i] >> 5] |= 1u << (at[i] & 31);
        }
        n = 0;
    };
    for (uint32_t c = first / HP; c * HP < end; c++) {
        for (uint32_t r = 0; r < NR; r++) {
            const uint32_t lo = c * HP + get_common_arg_val<uint32_t>(2 + 2 * r), hi = c * HP + get_common_arg_val<uint32_t>(3 + 2 * r);
            for (uint32_t p = lo < first ? first : lo; p < hi && p < end; p++) {
                noc_async_read(t_arr.get_noc_addr(p, 0), buf + n * 4096, 4096);
                at[n++] = p - first;
                if (n == NB) test();
            }
        }
    }
    test();
    noc_async_write(l1, t_map.get_noc_addr(row, 0), MB);
    noc_async_write_barrier();
}
