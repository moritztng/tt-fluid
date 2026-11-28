// Image placement and state collection (lower/image_dev.py upload_placed, download_state): one processor moves its
// share of a device array's 4 KB pages between the array and a staging tensor that holds only the pages that are
// not zero, in page order. A chip moves DRAM pages two orders of magnitude faster than the host's PCIe link, and
// most pages of an image and of a state are zero.
//
// Forward (BACK 0): each staged page is copied to its place and every other page is written as zeros, so the array
// ends up exactly as a full write from the host would leave it. Backward (BACK 1): each marked page of the array is
// copied to the stage, and worker 0 writes a zero page into the stage row the host names (where it looks up the
// pages that were not marked).
//
// Runtime args: the worker's row of the page map, its first page of the array, its page count, the staged page of
// its first set bit. Common: the addresses of the array, the stage and the map, and the zero row. Bit j of the map
// row is page first + j: set = staged.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);   // L1: the map row, a zero page, NB pages in flight
    constexpr uint32_t MB = get_compile_time_arg_val(1);   // bytes of a map row
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t BACK = get_compile_time_arg_val(3);
    constexpr auto a_arr = TensorAccessorArgs<4>();
    constexpr auto a_stg = TensorAccessorArgs<a_arr.next_compile_time_args_offset()>();
    constexpr auto a_map = TensorAccessorArgs<a_stg.next_compile_time_args_offset()>();
    const auto t_arr = TensorAccessor(a_arr, get_common_arg_val<uint32_t>(0), 4096);
    const auto t_stg = TensorAccessor(a_stg, get_common_arg_val<uint32_t>(1), 4096);
    const auto t_map = TensorAccessor(a_map, get_common_arg_val<uint32_t>(2), MB);
    const uint32_t row = get_arg_val<uint32_t>(0), first = get_arg_val<uint32_t>(1), count = get_arg_val<uint32_t>(2);
    uint32_t staged = get_arg_val<uint32_t>(3);
    const uint32_t l1 = get_write_ptr(cb), zero = l1 + MB, buf = zero + 4096;
    const uint32_t* map = reinterpret_cast<const uint32_t*>(l1);
    uint32_t* z = reinterpret_cast<uint32_t*>(zero);
    for (uint32_t i = 0; i < 1024; i++) z[i] = 0;
    noc_async_read(t_map.get_noc_addr(row, 0), l1, MB);
    noc_async_read_barrier();
    uint32_t to[NB], n = 0;
    for (uint32_t j = 0; j <= count; j++) {
        if (j < count && (map[j >> 5] >> (j & 31) & 1)) {
            noc_async_read(BACK ? t_arr.get_noc_addr(first + j, 0) : t_stg.get_noc_addr(staged, 0), buf + n * 4096, 4096);
            to[n++] = BACK ? staged : first + j;
            staged++;
        } else if (j < count && !BACK) {
            noc_async_write(zero, t_arr.get_noc_addr(first + j, 0), 4096);
        }
        if (n == NB || (j == count && n)) {   // the batch is in L1: send it on, and reuse the buffer once it has left
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; i++)
                noc_async_write(buf + i * 4096, BACK ? t_stg.get_noc_addr(to[i], 0) : t_arr.get_noc_addr(to[i], 0), 4096);
            noc_async_writes_flushed();
            n = 0;
        }
    }
    if (BACK && row == 0) noc_async_write(zero, t_stg.get_noc_addr(get_common_arg_val<uint32_t>(3), 0), 4096);
    noc_async_write_barrier();
}
