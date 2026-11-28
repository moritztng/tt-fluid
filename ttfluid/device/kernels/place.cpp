// Image placement (lower/image_dev.py place): one processor lays its share of a device array. The host wrote only the
// pages that hold a value into a staging tensor, in page order; here each of them is copied to its place and every
// other page is written as zeros, so the array ends up exactly as a full write from the host would leave it. A chip
// moves DRAM pages two orders of magnitude faster than the host's PCIe write, and most of an image is zero pages.
//
// Runtime args: the processor's row of the page map, its first page of the array, its page count, the staged page of
// its first set bit. Common: the addresses of the array, the staging tensor and the map. Bit j of the map row is page
// first + j: set = staged.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);   // L1: the map row, a zero page, NB pages in flight
    constexpr uint32_t MB = get_compile_time_arg_val(1);   // bytes of a map row
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr auto a_dst = TensorAccessorArgs<3>();
    constexpr auto a_src = TensorAccessorArgs<a_dst.next_compile_time_args_offset()>();
    constexpr auto a_map = TensorAccessorArgs<a_src.next_compile_time_args_offset()>();
    const auto t_dst = TensorAccessor(a_dst, get_common_arg_val<uint32_t>(0), 4096);
    const auto t_src = TensorAccessor(a_src, get_common_arg_val<uint32_t>(1), 4096);
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
        const bool set = j < count && (map[j >> 5] >> (j & 31) & 1);
        if (set) {
            noc_async_read(t_src.get_noc_addr(staged++, 0), buf + n * 4096, 4096);
            to[n++] = first + j;
        } else if (j < count) {
            noc_async_write(zero, t_dst.get_noc_addr(first + j, 0), 4096);
        }
        if (n == NB || (j == count && n)) {   // the batch is in L1: send it on, and reuse the buffer once it has left
            noc_async_read_barrier();
            for (uint32_t i = 0; i < n; i++) noc_async_write(buf + i * 4096, t_dst.get_noc_addr(to[i], 0), 4096);
            noc_async_writes_flushed();
            n = 0;
        }
    }
    noc_async_write_barrier();
}
