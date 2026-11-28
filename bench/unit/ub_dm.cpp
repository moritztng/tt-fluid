// Unit-cost bench, data mover (BRISC): input tiles into cb_v, wait for compute, header page and cb_v back to DRAM.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_v = get_compile_time_arg_val(0);
    constexpr uint32_t cb_hdr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_done = get_compile_time_arg_val(2);
    constexpr uint32_t nv = get_compile_time_arg_val(3);
    constexpr auto a_in = TensorAccessorArgs<4>();
    constexpr auto a_out = TensorAccessorArgs<a_in.next_compile_time_args_offset()>();
    const auto t_in = TensorAccessor(a_in, get_arg_val<uint32_t>(0), nv * 4096);
    const auto t_out = TensorAccessor(a_out, get_arg_val<uint32_t>(1), 4096 + nv * 4096);
    const uint32_t arena = get_write_ptr(cb_v);
    cb_reserve_back(cb_v, nv);
    noc_async_read(t_in.get_noc_addr(0), arena, nv * 4096);
    noc_async_read_barrier();
    cb_push_back(cb_v, nv);
    cb_wait_front(cb_done, 1);
    const uint64_t dst = t_out.get_noc_addr(0);
    noc_async_write(get_write_ptr(cb_hdr), dst, 4096);
    noc_async_write(arena, dst + 4096, nv * 4096);
    noc_async_write_barrier();
}
