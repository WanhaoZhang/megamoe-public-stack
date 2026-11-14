// #include "kernel_operator.h"
// #include "shmem.h"

#include "acl/acl.h"
#include "kernel_operator.h"

#include "ascend_deepep/kernels/launch.h"

using namespace AscendC;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void dispatch_kernel(
    GM_ADDR x,
    GM_ADDR topk_idx,
    GM_ADDR topk_weights,
    GM_ADDR symmetric_buffer,
    GM_ADDR recv_x,
    GM_ADDR recv_topk_idx,
    GM_ADDR recv_topk_weights,
    GM_ADDR handle,
    uint64_t ffts_addr) {
    (void)ffts_addr;

    // cce::printf("dispatch_kernel test\n");
    // 当前为空实现，仅作为调用实例
}

extern "C" ASCEND_DEEPEP_EXPORT void dispatch_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* x,
    uint8_t* topk_idx,
    uint8_t* topk_weights,
    uint8_t* symmetric_buffer,
    uint8_t* recv_x,
    uint8_t* recv_topk_idx,
    uint8_t* recv_topk_weights,
    uint8_t* handle,
    uint64_t ffts_addr) {
    dispatch_kernel<<<block_dim, nullptr, stream>>>(
        x,
        topk_idx,
        topk_weights,
        symmetric_buffer,
        recv_x,
        recv_topk_idx,
        recv_topk_weights,
        handle,
        ffts_addr);
}
