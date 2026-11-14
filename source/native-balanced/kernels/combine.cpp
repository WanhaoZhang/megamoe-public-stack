// #include "shmem.h"

#include "kernel_operator.h"

#include "ascend_deepep/kernels/launch.h"

using namespace AscendC;

extern "C" [[bisheng::core_ratio(0, 1)]] __global__ __aicore__ void combine_kernel(
    GM_ADDR expert_output,
    GM_ADDR handle,
    GM_ADDR topk_weights,
    GM_ADDR symmetric_buffer,
    GM_ADDR out,
    uint64_t ffts_addr) {
    (void)ffts_addr;
  
        // cce::printf("combine_kernel test\n");
        // 当前为空实现，仅作为调用实例
}

extern "C" ASCEND_DEEPEP_EXPORT void combine_kernel_do(
    uint32_t block_dim,
    void* stream,
    uint8_t* expert_output,
    uint8_t* handle,
    uint8_t* topk_weights,
    uint8_t* symmetric_buffer,
    uint8_t* out,
    uint64_t ffts_addr) {
    combine_kernel<<<block_dim, nullptr, stream>>>(
        expert_output,
        handle,
        topk_weights,
        symmetric_buffer,
        out,
        ffts_addr);
}
