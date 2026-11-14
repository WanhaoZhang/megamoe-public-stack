// Independent diagnostic ABI. No SHMEM, planner or main native ABI dependency.
#include <cstdint>
#include "kernel_operator.h"
using namespace AscendC;

#if defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
__mix__(1, 2)
#endif
__global__ __aicore__ void deepep_clock_probe_kernel(GM_ADDR output)
{
    InitSocState();
#if !defined(ASCEND_DEEPEP_USE_NEW_ASCENDC_FRONTEND)
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif
    // Logical lanes match the Timer exporter (AIC/AIV0/AIV1 per group).
    // Physical-core placement is not recorded and must not be claimed.
    const uint32_t group = g_coreType == AIC
        ? GetBlockIdx() / GetTaskRation() : GetBlockIdx() / 2;
    const uint32_t lane = 3 * group + (g_coreType == AIC ? 0 : 1 + GetBlockIdx() % 2);
    const int64_t cycle = static_cast<int64_t>(GetSystemCycle());
    AscendC::WriteGmByPassDCache<int64_t>(reinterpret_cast<__gm__ int64_t*>(output) + lane, cycle);
}

extern "C" __attribute__((visibility("default"))) int deepep_clock_probe_abi() { return 1; }
extern "C" __attribute__((visibility("default"))) void deepep_clock_probe_launch(void* output, void* stream)
{
    deepep_clock_probe_kernel<<<32, nullptr, stream>>>(reinterpret_cast<uint8_t*>(output));
}
