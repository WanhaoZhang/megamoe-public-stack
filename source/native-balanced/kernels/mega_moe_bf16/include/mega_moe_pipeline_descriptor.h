#pragma once
#include "mega_moe_device_descriptor.h"

// Forward-only extension; backward still uses the unmodified 19-word ABI.
struct MegaMoePipelineDescriptor {
    MegaMoeDeviceReplicaDescriptor device;
    uint64_t ready, trace;
};
static_assert(sizeof(MegaMoePipelineDescriptor)==21*sizeof(int64_t), "pipeline ABI");
