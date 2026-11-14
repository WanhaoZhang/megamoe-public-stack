#pragma once
#include "mega_moe_pipeline_descriptor.h"
struct MegaMoeStripeDescriptor {
    MegaMoeDeviceReplicaDescriptor device;
    uint64_t ready, trace, stripes;
};
static_assert(sizeof(MegaMoeStripeDescriptor)==22*sizeof(int64_t), "stripe ABI");
