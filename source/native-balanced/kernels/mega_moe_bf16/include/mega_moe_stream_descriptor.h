#pragma once
#include "mega_moe_stripe_descriptor.h"
struct MegaMoeStreamDescriptor {
    MegaMoeDeviceReplicaDescriptor device;
    uint64_t ready, trace, stripes, mode;
};
static_assert(sizeof(MegaMoeStreamDescriptor)==23*sizeof(int64_t), "stream ABI");
