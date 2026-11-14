#pragma once
#include "mega_moe_replica_descriptor.h"

// Independent versioned ABI. Legacy 14-word descriptors/libraries are unchanged.
struct MegaMoeDeviceReplicaDescriptor {
    MegaMoeReplicaDescriptor replica;
    uint64_t header, logical;
    int64_t epoch, capacity, abi;
};
static_assert(sizeof(MegaMoeDeviceReplicaDescriptor)==19*sizeof(int64_t), "device replica ABI");
