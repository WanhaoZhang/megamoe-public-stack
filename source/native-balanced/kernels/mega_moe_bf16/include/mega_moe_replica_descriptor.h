#pragma once
#include <cstdint>

// Host-built, passed by value as launch arguments. Plan storage is retained
// by the autograd context through backward; all addresses are on this device.
struct MegaMoeReplicaDescriptor {
    uint64_t base, plan, home1, home2, replica1, replica2, partial1, partial2;
    int64_t home, slots, hidden, ffn, rank, ranks;
};
static_assert(sizeof(MegaMoeReplicaDescriptor)==14*sizeof(int64_t), "replica ABI");
