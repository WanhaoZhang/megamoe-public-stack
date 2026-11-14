#pragma once
#include "mega_moe_stream_descriptor.h"
struct MegaMoeFullDescriptor {
    MegaMoeStreamDescriptor forward;
    uint64_t grad_ready, grad_trace;
};
static_assert(sizeof(MegaMoeFullDescriptor)==25*sizeof(int64_t), "full pipeline ABI");
