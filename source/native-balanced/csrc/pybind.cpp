#include <torch/extension.h>
#include <pybind11/stl.h>
#include "include/mega_moe_large_ep_limits.h"

void RegisterMegaMoeReplicaWorkspace();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    RegisterMegaMoeReplicaWorkspace();
    m.def("large_ep_limits",[](){return std::vector<int64_t>{MegaMoeLargeEP::MaxRanks,
        MegaMoeLargeEP::MaxExperts,MegaMoeLargeEP::MaxCopies,MegaMoeLargeEP::MaxLocalSlots,
        MegaMoeLargeEP::MaxPhysicalExperts};});
}
