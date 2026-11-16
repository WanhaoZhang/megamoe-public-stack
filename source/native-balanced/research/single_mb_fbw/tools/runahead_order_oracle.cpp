#include "mega_moe_runahead_order.h"
extern "C" __declspec(dllexport) int runahead_enabled(int mode,int bf16,int ready,int restore,int extent,int home) {
    return MegaFluxRunahead::Enabled(mode,bf16!=0,ready!=0,restore,extent,home);
}
extern "C" __declspec(dllexport) int runahead_home_first(int mode,int enabled,int producer) {
    return MegaFluxRunahead::HomeFirst(mode,enabled!=0,producer!=0);
}
extern "C" __declspec(dllexport) int redispatch_transport_rows(int bytes,int hidden,int tilingRows) {
    return MegaFluxRunahead::TransportRows(bytes,hidden,tilingRows);
}
