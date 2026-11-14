#include "mega_moe_moonep_direct.h"
#include <cassert>
#include <cstdint>

int main() {
    using MegaFluxMoonEPDirect::KeepHome;
    // Exact equality and the first rejected row, including a product that
    // exceeds int32. The helper is the same header compiled into the device TU.
    assert(KeepHome(1050, 1000, 8, 1050));
    assert(!KeepHome(1051, 1000, 8, 1050));
    assert(KeepHome(25804, 24576, 8, 1050));
    assert(!KeepHome(25805, 24576, 8, 1050));
    assert(KeepHome(10500000, 10000000, 8, 1050));
    assert(!KeepHome(10500001, 10000000, 8, 1050));
    assert(!KeepHome(1000, 1000, 8, 0));
    assert(!KeepHome(1000, 1000, 16, 1050));
    assert(!KeepHome(1000, 1000, 4, 1050));
    assert(!KeepHome(0, 0, 8, 1050));
}
