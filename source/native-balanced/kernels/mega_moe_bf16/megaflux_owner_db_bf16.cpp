// Independent SO with the accepted early ABI; select its directory explicitly.
#include "megaflux_early_bf16.cpp"
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_owner_db_version() {return 1;}
