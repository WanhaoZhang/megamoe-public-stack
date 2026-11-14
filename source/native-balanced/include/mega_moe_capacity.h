#pragma once
#ifndef MEGAMOE_MAX_LOCAL_EXPERTS
#define MEGAMOE_MAX_LOCAL_EXPERTS 32
#endif
static_assert(MEGAMOE_MAX_LOCAL_EXPERTS == 32 || MEGAMOE_MAX_LOCAL_EXPERTS == 64, "local expert capacity must be 32 or 64");
