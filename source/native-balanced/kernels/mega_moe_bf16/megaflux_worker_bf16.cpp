#include "mega_moe_capacity.h"
#define DEEPEP_BF16_WORKER_PIPELINE 1
#if defined(ENABLE_TIMER) && (!defined(MEGAFLUX_COORDINATE_TIMER) || MEGAFLUX_COORDINATE_TIMER)
#define DEEPEP_BF16_STAGE_TIMER 1
#endif
#define megaflux_full_bf16 megaflux_early_bf16
#define megaflux_full_bf16_kernel_do megaflux_early_bf16_kernel_do
#define megaflux_full_bf16_timer_enabled megaflux_early_bf16_timer_enabled
#include "kernel_operator.h"
using namespace AscendC; // TimerV2's original header uses unqualified device intrinsics.
#include "mega_moe_worker_queue.h"
#include "megaflux_full_bf16.cpp"
#ifndef MEGAFLUX_BACKWARD_NONE_GM
#define MEGAFLUX_BACKWARD_NONE_GM 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_backward_none_gm_version(){return MEGAFLUX_BACKWARD_NONE_GM ? 1 : 0;}
#ifndef MEGAFLUX_SHARED_EXPERT_BACKWARD
#define MEGAFLUX_SHARED_EXPERT_BACKWARD 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_forward_saved_modes(){
#if defined(MEGAFLUX_SHARED_EXPERT_FORWARD) && MEGAFLUX_SHARED_EXPERT_FORWARD
    return 7;
#else
    return 0;
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_backward_saved_modes(){return MEGAFLUX_SHARED_EXPERT_BACKWARD ? 7 : 0;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_backward_version(){return MEGAFLUX_SHARED_EXPERT_BACKWARD ? 1 : 0;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_backward_v2_version(){return MEGAFLUX_SHARED_EXPERT_BACKWARD ? 2 : 0;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_swiglu_decoupling_mode(){return DEEPEP_BF16_SWIGLU_DECOUPLE;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_gm_swiglu_tile_ready(){
#if defined(DEEPEP_BF16_GM_SWIGLU_TILE_READY) && DEEPEP_BF16_GM_SWIGLU_TILE_READY
    return 1;
#else
    return 0;
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_forward_timer_schema(){
#if defined(MEGAFLUX_SUMMARY_TIMER)
return 3;
#else
return 2;
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_tile_pipeline_version(){return 1;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_pipeline_version(){return 3;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_shared_expert_forward_abi(){
#if defined(MEGAFLUX_SHARED_EXPERT_FORWARD) && MEGAFLUX_SHARED_EXPERT_FORWARD
    return MegaMoeSharedExpertLayout::kAbi;
#else
    return 0;
#endif
}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_backward_replicated_dw1_early(){return MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_restore_panel_acquire_cache(){return MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_restore_p1_panel_acquire_cache(){return MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_pipeline_flags(){return MEGAFLUX_WORKER_FORWARD|(MEGAFLUX_WORKER_BACKWARD<<1);}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_gradient_priority(){return MEGAFLUX_GRADIENT_PRIORITY;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_early_dw1_wide(){return MEGAFLUX_EARLY_DW1_WIDE;}
#ifndef MEGAFLUX_OWNER_LATE_ONLY
#define MEGAFLUX_OWNER_LATE_ONLY 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_owner_late_only(){return MEGAFLUX_OWNER_LATE_ONLY;}
#ifndef MEGAFLUX_REDISPATCH_P1_RUNAHEAD
#define MEGAFLUX_REDISPATCH_P1_RUNAHEAD 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_redispatch_p1_runahead(){return MEGAFLUX_REDISPATCH_P1_RUNAHEAD;}
#ifndef MEGAFLUX_GRADIENT_KEY_CACHE
#define MEGAFLUX_GRADIENT_KEY_CACHE 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_gradient_key_cache(){return MEGAFLUX_GRADIENT_KEY_CACHE;}
#ifndef MEGAFLUX_REDISPATCH_TRANSPORT_BATCH
#define MEGAFLUX_REDISPATCH_TRANSPORT_BATCH 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_redispatch_transport_batch(){return MEGAFLUX_REDISPATCH_TRANSPORT_BATCH;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_shared_prefix(){return MEGAFLUX_WORKER_SHARED_PREFIX;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_external_refresh(){return MEGAFLUX_WORKER_EXTERNAL_REFRESH;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_owner_rows(){return MEGAFLUX_OWNER_ROWS;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_owner_compact(){return MEGAFLUX_OWNER_COMPACT;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_owner_early_ratio(){return DEEPEP_BF16_OWNER_RATIO;}
#ifndef MEGAFLUX_OWNER_AIV1_TAIL
#define MEGAFLUX_OWNER_AIV1_TAIL 0
#endif
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_worker_owner_aiv1_tail(){return MEGAFLUX_OWNER_AIV1_TAIL;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_backward_restore_mode(){return MEGAFLUX_BACKWARD_RESTORE;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_restore_operand_ready(){return MEGAFLUX_RESTORE_OPERAND_READY;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_restore_delay_cycles(){return MEGAFLUX_RESTORE_DELAY_CYCLES;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_max_ep(){return MegaMoeLargeEP::MaxRanks;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_max_copy_slots(){return MegaMoeLargeEP::MaxCopies;}

extern "C" ASCEND_DEEPEP_EXPORT int megaflux_dispatch_ready_largest(){return MEGAFLUX_DISPATCH_READY_LARGEST;}
extern "C" ASCEND_DEEPEP_EXPORT int megaflux_dispatch_stage_schema(){
#if defined(ENABLE_TIMER) && defined(MEGAFLUX_COORDINATE_TIMER) && !MEGAFLUX_COORDINATE_TIMER
    return 10; // Original per-expert TimerV2 only; no coordinate records.
#else
#if defined(DEEPEP_BF16_STAGE_TIMER)
#if defined(MEGAFLUX_SUMMARY_TIMER)
#if defined(MEGAFLUX_CALIBRATION_TIMER)
    return 9;
#else
    return MEGAFLUX_OWNER_AIV1_TAIL?8:7;
#endif
#else
    return 6;
#endif
#else
    return 0;
#endif
#endif
}

extern "C" ASCEND_DEEPEP_EXPORT int megaflux_max_local_experts(){return MEGAMOE_MAX_LOCAL_EXPERTS;}
