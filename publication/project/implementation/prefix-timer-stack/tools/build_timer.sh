set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
ROOT="$TASK/source/native-balanced"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS=64 MAX_JOBS=4
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1
export MEGAFLUX_SHARED_EXPERT_FORWARD=1 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 MEGAFLUX_OWNER_DOUBLE_BUFFER=OFF
cd "$ROOT"
cmake -S . -B build-worker-unified-cap64-calibration-timer -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON -DMEGAMOE_MAX_LOCAL_EXPERTS=64 -DASCEND_DEEPEP_ENABLE_TIMER=ON -DMEGAFLUX_SUMMARY_TIMER=ON -DMEGAFLUX_CALIBRATION_TIMER=ON -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" -DMEGAFLUX_WORKER_FORWARD=0 -DMEGAFLUX_WORKER_BACKWARD=0 -DMEGAFLUX_WORKER_SHARED_PREFIX=0 -DMEGAFLUX_WORKER_EXTERNAL_REFRESH=0 -DMEGAFLUX_OWNER_ROWS=16 -DMEGAFLUX_OWNER_COMPACT=0 -DMEGAFLUX_BACKWARD_RESTORE=2 -DMEGAFLUX_RESTORE_OPERAND_READY=1 -DMEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 -DMEGAFLUX_OWNER_DOUBLE_BUFFER=OFF -DMEGAFLUX_SHARED_EXPERT_FORWARD=1 -DMEGAFLUX_SHARED_EXPERT_BACKWARD=1
cmake --build build-worker-unified-cap64-calibration-timer --target ascend_deepep_megaflux_worker_bf16 ascend_deepep_link_calibration --parallel 4
mkdir -p worker-variants/unified-cap64-calibration-timer
cp build-worker-unified-cap64-calibration-timer/lib/libascend_deepep_megaflux_worker_bf16.so worker-variants/unified-cap64-calibration-timer/libascend_deepep_megaflux_early_bf16.so
cp build-worker-unified-cap64-calibration-timer/lib/libascend_deepep_link_calibration.so ascend_deepep/lib/libascend_deepep_link_calibration.so
sha256sum worker-variants/unified-cap64-calibration-timer/*.so
