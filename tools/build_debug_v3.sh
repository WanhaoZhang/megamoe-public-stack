#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
ROOT="$TASK/source/native-balanced"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS=64
export MAX_JOBS=4
export CXXFLAGS=-DMEGAFLUX_DEBUG_V3_GUARDS=1
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1
export MEGAFLUX_SHARED_EXPERT_FORWARD=1 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 MEGAFLUX_OWNER_DOUBLE_BUFFER=OFF
cd "$ROOT"
bash scripts/build_megaflux_worker_pipeline.sh ep8-cap64-v3guard 0 0 0
