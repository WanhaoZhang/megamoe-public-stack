#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/step-lifetime-prefix-timer-20261116/stack
BACKEND=$1
MODE=$2
TAG=$3
shift 3
OUT="$TASK/evidence/$TAG"
mkdir -p "$OUT"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH="$TASK/tools:$TASK/source/FSDPTurbo:$TASK/source/MindSpeed-LLM:$ASCEND_HOME_PATH/python/site-packages"
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export TRAINING_BACKEND=mindspeed_fsdp
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=1 CPU_AFFINITY_CONF=1
export HCCL_SOCKET_IFNAME=enp35s0f2 GLOO_SOCKET_IFNAME=enp35s0f2
export HCCL_CONNECT_TIMEOUT=180 HCCL_EXEC_TIMEOUT=240 HCCL_NPU_SOCKET_PORT_RANGE=auto
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2
export ASCEND_LAUNCH_BLOCKING=0
export ASCEND_PROCESS_LOG_PATH="$OUT/plog"
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS="${EP8_CAPACITY:-64}"
export MEGAFLUX_CLONE_OWNER_GRAD=1
unset V4LAB_HASH_ROUTE SHMEM_UID_SESSION_ID ASCEND_DEEPEP_FORCE_GM_PLANNER
if [ "$BACKEND" = moonep ]; then
 export PYTHONPATH="$TASK/source/MoonEP:$PYTHONPATH"
 export LD_LIBRARY_PATH="$TASK/source/MoonEP/npu_patch/lib:${LD_LIBRARY_PATH:-}"
else
 NATIVE_ROOT="$TASK/source/native-$BACKEND${EP8_NATIVE_SUFFIX:-}"
 export PYTHONPATH="$NATIVE_ROOT:$PYTHONPATH"
 DEFAULT_WORKER="ep8-cap${ASCEND_DEEPEP_MAX_LOCAL_EXPERTS}-aligned"
 export WORKER_NATIVE="$NATIVE_ROOT/worker-variants/${EP8_WORKER_VARIANT:-$DEFAULT_WORKER}"
 export LD_LIBRARY_PATH="$WORKER_NATIVE:$NATIVE_ROOT/ascend_deepep/lib:/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk/shmem/lib:/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
fi
if [ "$MODE" = meta ]; then
 python3 "$TASK/tools/probe_model.py"
 exit 0
fi
if [ "$MODE" = import ]; then
 if [ "$BACKEND" = moonep ]; then python3 "$TASK/tools/probe_moonep.py"; else python3 "$TASK/tools/probe_native.py"; fi
 exit 0
fi
if [ "$MODE" = lifetime ]; then
 python3 "$TASK/tools/test_plan_lifetime.py"
 exit 0
fi
if [ "$MODE" = links ]; then
 SCRIPT="$TASK/tools/calibrate_links.py"
 EXTRA=()
elif [ "$MODE" = scheduler ]; then
 SCRIPT="$TASK/tools/gate_temporal_scheduler.py"
 EXTRA=()
elif [ "$MODE" = lifetime_math ]; then
 SCRIPT="$TASK/tools/gate_lifetime_math.py"
 EXTRA=()
elif [ "$MODE" = temporal_math ]; then
 SCRIPT="$TASK/tools/gate_runset_math.py"
 EXTRA=()
elif [ "$MODE" = plan ]; then
 SCRIPT="$TASK/tools/gate_unified_plan.py"
 EXTRA=()
elif [ "$MODE" = moonpool ]; then
 SCRIPT="$TASK/tools/gate_moonep_pool.py"
 EXTRA=()
elif [ "$MODE" = sharedpool ]; then
 SCRIPT="$TASK/tools/gate_pool_shared.py"
 EXTRA=()
elif [ "$MODE" = norm ]; then
 SCRIPT="$TASK/tools/gate_global_norm.py"
 EXTRA=()
elif [ "$MODE" = pool ]; then
 SCRIPT="$TASK/tools/gate_pool_replay.py"
 EXTRA=()
elif [ "$MODE" = activation ]; then
 SCRIPT="$TASK/tools/gate_activation.py"
 EXTRA=()
elif [ "$MODE" = scaling ]; then
 SCRIPT="$TASK/tools/gate_fsdp_scaling.py"
 EXTRA=()
elif [ "$MODE" = validate ]; then
 SCRIPT="$TASK/tools/validate.py"
 EXTRA=()
else
 SCRIPT="$TASK/tools/train.py"
 EXTRA=(--root "$TASK")
fi
cd "$TASK"
python3 -m torch.distributed.run --nproc_per_node=8 --nnodes=1 --master_addr=127.0.0.1 --master_port="${EP8_MASTER_PORT:-29131}" "$SCRIPT" --backend "$BACKEND" --out "$OUT" "${EXTRA[@]}" "$@"
