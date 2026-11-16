#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${PLANNER_BENCH_GATED_RUN:?Run through host gate_and_run.py}"
: "${FSDP_ROOT:?Pinned FSDPTurbo reference root}"
: "${MOONEP_ROOT:?Pinned original MoonEP reference root}"
: "${SHMEM_HOME_PATH:?Pinned SHMEM installation}"
: "${WORKER_NATIVE:?Task-owned rebuilt worker variant}"
OUT=${1:?absolute output path};shift
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export TRAINING_BACKEND=mindspeed_fsdp
export PYTHONPATH="$ROOT:$FSDP_ROOT:$MOONEP_ROOT:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$WORKER_NATIVE:$ROOT/ascend_deepep/lib:$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
if [[ -n "${PLANNER_NATIVE:-}" ]]; then
  export LD_LIBRARY_PATH="$PLANNER_NATIVE:$LD_LIBRARY_PATH"
fi
export ASCEND_PROCESS_LOG_PATH="$OUT/plog"
export WUHAN_POOL_AUDIT=0 WUHAN_PUBLISH_SYNC=device WUHAN_CONTROL_TAPE_DEVICE=npu
export ASCEND_DEEPEP_FORCE_GM_PLANNER=0
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=180 ACLSHMEM_TIMEOUT=180
exec timeout --signal=TERM --kill-after=20s 1200s python -m torch.distributed.run \
 --nnodes=1 --nproc-per-node=8 --master-addr=127.0.0.1 --master-port="${BENCH_MASTER_PORT:-29991}" \
 "$ROOT/test/megamoe_planner_test/test_planner_megamoe.py" \
 --out "$OUT" --fsdp-root "$FSDP_ROOT" "$@"
