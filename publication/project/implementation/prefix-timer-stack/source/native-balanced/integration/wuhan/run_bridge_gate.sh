#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?candidate composite source root}
OUT=${2:?absolute evidence directory inside container}
EP=${3:?2, 4, or 8}
shift 3
[[ "$EP" =~ ^(2|4|8)$ ]]
ENTRY=${WUHAN_GATE_ENTRY:-test_bridge_npu.py}
case "$ENTRY" in
    test_bridge_npu.py) ;;
    test_expert_fsdp2.py) test "$EP" = 8 ;;
    replay_captured_layer.py) ;;
    *) echo 'Unsupported device gate entry' >&2; exit 2 ;;
esac
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
export PYTHONPATH="/project/python-runtime:$ROOT/ascend_deepep_hkq:$ROOT/FSDPTurbo:$ROOT/MindSpeed-LLM:/project/control-source-a2/MoonEP:${PYTHONPATH:-}"
VARIANT=${WUHAN_WORKER_VARIANT:-s9-ported}
[[ "$VARIANT" =~ ^[a-z0-9-]+$ ]]
export WORKER_NATIVE="$ROOT/ascend_deepep_hkq/worker-variants/$VARIANT"
export LD_LIBRARY_PATH="$WORKER_NATIVE:$ROOT/ascend_deepep_hkq/ascend_deepep/lib:${LD_LIBRARY_PATH:-}"
export MEGAFLUX_SHMEM_PORT=29872
export HCCL_EXEC_TIMEOUT=240
export ASCEND_PROCESS_LOG_PATH="$OUT/plog"
export OMP_NUM_THREADS=1
export WUHAN_POOL_AUDIT=1
mkdir -p "$OUT"
if [[ "${WUHAN_NNODES:-1}" != 1 ]]; then
    [[ "$ENTRY" == test_expert_fsdp2.py && "$EP" == 8 ]]
    python "$(dirname "$0")/validate_launch.py"
fi
python -m torch.distributed.run --nnodes="${WUHAN_NNODES:-1}" --node-rank="${WUHAN_NODE_RANK:-0}" --master-addr="${WUHAN_MASTER_ADDR:-127.0.0.1}" \
    --master-port="${WUHAN_MASTER_PORT:-29871}" --nproc-per-node="$EP" \
    "$ROOT/ascend_deepep_hkq/integration/wuhan/runtime_bootstrap.py" \
    "$ROOT/ascend_deepep_hkq/integration/wuhan/$ENTRY" --out "$OUT" "$@"
