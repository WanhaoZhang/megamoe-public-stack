#!/usr/bin/env bash
# Use on an admitted idle node/container with CANN, torch_npu, SHMEM and binding.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=;VARIANTS=;WORLD=8;PORT=32250
args=()
while (( $# )); do
  case "$1" in
    --out) OUT=${2:?};shift 2;;
    --variants) VARIANTS=${2:?};shift 2;;
    --world-size) WORLD=${2:?};shift 2;;
    --master-port) PORT=${2:?};shift 2;;
    *) args+=("$1");shift;;
  esac
done
: "${OUT:?--out NEW_DIRECTORY required}"
: "${VARIANTS:?--variants /absolute/path/worker-variants/prefix required}"
[[ "$WORLD" =~ ^(2|4|8)$ && "$PORT" =~ ^[0-9]+$ ]] || exit 2
[[ "$OUT" = /* && "$VARIANTS" = /* ]] || { echo 'Use absolute out/variants paths';exit 2; }
test ! -e "$OUT" || { echo 'Output exists; choose a fresh directory';exit 2; }
for mode in off on; do test -f "$VARIANTS-$mode/timer-capture-build.json";done
mkdir -p "$(dirname "$OUT")";mkdir "$OUT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export WUHAN_PUBLISH_SYNC=device WUHAN_CONTROL_TAPE_DEVICE=npu
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
unset WUHAN_REPLAY_CAPTURE_DIR
BASE_LD=${LD_LIBRARY_PATH:-}
for mode in off on; do
  export WORKER_NATIVE="$VARIANTS-$mode"
  export LD_LIBRARY_PATH="$WORKER_NATIVE:$ROOT/ascend_deepep/lib:$BASE_LD"
  timer=0;reference=()
  if [[ "$mode" == on ]]; then timer=1;reference=(--reference "$OUT/control");fi
  kind=control;if [[ "$mode" == on ]];then kind=timer;fi
  mkdir "$OUT/$kind"
  torchrun --nnodes=1 --nproc-per-node="$WORLD" --master-port="$PORT" \
    "$ROOT/integration/wuhan/runtime_bootstrap.py" --native-only \
    "$ROOT/integration/wuhan/capture_kernel_timer.py" \
    --out "$OUT/$kind" --timer "$timer" "${reference[@]}" "${args[@]}" \
    >"$OUT/$kind/run.log" 2>&1
done
python3 "$ROOT/integration/wuhan/export_timer_trace.py" \
  --input "$OUT/timer" --out "$OUT/kernel-trace" --world-size "$WORLD"
python3 "$ROOT/integration/wuhan/verify_timer_capture.py" --out "$OUT" --world-size "$WORLD"
printf 'Timer trace completed: %s/kernel-trace\n' "$OUT"
