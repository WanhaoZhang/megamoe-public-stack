#!/usr/bin/env bash
# Fast BigOpV2 precision matrix: one torchrun/HCCL/SHMEM session for all cases;
# recreate the shape-dependent MegaMoe handle for each case.
# This is also run_precision.sh's default path.  Set PRECISION_ISOLATE_CASES=1
# on run_precision.sh only when per-case process isolation is needed.
# Backward hard gate: finite + cosine>0.99 + double-hundred (including the
# BF16 small-value absolute-error rule). Double-thousand is warning-only.
# Saved-state path selection:
#   MEGAMOE_SAVE_GMM1_OUTPUT=bf16|mxfp8|none|auto (default bf16)
#   MXFP8 uses a mode-aware backward golden for the hard gate and reports
#   exact-BF16 compression drift separately as INFO.
#   MEGAMOE_SWIGLU_LIMIT=<float> (default 0)
set -eo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
IP_PORT=${ASCEND_DEEPEP_SHMEM_IP_PORT:-tcp://127.0.0.1:8899}
TEST_MODE=${1:-both}
FUSED_MODE=${2:-train}
NPROC=${PRECISION_NPROC:-2}
TIMEOUT=${PRECISION_FAST_TIMEOUT:-3600}
MASTER_PORT=${PRECISION_MASTER_PORT:-29517}
MEGAMOE_SAVE_GMM1_OUTPUT=${MEGAMOE_SAVE_GMM1_OUTPUT:-bf16}
MEGAMOE_SWIGLU_LIMIT=${MEGAMOE_SWIGLU_LIMIT:-0}

case "$MEGAMOE_SAVE_GMM1_OUTPUT" in
    auto|none|bf16|mxfp8) ;;
    *)
        echo "[error] MEGAMOE_SAVE_GMM1_OUTPUT must be auto, none, bf16 or mxfp8; got '$MEGAMOE_SAVE_GMM1_OUTPUT'" >&2
        exit 2
        ;;
esac

DEFAULT_CASES=(
    "16:512:512:16:1"
    "1000:512:512:8:2"
    "128:1024:2048:16:2"
    "4096:1024:2048:16:2"
    "512:2048:3072:16:2"
    "256:2048:2048:16:8"
)
EP8_CASES=(
    "16:512:512:16:1"
    "1000:512:512:8:2"
    "128:1024:2048:16:2"
    "4096:1024:2048:16:2"
    "128:1024:2048:32:4"
    "512:2048:3072:64:2"
)

if [[ -z "${PRECISION_CASES:-}" ]]; then
    if [[ "$NPROC" -eq 8 ]]; then
        PRECISION_CASES="${EP8_CASES[*]}"
    else
        PRECISION_CASES="${DEFAULT_CASES[*]}"
    fi
fi

export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_DEEPEP_SHMEM_IP_PORT="$IP_PORT"
export SHMEM_UID_SESSION_ID="${IP_PORT#tcp://}"
export TEST_MODE FUSED_MODE PRECISION_CASES
export MEGAMOE_SAVE_GMM1_OUTPUT MEGAMOE_SWIGLU_LIMIT
cd "$HERE"

if ! python3 -c "import torch_npu; import ascend_deepep; print('[ok] ascend_deepep import OK')"; then
    echo "[error] ascend_deepep import FAILED (see stderr above)." >&2
    exit 1
fi

LOG=run_precision_fast.log
echo "[run] $(date +%H:%M:%S) fast precision EP${NPROC} ip=$IP_PORT " \
     "test=$TEST_MODE fused=$FUSED_MODE cases='$PRECISION_CASES' " \
     "save_gmm1=$MEGAMOE_SAVE_GMM1_OUTPUT swiglu_limit=$MEGAMOE_SWIGLU_LIMIT " \
     "timeout=${TIMEOUT}s" | tee "$LOG"

timeout -k 30 "$TIMEOUT" torchrun --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" test_precision_fast.py 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
if [[ "$RC" -ne 0 ]]; then
    echo "[summary] fast precision Failed rc=$RC; log: $LOG" | tee -a "$LOG"
    exit "$RC"
fi
if grep -q '^\[fast summary\] Failed:' "$LOG"; then
    echo "[summary] fast precision Failed; log: $LOG" | tee -a "$LOG"
    exit 2
elif grep -q '^\[fast summary\] WARNING:' "$LOG"; then
    echo "[summary] fast precision WARNING; hard gate passed, see: $LOG" | tee -a "$LOG"
else
    echo "[summary] fast precision SUCCESS; log: $LOG" | tee -a "$LOG"
fi
