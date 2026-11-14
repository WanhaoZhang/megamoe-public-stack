#!/bin/bash
# Unified run_speed: single-node AND multi-node auto-detect.
#
# Single-node (default, no extra setup):
#   NPROC=8 bash test/bigop_fused/run_speed.sh 4096 3 10
#
# Multi-node (set NODE_RANK to activate; MASTER_IP required):
#   Machine A (node 0, IP 10.0.0.1):
#     NNODES=2 NODE_RANK=0 MASTER_IP=10.0.0.1 NPROC=8 \
#       bash test/bigop_fused/run_speed.sh 4096 3 10
#   Machine B (node 1):
#     NNODES=2 NODE_RANK=1 MASTER_IP=10.0.0.1 NPROC=8 \
#       bash test/bigop_fused/run_speed.sh 4096 3 10
#
# Multi-node detection: NODE_RANK is set (and non-empty) -> multi-node.
# All node config via env vars; positional args identical in both modes.
#
# Usage (positional, same for single/multi):
#   bash test/bigop_fused/run_speed.sh [seq] [warmup] [iters]
#     seq    default 128 (comma-separated for multiple; <=16384)
#     warmup default 0
#     iters  default 1
#
# Shape / parallelism via env vars:
#   NPROC                  cards per node (default 2)
#   NNODES                 number of nodes (default 2; only used when NODE_RANK set)
#   NODE_RANK              set to 0..N-1 to enable multi-node mode
#   MASTER_IP              required for multi-node; master node's real IP
#   MASTER_PORT            torchrun port (default 29518)
#   SHMEM_PORT             SHMEM symmetric window port (default 8899)
#   BENCH_HIDDEN           K (hidden size, %256==0)
#   BENCH_FFN              F (SwiGLU output width, not x2; N=2F, %512==0)
#   BENCH_EXPERTS          E total experts (must divide NPROC*NNODES)
#   BENCH_TOPK             topK (<=E)
#   BENCH_LOCAL_MEM_BYTES  SHMEM symmetric window (default 1GiB)
#   BIGOP_BWD_RECOMPUTE    bigop backward regime (default 1 = recompute)
#   BIGOP_TIMING           =1 prints bigop per-step tables + bwd decomposition
#   V2_LAYOUT_TIMING       =1 prints v2 router / weight view / fused forward
#   MEGAMOE_TRACE          =1 prints v2 forward total + backward stage table
#                          (requires --enable-timer build)
#   BENCH_JSON_OUT         =<path> dumps structured JSON results
#
# Production example (single-node 8 cards):
#   NPROC=8 BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
#     bash test/bigop_fused/run_speed.sh 4096 3 10
#
# Production example (2-node 16 cards, E=64/EP16/EPR4):
#   # Machine A:
#   NNODES=2 NODE_RANK=0 MASTER_IP=10.0.0.1 NPROC=8 \
#     BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
#     bash test/bigop_fused/run_speed.sh 4096 3 10
#   # Machine B:
#   NNODES=2 NODE_RANK=1 MASTER_IP=10.0.0.1 NPROC=8 \
#     BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
#     bash test/bigop_fused/run_speed.sh 4096 3 10
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
SEQS=${1:-"128"}
WARMUP=${2:-0}
ITERS=${3:-1}

NPROC=${NPROC:-2}
MASTER_PORT=${MASTER_PORT:-29518}
SHMEM_PORT=${SHMEM_PORT:-8899}

# --- Single/multi-node detection ---
NODE_RANK=${NODE_RANK:-}
if [[ -n "$NODE_RANK" ]]; then
    # Multi-node mode
    NNODES=${NNODES:-2}
    MASTER_IP=${MASTER_IP:?Multi-node requires MASTER_IP}
    IP_PORT="tcp://${MASTER_IP}:${SHMEM_PORT}"
    TORCHRUN_NODE_ARGS="--nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP}"
    MODE="multi-node nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_IP}"
else
    # Single-node mode (default)
    IP_PORT=${ASCEND_DEEPEP_SHMEM_IP_PORT:-tcp://127.0.0.1:${SHMEM_PORT}}
    TORCHRUN_NODE_ARGS=""
    MODE="single-node"
fi

# Self-contained: PYTHONPATH points to bigop_fused itself.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_DEEPEP_SHMEM_IP_PORT="$IP_PORT"
export SHMEM_UID_SESSION_ID="${IP_PORT#tcp://}"
export BENCH_SEQS="$SEQS"
export BENCH_WARMUP="$WARMUP"
export BENCH_ITERS="$ITERS"
cd "$HERE"

if ! python -c "import torch_npu; import ascend_deepep; print('[ok] ascend_deepep import OK')"; then
    echo "[error] ascend_deepep import FAILED (see stderr above)." >&2
    exit 1
fi

# Multi-node nodes may share one checkout (e.g. shared mount launched by
# scripts/batch_launch.sh); keep one log per node, single-node keeps the
# historical name.
if [[ -n "$NODE_RANK" ]]; then
    LOG="run_speed.node${NODE_RANK}.log"
else
    LOG="run_speed.log"
fi
echo "[run] $(date +%H:%M:%S) BigOpV2 bench [${MODE}] EP=$((NPROC * ${NNODES:-1})) ip=$IP_PORT" \
     "seqs=$SEQS warmup=$WARMUP iters=$ITERS" \
     "K=${BENCH_HIDDEN:-1024} F=${BENCH_FFN:-2048} E=${BENCH_EXPERTS:-16} topK=${BENCH_TOPK:-2}" \
     "local_mem=${BENCH_LOCAL_MEM_BYTES:-1GiB} save_gmm1=${MEGAMOE_SAVE_GMM1_OUTPUT:-bf16}" \
     "bigop_timing=${BIGOP_TIMING:-0} v2_layout_timing=${V2_LAYOUT_TIMING:-0}" \
     "megamoe_trace=${MEGAMOE_TRACE:-0}" | tee "$LOG"

torchrun ${TORCHRUN_NODE_ARGS} \
    --nproc_per_node="$NPROC" \
    --master_port="$MASTER_PORT" \
    test_speed.py 2>&1 | tee -a "$LOG"
RUN_RC=${PIPESTATUS[0]}
echo "[run] done rc=$RUN_RC. log: $LOG" | tee -a "$LOG"
exit $RUN_RC
