#!/usr/bin/env bash
# Run only after every participating host has passed its own idle/health gate.
# One process per physical NPU; this entry never builds or launches training.
set -euo pipefail
: "${EP_SIZE:?Set true EP_SIZE=16/32/64/128/256}"
: "${NODE_RANK:?Set this host's node rank}"
: "${MASTER_ADDR:?Set a routable address on node 0}"
: "${WORKER_NATIVE:?Set this build's large-ep-ready directory}"
: "${MEGAFLUX_SHMEM_PORT:?Reserve a separate bootstrap port}"
: "${OUT:?Set a new evidence directory}"
case "$EP_SIZE" in 16|32|64|128|256) ;; *) echo 'Unsupported EP_SIZE' >&2; exit 2 ;; esac
[[ "$MASTER_ADDR" != 127.* && "$MASTER_ADDR" != localhost ]]
[[ "$NODE_RANK" =~ ^[0-9]+$ && "$NODE_RANK" -lt $((EP_SIZE/8)) ]]
[[ ! -e "$OUT" ]] || { echo 'Refuse to replace evidence' >&2; exit 2; }
[[ "${PLANNER:-moonep}" == moonep || "${PLANNER:-moonep}" == native ]]
mkdir -p "$OUT"
export OMP_NUM_THREADS=1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-180}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-300}"
export ASCEND_DEEPEP_FORCE_GM_PLANNER=0
export WUHAN_POOL_AUDIT=1
export ASCEND_PROCESS_LOG_PATH="$OUT/plog"
tools=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
printf '%s\n' "true_ep=$EP_SIZE local_world=8 nodes=$((EP_SIZE/8)) node_rank=$NODE_RANK planner=${PLANNER:-moonep}" > "$OUT/topology.txt"
exec timeout --signal=TERM --kill-after=30s 900s python -m torch.distributed.run \
 --nnodes="$((EP_SIZE/8))" --nproc-per-node=8 --node-rank="$NODE_RANK" \
 --master-addr="$MASTER_ADDR" --master-port="${MASTER_PORT:-29871}" \
 "$tools/runtime_bootstrap.py" "$tools/test_bridge_npu.py" \
 --out "$OUT" --planner "${PLANNER:-moonep}" --restore panel \
 --tokens 2049 --ffn 512 --experts "$((EP_SIZE*2))" --slots 2 --capacity 8192 \
 --checkpoint --parameter-fp32 --require-replicas \
 --expect-operand-ready 1 --expect-delay-cycles 0
