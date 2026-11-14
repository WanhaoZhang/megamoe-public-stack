#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICES=${1:-}
if [[ -z "${DEVICES}" || $# -ne 1 ]]; then
    echo "Usage: bash run_remote_mte_bi_rows.sh <device0,device1,...>" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
WORLD_SIZE=${#DEVICE_LIST[@]}
if (( WORLD_SIZE < 2 )); then
    echo "[ERROR] Remote MTE BI row sweep requires at least two devices" >&2
    exit 2
fi
declare -A SEEN_DEVICES=()
for device in "${DEVICE_LIST[@]}"; do
    if [[ ! "${device}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_DEVICES[${device}]:-}" ]]; then
        echo "[ERROR] Invalid or duplicate device ID: ${device}" >&2
        exit 2
    fi
    SEEN_DEVICES[${device}]=1
done

ROW_SIZES=${DEEPEP_BENCH_REMOTE_MTE_ROW_SIZES:-"2048,4096,8192"}
TOTAL_ROWS=${DEEPEP_BENCH_REMOTE_MTE_TOTAL_ROWS:-"16,64,256,1024"}
BLOCKS=${DEEPEP_BENCH_REMOTE_MTE_ROW_BLOCKS:-"1,2"}
WARMUP_OPERATIONS=${DEEPEP_BENCH_REMOTE_MTE_WARMUP_OPERATIONS:-${DEEPEP_BENCH_REMOTE_MTE_ROW_WARMUP:-100}}
MEASURED_GROUPS=${DEEPEP_BENCH_REMOTE_MTE_MEASURED_GROUPS:-100}
RUNNER=${DEEPEP_TRANSPORT_BENCHMARK_RUNNER:-"${SCRIPT_DIR}/install/bin/deepep_transport_benchmark"}
TIMEOUT_SECONDS=${DEEPEP_TRANSPORT_BENCHMARK_TIMEOUT:-1800}

if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "[ERROR] Build it first: bash ${SCRIPT_DIR}/build.sh" >&2
    exit 1
fi

PARENT_OUTPUT_DIR=${DEEPEP_TRANSPORT_BENCHMARK_PARENT_OUTPUT:-}
if [[ -n "${PARENT_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PARENT_OUTPUT_DIR}/remote_mte_bi_row_sweep"
else
    RUN_ID="remote_mte_bi_row_sweep_$(date +%Y%m%d_%H%M%S)_$$"
    OUTPUT_DIR="${SCRIPT_DIR}/output/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"
PORT=$((20000 + ($$ % 20000)))
IP_PORT="tcp://127.0.0.1:${PORT}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export SHMEM_UID_SESSION_ID="127.0.0.1:${PORT}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/install/lib:${LD_LIBRARY_PATH:-}"

cat >"${OUTPUT_DIR}/config.txt" <<EOF
devices=${DEVICES}
world_size=${WORLD_SIZE}
row_sizes=${ROW_SIZES}
total_rows=${TOTAL_ROWS}
blocks=${BLOCKS}
warmup_operations=${WARMUP_OPERATIONS}
measured_groups=${MEASURED_GROUPS}
shmem_commit=$(git -C "${SCRIPT_DIR}/../../3rdparty/shmem" rev-parse HEAD 2>/dev/null || echo unknown)
deepep_commit=$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD 2>/dev/null || echo unknown)
EOF

echo "[INFO] Remote MTE BI row sweep devices=${DEVICES}"
echo "[INFO] row_sizes=${ROW_SIZES} total_rows=${TOTAL_ROWS} blocks=${BLOCKS} warmup_operations=${WARMUP_OPERATIONS} measured_groups=${MEASURED_GROUPS}"
echo "[INFO] output=${OUTPUT_DIR}"

pids=()
logs=()
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    log_path="${OUTPUT_DIR}/rank${rank}.log"
    timeout "${TIMEOUT_SECONDS}" "${RUNNER}" remote_mte_bi_rows \
        "${rank}" "${WORLD_SIZE}" "${IP_PORT}" \
        "${ROW_SIZES}" "${TOTAL_ROWS}" "${BLOCKS}" \
        "${WARMUP_OPERATIONS}" "${MEASURED_GROUPS}" \
        >"${log_path}" 2>&1 &
    pids+=("$!")
    logs+=("${log_path}")
done

failed=0
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    status=0
    wait "${pids[rank]}" || status=$?
    echo "===== Remote MTE BI rows rank=${rank} (exit=${status}) ====="
    grep -E '^\[(RESULT|VERIFY|SUCCESS|ERROR)\]' "${logs[rank]}" || true
    if (( status != 0 )); then
        failed=1
    fi
done

if (( failed != 0 )); then
    echo "[ERROR] Remote MTE BI row sweep failed; logs retained in ${OUTPUT_DIR}" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/analyze_remote_mte_bi_rows.py" \
    --input-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/results.csv" \
    --output-markdown "${OUTPUT_DIR}/summary.md"
cat "${OUTPUT_DIR}/summary.md"

echo "[SUCCESS] Remote MTE BI row sweep completed"
echo "[INFO] Results: ${OUTPUT_DIR}/results.csv"
echo "[INFO] Summary: ${OUTPUT_DIR}/summary.md"
