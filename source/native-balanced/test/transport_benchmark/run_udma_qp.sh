#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICES=${1:-}
if [[ -z "${DEVICES}" || $# -ne 1 ]]; then
    echo "Usage: bash run_udma_qp.sh <device0,device1>" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
WORLD_SIZE=${#DEVICE_LIST[@]}
if (( WORLD_SIZE != 2 )); then
    echo "[ERROR] UDMA QP uni/bi benchmark requires exactly two devices" >&2
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

ROW_SIZES=${DEEPEP_BENCH_QP_ROW_SIZES:-"2048,4096,8192"}
ROW_COUNTS=${DEEPEP_BENCH_QP_ROW_COUNTS:-"16,64,256,1024"}
QP_COUNTS=${DEEPEP_BENCH_QP_COUNTS:-${DEEPEP_BENCH_QP_COUNT:-"1,4,16,28"}}
WARMUP_OPERATIONS=${DEEPEP_BENCH_QP_WARMUP_OPERATIONS:-100}
MEASURED_GROUPS=${DEEPEP_BENCH_QP_MEASURED_GROUPS:-${DEEPEP_BENCH_QP_MEASURED_BATCHES:-100}}
RUNNER=${DEEPEP_TRANSPORT_BENCHMARK_RUNNER:-"${SCRIPT_DIR}/install/bin/deepep_transport_benchmark"}
TIMEOUT_SECONDS=${DEEPEP_TRANSPORT_BENCHMARK_TIMEOUT:-1800}

IFS=',' read -r -a QP_COUNT_LIST <<< "${QP_COUNTS}"
declare -A SEEN_QP_COUNTS=()
for qp_count in "${QP_COUNT_LIST[@]}"; do
    if [[ ! "${qp_count}" =~ ^[1-9][0-9]*$ ]] ||
       [[ -n "${SEEN_QP_COUNTS[${qp_count}]:-}" ]]; then
        echo "[ERROR] Invalid or duplicate QP count: ${qp_count}" >&2
        exit 2
    fi
    SEEN_QP_COUNTS[${qp_count}]=1
done

if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "[ERROR] Build it first: bash ${SCRIPT_DIR}/build.sh" >&2
    exit 1
fi

PARENT_OUTPUT_DIR=${DEEPEP_TRANSPORT_BENCHMARK_PARENT_OUTPUT:-}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [[ -n "${PARENT_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PARENT_OUTPUT_DIR}/udma_qp"
else
    RUN_ID="udma_qp_${TIMESTAMP}_$$"
    OUTPUT_DIR="${SCRIPT_DIR}/output/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"
SUMMARY_PATH="${OUTPUT_DIR}/summary_udma_qp_${TIMESTAMP}.md"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/install/lib:${LD_LIBRARY_PATH:-}"

cat >"${OUTPUT_DIR}/config.txt" <<EOF
devices=${DEVICES}
world_size=${WORLD_SIZE}
row_sizes=${ROW_SIZES}
row_counts=${ROW_COUNTS}
qp_counts=${QP_COUNTS}
warmup_operations=${WARMUP_OPERATIONS}
measured_groups=${MEASURED_GROUPS}
pipe=pipe_s
shmem_commit=$(git -C "${SCRIPT_DIR}/../../3rdparty/shmem" rev-parse HEAD 2>/dev/null || echo unknown)
deepep_commit=$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD 2>/dev/null || echo unknown)
EOF

echo "[INFO] UDMA QP uni/bi row benchmark devices=${DEVICES} qp_counts=${QP_COUNTS}"
echo "[INFO] row_sizes=${ROW_SIZES} row_counts=${ROW_COUNTS}"
echo "[INFO] output=${OUTPUT_DIR}"

failed=0
for qp_count in "${QP_COUNT_LIST[@]}"; do
    # QP count is frozen by aclshmemx_set_qp_num before SHMEM initialization,
    # so every QP-count point must use a fresh pair of rank processes.
    port=$((20000 + (($$ + qp_count * 97) % 20000)))
    ip_port="tcp://127.0.0.1:${port}"
    export SHMEM_UID_SESSION_ID="127.0.0.1:${port}"
    echo "[INFO] Running qp_count=${qp_count}"

    pids=()
    logs=()
    for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
        log_path="${OUTPUT_DIR}/rank${rank}_qp${qp_count}.log"
        timeout "${TIMEOUT_SECONDS}" "${RUNNER}" udma_qp \
            "${rank}" "${WORLD_SIZE}" "${ip_port}" \
            "${ROW_SIZES}" "${ROW_COUNTS}" "${qp_count}" \
            "${WARMUP_OPERATIONS}" "${MEASURED_GROUPS}" \
            >"${log_path}" 2>&1 &
        pids+=("$!")
        logs+=("${log_path}")
    done

    for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
        status=0
        wait "${pids[rank]}" || status=$?
        echo "===== qp ${qp_count} rank ${rank} (exit=${status}) ====="
        grep -E '^\[(RESULT|VERIFY|SUCCESS|ERROR)\]' "${logs[rank]}" || true
        if (( status != 0 )); then
            failed=1
        fi
    done
done

python3 "${SCRIPT_DIR}/analyze_udma_qp.py" \
    --input-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/results.csv" \
    --output-markdown "${SUMMARY_PATH}"
cat "${SUMMARY_PATH}"

if (( failed != 0 )); then
    echo "[ERROR] At least one UDMA QP configuration failed; logs retained in ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "[SUCCESS] UDMA QP benchmark completed"
echo "[INFO] Results: ${OUTPUT_DIR}/results.csv"
echo "[INFO] Summary: ${SUMMARY_PATH}"
