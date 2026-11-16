#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICES=${1:-}
if [[ -z "${DEVICES}" || $# -ne 1 ]]; then
    echo "Usage: bash run.sh <device0,device1,...>" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
WORLD_SIZE=${#DEVICE_LIST[@]}
if (( WORLD_SIZE < 2 )); then
    echo "[ERROR] At least two devices are required" >&2
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

DEFAULT_UDMA_BANDWIDTH_SIZES="256,512,1024,2048,4096,8192,16384,32768,65536"
DEFAULT_UDMA_BANDWIDTH_SIZES+=",131072,262144,524288,917504,1048576"
REMOTE_MTE_BANDWIDTH_SIZES=${DEEPEP_BENCH_REMOTE_MTE_BANDWIDTH_SIZES:-"1048576"}
LOCAL_MTE_BANDWIDTH_SIZES=${DEEPEP_BENCH_LOCAL_MTE_BANDWIDTH_SIZES:-"1048576,16777216"}
UDMA_BANDWIDTH_SIZES=${DEEPEP_BENCH_UDMA_BANDWIDTH_SIZES:-"${DEFAULT_UDMA_BANDWIDTH_SIZES}"}
BENCH_WARMUP=${DEEPEP_BENCH_WARMUP:-100}
BENCH_LOOPS=${DEEPEP_BENCH_LOOPS:-1000}
DEFAULT_BLOCK_COUNTS="1"
REMOTE_MTE_BLOCKS=${DEEPEP_BENCH_REMOTE_MTE_BLOCKS:-"1,2"}
LOCAL_MTE_BLOCKS=${DEEPEP_BENCH_LOCAL_MTE_BLOCKS:-"1,4,16"}
LOCAL_SDMA_BLOCKS=${DEEPEP_BENCH_LOCAL_SDMA_BLOCKS:-"${DEFAULT_BLOCK_COUNTS}"}
BATCH_SIZE=${DEEPEP_BENCH_BATCH_SIZE:-0}
PIPE_MODE=${DEEPEP_BENCH_UDMA_PIPE:-pipe_s}
RUNNER=${DEEPEP_TRANSPORT_BENCHMARK_RUNNER:-"${SCRIPT_DIR}/install/bin/deepep_transport_benchmark"}
TIMEOUT_SECONDS=${DEEPEP_TRANSPORT_BENCHMARK_TIMEOUT:-1800}
RUN_UDMA_LOOP_SWEEP=${DEEPEP_BENCH_RUN_UDMA_LOOP_SWEEP:-1}
RUN_REMOTE_MTE_BI_ROW_SWEEP=${DEEPEP_BENCH_RUN_REMOTE_MTE_BI_ROW_SWEEP:-1}

for flag_name in RUN_UDMA_LOOP_SWEEP RUN_REMOTE_MTE_BI_ROW_SWEEP; do
    flag_value=${!flag_name}
    if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
        echo "[ERROR] ${flag_name} must be 0 or 1" >&2
        exit 2
    fi
done

if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "[ERROR] Build it first: bash ${SCRIPT_DIR}/build.sh" >&2
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_ID="run_${TIMESTAMP}_$$"
OUTPUT_DIR="${SCRIPT_DIR}/output/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
SUMMARY_PATH="${OUTPUT_DIR}/summary_mte_udma_${TIMESTAMP}.md"
PORT=$((20000 + ($$ % 20000)))
IP_PORT="tcp://127.0.0.1:${PORT}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export SHMEM_UID_SESSION_ID="127.0.0.1:${PORT}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/install/lib:${LD_LIBRARY_PATH:-}"

cat >"${OUTPUT_DIR}/config.txt" <<EOF
devices=${DEVICES}
world_size=${WORLD_SIZE}
remote_mte_bandwidth_sizes=${REMOTE_MTE_BANDWIDTH_SIZES}
local_mte_bandwidth_sizes=${LOCAL_MTE_BANDWIDTH_SIZES}
udma_bandwidth_sizes=${UDMA_BANDWIDTH_SIZES}
warmup=${BENCH_WARMUP}
loops=${BENCH_LOOPS}
remote_mte_blocks=${REMOTE_MTE_BLOCKS}
local_mte_blocks=${LOCAL_MTE_BLOCKS}
local_sdma_blocks=${LOCAL_SDMA_BLOCKS}
batch_size=${BATCH_SIZE}
udma_pipe=${PIPE_MODE}
run_udma_loop_sweep=${RUN_UDMA_LOOP_SWEEP}
run_remote_mte_bi_row_sweep=${RUN_REMOTE_MTE_BI_ROW_SWEEP}
shmem_commit=$(git -C "${SCRIPT_DIR}/../../3rdparty/shmem" rev-parse HEAD 2>/dev/null || echo unknown)
deepep_commit=$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD 2>/dev/null || echo unknown)
EOF

echo "[INFO] devices=${DEVICES} world_size=${WORLD_SIZE}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] Live progress: tail -f ${OUTPUT_DIR}/rank0.log"

pids=()
logs=()
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    log_path="${OUTPUT_DIR}/rank${rank}.log"
    timeout "${TIMEOUT_SECONDS}" "${RUNNER}" \
        "${rank}" "${WORLD_SIZE}" "${IP_PORT}" \
        "${REMOTE_MTE_BANDWIDTH_SIZES}" "${LOCAL_MTE_BANDWIDTH_SIZES}" \
        "${UDMA_BANDWIDTH_SIZES}" \
        "${BENCH_WARMUP}" "${BENCH_LOOPS}" \
        "${REMOTE_MTE_BLOCKS}" "${LOCAL_MTE_BLOCKS}" \
        "${LOCAL_SDMA_BLOCKS}" "${BATCH_SIZE}" \
        "${PIPE_MODE}" >"${log_path}" 2>&1 &
    pids+=("$!")
    logs+=("${log_path}")
done

failed=0
for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
    status=0
    wait "${pids[rank]}" || status=$?
    echo "===== rank ${rank} (exit=${status}) ====="
    grep -E '^\[(RESULT|VERIFY|SKIP|SUCCESS|ERROR)\]' "${logs[rank]}" || true
    if (( status != 0 )); then
        failed=1
    fi
done

if (( failed != 0 )); then
    python3 "${SCRIPT_DIR}/analyze.py" \
        --input-dir "${OUTPUT_DIR}" \
        --output-csv "${OUTPUT_DIR}/results.csv" \
        --output-parameters-csv "${OUTPUT_DIR}/model_parameters.csv" \
        --output-markdown "${SUMMARY_PATH}"
    cat "${SUMMARY_PATH}"
    echo "[ERROR] Transport benchmark failed; logs retained in ${OUTPUT_DIR}" >&2
    exit 1
fi

loop_sweep_status=0
if [[ "${RUN_UDMA_LOOP_SWEEP}" == "1" && "${WORLD_SIZE}" == "2" ]]; then
    echo "[INFO] Running bidirectional ordinary UDMA 64 KiB loop sweep"
    DEEPEP_TRANSPORT_BENCHMARK_PARENT_OUTPUT="${OUTPUT_DIR}" \
        bash "${SCRIPT_DIR}/run_udma_loop_sweep.sh" "${DEVICES}" || loop_sweep_status=$?
fi

if (( WORLD_SIZE != 2 )) && [[ "${RUN_UDMA_LOOP_SWEEP}" == "1" ]]; then
    echo "[SKIP] Bidirectional UDMA loop sweep requires exactly two devices"
fi

remote_mte_row_sweep_status=0
if [[ "${RUN_REMOTE_MTE_BI_ROW_SWEEP}" == "1" ]]; then
    echo "[INFO] Running Remote MTE BI token-row sweep"
    DEEPEP_TRANSPORT_BENCHMARK_PARENT_OUTPUT="${OUTPUT_DIR}" \
        bash "${SCRIPT_DIR}/run_remote_mte_bi_rows.sh" "${DEVICES}" || \
        remote_mte_row_sweep_status=$?
fi

analyze_args=(
    --input-dir "${OUTPUT_DIR}"
    --output-csv "${OUTPUT_DIR}/results.csv"
    --output-parameters-csv "${OUTPUT_DIR}/model_parameters.csv"
    --output-markdown "${SUMMARY_PATH}"
)
if [[ -f "${OUTPUT_DIR}/udma_loop_sweep/results.csv" ]]; then
    analyze_args+=(
        --udma-loop-sweep-csv "${OUTPUT_DIR}/udma_loop_sweep/results.csv"
    )
fi
if [[ -f "${OUTPUT_DIR}/remote_mte_bi_row_sweep/results.csv" ]]; then
    analyze_args+=(
        --remote-mte-bi-row-sweep-csv \
        "${OUTPUT_DIR}/remote_mte_bi_row_sweep/results.csv"
    )
fi
python3 "${SCRIPT_DIR}/analyze.py" "${analyze_args[@]}"
cat "${SUMMARY_PATH}"

if (( loop_sweep_status != 0 )); then
    echo "[ERROR] UDMA loop sweep failed: status=${loop_sweep_status}; logs retained in ${OUTPUT_DIR}" >&2
    exit 1
fi
if (( remote_mte_row_sweep_status != 0 )); then
    echo "[ERROR] Remote MTE BI row sweep failed: status=${remote_mte_row_sweep_status}; logs retained in ${OUTPUT_DIR}" >&2
    exit 1
fi

echo "[SUCCESS] Transport benchmark completed"
echo "[INFO] Results: ${OUTPUT_DIR}/results.csv"
echo "[INFO] Cost model parameters: ${OUTPUT_DIR}/model_parameters.csv"
echo "[INFO] Summary: ${SUMMARY_PATH}"
