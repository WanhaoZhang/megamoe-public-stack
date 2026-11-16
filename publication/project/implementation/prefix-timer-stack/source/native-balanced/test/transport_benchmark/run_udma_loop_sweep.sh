#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICES=${1:-}
if [[ -z "${DEVICES}" || $# -ne 1 ]]; then
    echo "Usage: bash run_udma_loop_sweep.sh <device0,device1>" >&2
    exit 2
fi

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
WORLD_SIZE=${#DEVICE_LIST[@]}
if (( WORLD_SIZE != 2 )); then
    echo "[ERROR] UDMA loop sweep requires exactly two devices" >&2
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

MESSAGE_BYTES=65536
LOOP_COUNTS=${DEEPEP_BENCH_UDMA_LOOP_COUNTS:-"8,16,32,64,128,256,512"}
WARMUP=${DEEPEP_BENCH_UDMA_LOOP_WARMUP:-100}
RUNNER=${DEEPEP_TRANSPORT_BENCHMARK_RUNNER:-"${SCRIPT_DIR}/install/bin/deepep_transport_benchmark"}
TIMEOUT_SECONDS=${DEEPEP_TRANSPORT_BENCHMARK_TIMEOUT:-1800}

if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "[ERROR] Build it first: bash ${SCRIPT_DIR}/build.sh" >&2
    exit 1
fi

IFS=',' read -r -a LOOP_LIST <<< "${LOOP_COUNTS}"
for loops in "${LOOP_LIST[@]}"; do
    if [[ ! "${loops}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] Invalid loop count: ${loops}" >&2
        exit 2
    fi
done

PARENT_OUTPUT_DIR=${DEEPEP_TRANSPORT_BENCHMARK_PARENT_OUTPUT:-}
if [[ -n "${PARENT_OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${PARENT_OUTPUT_DIR}/udma_loop_sweep"
else
    RUN_ID="udma_loop_sweep_$(date +%Y%m%d_%H%M%S)_$$"
    OUTPUT_DIR="${SCRIPT_DIR}/output/${RUN_ID}"
fi
mkdir -p "${OUTPUT_DIR}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/install/lib:${LD_LIBRARY_PATH:-}"

cat >"${OUTPUT_DIR}/config.txt" <<EOF
devices=${DEVICES}
world_size=${WORLD_SIZE}
message_bytes=${MESSAGE_BYTES}
loop_counts=${LOOP_COUNTS}
warmup=${WARMUP}
batch=0
blocks=1
pipe=pipe_s
shmem_commit=$(git -C "${SCRIPT_DIR}/../../3rdparty/shmem" rev-parse HEAD 2>/dev/null || echo unknown)
deepep_commit=$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD 2>/dev/null || echo unknown)
EOF

echo "[INFO] Bidirectional ordinary UDMA loop sweep devices=${DEVICES} bytes=${MESSAGE_BYTES}"
echo "[INFO] loops=${LOOP_COUNTS} output=${OUTPUT_DIR}"

failed=0
case_index=0
for loops in "${LOOP_LIST[@]}"; do
    port=$((20000 + (($$ + case_index) % 20000)))
    ip_port="tcp://127.0.0.1:${port}"
    export SHMEM_UID_SESSION_ID="127.0.0.1:${port}"
    echo "[INFO] Running loops=${loops}"

    pids=()
    logs=()
    for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
        log_path="${OUTPUT_DIR}/loop${loops}_rank${rank}.log"
        timeout "${TIMEOUT_SECONDS}" "${RUNNER}" \
            "${rank}" "${WORLD_SIZE}" "${ip_port}" \
            "${MESSAGE_BYTES}" "${MESSAGE_BYTES}" "${MESSAGE_BYTES}" \
            "${WARMUP}" "${loops}" \
            "1" "1" "1" "0" "pipe_s" "udma_loop_sweep" \
            >"${log_path}" 2>&1 &
        pids+=("$!")
        logs+=("${log_path}")
    done

    for ((rank = 0; rank < WORLD_SIZE; ++rank)); do
        status=0
        wait "${pids[rank]}" || status=$?
        echo "===== loops=${loops} rank=${rank} (exit=${status}) ====="
        grep -E '^\[(RESULT|VERIFY|SUCCESS|ERROR)\]' "${logs[rank]}" || true
        if (( status != 0 )); then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        break
    fi
    case_index=$((case_index + 1))
done

if (( failed != 0 )); then
    echo "[ERROR] UDMA loop sweep failed; logs retained in ${OUTPUT_DIR}" >&2
    exit 1
fi

python3 "${SCRIPT_DIR}/analyze_udma_loop_sweep.py" \
    --input-dir "${OUTPUT_DIR}" \
    --output-csv "${OUTPUT_DIR}/results.csv" \
    --output-markdown "${OUTPUT_DIR}/summary.md"
cat "${OUTPUT_DIR}/summary.md"

echo "[SUCCESS] UDMA loop sweep completed"
echo "[INFO] Results: ${OUTPUT_DIR}/results.csv"
echo "[INFO] Summary: ${OUTPUT_DIR}/summary.md"
