#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICES=${1:-}
if [[ -z "${DEVICES}" ]]; then
    echo "Usage: bash test_mega_moe.sh <device0,device1,...> [options]" >&2
    echo "Options: --csv-path PATH --data-dir PATH --python PATH --warmup N --loops N" >&2
    exit 1
fi
shift

CSV_PATH="${SCRIPT_DIR}/test_shapes.csv"
DATA_DIR="${SCRIPT_DIR}/output/run_$(date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN=python3
WARMUP=0
LOOPS=1
SEED=831
ABSOLUTE_ERROR_THRESHOLD=0.01
MAX_ERROR_RATIO=0.05
TEST_TIMEOUT=600
RUNNER=${MEGA_MOE_INDEPENDENT_RUNNER:-"${SCRIPT_DIR}/install/bin/mega_moe_stub"}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv-path) CSV_PATH=${2:?}; shift 2 ;;
        --data-dir) DATA_DIR=${2:?}; shift 2 ;;
        --python) PYTHON_BIN=${2:?}; shift 2 ;;
        --warmup) WARMUP=${2:?}; shift 2 ;;
        --loops) LOOPS=${2:?}; shift 2 ;;
        --seed) SEED=${2:?}; shift 2 ;;
        --absolute-error-threshold) ABSOLUTE_ERROR_THRESHOLD=${2:?}; shift 2 ;;
        --max-error-ratio) MAX_ERROR_RATIO=${2:?}; shift 2 ;;
        --timeout) TEST_TIMEOUT=${2:?}; shift 2 ;;
        --runner) RUNNER=${2:?}; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
EP=${#DEVICE_LIST[@]}
if (( EP < 2 )); then
    echo "[ERROR] MegaMoe requires at least two devices" >&2
    exit 1
fi
if (( EP > 8 )); then
    echo "[ERROR] At most eight local devices are supported by this runner" >&2
    exit 1
fi
declare -A seen_devices=()
for device in "${DEVICE_LIST[@]}"; do
    if [[ ! "${device}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid device ID: ${device}" >&2
        exit 1
    fi
    if [[ -n "${seen_devices[${device}]:-}" ]]; then
        echo "[ERROR] Duplicate device ID: ${device}" >&2
        exit 1
    fi
    seen_devices[${device}]=1
done
if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}" >&2
    echo "[ERROR] Build it first with: bash ${SCRIPT_DIR}/build.sh" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set; source the CANN set_env.sh first" >&2
    exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] The Linux 'timeout' command is required" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}"
DATA_DIR=$(cd "${DATA_DIR}" && pwd)
CSV_PATH=$(cd "$(dirname "${CSV_PATH}")" && pwd)/$(basename "${CSV_PATH}")

echo "[INFO] Independent MegaMoe data directory: ${DATA_DIR}"
echo "[INFO] Generating inputs and CPU golden without ascend_deepep registration"
"${PYTHON_BIN}" "${SCRIPT_DIR}/independent_data.py" generate \
    --csv-path "${CSV_PATH}" --ep "${EP}" --data-dir "${DATA_DIR}" \
    --seed "${SEED}" --warmup "${WARMUP}" --loops "${LOOPS}"

SHMEM_HOME_PATH="${SCRIPT_DIR}/../../../3rdparty/shmem/install"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/install/lib:${SHMEM_HOME_PATH}/shmem/lib:${SHMEM_HOME_PATH}/memfabric_hybrid/lib:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"

case_index=0
while IFS= read -r runner_csv; do
    case_dir=$(dirname "${runner_csv}")
    case_id="independent"
    port=$((20000 + ($$ + case_index) % 20000))
    ip_port="tcp://127.0.0.1:${port}"
    export SHMEM_UID_SESSION_ID="127.0.0.1:${port}"
    echo "[INFO] Running shape=$(basename "${case_dir}"), EP=${EP}, devices=${DEVICES}, SHMEM=${ip_port}"

    pids=()
    logs=()
    for ((rank = 0; rank < EP; ++rank)); do
        log_path="${case_dir}/rank${rank}.log"
        timeout "${TEST_TIMEOUT}" "${RUNNER}" "${rank}" "${ip_port}" "${runner_csv}" \
            "${case_id}" "${case_dir}" >"${log_path}" 2>&1 &
        pids+=("$!")
        logs+=("${log_path}")
    done

    failed=0
    for ((rank = 0; rank < EP; ++rank)); do
        status=0
        wait "${pids[rank]}" || status=$?
        echo "===== rank ${rank} (exit=${status}) ====="
        sed -n '1,240p' "${logs[rank]}"
        if (( status != 0 )); then
            failed=1
        fi
    done
    if (( failed != 0 )); then
        echo "[ERROR] Independent MegaMoe execution failed; data retained at ${DATA_DIR}" >&2
        exit 1
    fi
    case_index=$((case_index + 1))
done < <(find "${DATA_DIR}" -type f -name runner_case.csv -print | sort)

if (( case_index == 0 )); then
    echo "[ERROR] No generated runner_case.csv found under ${DATA_DIR}" >&2
    exit 1
fi

echo "[INFO] Comparing standalone fused output with CPU golden"
"${PYTHON_BIN}" "${SCRIPT_DIR}/independent_data.py" compare \
    --csv-path "${CSV_PATH}" --ep "${EP}" --data-dir "${DATA_DIR}" \
    --absolute-error-threshold "${ABSOLUTE_ERROR_THRESHOLD}" \
    --max-error-ratio "${MAX_ERROR_RATIO}"

echo "[SUCCESS] Independent MegaMoe test passed"
echo "[INFO] Inputs, outputs, golden and rank logs retained at ${DATA_DIR}"
