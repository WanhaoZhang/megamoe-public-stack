#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEVICES=${1:-}
if [[ -z "${DEVICES}" ]]; then
    echo "Usage: bash test_mega_moe_w4a8.sh <device0,device1,...> [--csv-path PATH] [--data-dir PATH] [--enable-timer]" >&2
    exit 1
fi
shift

CSV_PATH="${SCRIPT_DIR}/test_shapes.csv"
DATA_DIR="${SCRIPT_DIR}/output/run_$(date +%Y%m%d_%H%M%S)_$$"
PYTHON_BIN=python3
WARMUP=10
LOOPS=100
TEST_TIMEOUT=600
ENABLE_TIMER=0
RUNNER=${MEGA_MOE_W4A8_INDEPENDENT_RUNNER:-"${SCRIPT_DIR}/install/bin/mega_moe_w4a8_stub"}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv-path) CSV_PATH=${2:?}; shift 2 ;;
        --data-dir) DATA_DIR=${2:?}; shift 2 ;;
        --python) PYTHON_BIN=${2:?}; shift 2 ;;
        --warmup) WARMUP=${2:?}; shift 2 ;;
        --loops) LOOPS=${2:?}; shift 2 ;;
        --timeout) TEST_TIMEOUT=${2:?}; shift 2 ;;
        --enable-timer) ENABLE_TIMER=1; shift ;;
        --runner) RUNNER=${2:?}; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
EP=${#DEVICE_LIST[@]}
if (( EP < 2 || EP > 8 )); then
    echo "[ERROR] The standalone runner requires 2-8 devices" >&2
    exit 1
fi
if [[ ! -x "${RUNNER}" ]]; then
    echo "[ERROR] Runner not found: ${RUNNER}; run bash ${SCRIPT_DIR}/build.sh first" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set; source the CANN set_env.sh first" >&2
    exit 1
fi
if ! command -v timeout >/dev/null 2>&1; then
    echo "[ERROR] The Linux timeout command is required" >&2
    exit 1
fi
mkdir -p "${DATA_DIR}"
DATA_DIR=$(cd "${DATA_DIR}" && pwd)
CSV_PATH=$(cd "$(dirname "${CSV_PATH}")" && pwd)/$(basename "${CSV_PATH}")

IFS=',' read -r M K N TOPK EXPERTS CSV_EP < <(sed -n '2p' "${CSV_PATH}")
if [[ -z "${M:-}" || "${CSV_EP}" -ne "${EP}" ]]; then
    echo "[ERROR] test CSV must contain one case whose EP=${EP}" >&2
    exit 1
fi
if [[ $(sed -n '3p' "${CSV_PATH}") ]]; then
    echo "[ERROR] W4A8 golden generator currently accepts exactly one CSV case per run" >&2
    exit 1
fi
MAX_RECV=$((M * EP * (TOPK < EXPERTS ? TOPK : EXPERTS)))
RUNNER_CSV="${DATA_DIR}/runner_case.csv"
printf '%s\n' 'case_id,M,K,N,TOPK,expertPerRank,EP,maxRecvTokenNum,localMemBytes,warmup,loops' >"${RUNNER_CSV}"
printf 'w4a8,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "${M}" "${K}" "${N}" "${TOPK}" "${EXPERTS}" "${EP}" \
    "${MAX_RECV}" 1073741824 "${WARMUP}" "${LOOPS}" >>"${RUNNER_CSV}"

echo "[INFO] Generating migrated W4A8 input and CPU golden in ${DATA_DIR}"
MEGA_MOE_W4A8_CSV_PATH="${CSV_PATH}" MEGA_MOE_W4A8_DATA_DIR="${DATA_DIR}" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/gen_input_and_golden_w4a8.py"

SHMEM_HOME_PATH="${SCRIPT_DIR}/../../../3rdparty/shmem/install"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICES}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/install/lib:${SHMEM_HOME_PATH}/shmem/lib:${SHMEM_HOME_PATH}/memfabric_hybrid/lib:${ASCEND_HOME_PATH}/lib64:${ASCEND_HOME_PATH}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
port=$((20000 + $$ % 20000))
ip_port="tcp://127.0.0.1:${port}"
export SHMEM_UID_SESSION_ID="127.0.0.1:${port}"
pids=()
logs=()
for ((rank = 0; rank < EP; ++rank)); do
    log_path="${DATA_DIR}/rank${rank}.log"
    timeout "${TEST_TIMEOUT}" "${RUNNER}" "${rank}" "${ip_port}" \
        "${RUNNER_CSV}" w4a8 "${DATA_DIR}" "${ENABLE_TIMER}" \
        >"${log_path}" 2>&1 &
    pids+=("$!")
    logs+=("${log_path}")
done
failed=0
for ((rank = 0; rank < EP; ++rank)); do
    status=0
    wait "${pids[rank]}" || status=$?
    echo "===== rank ${rank} (exit=${status}) ====="
    sed -n '1,240p' "${logs[rank]}"
    (( status == 0 )) || failed=1
done
if (( failed != 0 )); then
    echo "[ERROR] Independent W4A8 execution failed; artifacts retained at ${DATA_DIR}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_output.py" \
    --data-dir "${DATA_DIR}" --m "${M}" --k "${K}" --n "${N}" --ep "${EP}"
if (( ENABLE_TIMER != 0 )); then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/../parse_timer_csv.py" \
        --dir "${DATA_DIR}" --rank-start 0 --rank-end "$((EP - 1))"
fi
echo "[SUCCESS] Independent MegaMoe W4A8 test passed"
echo "[INFO] Artifacts retained at ${DATA_DIR}"
