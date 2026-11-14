#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

# set_env.sh is harmless when the caller already sourced the CANN environment,
# and makes the standalone script convenient on a fresh shell.
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # Some toolkit versions inspect optional variables while being sourced, so
    # enable nounset only after the environment script has completed.
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -u

PHASE=${1:-${MEGA_MOE_PROFILE_PHASE:-backward}}
if [[ $# -gt 0 ]]; then
    shift
fi
DEVICES=${1:-${MEGA_MOE_PROFILE_DEVICES:-0,1}}
if [[ $# -gt 0 ]]; then
    shift
fi

case "${PHASE}" in
    forward)
        KERNEL_NAME=mega_moe_bf16
        ;;
    backward)
        KERNEL_NAME=mega_moe_bf16_backward
        ;;
    *)
        echo "[ERROR] phase must be 'forward' or 'backward', got: ${PHASE}" >&2
        exit 2
        ;;
esac

if ! command -v msprof >/dev/null 2>&1; then
    echo "[ERROR] msprof is unavailable after sourcing the CANN environment." >&2
    echo "[ERROR] Check ASCEND_HOME and \$ASCEND_HOME/tools/msopprof/bin/msopprof." >&2
    exit 1
fi

PYTHON_BIN=${MEGA_MOE_PROFILE_PYTHON:-python3}
SOC_VERSION=${MEGA_MOE_PROFILE_SOC-Ascend950}
SIMULATOR_LIB=${MEGA_MOE_PROFILE_SIM_LIB:-}
CORE_ID=${MEGA_MOE_PROFILE_CORE_ID:-0}
TIMEOUT_MINUTES=${MEGA_MOE_PROFILE_TIMEOUT:-10}
OUTPUT_DIR=${MEGA_MOE_PROFILE_OUTPUT:-"/tmp/megamoe_${PHASE}_sim_$(date +%Y%m%d_%H%M%S)"}
M=${MEGA_MOE_PROFILE_M:-64}
K=${MEGA_MOE_PROFILE_K:-512}
N=${MEGA_MOE_PROFILE_N:-1024}
TOPK=${MEGA_MOE_PROFILE_TOPK:-2}
EXPERT_PER_RANK=${MEGA_MOE_PROFILE_EXPERT_PER_RANK:-2}

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MSPROF_ARGS=(
    op
    simulator
    "--kernel-name=${KERNEL_NAME}"
    "--launch-count=1"
    "--timeout=${TIMEOUT_MINUTES}"
    "--output=${OUTPUT_DIR}"
)
if [[ -n "${SIMULATOR_LIB}" ]]; then
    if [[ ! -d "${SIMULATOR_LIB}" ]]; then
        echo "[ERROR] simulator lib directory does not exist: ${SIMULATOR_LIB}" >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${SIMULATOR_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    SOC_VERSION=""
elif [[ -n "${SOC_VERSION}" ]]; then
    MSPROF_ARGS+=("--soc-version=${SOC_VERSION}")
else
    echo "[ERROR] set MEGA_MOE_PROFILE_SOC or MEGA_MOE_PROFILE_SIM_LIB" >&2
    exit 1
fi
if [[ -n "${CORE_ID}" ]]; then
    MSPROF_ARGS+=("--core-id=${CORE_ID}")
fi

RUNNER_ARGS=(
    "${SCRIPT_DIR}/profile_mega_moe_bf16.py"
    "${DEVICES}"
    "--phase=${PHASE}"
    "--m=${M}"
    "--k=${K}"
    "--n=${N}"
    "--topk=${TOPK}"
    "--expert-per-rank=${EXPERT_PER_RANK}"
    --warmup=0
    --loops=1
)
RUNNER_ARGS+=("$@")

echo "[PROFILE] phase=${PHASE} kernel=${KERNEL_NAME} devices=${DEVICES}"
echo "[PROFILE] soc=${SOC_VERSION:-library-selected} "\
"simulator_lib=${SIMULATOR_LIB:-none} core_id=${CORE_ID:-all} output=${OUTPUT_DIR}"
printf '[PROFILE] command:'
printf ' %q' msprof "${MSPROF_ARGS[@]}" "${PYTHON_BIN}" "${RUNNER_ARGS[@]}"
printf '\n'

exec msprof "${MSPROF_ARGS[@]}" "${PYTHON_BIN}" "${RUNNER_ARGS[@]}"
