#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
set -u

# Fixed smoke/profile case.  Override one value through the environment only
# when a different case is needed; the normal invocation takes no arguments.
PYTHON_BIN=${MEGA_MOE_PROFILE_PYTHON:-python3}
DEVICES=${MEGA_MOE_PROFILE_DEVICES:-0,1}
PHASE=${MEGA_MOE_PROFILE_PHASE:-backward}
M=${MEGA_MOE_PROFILE_M:-64}
K=${MEGA_MOE_PROFILE_K:-512}
N=${MEGA_MOE_PROFILE_N:-1024}
TOPK=${MEGA_MOE_PROFILE_TOPK:-2}
EXPERT_PER_RANK=${MEGA_MOE_PROFILE_EXPERT_PER_RANK:-2}
WARMUP=${MEGA_MOE_PROFILE_WARMUP:-0}
LOOPS=${MEGA_MOE_PROFILE_LOOPS:-1}
ENABLE_TORCH_PROFILE=${MEGA_MOE_TORCH_PROFILE:-0}
TORCH_PROFILE_DIR=${MEGA_MOE_TORCH_PROFILE_DIR:-/tmp/megamoe_bf16_torch_profile}
TORCH_PROFILE_RANKS=${MEGA_MOE_TORCH_PROFILE_RANKS:-0}
TORCH_PROFILE_AIC_METRICS=${MEGA_MOE_TORCH_PROFILE_AIC_METRICS:-0}
CAPTURE_BACKWARD_TIMER=${MEGA_MOE_PROFILE_CAPTURE_BACKWARD_TIMER:-0}

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

COMMAND=(
    "${PYTHON_BIN}"
    "${SCRIPT_DIR}/profile_mega_moe_bf16.py"
    "${DEVICES}"
    "--phase=${PHASE}"
    "--m=${M}"
    "--k=${K}"
    "--n=${N}"
    "--topk=${TOPK}"
    "--expert-per-rank=${EXPERT_PER_RANK}"
    "--warmup=${WARMUP}"
    "--loops=${LOOPS}"
)
if [[ "${ENABLE_TORCH_PROFILE}" == "1" ]]; then
    COMMAND+=(
        --profile
        "--profile-dir=${TORCH_PROFILE_DIR}"
        "--profile-ranks=${TORCH_PROFILE_RANKS}"
    )
    if [[ "${TORCH_PROFILE_AIC_METRICS}" == "1" ]]; then
        COMMAND+=(--profile-aic-metrics)
    fi
fi
if [[ "${CAPTURE_BACKWARD_TIMER}" == "1" ]]; then
    COMMAND+=(--capture-backward-timer)
fi

printf '[RUN] command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

exec "${COMMAND[@]}"
