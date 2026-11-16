#!/usr/bin/env bash
set -euo pipefail
name=${1:?new name};steps=${2:?steps};timer=${3:?timer};port=${4:?master port}
[[ "$name" =~ ^[a-zA-Z0-9_.-]+$ && "$timer" =~ ^[01]$ ]]
export WHOLE_TIMER="$timer" WHOLE_CAPTURE_STEP=10 WHOLE_FRAMEWORK_PROFILE=False
suffix=off
if [[ "$timer" == 1 ]]; then suffix=on;export WHOLE_FRAMEWORK_PROFILE=True;fi
export WUHAN_WORKER_VARIANT="whole-python-balanced-a1-$suffix"
export WUHAN_RECV_CAPACITY=20480 BALANCED_RESTORE_MODE=panel
export WUHAN_MASTER_PORT="$port" WUHAN_DATA_NO_SHARED_STORAGE=1
exec bash /project/tools/run-whole-a1.sh /project/source "/project/evidence/$name" C 0 recompute "$steps" 8
