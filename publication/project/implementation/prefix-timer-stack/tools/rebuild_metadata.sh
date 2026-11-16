#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
NAME=$1
CAP=$2
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
cd "$TASK/source/native-$NAME"
cmake --build "build-cap$CAP" --target ascend_deepep_large_ep_planner --parallel 2
cp "build-cap$CAP/lib/libascend_deepep_large_ep_planner.so" ascend_deepep/lib/
