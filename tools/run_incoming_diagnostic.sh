#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- \
  python3 tools/controller.py --backend balanced --mode validate --seconds 1800 \
  --tag diagnostic-incoming-native-self-vio5-v1 --incoming-gradient-diagnostic \
  --root "$task" --planner native --maxvio 5 --steps 3
