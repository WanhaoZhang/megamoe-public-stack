#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
cd "$task"
mkdir -p receipts
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- \
  python3 tools/controller.py --backend balanced --mode plan --tag budget-common-plan-v2 --seconds 900
