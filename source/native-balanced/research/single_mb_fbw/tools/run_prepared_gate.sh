#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/prepared_plan_gate.py
