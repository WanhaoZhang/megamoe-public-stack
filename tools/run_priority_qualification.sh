#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
docker exec single-mb-fbw-planner-study bash --noprofile --norc "$task/tools/check_priority_capabilities.sh"
docker exec single-mb-fbw-planner-study bash --noprofile --norc "$task/tools/seal_workers.sh"
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/priority_qualification.py
