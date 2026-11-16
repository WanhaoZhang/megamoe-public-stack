#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
docker exec -e PRIORITY_VARIANT_PREFIX=wide-priority -e EARLY_DW1_WIDE=1 \
  single-mb-fbw-planner-study bash --noprofile --norc "$task/tools/check_priority_capabilities.sh"
docker exec -e PRIORITY_VARIANT_PREFIX=wide-priority single-mb-fbw-planner-study \
  bash --noprofile --norc "$task/tools/seal_workers.sh"
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- \
  python3 tools/priority_qualification.py --wide --revision v2
