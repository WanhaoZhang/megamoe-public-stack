#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
# This script is staged in artifacts, outside the live queue's source tree.
# Do not overlap CPU compilation with any preceding measured device job.
python3 - "$task" <<'PY'
import json,subprocess,sys,time
from pathlib import Path
task=Path(sys.argv[1])
for _ in range(1200):
    path=task/'evidence/redispatch-runahead-stage-v1.json'
    if path.is_file() and json.loads(path.read_text()).get('complete'):
        processes=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,comm'],text=True)
        if not any(any(x in line for x in ('python','pt_elastic','bisheng','clang')) for line in processes.splitlines()[1:]):
            break
    time.sleep(3)
else:raise RuntimeError('Prior queue not complete. Inspect its exact handle; never supersede it.')
print('PRIOR_RUNAHEAD_STAGE_COMPLETE_CPU_BUILD_ADMITTED',flush=True)
PY
docker exec -i single-mb-fbw-planner-study bash --noprofile --norc "$task/artifacts/build_key_cache.sh"
