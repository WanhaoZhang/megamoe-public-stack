#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
python3 - "$task" <<'PY'
import json,subprocess,sys,time
from pathlib import Path
task=Path(sys.argv[1])
for _ in range(4800):
 path=task/'evidence/fast-redispatch-stage-v1.json'
 if path.is_file() and json.loads(path.read_text()).get('complete'):
  ps=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,comm'],text=True)
  if not any(any(w in line for w in ('python','pt_elastic','bisheng','clang')) for line in ps.splitlines()[1:]):break
 time.sleep(3)
else:raise RuntimeError('Prior fast pipeline stage incomplete; inspect exact session before recovery')
PY
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 artifacts/quota_floor_stage.py
