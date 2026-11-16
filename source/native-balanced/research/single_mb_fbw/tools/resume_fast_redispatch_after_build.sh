#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
# Deploy only this sealed set after all preceding measured jobs have retired.
python3 - "$task" <<'PY'
import hashlib,json,sys,tarfile
from pathlib import Path
task=Path(sys.argv[1]);archive=task/'artifacts/native-fast-redispatch-source.tar'
names=['train.py','validate.py','gate_pool_replay.py','capture_lite_timers.py','fast_redispatch_stage.py','check_fast_redispatch.sh']
with tarfile.open(archive) as t:
 for name in names:
  payload=t.extractfile('research/single_mb_fbw/tools/'+name).read()
  (task/'tools'/name).write_bytes(payload)
 payload=t.extractfile('ascend_deepep/megaflux_worker_timer.py').read()
 (task/'source/native-balanced/ascend_deepep/megaflux_worker_timer.py').write_bytes(payload)
(task/'evidence/fast-redispatch-harness-deployment-v1.json').write_text(json.dumps(dict(
 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
 files={n:hashlib.sha256((task/'tools'/n).read_bytes()).hexdigest() for n in names},
 runtime_python_sha256=hashlib.sha256(payload).hexdigest()),indent=2))
PY
docker exec single-mb-fbw-planner-study bash --noprofile --norc "$task/tools/check_fast_redispatch.sh"
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/fast_redispatch_stage.py
