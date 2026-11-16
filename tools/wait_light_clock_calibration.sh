#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
python3 - "$task" <<'PY'
import json,subprocess,sys,time
from pathlib import Path
task=Path(sys.argv[1])
for _ in range(6000):
 path=task/'evidence/quota-floor-stage-v1.json'
 if path.is_file() and json.loads(path.read_text()).get('complete'):
  ps=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,comm'],text=True)
  if not any(any(w in line for w in ('python','pt_elastic','bisheng','clang')) for line in ps.splitlines()[1:]):break
 time.sleep(3)
else:raise RuntimeError('Prior native diagnostic queue incomplete')
PY
cd "$task"
# Reuse the frozen clock/link calibration program and its recorded loaded SO.
# Do not reuse a different node's Event/GetSystemCycle conversion.
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/controller.py --backend balanced --mode links --tag light-timer-node-clock-calibration-v1 --worker-variant fastpipe-r0-t0 --seconds 900
python3 - "$task" <<'PY'
import json,sys
from pathlib import Path
task=Path(sys.argv[1]);root=task/'evidence/light-timer-node-clock-calibration-v1'
status=json.loads((root/'status.json').read_text())
assert status['qualified'] and status['source_unchanged']
clocks={str(r):json.loads((root/f'links-rank{r}.json').read_text())['ticks_per_ms'] for r in range(8)}
assert all(v>0 for v in clocks.values())
(task/'evidence/light-timer-node-clocks-v1.json').write_text(json.dumps(dict(clocks=clocks,
 diagnostic_only=True,calibration_tag=root.name,scope='Same admitted node; rank-local Event/GetSystemCycle ratio, not clock synchronization'),indent=2))
print('CURRENT_NODE_CLOCK_CALIBRATION_COMPLETE',clocks,flush=True)
PY
