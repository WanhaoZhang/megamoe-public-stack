#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 artifacts/quota_floor_regrouped_stage_v3.py
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/controller.py --backend balanced --mode links --tag light-timer-node-clock-only-v2 --worker-variant fastpipe-r0-t0 --seconds 900 --clock-only
# Publish the dependency marker only AFTER releasing the hardware guard.
python3 - "$task" <<'PY'
import json,sys
from pathlib import Path

task=Path(sys.argv[1]);root=task/'evidence/light-timer-node-clock-only-v2'
status=json.loads((root/'status.json').read_text());assert status['qualified'] and status['source_unchanged']
rows=[json.loads((root/f'clock-rank{r}.json').read_text()) for r in range(8)]
assert all(len(row['clock'])==5 and row['ticks_per_ms']>0 for row in rows)
assert len({r['library_sha256'] for r in rows})==1
for rank,row in enumerate(rows):
 loaded=json.loads((root/f'libraries-rank{rank}.json').read_text())['libraries']
 clock=[v for k,v in loaded.items() if k.endswith('/libascend_deepep_link_calibration.so')]
 assert len(clock)==1 and clock[0]['sha256']==row['library_sha256']
clocks={str(r):row['ticks_per_ms'] for r,row in enumerate(rows)}
(task/'evidence/light-timer-node-clocks-v1.json').write_text(json.dumps(dict(clocks=clocks,
 diagnostic_only=True,calibration_tag=root.name,clock_library_sha256=rows[0]['library_sha256'],
 scope='Same admitted node; independent Event/GetSystemCycle ratios only; no clock synchronization or link-bandwidth qualification',
 failed_full_link_calibration_preserved='light-timer-node-clock-calibration-v1'),indent=2))
print('CURRENT_NODE_CLOCK_ONLY_CALIBRATION_COMPLETE',clocks,flush=True)
PY
