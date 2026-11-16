#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
python3 - "$task" <<'PY'
import hashlib,json,subprocess,sys,tarfile
from pathlib import Path

task=Path(sys.argv[1])
ps=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,comm'],text=True)
assert not any(any(w in line for w in ('python','pt_elastic','bisheng','clang')) for line in ps.splitlines()[1:])
assert json.loads((task/'evidence/quota-floor-stage-v1.json').read_text())['complete']
archive=task/'artifacts/quota-clock-recovery-v2.tar'
sha=hashlib.sha256(archive.read_bytes()).hexdigest();assert sha=='8a52efc62a96dc2586979ea9625f50f400bb7b7c0fa83ee9a720e7e4669838ae'
with tarfile.open(archive) as t:
 manifest=json.loads(t.extractfile('manifest.json').read())
 for name,expected in manifest['expected_old'].items():
  p=task/name;actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
  if actual!=expected:raise RuntimeError('Unexpected recovery target drift: '+name)
 backup=task/'artifacts/quota-clock-before-v2.tar';assert not backup.exists()
 with tarfile.open(backup,'w') as b:
  for name in manifest['expected_old']:
   if (task/name).is_file():b.add(task/name,arcname=name,recursive=False)
 for name,expected in manifest['new'].items():
  payload=t.extractfile(name).read();assert hashlib.sha256(payload).hexdigest()==expected
  (task/name).write_bytes(payload)
 (task/'evidence/quota-clock-recovery-deployment-v2.json').write_text(json.dumps(dict(
  archive_sha256=sha,files=manifest['new'],expected_old=manifest['expected_old'],
  old_failed_runs_preserved=True,backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest()),indent=2))
print('QUOTA_CLOCK_RECOVERY_DEPLOYED_WHILE_IDLE',flush=True)
PY
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 artifacts/quota_floor_regrouped_stage_v2.py
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
