#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
python3 - "$task" <<'PY'
import hashlib,io,json,subprocess,sys,tarfile,time
from pathlib import Path

task=Path(sys.argv[1])
for _ in range(9600):
 path=task/'evidence/light-timer-node-clocks-v1.json'
 if path.is_file() and len(json.loads(path.read_text()).get('clocks',{}))==8:
  ps=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,comm'],text=True)
  if not any(any(w in line for w in ('python','pt_elastic','bisheng','clang')) for line in ps.splitlines()[1:]):break
 time.sleep(3)
else:raise RuntimeError('Prior pipeline/quota/calibration queues incomplete')
archive=task/'artifacts/checkpoint-home-reuse-payload-v1.tar'
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert sha=='1e09bf0ecc59c016189504c8b5641f8e9c67e7c2d7ce084dd7842a9d4aba225d'
with tarfile.open(archive) as t:
 manifest=json.loads(t.extractfile('manifest.json').read())
 old=manifest['expected_old'];new=manifest['new']
 for name,expected in old.items():
  path=task/name
  actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
  if actual!=expected:raise RuntimeError('Audited source drift; refusing deployment: '+name)
 backup=task/'artifacts/checkpoint-home-reuse-before-v1.tar'
 if backup.exists():raise RuntimeError('Prior deployment exists; inspect instead of repeating')
 with tarfile.open(backup,'w') as b:
  for name in old:
   if (task/name).is_file():b.add(task/name,arcname=name,recursive=False)
 for name,expected in new.items():
  payload=t.extractfile(name).read()
  assert hashlib.sha256(payload).hexdigest()==expected
  (task/name).write_bytes(payload)
 receipt=dict(source_commits=manifest['source_commits'],files=new,expected_old=old,
              archive_sha256=sha,backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),
              qualification='default-off probe; NPU math pending')
 (task/'evidence/checkpoint-home-reuse-deployment-v1.json').write_text(json.dumps(receipt,indent=2))
print('HOME_REUSE_DEPLOYED_AFTER_ALL_OLD_QUEUES_RETIRED',flush=True)
PY
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/checkpoint_home_reuse_stage.py
