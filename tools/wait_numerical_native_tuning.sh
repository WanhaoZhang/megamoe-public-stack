#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
python3 - "$task" <<'PY'
import hashlib,json,subprocess,sys,tarfile,time
from pathlib import Path

task=Path(sys.argv[1])
for _ in range(9600):
 path=task/'evidence/checkpoint-home-reuse-stage-v1.json'
 if path.is_file() and json.loads(path.read_text()).get('complete'):
  ps=subprocess.check_output(['docker','top','single-mb-fbw-planner-study','-eo','pid,comm'],text=True)
  if not any(any(w in line for w in ('python','pt_elastic','bisheng','clang')) for line in ps.splitlines()[1:]):break
 time.sleep(3)
else:raise RuntimeError('Prior home probe incomplete; inspect controller rather than mutating live code')
archive=task/'artifacts/numerical-native-tuning-payload-v1.tar'
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
assert sha=='32b347da797cefee650403f5b4b41341a0e41e11a2ef29cd2b23c1a597b9f4a4'
with tarfile.open(archive) as t:
 manifest=json.loads(t.extractfile('manifest.json').read());old=manifest['expected_old'];new=manifest['new']
 for name,expected in old.items():
  path=task/name;actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
  if actual!=expected:raise RuntimeError('Source/bank drift; refusing deployment: '+name)
 backup=task/'artifacts/numerical-native-tuning-before-v1.tar'
 if backup.exists():raise RuntimeError('Prior deployment exists; inspect before repeating')
 with tarfile.open(backup,'w') as b:
  for name in old:
   if (task/name).is_file():b.add(task/name,arcname=name,recursive=False)
 for name,expected in new.items():
  payload=t.extractfile(name).read();assert hashlib.sha256(payload).hexdigest()==expected
  (task/name).write_bytes(payload)
 (task/'evidence/numerical-native-tuning-deployment-v1.json').write_text(json.dumps(dict(
  source_commit=manifest['source_commit'],files=new,expected_old=old,archive_sha256=sha,
  backup_sha256=hashlib.sha256(backup.read_bytes()).hexdigest(),qualification='NPU provenance/tuning pending'),indent=2))
print('NUMERICAL_NATIVE_TUNING_DEPLOYED_AFTER_HOME_QUEUE',flush=True)
PY
cd "$task"
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/numerical_and_native_tuning_stage.py
