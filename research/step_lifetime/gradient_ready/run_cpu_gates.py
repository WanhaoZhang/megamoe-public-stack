"""Apply one math-gate patch, then run locked CPU correctness oracles."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import remote

ROOT=Path(__file__).resolve().parent
TAG='gradient-ready-cpu-v1'
if (ROOT/'evidence'/TAG).exists():raise SystemExit('CPU qualification already exists')
relative='tools/gate_lifetime_math.py'
before=hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT/'stack'),'show','78a9c39:'+relative])).hexdigest()
after=hashlib.sha256((ROOT/'stack'/relative).read_bytes()).hexdigest()
bundle=ROOT/'math-gate-patch.tar'
with tarfile.open(bundle,'w') as archive:archive.add(ROOT/'stack'/relative,arcname=relative)
subprocess.run(['scp',str(bundle),remote.HOST+':'+remote.REMOTE+'/math-gate-patch.tar'],check=True)
script=f'''import fcntl,hashlib,importlib.util,json,pathlib,subprocess,sys,tarfile
task=pathlib.Path({remote.REMOTE!r})/'stack';out=task/'evidence'/{TAG!r}
with pathlib.Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 target=task/{relative!r}
 assert hashlib.sha256(target.read_bytes()).hexdigest()=={before!r}
 with tarfile.open(task.parent/'math-gate-patch.tar') as archive:
  assert archive.getnames()==[{relative!r}]
  payload=archive.extractfile({relative!r}).read()
 assert hashlib.sha256(payload).hexdigest()=={after!r}
 target.write_bytes(payload)
 spec=importlib.util.spec_from_file_location('controller',str(task/'tools/controller.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 out.mkdir(exist_ok=False,parents=True)
 admission=module.idle();sources=module.source_identity()
 (out/'source-identity.json').write_text(json.dumps(sources,indent=2))
 sys.path.insert(0,str(task/'tools'))
 from source_freeze import freeze_snapshot
 freeze_snapshot(task,{TAG!r},sources)
 command="source /home/pkg/b070/ascend-toolkit/set_env.sh; source /home/pkg/b070/cann-9.2.0/set_env.sh; export PATH=/usr/local/python3.12.13/bin:$PATH; export PYTHONPATH="+str(task/'tools')+"; export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35; export OMP_NUM_THREADS=1; python3 -m unittest -v test_gradient_bank_ready test_lifetime_delivery test_gradient_ready_capacity"
 with (out/'stdout.log').open('w') as log:
  result=subprocess.run(['docker','exec','-w',str(task),{remote.CONTAINER!r},'timeout','-k','10','240','bash','-lc',command],stdout=log,stderr=subprocess.STDOUT)
 status=dict(exit=result.returncode,source_unchanged=sources==module.source_identity(),admission=admission,math_gate_patch=dict(before={before!r},after={after!r}),cpu_only=True)
 status['passed']=status['exit']==0 and status['source_unchanged']
 (out/'status.json').write_text(json.dumps(status,indent=2))
 print(json.dumps(status))
 raise SystemExit(0 if status['passed'] else 1)
'''
with (ROOT/'diagnostics/cpu-driver.log').open('wb') as handle:
    result=subprocess.run(['ssh',remote.HOST,'python3','-'],input=script.encode(),stdout=handle,stderr=subprocess.STDOUT)
subprocess.run(['scp','-r',remote.HOST+':'+remote.REMOTE+'/stack/evidence/'+TAG,str(ROOT/'evidence')],check=True)
state=json.loads((ROOT/'evidence'/TAG/'status.json').read_text())
(ROOT/'cpu-v1.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(state,indent=2))
raise SystemExit(result.returncode)
