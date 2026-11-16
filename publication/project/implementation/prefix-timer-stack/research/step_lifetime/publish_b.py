"""Stage local patches, then publish only while holding the shared run lock."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parents[3]
lane='/home/z00956592/megaflux/step-lifetime-kernel-20261115'
files=sys.argv[1:]
if not files:raise SystemExit('pass stack-relative file paths')
manifest={}
for name in files:
 source=(root/'stack'/name).resolve()
 if not source.is_relative_to((root/'stack').resolve()):raise ValueError('outside isolated source')
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
stage=lane+'/staging/'+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
subprocess.run(['ssh','pod6-1-root','mkdir','-p',stage],check=True)
for index,name in enumerate(files):
 subprocess.run(['scp',str(root/'stack'/name),'pod6-1-root:'+stage+'/'+str(index)],check=True)
script="""import fcntl,hashlib,json,os,shutil
from pathlib import Path
lane=Path(LANE)
source=Path(STAGE)
files=MANIFEST
with Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 for index,(name,expected) in enumerate(files.items()):
  target=(lane/'stack'/name).resolve()
  if not target.is_relative_to((lane/'stack').resolve()):raise ValueError('outside task')
  incoming=source/str(index)
  if hashlib.sha256(incoming.read_bytes()).hexdigest()!=expected:raise ValueError('staging mismatch')
  target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.incoming')
  shutil.copy2(incoming,temporary);os.replace(temporary,target)
 print(json.dumps({'published':files,'staging':str(source)}))
""".replace('LANE',repr(lane)).replace('STAGE',repr(stage)).replace('MANIFEST',repr(manifest))
result=subprocess.run(['ssh','pod6-1-root','python3','-'],input=script.encode(),stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(root/'implementation'/('publish-'+Path(stage).name+'.log')).write_bytes(result.stdout)
raise SystemExit(result.returncode)
