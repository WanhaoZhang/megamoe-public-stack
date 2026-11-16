"""Publish selected diagnostic files atomically under the common device lock."""
from pathlib import Path
import hashlib,json,subprocess,sys
root=Path(__file__).resolve().parent
lane='/home/z00956592/megaflux/step-lifetime-raw-control-20261116'
files=sys.argv[1:]
if not files:raise SystemExit('Pass stack-relative diagnostic files')
manifest={}
for name in files:
 source=(root/'stack'/name).resolve()
 if not source.is_relative_to((root/'stack').resolve()) or name=='tools/controller.py':raise ValueError('Unsafe or unrewritten path')
 manifest[name]=hashlib.sha256(source.read_bytes()).hexdigest()
stage=lane+'/staging/'+hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:12]
subprocess.run(['ssh','pod6-1-root','mkdir','-p',stage],check=True)
for i,name in enumerate(files):subprocess.run(['scp',str(root/'stack'/name),'pod6-1-root:'+stage+'/'+str(i)],check=True)
script='''import pathlib,fcntl,hashlib,shutil,os
lane=pathlib.Path(LANE);stage=pathlib.Path(STAGE)
with pathlib.Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 for index,(name,digest) in enumerate(MANIFEST.items()):
  source=stage/str(index);assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
  target=(lane/'stack'/name).resolve();assert target.is_relative_to((lane/'stack').resolve())
  temp=target.with_name(target.name+'.incoming');shutil.copy2(source,temp);os.replace(temp,target)
 print('Published',len(MANIFEST),'diagnostic files')
'''.replace('LANE',repr(lane)).replace('STAGE',repr(stage)).replace('MANIFEST',repr(manifest))
subprocess.run(['ssh','pod6-1-root','python3','-'],input=script.encode(),check=True)
