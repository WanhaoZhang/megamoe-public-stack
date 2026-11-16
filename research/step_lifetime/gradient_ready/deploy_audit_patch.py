"""Add actual loaded-library records to the math gate before any NPU runs."""
from pathlib import Path
import hashlib
import json
import subprocess
import remote

ROOT=Path(__file__).resolve().parent
relative='tools/gate_lifetime_math.py'
old=hashlib.sha256(subprocess.check_output(['git','-C',str(ROOT/'stack'),'show','e82c74b:'+relative])).hexdigest()
new=hashlib.sha256((ROOT/'stack'/relative).read_bytes()).hexdigest()
payload=ROOT/'math-gate-audited.py'
payload.write_bytes((ROOT/'stack'/relative).read_bytes())
subprocess.run(['scp',str(payload),remote.HOST+':'+remote.REMOTE+'/math-gate-audited.py'],check=True)
script=f'''import fcntl,hashlib,pathlib,json
task=pathlib.Path({remote.REMOTE!r})
with pathlib.Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 target=task/'stack'/{relative!r}
 assert hashlib.sha256(target.read_bytes()).hexdigest()=={old!r}
 data=(task/'math-gate-audited.py').read_bytes()
 assert hashlib.sha256(data).hexdigest()=={new!r}
 target.write_bytes(data)
print(json.dumps(dict(passed=True,before={old!r},after={new!r})))
'''
result=subprocess.run(['ssh',remote.HOST,'python3','-'],input=script.encode(),capture_output=True)
(ROOT/'diagnostics/math-library-audit-patch.json').write_text(json.dumps(dict(exit=result.returncode,before=old,after=new,
    stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n',encoding='utf-8',newline='\n')
result.check_returncode()
print(result.stdout.decode())
