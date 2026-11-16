"""Deploy one reviewed tracer-only patch after a terminal failed coverage run."""
from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import remote

ROOT=Path(__file__).resolve().parent
relative='tools/capture_host_boundary.py'
old=subprocess.check_output(['git','-C',str(ROOT/'stack'),'show','35e1fb4:'+relative])
old_sha=hashlib.sha256(old).hexdigest()
new_sha=hashlib.sha256((ROOT/'stack'/relative).read_bytes()).hexdigest()
bundle=ROOT/'trace-patch-v4.tar'
with tarfile.open(bundle,'w') as archive:archive.add(ROOT/'stack'/relative,arcname=relative)
subprocess.run(['scp',str(bundle),remote.HOST+':'+remote.REMOTE+'/trace-patch-v4.tar'],check=True)
script=f'''set -euo pipefail
task={remote.REMOTE}
exec 9>/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock
flock -x 9
test "$(sha256sum "$task/stack/{relative}" | cut -d' ' -f1)" = {old_sha}
test "$(tar -tf "$task/trace-patch-v4.tar")" = {relative}
tar -xf "$task/trace-patch-v4.tar" -C "$task/stack"
test "$(sha256sum "$task/stack/{relative}" | cut -d' ' -f1)" = {new_sha}
'''
result=subprocess.run(['ssh',remote.HOST,'bash','-s'],input=script.encode(),capture_output=True)
(ROOT/'diagnostics/trace-patch-v4.json').write_text(json.dumps(dict(relative=relative,before=old_sha,after=new_sha,
    exit=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode()),indent=2)+'\n',newline='\n')
result.check_returncode()
print(json.dumps(dict(passed=True,before=old_sha,after=new_sha)))
