"""Build the isolated ABI2 worker under the common task lock, then freeze it."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
lane='/home/z00956592/megaflux/step-lifetime-priority-cache-20261116'
container='step-lifetime-priority-cache-20261116'
variant='ep8-cap64-lifetime-priority-cache-v1'
script=root/'implementation/build-priority-worker.sh'
subprocess.run(['scp',str(script),'pod6-1-root:'+lane+'/build-priority-worker.sh'],check=True)
command=f'''set -euo pipefail
exec 9>/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock
flock -x 9
test -f {lane}/stack/tools/controller.py
docker exec {container} bash {lane}/build-priority-worker.sh
'''
with (root/'implementation/build.log').open('wb') as log:
 result=subprocess.run(['ssh','pod6-1-root','bash','-s'],input=command.encode(),stdout=log,stderr=subprocess.STDOUT)
(root/'implementation/build-status.json').write_text(json.dumps(dict(exit=result.returncode,variant=variant))+'\n')
if result.returncode:raise SystemExit(result.returncode)
for remote,local in (
 ('stack/configs/lifetime-runtime-freeze.json','stack/configs/lifetime-runtime-freeze.json'),
 ('stack/source/native-balanced/worker-variants/'+variant+'/build.json','implementation/worker-build.json'),
 ('stack/source/native-balanced/worker-variants/'+variant+'/libascend_deepep_megaflux_early_bf16.so','implementation/worker.so')):
 subprocess.run(['scp','pod6-1-root:'+lane+'/'+remote,str(root/local)],check=True)
manifest=json.loads((root/'stack/configs/lifetime-runtime-freeze.json').read_text())
sha=hashlib.sha256((root/'implementation/worker.so').read_bytes()).hexdigest()
if set(manifest['binaries'].values())!={sha}:raise RuntimeError('Collected ABI2 SO mismatch')
print(json.dumps(dict(variant=variant,sha256=sha,build_passed=True)))
