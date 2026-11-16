"""Read actual immutable generated build inputs; no compilation or NPU calls."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
program='''
from pathlib import Path
import json,hashlib
items={}
for name,task,variant in (
 ('reference','step-lifetime-raw-control-20261116','ep8-cap64-lifetime-raw-control-v2'),
 ('candidate','step-lifetime-priority-cache-20261116','ep8-cap64-lifetime-priority-cache-v1')):
 source=Path('/home/z00956592/megaflux')/task/'stack/source/native-balanced'
 meta=json.loads((source/'worker-variants'/variant/'build.json').read_text())
 build=source/('build-worker-'+variant)
 values={}
 for relative,expected in {**meta['compile_files'],'CMakeCache.txt':meta['cmake_cache_sha256']}.items():
  data=(build/relative).read_bytes()
  if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('Generated build input differs from frozen metadata: '+name+relative)
  values[relative]=data.decode()
 items[name]=dict(task=task,variant=variant,files=values)
print(json.dumps(items))
'''
answer=subprocess.run(['ssh','pod6-1-root','python3','-'],input=program.encode(),capture_output=True,check=True)
data=json.loads(answer.stdout)
(root/'diagnostics/raw-build-options.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
print('Collected and hash-checked both compile flag sets and CMake caches')
