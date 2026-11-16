"""Stage subsequent provenance/tuning diagnostics without touching live source."""
from pathlib import Path
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/z00956592/megaflux/single-mb-fbw-planner-study';HOST='pod5-8-root'
FILES={f'tools/{n}.py':ROOT/f'tools/{n}.py' for n in ('gradient_incoming','numerical_and_native_tuning_stage')}
FILES.update({f'artifacts/native-tune-vio20-{family}.{suffix}':ROOT/f'artifacts/native-tune-vio20-{family}.{suffix}'
              for family in ('tile384','tile1536','penalty4') for suffix in ('npz','json')})
code='''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);names=json.loads(sys.argv[2])
print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() if (root/n).is_file() else None for n in names}))
'''
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,
         'python3 -c '+shlex.quote(code)+' '+shlex.quote(REMOTE)+' '+shlex.quote(json.dumps(list(FILES)))]
old=json.loads(subprocess.run(command,capture_output=True,check=True).stdout)
new={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in FILES.items()}
manifest=dict(expected_old=old,new=new,source_commit=subprocess.check_output(['git','-C',str(ROOT/'source/native-balanced'),'rev-parse','HEAD'],text=True).strip(),
              qualification='2100 exact CPU Native cases; determinism hypotheses and prepared banks have no new NPU result')
archive=ROOT/'artifacts/numerical-native-tuning-payload-v1.tar'
if archive.exists():raise RuntimeError('Immutable diagnostic payload exists')
with tarfile.open(archive,'w') as t:
 for n,p in FILES.items():t.add(p,arcname=n,recursive=False)
 payload=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
subprocess.run(['scp','-q','-o','BatchMode=yes',str(archive),HOST+':'+REMOTE+'/artifacts/'],check=True)
remote_sha=subprocess.check_output(['ssh','-o','BatchMode=yes',HOST,'sha256sum '+shlex.quote(REMOTE+'/artifacts/'+archive.name)],text=True).split()[0]
assert remote_sha==sha
(ROOT/'evidence/numerical-native-tuning-staged-v1.json').write_text(json.dumps(dict(archive_sha256=sha,manifest=manifest,runtime_deployed=False),indent=2))
print('NUMERICAL_NATIVE_TUNING_ARTIFACT_STAGED',sha)
