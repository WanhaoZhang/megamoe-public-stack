"""Stage immutable home-reuse payload only; no audited runtime mutations."""
from pathlib import Path
import hashlib
import io
import json
import shlex
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
REMOTE='/home/z00956592/megaflux/single-mb-fbw-planner-study'
HOST='pod5-8-root'
FILES={
 'source/native-balanced/ascend_deepep/fsdp2_bridge.py':ROOT/'source/native-balanced/ascend_deepep/fsdp2_bridge.py',
 'source/native-balanced/ascend_deepep/replay_weight_lease.py':ROOT/'source/native-balanced/ascend_deepep/replay_weight_lease.py',
 'source/FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py':ROOT/'source/FSDPTurbo/fsdp_turbo/fsdp_turbo_config.py',
 'source/FSDPTurbo/fsdp_turbo/distributed/expert_parallel/dispatcher_megamoe_balanced.py':ROOT/'source/FSDPTurbo/fsdp_turbo/distributed/expert_parallel/dispatcher_megamoe_balanced.py',
 **{f'tools/{n}.py':ROOT/f'tools/{n}.py' for n in ('train','gate_pool_replay','checkpoint_home_reuse_stage')}}
code='''import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);names=json.loads(sys.argv[2])
print(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() if (root/n).is_file() else None for n in names}))
'''
command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,
         'python3 -c '+shlex.quote(code)+' '+shlex.quote(REMOTE)+' '+shlex.quote(json.dumps(list(FILES)))]
old=json.loads(subprocess.run(command,capture_output=True,check=True).stdout)
new={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in FILES.items()}
commits={c:subprocess.check_output(['git','-C',str(ROOT/'source'/c),'rev-parse','HEAD'],text=True).strip() for c in ('native-balanced','FSDPTurbo')}
manifest=dict(expected_old=old,new=new,source_commits=commits,qualification='39 CPU control cases only; no NPU correctness/performance qualification')
archive=ROOT/'artifacts/checkpoint-home-reuse-payload-v1.tar'
if archive.exists():raise RuntimeError('Immutable payload already exists; use a new version')
with tarfile.open(archive,'w') as t:
 for n,p in FILES.items():t.add(p,arcname=n,recursive=False)
 payload=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
sha=hashlib.sha256(archive.read_bytes()).hexdigest()
subprocess.run(['scp','-q','-o','BatchMode=yes',str(archive),HOST+':'+REMOTE+'/artifacts/'],check=True)
remote_sha=subprocess.check_output(['ssh','-o','BatchMode=yes',HOST,'sha256sum '+shlex.quote(REMOTE+'/artifacts/'+archive.name)],text=True).split()[0]
assert remote_sha==sha
receipt=dict(archive_sha256=sha,remote_archive=REMOTE+'/artifacts/'+archive.name,runtime_deployed=False,manifest=manifest)
(ROOT/'evidence/checkpoint-home-reuse-staged-v1.json').write_text(json.dumps(receipt,indent=2))
print('HOME_REUSE_IMMUTABLE_ARTIFACT_STAGED',sha)
