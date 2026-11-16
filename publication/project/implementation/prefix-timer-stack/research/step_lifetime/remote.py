"""Deploy an isolated frozen checkout and execute finite, admitted study gates."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tarfile
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
STACK = ROOT / 'stack'
HOST = 'pod6-1-root'
REMOTE = '/home/z00956592/megaflux/step-lifetime-study-20261115'
CONTAINER = 'step-lifetime-study-20261115'


def call(arguments, **kwargs):
    return subprocess.run(arguments, check=True, **kwargs)


def deploy():
    archive = ROOT / 'source.tar'
    with archive.open('wb') as handle:
        call(['git','-c','core.longpaths=true','-C',str(STACK),'archive','HEAD'], stdout=handle)
    with tarfile.open(archive, 'a') as handle:
        changed = subprocess.check_output(['git','-c','core.longpaths=true','-C',str(STACK),'diff','--name-only'],text=True).splitlines()
        untracked = subprocess.check_output(['git','-c','core.longpaths=true','-C',str(STACK),'ls-files','--others','--exclude-standard'],text=True).splitlines()
        for name in changed + untracked:
            handle.add(STACK / name, arcname=name)
    hashes = {name: hashlib.sha256((STACK/name).read_bytes()).hexdigest() for name in changed+untracked}
    (ROOT/'source-overlay.json').write_text(json.dumps(hashes, indent=2))
    call(['ssh', HOST, 'mkdir', '-p', REMOTE])
    call(['scp', str(archive), f'{HOST}:{REMOTE}/source.tar'])
    script = f'''set -euo pipefail
task={REMOTE}
exec 9>/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock
flock -x 9
base=/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack
mkdir -p "$task/stack" "$task/evidence"
if [ ! -d "$task/stack/source" ]; then
 mkdir -p "$task/stack/source/native-balanced/ascend_deepep"
 cp -a --reflink=auto "$base/source/native-balanced/ascend_deepep/lib" "$task/stack/source/native-balanced/ascend_deepep/"
 cp -a --reflink=auto "$base/source/native-balanced/worker-variants" "$task/stack/source/native-balanced/"
 cp -a --reflink=auto "$base/routes" "$task/stack/routes"
 cp -a --reflink=auto /home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes "$task/stack/small-routes"
fi
tar -xf "$task/source.tar" -C "$task/stack"
python3 - "$base" "$task/stack" <<'STAGE'
from pathlib import Path
import json,shutil,sys
base,dest=map(Path,sys.argv[1:])
for name in ('runtime-freeze.json','temporal-scheduler-freeze.json'):
 manifest=json.loads((dest/'configs'/name).read_text())
 for relative in manifest['binaries']:
  target=dest/relative
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True)
   shutil.copy2(base/relative,target)
STAGE
if ! docker container inspect {CONTAINER} >/dev/null 2>&1; then
 docker run -d --name {CONTAINER} --label codex.task=step-lifetime-study-20261115 --privileged --network host --ipc host -v /home:/home -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/dcmi:/usr/local/dcmi -v /etc/ascend_install.info:/etc/ascend_install.info --entrypoint /bin/bash planner-executor-ep8-runtime:20261114 -lc 'sleep infinity'
fi
test "$(docker inspect {CONTAINER} --format '{{{{index .Config.Labels "codex.task"}}}}')" = step-lifetime-study-20261115
docker start {CONTAINER} >/dev/null
'''
    launcher = re.sub(r'^TASK=.*$', f'TASK={REMOTE}/stack', (STACK/'tools/launch.sh').read_text(), flags=re.M)
    controller = re.sub(r"^TASK = pathlib.Path\(.*$", f"TASK = pathlib.Path('{REMOTE}/stack')", (STACK/'tools/controller.py').read_text(), flags=re.M)
    controller = re.sub(r"^CONTAINER = .*$", f"CONTAINER = '{CONTAINER}'", controller, flags=re.M)
    for name, contents in [('launch.sh',launcher), ('controller.py',controller)]:
        path = ROOT/name
        path.write_text(contents, encoding='utf-8', newline='\n')
    call(['scp', str(ROOT/'launch.sh'), str(ROOT/'controller.py'), f'{HOST}:{REMOTE}/'])
    script += '\ncp "$task/launch.sh" "$task/controller.py" "$task/stack/tools/"\n'
    call(['ssh', HOST, 'bash', '-s'], input=script.encode())


def gate(tag, extra, *, metadata_only=False):
    script = f'''set -euo pipefail
cd {REMOTE}/stack
mkdir -p evidence/source-snapshots
python3 tools/controller.py --backend balanced --tag {tag} --worker-variant ep8-cap64-aligned --seconds 900 {' '.join(extra)}
'''
    began = datetime.now(timezone.utc).isoformat()
    with (ROOT/f'{tag}-remote.log').open('wb') as handle:
        result = subprocess.run(['ssh', HOST, 'bash', '-s'], input=script.encode(), stdout=handle, stderr=subprocess.STDOUT)
    receipt = dict(tag=tag, started=began, exit=result.returncode)
    (ROOT/f'{tag}-receipt.json').write_text(json.dumps(receipt, indent=2))
    if metadata_only:
        # Full tensors remain in this run's project-owned remote directory.
        # This avoids retransmitting multi-GiB PT files for every qualification.
        bundle=f'{REMOTE}/stack/evidence/{tag}-metadata.tar.gz'
        packed=subprocess.run(['ssh',HOST,'tar','-czf',bundle,'--exclude=*.pt','-C',
                               f'{REMOTE}/stack/evidence',tag])
        packed.check_returncode()
        collected=subprocess.run(['scp',f'{HOST}:{bundle}',str(ROOT/'evidence')])
        if collected.returncode==0:
            with tarfile.open(ROOT/'evidence'/f'{tag}-metadata.tar.gz') as handle:
                handle.extractall(ROOT/'evidence',filter='data')
        receipt['large_tensors_remote']=f'{REMOTE}/stack/evidence/{tag}'
    else:
        collected = subprocess.run(['scp', '-r', f'{HOST}:{REMOTE}/stack/evidence/{tag}', str(ROOT/'evidence')])
    receipt['collection_exit'] = collected.returncode
    snapshots = ROOT/'evidence'/'source-snapshots'
    snapshots.mkdir(exist_ok=True)
    current=ROOT/'evidence'/tag/'source-identity.json'
    duplicate=None
    if current.exists():
        identity=json.loads(current.read_text())
        for candidate in (ROOT/'evidence').glob('*/source-identity.json'):
            other=candidate.parent.name
            other_receipt=ROOT/f'{other}-receipt.json'
            if other==tag or not (snapshots/f'{other}.tar.gz').exists() or not other_receipt.exists():continue
            if json.loads(other_receipt.read_text()).get('collection_exit')!=0:continue
            if json.loads(candidate.read_text())==identity:
                duplicate=other;break
    if duplicate is None and current.exists():
        copied=subprocess.run(['scp',f'{HOST}:{REMOTE}/stack/evidence/source-snapshots/{tag}.tar.gz',str(snapshots)])
        receipt['snapshot_collection_exit']=copied.returncode
    elif duplicate is not None:
        (snapshots/f'{tag}-source-reference.json').write_text(json.dumps(dict(
            same_guarded_files_as=duplicate,archive=f'{duplicate}.tar.gz'),indent=2))
        receipt['source_snapshot_reused']=duplicate
    for suffix in ('-runtime.json','-temporal-runtime.json'):
        if current.exists():
            subprocess.run(['scp',f'{HOST}:{REMOTE}/stack/evidence/source-snapshots/{tag}{suffix}',str(snapshots)])
    if '--diagnostic-timer' in extra and current.exists():
        subprocess.run(['scp',f'{HOST}:{REMOTE}/stack/evidence/source-snapshots/{tag}-timer-worker-runtime.json',str(snapshots)])
    (ROOT/f'{tag}-receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('deploy','gate'))
    parser.add_argument('--tag')
    parser.add_argument('--metadata-only', action='store_true')
    args, extra = parser.parse_known_args()
    (ROOT/'evidence').mkdir(exist_ok=True)
    if args.action == 'deploy':
        deploy()
    else:
        if not args.tag or not all(character.isalnum() or character in '-_.' for character in args.tag):
            parser.error('a safe run tag is required')
        if any(not all(character.isalnum() or character in '-_./:' for character in item) for item in extra):
            parser.error('unexpected shell metacharacter in gate arguments')
        gate(args.tag, extra, metadata_only=args.metadata_only)
