"""Serialize task-owned EP8 jobs, driver HBM sampling, and admission checks."""
from __future__ import annotations
import argparse
import fcntl
import datetime
import json
import pathlib
import re
import subprocess
import time
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor

TASK = pathlib.Path('/home/z00956592/megaflux/step-lifetime-kernel-20261115/stack')
CONTAINER = 'step-lifetime-kernel-20261115'

def source_identity(arguments=()):
    paths={}
    for component in ('native-balanced','FSDPTurbo','MindSpeed-LLM'):
        root=TASK/'source'/component
        for directory,children,names in os.walk(root):
            children[:] = sorted(name for name in children if name not in
                ('.git','__pycache__','3rdparty','worker-variants') and not name.startswith('build'))
            for name in sorted(names):
                path=pathlib.Path(directory)/name
                if path.is_symlink():continue
                if path.suffix in ('.py','.cpp','.h','.hpp','.cmake') or path.name=='CMakeLists.txt':
                    paths[str(path.relative_to(TASK))]=path
    for directory in ('tools','configs','routes'):
        for path in sorted((TASK/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.sh','.json','.yaml','.yml','.npy'):
                paths[str(path.relative_to(TASK))]=path
    if '--route-bank' in arguments:
        bank=pathlib.Path(arguments[arguments.index('--route-bank')+1])
        if not bank.is_absolute():bank=TASK/bank
        for path in sorted(bank.iterdir()):
            if path.is_file() and path.suffix in ('.json','.npy'):
                key=str(path.relative_to(TASK)) if path.is_relative_to(TASK) else str(path)
                paths[key]=path
    if '--temporal-schedule-replay' in arguments:
        replay=pathlib.Path(arguments[arguments.index('--temporal-schedule-replay')+1])
        if not replay.is_absolute():replay=TASK/replay
        paths[str(replay.relative_to(TASK)) if replay.is_relative_to(TASK) else str(replay)]=replay
    def digest(item):
        name,path=item
        return name,hashlib.sha256(path.read_bytes()).hexdigest()
    with ThreadPoolExecutor(max_workers=16) as readers:
        return dict(readers.map(digest,sorted(paths.items())))

def hbm_sample():
    result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True, timeout=20)
    cards = []
    current_rank = None
    for line in result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if len(cells) == 4 and cells[0].isdigit() and cells[1].startswith('Ascend'):
            current_rank = int(cells[0])
        if len(cells) != 4 or cells[2] != 'NA':
            continue
        numbers = cells[3].split()
        if len(numbers) >= 3 and numbers[-2] == '/':
            cards.append(dict(rank=current_rank, used_MiB=int(numbers[-3]), total_MiB=int(numbers[-1]),
                              utilization=int(numbers[0])))
    if len(cards) != 8:
        raise RuntimeError('Could not parse eight driver HBM readings')
    return dict(timestamp=time.time(), cards=cards, no_process_cards=result.stdout.count('No running processes found'))

def idle():
    owners = subprocess.run(['fuser', *[f'/dev/davinci{rank}' for rank in range(8)]],
                            capture_output=True, text=True)
    if owners.stdout.strip():
        raise RuntimeError(f'GPUs occupied: {owners.stdout.strip()}')
    sample = hbm_sample()
    if sample['no_process_cards'] != 8:
        raise RuntimeError('Driver reports GPU processes before admission')
    if any(card['utilization'] > 0 for card in sample['cards']):
        raise RuntimeError('GPU utilization is nonzero before admission')
    return sample


def foreign_device_owners(container_id):
    """Observe overlap without signalling another project's processes."""
    result=subprocess.run(['fuser',*[f'/dev/davinci{rank}' for rank in range(8)]],
                          capture_output=True,text=True,timeout=10)
    foreign=[]
    for value in set(result.stdout.split()):
        if not value.isdigit():continue
        try:group=pathlib.Path(f'/proc/{value}/cgroup').read_text()
        except (FileNotFoundError,PermissionError):continue
        if container_id not in group and container_id[:12] not in group:
            foreign.append(int(value))
    return sorted(foreign)

def _run_job(backend, mode, tag, arguments, seconds=1800, worker=None, capacity=64):
    diagnostic='--diagnostic-timer' in arguments
    if '--temporal-schedule-replay' in arguments and '--diagnostic-only' not in arguments:
        raise RuntimeError('Offline task schedules cannot be submitted as online training performance')
    if worker=='temporal-ep8-cap64-timer' and not diagnostic:
        raise RuntimeError('TimerON worker is diagnostic-only, never a performance backend')
    out = TASK / 'evidence' / tag
    out.mkdir(parents=True, exist_ok=False)
    deadline=time.monotonic()+3600
    while True:
        try:
            admission = idle()
            break
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            (out/'queue.json').write_text(json.dumps(dict(state='waiting_for_idle_devices',
                timestamp=time.time(),reason=str(error),admission_limit_seconds=3600),indent=2))
            if time.monotonic()>=deadline:
                status=dict(exit=1,qualified=False,source_unchanged=None,rank_completions=0,
                            started=False,outcome='admission_timeout',reason=str(error),peak_driver_MiB=0)
                (out/'status.json').write_text(json.dumps(status,indent=2))
                return status
            time.sleep(5)
    (out / 'admission.json').write_text(json.dumps(admission, indent=2))
    snapshots=TASK/'evidence'/'source-snapshots'
    snapshots.mkdir(exist_ok=True)
    subprocess.run(['python3',str(TASK/'tools/verify_frozen_runtime.py'),'--manifest',
                    str(TASK/'configs/runtime-freeze.json'),'--out',str(snapshots/f'{tag}-runtime.json')],check=True)
    if diagnostic:
        manifest=TASK/'configs/temporal-worker-timer-freeze.json'
        entries=json.loads(manifest.read_text())['binaries']
        prefix=f'source/native-balanced/worker-variants/{worker}/'
        if not entries or any(not name.startswith(prefix) for name in entries):
            raise RuntimeError('Diagnostic worker does not match its separately frozen TimerON build')
        subprocess.run(['python3',str(TASK/'tools/verify_frozen_runtime.py'),'--manifest',str(manifest),
            '--out',str(snapshots/f'{tag}-timer-worker-runtime.json')],check=True)
    if mode=='scheduler' or ('--temporal-scheduler' in arguments and
            arguments[arguments.index('--temporal-scheduler')+1]=='ascendc'):
        subprocess.run(['python3',str(TASK/'tools/verify_frozen_runtime.py'),'--manifest',
            str(TASK/'configs/temporal-scheduler-freeze.json'),'--out',
            str(snapshots/f'{tag}-temporal-runtime.json')],check=True)
    sources=source_identity(arguments)
    (out/'source-identity.json').write_text(json.dumps(sources,indent=2))
    # Cache only byte-identical, verified snapshots from unchanged prior runs.
    from source_freeze import freeze_snapshot
    freeze_snapshot(TASK,tag,sources)
    # The source freeze may take time on NFS. Another project need not share
    # this study's flock, so check hardware ownership again before launching.
    final_admission=idle()
    (out/'launch-admission.json').write_text(json.dumps(final_admission,indent=2))
    container_id=subprocess.check_output(['docker','inspect',CONTAINER,'--format','{{.Id}}'],text=True).strip()
    command = ['docker', 'exec', '-w', str(TASK), '-e', f'EP8_MASTER_PORT={29200 + int(time.time()) % 2000}',
               CONTAINER, 'timeout', '-k', '30', str(seconds), 'bash', 'tools/launch.sh',
               backend, mode, tag, *arguments]
    if worker:
        command[2:2] = ['-e', f'EP8_WORKER_VARIANT={worker}']
    if capacity == 32:
        command[2:2] = ['-e', 'EP8_CAPACITY=32', '-e', 'EP8_NATIVE_SUFFIX=32']
    started = time.time()
    max_hbm = max(card['used_MiB'] for card in admission['cards'])
    overlap=set()
    with (out / 'stdout.log').open('w') as log, (out / 'hbm.jsonl').open('w') as monitor:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                sample = hbm_sample()
                owners=foreign_device_owners(container_id)
                sample['foreign_device_owners']=owners
                overlap.update(owners)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                monitor.write(json.dumps(dict(timestamp=time.time(), sample_error=str(error))) + '\n')
                monitor.flush()
                continue
            max_hbm = max(max_hbm, *(card['used_MiB'] for card in sample['cards']))
            monitor.write(json.dumps(sample) + '\n')
            monitor.flush()
            time.sleep(1)
        status = dict(command=command, exit=process.returncode, start=started,
                      finish=time.time(), peak_driver_MiB=max_hbm,
                      foreign_device_owners=sorted(overlap),exclusive_devices=not overlap)
    if mode == 'train':
        status['rank_completions'] = len(list(out.glob('complete-rank*.json')))
        status['qualified'] = status['exit'] == 0 and status['rank_completions'] == 8
    else:
        status['qualified'] = status['exit'] == 0
    if overlap:status['qualified']=False
    status['source_unchanged']=sources==source_identity(arguments)
    if not status['source_unchanged']:status['qualified']=False
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(dict(tag=tag, **status)), flush=True)
    return status

def run_job(backend, mode, tag, arguments, seconds=1800, worker=None, capacity=64):
    with pathlib.Path('/home/z00956592/megaflux/temporal-replica-study-20261114/stack/evidence/device-jobs.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run_job(backend, mode, tag, arguments, seconds, worker, capacity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['balanced', 'shared', 'moonep'])
    parser.add_argument('--mode', default='train', choices=['train', 'validate', 'activation', 'scaling', 'pool', 'norm', 'moonpool', 'sharedpool','plan','links','scheduler','temporal_math'])
    parser.add_argument('--tag')
    parser.add_argument('--worker-variant')
    parser.add_argument('--capacity', type=int, choices=[32,64], default=64)
    parser.add_argument('--campaign', action='store_true')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=1800)
    args, extra = parser.parse_known_args()
    if not args.campaign:
        if args.mode == 'train':
            extra += ['--layers', str(args.layers)]
        status = run_job(args.backend, args.mode, args.tag, extra, args.seconds, args.worker_variant, args.capacity)
        raise SystemExit(0 if status['qualified'] else 1)
    matrix = []
    arms = ['moonep', 'balanced', 'shared']
    for repeat in range(3):
        ordered = arms[repeat:] + arms[:repeat]
        for maxvio in (.2, 1, 5, 10, 15, 20):
            for backend in ordered:
                tag = f'formal-L{args.layers}-vio{maxvio:g}-{backend}-r{repeat+1}'
                status = run_job(backend, 'train', tag,
                         ['--maxvio', str(maxvio), '--layers', str(args.layers)], args.seconds, args.worker_variant, args.capacity)
                matrix.append(dict(tag=tag, backend=backend, maxvio=maxvio, repeat=repeat+1, **status))
                (TASK / 'evidence' / 'campaign.json').write_text(json.dumps(matrix, indent=2))
                if not status['qualified']:
                    raise RuntimeError(f'Unqualified run: {tag}')
                if status['peak_driver_MiB'] > 90 * 1024:
                    raise RuntimeError(f'Common shape memory gate exceeded: {tag}')
    print('CAMPAIGN_COMPLETE', flush=True)

if __name__ == '__main__':
    main()
