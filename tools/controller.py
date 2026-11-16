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

TASK = pathlib.Path('/home/z00956592/megaflux/single-mb-fbw-planner-study')
CONTAINER = 'single-mb-fbw-planner-study'

def source_identity(worker=None):
    files={}
    for component in ('native-balanced','FSDPTurbo','MindSpeed-LLM'):
        root=TASK/'source'/component
        for path in sorted(root.rglob('*')):
            relative=path.relative_to(TASK)
            if not path.is_file() or path.is_symlink() or any(p in ('.git','__pycache__','3rdparty','worker-variants') or p.startswith('build') for p in relative.parts[:-1]):continue
            if path.suffix in ('.py','.cpp','.h','.hpp','.cmake') or path.name in ('CMakeLists.txt',):
                files[str(relative)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((TASK/'tools').iterdir()):
        if path.is_file() and path.suffix in ('.py','.sh','.cpp'):
            files[str(path.relative_to(TASK))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for path in [*sorted((TASK/'configs').rglob('*.json')),
                 *sorted((TASK/'routes').rglob('*.npy')),*sorted((TASK/'routes').rglob('*.json'))]:
        if path.is_file():files[str(path.relative_to(TASK))]=hashlib.sha256(path.read_bytes()).hexdigest()
    native=TASK/'source/native-balanced'
    binary_paths=list((native/'ascend_deepep/lib').glob('*.so'))+list((native/'ascend_deepep').glob('_C*.so'))
    if not worker:
        binary_paths+=list((native/'worker-variants/ep8-cap64-aligned').glob('*.so'))
    for path in sorted(binary_paths):
        if path.is_file():files[str(path.relative_to(TASK))]=hashlib.sha256(path.read_bytes()).hexdigest()
    if worker:
        variant=TASK/'source/native-balanced/worker-variants'/worker
        if variant.parent.resolve()!=(TASK/'source/native-balanced/worker-variants').resolve():
            raise ValueError('worker must name a task-local variant')
        for path in sorted(variant.glob('*')):
            if path.is_file() and path.suffix in ('.so','.json'):
                files[str(path.relative_to(TASK))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return files

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

def _run_job(backend, mode, tag, arguments, seconds=1800, worker=None, capacity=64):
    out = TASK / 'evidence' / tag
    out.mkdir(parents=True, exist_ok=False)
    for attempt in range(30):
        try:
            admission = idle()
            break
        except RuntimeError:
            if attempt == 29:
                raise
            time.sleep(1)
    (out / 'admission.json').write_text(json.dumps(admission, indent=2))
    sources=source_identity(worker)
    (out/'source-identity.json').write_text(json.dumps(sources,indent=2))
    command = ['docker', 'exec', '-w', str(TASK), '-e', f'EP8_MASTER_PORT={29200 + int(time.time()) % 2000}',
               CONTAINER, 'timeout', '-k', '30', str(seconds), 'bash', 'tools/launch.sh',
               backend, mode, tag, *arguments]
    if worker:
        command[2:2] = ['-e', f'EP8_WORKER_VARIANT={worker}']
    if capacity == 32:
        command[2:2] = ['-e', 'EP8_CAPACITY=32', '-e', 'EP8_NATIVE_SUFFIX=32']
    started = time.time()
    max_hbm = max(card['used_MiB'] for card in admission['cards'])
    with (out / 'stdout.log').open('w') as log, (out / 'hbm.jsonl').open('w') as monitor:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                sample = hbm_sample()
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                monitor.write(json.dumps(dict(timestamp=time.time(), sample_error=str(error))) + '\n')
                monitor.flush()
                continue
            max_hbm = max(max_hbm, *(card['used_MiB'] for card in sample['cards']))
            monitor.write(json.dumps(sample) + '\n')
            monitor.flush()
            time.sleep(1)
        status = dict(command=command, exit=process.returncode, start=started,
                      finish=time.time(), peak_driver_MiB=max_hbm)
    if mode in ('train','prepared'):
        status['rank_completions'] = len(list(out.glob('complete-rank*.json')))
        status['qualified'] = status['exit'] == 0 and status['rank_completions'] == 8
    else:
        status['qualified'] = status['exit'] == 0
    status['source_unchanged']=sources==source_identity(worker)
    if not status['source_unchanged']:status['qualified']=False
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    print(json.dumps(dict(tag=tag, **status)), flush=True)
    return status

def run_job(backend, mode, tag, arguments, seconds=1800, worker=None, capacity=64):
    with (TASK / 'evidence' / 'device-jobs.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run_job(backend, mode, tag, arguments, seconds, worker, capacity)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['balanced', 'shared', 'moonep'])
    parser.add_argument('--mode', default='train', choices=['train','prepared','prepared_validate','validate', 'activation', 'scaling', 'pool', 'norm', 'moonpool', 'sharedpool','plan','links'])
    parser.add_argument('--tag')
    parser.add_argument('--worker-variant')
    parser.add_argument('--capacity', type=int, choices=[32,64], default=64)
    parser.add_argument('--campaign', action='store_true')
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seconds', type=int, default=1800)
    args, extra = parser.parse_known_args()
    if not args.campaign:
        if args.mode in ('train','prepared'):
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
