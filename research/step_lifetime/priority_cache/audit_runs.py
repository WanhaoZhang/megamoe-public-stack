"""Audit actual admitted runs; never treat 20 within-process steps as 20 repeats."""
from pathlib import Path
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parent
WORKERS = {
    'ep8-cap64-lifetime-raw-control-v2': '72653abf3c22fb951624dd5dc420c488d2511d1c22e5e605a3621cb4210502bf',
    'ep8-cap64-lifetime-priority-cache-v1': 'e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed',
}

def load(path):
    return json.loads(path.read_text())

def canonical_groups(folder, rank):
    rows = [json.loads(line) for line in (folder/f'temporal-groups-rank{rank}.jsonl').read_text().splitlines()]
    return [{key: row[key] for key in ('step', 'group', 'keys', 'ready_keys', 'native_waves')} for row in rows]

def audit(tag, variant, *, reference=None, lifetime='b1'):
    folder = ROOT/'evidence'/tag
    status = load(folder/'status.json')
    checks = {key: bool(status.get(key)) for key in ('qualified', 'source_unchanged', 'exclusive_devices')}
    checks['hbm_budget'] = status['peak_driver_MiB'] <= 90*1024
    receipt = load(ROOT/f'{tag}-receipt.json')
    checks['collected'] = receipt['exit'] == receipt['collection_exit'] == 0
    sources = load(folder/'source-identity.json')
    for name, digest in sources.items():
        # Deployment explicitly rewrites TASK/CONTAINER paths in these two launch files.
        local = ROOT/name.removeprefix('tools/') if name in ('tools/controller.py', 'tools/launch.sh') else ROOT/'stack'/name
        # Runtime snapshots include all guarded source files, not generated binary files.
        if local.is_file():
            checks['current_source:'+name] = hashlib.sha256(local.read_bytes()).hexdigest() == digest
    all_steps = []
    worker_records = []
    for rank in range(8):
        rows = [json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        all_steps.append(rows)
        checks[f'rank{rank}:25_finite_updates'] = len(rows) == 25 and all(
            all(math.isfinite(row[key]) for key in ('loss', 'grad_norm', 'milliseconds')) for row in rows)
        # train.py samples parameters only for the first three updates.
        checks[f'rank{rank}:three_parameter_updates'] = len(load(folder/f'updates-rank{rank}.json')) == 3 and all(
            row['changed'] for row in load(folder/f'updates-rank{rank}.json'))
        complete = load(folder/f'complete-rank{rank}.json')
        checks[f'rank{rank}:complete'] = complete['steps'] == 25 and complete['exit'] == 0
        libraries = load(folder/f'libraries-rank{rank}.json')
        workers = [(name, info['sha256']) for name, info in libraries['libraries'].items() if '/worker-variants/' in name and name.endswith('/libascend_deepep_megaflux_early_bf16.so')]
        checks[f'rank{rank}:worker'] = len(workers) == 1 and '/'+variant+'/' in workers[0][0] and workers[0][1] == WORKERS[variant]
        checks[f'rank{rank}:one_shmem'] = len(libraries['shmem']) == 1
        worker_records.append(dict(rank=rank, workers=workers, shmem=libraries['shmem']))
        routes = load(folder/f'route-identity-rank{rank}.json')
        calls = load(folder/f'route-calls-rank{rank}.json')
        checks[f'rank{rank}:route_replay'] = len(routes) == len(calls) == 400 and set(calls.values()) == {2}
        if lifetime != 'off':
            config = load(folder/f'lifetime-config-rank{rank}.json')
            stats = load(folder/f'lifetime-audit-rank{rank}.json')
            checks[f'rank{rank}:abi_mode'] = config['worker_capability'] == 2 and config['raw_schedule'] == 'preserve' and config['mode'] == lifetime
            checks[f'rank{rank}:delivered'] = stats['pending'] == 0
        if reference:
            other = ROOT/'evidence'/reference
            for stem in ('initial', 'persistent-initial', 'input-identity', 'route-identity', 'route-calls'):
                checks[f'rank{rank}:same_{stem}'] = load(folder/f'{stem}-rank{rank}.json') == load(other/f'{stem}-rank{rank}.json')
            checks[f'rank{rank}:same_groups'] = canonical_groups(folder, rank) == canonical_groups(other, rank)
    if reference:
        checks['same_actual_guarded_sources'] = sources == load(ROOT/'evidence'/reference/'source-identity.json')
    times = [max(rows[step]['milliseconds'] for rows in all_steps) for step in range(5, 25)]
    return dict(tag=tag, passed=all(checks.values()), failed_checks=[key for key, value in checks.items() if not value],
                checks=checks, actual_libraries=worker_records, median_step_ms=statistics.median(times),
                measured_rank_max_step_ms=times, peak_driver_MiB=status['peak_driver_MiB'],
                independent_processes=1, warmup_updates=5, measured_updates=20,
                full_actual_plan_dump=False, reference=reference)

if __name__ == '__main__':
    results = [audit('priority-pilot-reference-v1', next(iter(WORKERS))),
               audit('priority-pilot-cached_keys-v1', list(WORKERS)[1], reference='priority-pilot-reference-v1')]
    output = dict(passed=all(row['passed'] for row in results), runs=results,
                  actual_plan_evidence='Immutable route/task identity and order-equivalence proof; no full plan tensor dump in timed runs.')
    (ROOT/'diagnostics/pilot-identity-audit.json').write_text(json.dumps(output, indent=2)+'\n', newline='\n')
    print(json.dumps(dict(passed=output['passed'], failures=[(row['tag'], row['failed_checks']) for row in results])))
    raise SystemExit(0 if output['passed'] else 1)
