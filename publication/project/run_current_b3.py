"""Numerically qualify host-only wrappers, then inspect warm host boundaries."""
from pathlib import Path
import json
import math
import subprocess
import remote
ROOT = Path(__file__).resolve().parent
VERSION = 'v5'
STATE = ROOT / ('diagnosis-' + VERSION + '.json')
if STATE.exists():
    raise SystemExit('Immutable diagnosis already started; inspect its state.')
state = dict(stage='small', runs=[], diagnostic_only=True, new_planner=False, schedule=[['b3_cached', 'b3', 'preserve', 'ep8-cap64-lifetime-priority-cache-v1']])

def save():
    STATE.write_text(json.dumps(state, indent=2) + '\n', newline='\n')

def gate(tag, args, steps, layers):
    try:
        remote.gate(tag, args, metadata_only=True)
    except SystemExit as done:
        code = done.code
    folder = ROOT / 'evidence' / tag
    status = json.loads((folder / 'status.json').read_text())
    checks = {key: bool(status.get(key)) for key in ('qualified', 'source_unchanged', 'exclusive_devices')}
    checks['exit_and_memory'] = code == 0 and status['peak_driver_MiB'] <= 90 * 1024
    if not all(checks.values()):
        return (folder, dict(tag=tag, passed=False, failed_checks=[key for key, value in checks.items() if not value], peak_driver_MiB=status['peak_driver_MiB'], diagnostic_only=True, actual_exit=status['exit']))
    for rank in range(8):
        records = [json.loads(line) for line in (folder / f'steps-rank{rank}.jsonl').read_text().splitlines()]
        checks[f'rank{rank}:finite_steps'] = len(records) == steps and all((math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in records))
        updates = json.loads((folder / f'updates-rank{rank}.json').read_text())
        checks[f'rank{rank}:three_adam_updates'] = len(updates) == 3 and all((row['changed'] for row in updates))
        contract = json.loads((folder / f'host-boundary-contract-rank{rank}.json').read_text())
        checks[f'rank{rank}:trace_contract'] = all((contract[key] for key in ('diagnostic_only', 'no_new_device_events', 'no_new_device_syncs', 'no_tensor_snapshots', 'dump_after_training'))) and (not contract['host_return_is_device_completion'])
        trace = [json.loads(line) for line in (folder / f'host-boundary-rank{rank}.jsonl').read_text().splitlines()]
        checks[f'rank{rank}:paired_scopes'] = len(trace) == contract['records'] and len(set((row['sequence'] for row in trace))) == len(trace) and all((row['host_return_ns'] >= row['host_begin_ns'] and (not row['failed']) for row in trace))
        for step in range(steps):
            calls = [row for row in trace if row['name'] == 'operator.backward' and row['update'] == step]
            checks[f'rank{rank}:step{step}_backward_coverage'] = len(calls) == layers * 4
        libraries = json.loads((folder / f'libraries-rank{rank}.json').read_text())
        variant = args[args.index('--worker-variant') + 1]
        digest = '72653abf3c22fb951624dd5dc420c488d2511d1c22e5e605a3621cb4210502bf' if variant.endswith('raw-control-v2') else 'e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed'
        workers = [info['sha256'] for name, info in libraries['libraries'].items() if '/worker-variants/' in name and name.endswith('.so')]
        checks[f'rank{rank}:actual_worker'] = workers == [digest] and len(libraries['shmem']) == 1
    return (folder, dict(tag=tag, passed=all(checks.values()), failed_checks=[key for key, value in checks.items() if not value], peak_driver_MiB=status['peak_driver_MiB'], diagnostic_only=True))
save()
try:
    for phase in ('small', 'main'):
        state['stage'] = phase
        save()
        for name, mode, raw, variant in state['schedule']:
            tag = 'host-' + phase + '-' + name + '-' + VERSION
            state['current'] = tag
            save()
            small = phase == 'small'
            steps, layers = (3, 2) if small else (11, 4)
            args = ['--layers', str(layers), '--steps', str(steps), '--tokens', '256' if small else '16384', '--hidden', '512' if small else '4096', '--ffn', '512' if small else '1024', '--microbatches', '4', '--forward-window', '2', '--recv-capacity', '4096' if small else '262144', '--maxvio', '10', '--planner', 'moonep' if small else 'native', '--temporal-pattern', 'same', '--temporal-policy', 'fixed_native', '--worker-variant', variant, '--grad-sync', '--hold-fsdp-forward', '--temporal-prefix-reuse', '--lifetime-mode', mode, '--lifetime-raw-schedule', raw, '--host-boundary-trace']
            if small:
                args += ['--dump-grad-step0', '--finite-diagnostics', '--route-bank', remote.REMOTE + '/stack/small-routes']
            else:
                args += ['--temporal-scheduler', 'ascendc', '--temporal-scheduler-library', remote.REMOTE + '/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
            folder, result = gate(tag, args, steps, layers)
            if small and result['passed']:
                path = remote.REMOTE + '/stack/evidence/' + tag + '-gradient-comparison.json'
                command = ['ssh', remote.HOST, 'docker', 'exec', remote.CONTAINER, '/usr/local/python3.12.13/bin/python3', remote.REMOTE + '/stack/tools/compare_temporal_gradients.py', '/home/z00956592/megaflux/step-lifetime-raw-control-20261116/stack/evidence/raw-model-preserve-v1', remote.REMOTE + '/stack/evidence/' + tag, '--out', path]
                with (ROOT / 'diagnostics' / (tag + '-comparison.log')).open('wb') as handle:
                    comparison = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
                local = ROOT / 'diagnostics' / (tag + '-gradient-comparison.json')
                subprocess.run(['scp', remote.HOST + ':' + path, str(local)], check=True)
                result['passed'] &= comparison.returncode == 0 and json.loads(local.read_text())['passed']
                result['full_step0_gradient_reference'] = 'raw-model-preserve-v1'
            state['runs'].append(result)
            save()
            if not result['passed']:
                raise RuntimeError('Host-only qualification failed: ' + tag + ' ' + str(result['failed_checks']))
    state.update(stage='complete', passed=True)
    state.pop('current', None)
    save()
except BaseException as error:
    state.update(stage='failed', error=str(error))
    save()
    raise
