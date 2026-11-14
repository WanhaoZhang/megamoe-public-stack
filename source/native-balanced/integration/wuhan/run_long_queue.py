"""Run explicit 200-step single-node cases with gates, HBM and no retries."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from subprocess_bounds import bound_command


def training_command(source, container, case, worker_variant='restore-panel-ready'):
    name = case['name']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
        raise ValueError('unsafe case name')
    if case['method'] not in ('A', 'B', 'C', 'D') or case['rho'] not in (0, 3, 6):
        raise ValueError('unknown method/rho')
    if not re.fullmatch(r'[a-z0-9-]+', worker_variant):
        raise ValueError('unsafe worker variant')
    command = ['docker', 'exec']
    environment = dict(WUHAN_NNODES='1', WUHAN_NODE_RANK='0', WUHAN_EP_SIZE='4',
        WUHAN_EXPERT_FSDP='2', WUHAN_LOCAL_WORLD_SIZE='8', WUHAN_MASTER_ADDR='127.0.0.1',
        WUHAN_MASTER_PORT='29873', WUHAN_WORKER_VARIANT=worker_variant,
        BALANCED_RESTORE_MODE='panel', WUHAN_PUBLISH_SYNC='device',
        WUHAN_CONTROL_TAPE_DEVICE='npu', WUHAN_NATIVE_SLOTS='2',
        WUHAN_RECV_CAPACITY='20480', WUHAN_CAPTURE_REPLAY='0',
        WUHAN_DATA_NO_SHARED_STORAGE='1')
    for key, value in environment.items():
        command += ['-e', key+'='+value]
    command += [container, 'bash', source+'/ascend_deepep_hkq/integration/wuhan/run_training_case.sh',
        source, '/project/evidence/'+name, case['method'], str(case['rho']), 'recompute', '200', '8']
    return bound_command(command, 2400)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--worker-variant', default='restore-panel-ready',
                        help='isolated operand-ready=1, delay=0 worker build name')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.manifest.read_text(encoding='utf-8'))
    if not cases or len({c['name'] for c in cases}) != len(cases):
        raise ValueError('empty/duplicate cases')
    commands = [training_command(args.source, args.container, c, args.worker_variant) for c in cases]
    if any((args.project/'evidence'/c['name']).exists() for c in cases):
        raise ValueError('evidence already exists; use a new reviewed manifest')
    tools = Path(__file__).resolve().parent
    with args.record.open('x', encoding='utf-8', buffering=1) as record:
        for case, bounded in zip(cases, commands):
            out = args.project/'evidence'/case['name']
            command = [sys.executable, str(tools/'gate_and_run.py'), '--project', str(args.project),
                '--name', case['name'], '--', sys.executable, str(tools/'fast_memory_command.py'),
                '--out', str(out), '--', *bounded]
            record.write(json.dumps(dict(event='start', case=case, command=command))+'\n')
            print('LONG_CASE_START', case['name'], flush=True)
            rc = subprocess.run(command, stdin=subprocess.DEVNULL).returncode
            record.write(json.dumps(dict(event='exit', name=case['name'], rc=rc))+'\n')
            if rc:
                raise SystemExit(rc)
            # Sealing and the 200x8 completeness audit happen after the device
            # post-gate, before the next case can acquire the experiment lock.
            subprocess.run([sys.executable, str(tools/'seal_evidence_origin.py'),
                '--runs', str(out)], check=True, stdin=subprocess.DEVNULL)
            subprocess.run([sys.executable, str(tools/'summarize_training.py'), '--runs', str(out),
                '--steps', '200', '--warmup', '10', '--out', str(out/'training-summary.json')],
                check=True, stdin=subprocess.DEVNULL)
            print('LONG_CASE_PASS', case['name'], flush=True)
    print('LONG_QUEUE_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
