"""Explicit serial protocol gates; no retries, device resets, or lost rank receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    assert cases and len({case['name'] for case in cases}) == len(cases)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open('x', buffering=1) as log:
        for case in cases:
            assert case['entry'] in ('test_expert_fsdp2.py', 'test_bridge_npu.py')
            assert case['ep'] in (2, 4, 8)
            assert case['planner'] in ('native', 'moonep')
            restore = case['restore']
            variant = {'staged': 's9-ported', 'expert': 'restore-expert', 'panel': 'restore-panel'}[restore]
            command = [sys.executable, str(args.project/'snapshots/gate_and_run.py'),
                       '--project', str(args.project), '--name', case['name'], '--',
                       'docker', 'exec', '-e', 'WUHAN_GATE_ENTRY='+case['entry'],
                       '-e', 'WUHAN_PUBLISH_SYNC='+case.get('publish_sync', 'device'),
                       '-e', 'WUHAN_WORKER_VARIANT='+variant, args.container, 'bash',
                       args.source+'/ascend_deepep_hkq/integration/wuhan/run_bridge_gate.sh',
                       args.source, '/project/evidence/'+case['name'], str(case['ep']),
                       '--planner', case['planner'], '--restore', restore]
            for key in ('tokens', 'ffn', 'limit', 'slots', 'capacity'):
                if key in case:
                    command += ['--'+key, str(case[key])]
            for key in ('checkpoint', 'require_replicas', 'parameter_fp32'):
                if case.get(key, False):
                    command += ['--'+key.replace('_', '-')]
            print('GATE_START', case['name'], flush=True)
            log.write(json.dumps(dict(event='start', case=case, command=command))+'\n')
            rc = subprocess.run(command).returncode
            log.write(json.dumps(dict(event='exit', name=case['name'], rc=rc))+'\n')
            if rc:
                raise SystemExit(rc)
            evidence = args.project/'evidence'/case['name']
            for rank in range(case['ep']):
                receipt = json.loads((evidence/f'rank{rank}.json').read_text())
                assert receipt['pass_'] and receipt['rank'] == rank
            log.write(json.dumps(dict(event='rank_receipts_verified', name=case['name'], ranks=case['ep']))+'\n')
            print('GATE_ACCEPTED', case['name'], flush=True)
    print('GATE_QUEUE_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
