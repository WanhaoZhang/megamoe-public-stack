"""Host-side explicit serial case list; stop on any failed gate or training.

This is not a retry service. Every evidence name must be new, every case goes
through the root device gate, and a failure leaves remaining cases unstarted.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--container', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    assert cases and len({case['name'] for case in cases}) == len(cases)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    with args.record.open('x', buffering=1) as log:
        for case in cases:
            assert case['method'] in 'ABCD' and len(case['method']) == 1
            assert case['rho'] in (0, 3, 6) and case['activation'] in ('recompute', 'uam')
            assert isinstance(case['steps'], int) and case['steps'] > 0
            gate = args.project/'snapshots/gate_and_run.py'
            runner = args.source+'/ascend_deepep_hkq/integration/wuhan/run_training_case.sh'
            restore = case.get('restore', 'staged')
            assert restore in ('staged', 'expert', 'panel')
            variant = {'staged': 's9-ported', 'expert': 'restore-expert', 'panel': 'restore-panel'}[restore]
            command = [sys.executable, str(gate), '--project', str(args.project), '--name', case['name'],
                       '--', 'docker', 'exec', '-e', 'WUHAN_WORKER_VARIANT='+variant,
                       '-e', 'BALANCED_RESTORE_MODE='+restore,
                       '-e', 'WUHAN_CAPTURE_REPLAY='+('1' if case.get('capture_replay', False) else '0'),
                       args.container, 'bash', runner, args.source,
                       '/project/evidence/'+case['name'], case['method'], str(case['rho']),
                       case['activation'], str(case['steps']), str(case.get('layers', 8))]
            print('CASE_START', case['name'], flush=True)
            log.write(json.dumps(dict(event='start', case=case, command=command))+'\n')
            rc = subprocess.run(command).returncode
            log.write(json.dumps(dict(event='exit', name=case['name'], rc=rc))+'\n')
            print('CASE_EXIT', case['name'], rc, flush=True)
            if rc:
                raise SystemExit(rc)
    print('CASE_QUEUE_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
