"""Explicit frozen paired queue, with device gates and whole-device HBM."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--container', required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    cases = json.loads(a.manifest.read_text())
    assert len({c['name'] for c in cases}) == len(cases)
    with a.record.open('x', buffering=1) as handle:
        for case in cases:
            name = case['name']
            restore = case.get('restore', 'panel')
            variant = {'staged':'s9-ported', 'expert':'restore-expert', 'panel':'restore-panel'}[restore]
            command = [sys.executable, str(a.project/'snapshots/gate_and_run.py'),
                '--project', str(a.project), '--name', name, '--', sys.executable,
                str(a.project/'snapshots/fast_memory_command.py'),
                '--out', str(a.project/'evidence'/name), '--', 'docker', 'exec',
                '-e', 'WUHAN_WORKER_VARIANT='+variant, '-e', 'BALANCED_RESTORE_MODE='+restore,
                '-e', 'WUHAN_PUBLISH_SYNC='+case.get('publish_sync', 'device'),
                '-e', 'WUHAN_NATIVE_SLOTS='+str(case.get('native_slots', 4)),
                '-e', 'WUHAN_RECV_CAPACITY='+str(case.get('recv_capacity', '')),
                '-e', 'WUHAN_CAPTURE_REPLAY=0', a.container, 'bash',
                a.source+'/ascend_deepep_hkq/integration/wuhan/run_training_case.sh',
                a.source, '/project/evidence/'+name, case['method'], str(case['rho']),
                case.get('activation', 'recompute'), str(case.get('steps', 40)), '8']
            print('CASE_START', name, flush=True)
            handle.write(json.dumps(dict(event='start', case=case, command=command))+'\n')
            rc = subprocess.run(command).returncode
            handle.write(json.dumps(dict(event='exit', name=name, rc=rc))+'\n')
            print('CASE_EXIT', name, rc, flush=True)
            if rc:
                raise SystemExit(rc)
    print('FAST_QUEUE_COMPLETE', len(cases), flush=True)


if __name__ == '__main__':
    main()
