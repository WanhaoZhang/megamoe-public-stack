"""Run on the host as root. Save immediate all-eight-card gates and real exit."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from seal_evidence_origin import node_identity
from device_gate_parse import transient_monitor_only


def gate(out, stage):
    smi = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
    (out/f'{stage}.npu-smi.txt').write_text(smi.stdout+smi.stderr)
    devices = [f'/dev/davinci{i}' for i in range(8)]+['/dev/davinci_manager', '/dev/hisi_hdc']
    missing = [p for p in devices if not Path(p).exists()]
    holders = subprocess.run(['fuser', '-v', *devices], capture_output=True, text=True)
    health = re.findall(r'\|\s*(\d+)\s*\|\s*Ascend950\S*\s*\|\s*(\S+)', smi.stdout)
    idle = sorted(map(int, re.findall(r'No running processes found in NPU (\d+)', smi.stdout)))
    observations = []
    for attempt in range(5):
        observations.append(dict(utc=datetime.now(timezone.utc).isoformat(), rc=holders.returncode,
                                 stdout=holders.stdout, stderr=holders.stderr))
        monitor_only = transient_monitor_only(holders.returncode, holders.stdout,
                                               holders.stderr, idle)
        if not monitor_only or attempt == 4:
            break
        # Observe a short-lived query retire; do not kill it or ignore holders.
        time.sleep(.25)
        holders = subprocess.run(['fuser', '-v', *devices], capture_output=True, text=True)
    (out/f'{stage}.root-fuser.txt').write_text(holders.stdout+holders.stderr)
    passed = (os.geteuid() == 0 and smi.returncode == 0 and not missing and
              sorted((int(r), h) for r,h in health) == [(r,'OK') for r in range(8)] and
              idle == list(range(8)) and holders.returncode == 1 and not holders.stdout.strip())
    record = dict(pass_=passed, utc=datetime.now(timezone.utc).isoformat(),
                  health=health, idle=idle, missing=missing, fuser_rc=holders.returncode,
                  node=node_identity(), fuser_observations=observations)
    (out/f'{stage}.json').write_text(json.dumps(record, indent=2))
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', args.name):
        raise SystemExit('invalid run name')
    out = args.project/'evidence'/args.name
    out.mkdir(parents=True, exist_ok=False)
    lock = open(args.project/'.device-experiment.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    (out/'command.json').write_text(json.dumps(command, indent=2))
    if not gate(out, 'pre'):
        (out/'NOT_STARTED').write_text('all-eight-card pre-gate failed\n')
        raise SystemExit(75)
    with (out/'run.log').open('w') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    (out/'run.exit').write_text(str(rc)+'\n')
    post = gate(out, 'post')
    (out/'acceptance.json').write_text(json.dumps(dict(run_exit=rc, post_gate=post,
        eligible=rc==0 and post, note='test assertions and rank receipts must also pass'), indent=2))
    raise SystemExit(rc if rc else (0 if post else 76))


if __name__ == '__main__':
    main()
