"""Root-owned node worker. Hold the experiment lock across pre/start/post."""
import argparse
import base64
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gate_and_run import gate
from subprocess_bounds import bound_command


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--command-base64', required=True)
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if not a.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in a.name):
        raise ValueError('unsafe run name')
    out = a.project/'evidence'/a.name
    out.mkdir(parents=True, exist_ok=False)
    command = json.loads(base64.b64decode(a.command_base64))
    assert isinstance(command, list) and command and all(isinstance(x, str) for x in command)
    (out/'command.json').write_text(json.dumps(command, indent=2))
    executed, host_timeout = bound_command(command, a.timeout)
    (out/'executed-command.json').write_text(json.dumps(executed, indent=2))
    with (a.project/'.device-experiment.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        passed = gate(out, 'pre')
        (out/'ready.json').write_text(json.dumps(dict(ready=passed, pid=os.getpid())))
        if not passed:
            (out/'NOT_STARTED').write_text('pre-gate failed\n')
            return 75
        deadline = time.monotonic()+180
        while not (out/'release').exists():
            if (out/'abort').exists() or time.monotonic() >= deadline:
                (out/'NOT_STARTED').write_text('coordinator aborted or rendezvous release timed out\n')
                gate(out, 'post')
                return 77
            time.sleep(.5)
        with (out/'run.log').open('w') as log:
            proc = subprocess.Popen(executed, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (out/'process.json').write_text(json.dumps(dict(pid=proc.pid, pgid=proc.pid)))
            try:
                rc = proc.wait(timeout=host_timeout)
            except subprocess.TimeoutExpired:
                (out/'TIMEOUT').write_text('terminating this job process group only\n')
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                rc = 124
        (out/'run.exit').write_text(str(rc)+'\n')
        post = gate(out, 'post')
        (out/'acceptance.json').write_text(json.dumps(dict(run_exit=rc, post_gate=post,
            eligible=rc == 0 and post, scope='node receipt; all peers must also pass'), indent=2))
        if rc == 0 and post and (out/'launch.json').exists():
            subprocess.run([sys.executable, str(Path(__file__).with_name('seal_evidence_origin.py')),
                            '--runs', str(out)], check=True)
        return rc if rc else (0 if post else 76)


if __name__ == '__main__':
    raise SystemExit(main())
