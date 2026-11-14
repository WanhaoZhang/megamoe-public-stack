"""CPU-only Docker timeout probe; creates and observes only its own child."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import uuid

from subprocess_bounds import bound_command


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--container', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    marker = '/tmp/wuhan-timeout-'+uuid.uuid4().hex
    child = ('import pathlib,subprocess,time; '
             'p=subprocess.Popen(["sleep","60"]); '
             f'pathlib.Path({marker!r}).write_text(str(p.pid)); time.sleep(60)')
    command, deadline = bound_command(['docker','exec',args.container,'python','-c',child], 2)
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=deadline)
    elapsed = time.monotonic()-started
    inspect = ('import pathlib,json; '
               f'm=pathlib.Path({marker!r}); pid=int(m.read_text()); '
               'p=pathlib.Path(f"/proc/{pid}/stat"); '
               'state=p.read_text().split()[2] if p.exists() else "absent"; '
               'print(json.dumps(dict(pid=pid,state=state))); m.unlink()')
    observed = subprocess.run(['docker','exec',args.container,'python','-c',inspect],
                              capture_output=True,text=True,check=True)
    status = json.loads(observed.stdout)
    passed = result.returncode == 124 and status['state'] in ('absent','Z')
    args.out.write_text(json.dumps(dict(pass_=passed,exit=result.returncode,elapsed_s=elapsed,
        child=status,stderr=result.stderr,scope='CPU-only timeout process-group probe'),indent=2)+'\n')
    assert passed, args.out.read_text()
    print('CONTAINER_TIMEOUT_CHILD_RETIRED',status['state'])


if __name__ == '__main__':
    main()
