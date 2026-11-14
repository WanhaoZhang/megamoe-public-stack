"""Host wrapper: sample whole-device HBM while preserving child exit status."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess
import threading
import time


def sample(out, stop):
    with (out/'hbm-samples.jsonl').open('x', buffering=1) as handle:
        while not stop.is_set():
            started = time.monotonic()
            result = subprocess.run(['npu-smi', 'info'], capture_output=True, text=True)
            raw = result.stdout + result.stderr
            values = {}
            rank = None
            for line in raw.splitlines():
                match = re.match(r'\|\s*(\d+)\s*\|\s*Ascend950\S*\s*\|', line)
                if match:
                    rank = int(match[1])
                elif rank is not None:
                    pair = re.search(r'(\d+)\s*/\s*(\d+)\s*\|\s*$', line)
                    if pair:
                        values[rank] = dict(used_mb=int(pair[1]), total_mb=int(pair[2]))
                        rank = None
            handle.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                monotonic=started, elapsed_s=time.monotonic()-started,
                rc=result.returncode, hbm=values, raw=raw))+'\n')
            stop.wait(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    worker = threading.Thread(target=sample, args=(a.out, stop))
    worker.start()
    try:
        command = a.command[1:] if a.command[:1] == ['--'] else a.command
        rc = subprocess.run(command).returncode
    finally:
        stop.set()
        worker.join()
    raise SystemExit(rc)


if __name__ == '__main__':
    main()
