"""Reject binary drift before opening any device context."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
expected = json.loads(args.manifest.read_text())['binaries']
observed = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in expected}
mismatches = [name for name in expected if expected[name] != observed[name]]
args.out.write_text(json.dumps(dict(qualified=not mismatches, binaries=observed,mismatches=mismatches),indent=2))
if mismatches:
    raise RuntimeError(f'Frozen runtime binary mismatch: {mismatches}')
print('FROZEN_RUNTIME_PASS', len(observed), flush=True)
