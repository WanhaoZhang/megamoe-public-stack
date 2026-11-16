"""Verify every native/runtime file against the frozen migration manifest."""
import hashlib
import json
from pathlib import Path
import sys

project, manifest_file = map(Path, sys.argv[1:])
manifest = json.loads(manifest_file.read_text())
bad = []
for name, expected in manifest['files'].items():
    path = project/name
    got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if got != expected:
        bad.append(dict(path=name, expected=expected, actual=got))
record = dict(pass_=not bad, files=len(manifest['files']), mismatches=bad)
(project/'evidence/migration-verify.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
raise SystemExit(bool(bad))
