"""Verify the exact archived baseline bytes without extracting over live files."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'artifacts/sources/BASELINE-SNAPSHOT.json').read_text())
archive=root/manifest['archive']
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
seen=set()
with tarfile.open(archive,'r:gz') as stream:
    for member in stream:
        assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
        if member.name=='evidence/BASELINE-FREEZE.json':continue
        assert member.name in manifest['members'] and member.name not in seen
        payload=stream.extractfile(member).read();expected=manifest['members'][member.name]
        assert len(payload)==expected['bytes'] and hashlib.sha256(payload).hexdigest()==expected['sha256'],member.name
        seen.add(member.name)
assert seen==set(manifest['members'])
result=dict(passed=True,archive_sha256=manifest['sha256'],files=len(seen),
    baseline_source_files=manifest['source_files'],baseline_binary_files=manifest['binary_files'],
    archive_stream_verified=True,no_live_file_overwritten=True)
(root/'artifacts/sources/BASELINE-SNAPSHOT-LOCAL-VERIFY.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
