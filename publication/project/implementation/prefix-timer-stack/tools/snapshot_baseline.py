"""Preserve exact frozen baseline bytes before later policy-only additions."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
from controller import TASK, source_identity

freeze_path=TASK/'evidence/BASELINE-FREEZE.json'
freeze=json.loads(freeze_path.read_text())
assert source_identity()==freeze['source'],'baseline sources already changed'
destination=TASK/'artifacts/sources';destination.mkdir(exist_ok=True,parents=True)
target=destination/'baseline-measured-source-and-binaries.tar.gz'
manifest_path=destination/'BASELINE-SNAPSHOT.json'
if target.exists() or manifest_path.exists():raise RuntimeError('refuse to overwrite the baseline snapshot')
expected={**freeze['source'],**freeze['binaries']}
manifest={}
with tarfile.open(target,'w:gz',compresslevel=1) as archive:
    for name,digest in sorted(expected.items()):
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe frozen path')
        payload=(TASK/relative).read_bytes();actual=hashlib.sha256(payload).hexdigest()
        if actual!=digest:raise RuntimeError('frozen bytes changed: '+name)
        item=tarfile.TarInfo(name);item.size=len(payload);item.mode=0o644
        archive.addfile(item,io.BytesIO(payload));manifest[name]=dict(sha256=actual,bytes=len(payload))
    payload=freeze_path.read_bytes();item=tarfile.TarInfo('evidence/BASELINE-FREEZE.json');item.size=len(payload)
    archive.addfile(item,io.BytesIO(payload))
assert source_identity()==freeze['source'],'source changed while preserving baseline'
result=dict(passed=True,source_files=len(freeze['source']),binary_files=len(freeze['binaries']),
    archive=target.relative_to(TASK).as_posix(),bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),members=manifest,
    purpose='exact original existing-policy matrix bytes, independent of later step_time or link-diagnostic additions',
    working_sources_unchanged=True)
manifest_path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='members'}),flush=True)
