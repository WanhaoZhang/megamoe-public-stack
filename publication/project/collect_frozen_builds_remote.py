"""Read-only binary/build export; installed libraries are never overwritten."""
from pathlib import Path
import hashlib
import io
import json
import tarfile

root=Path('/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack')
output=root/'evidence/frozen-build-export.tar.gz'
manifests=('runtime-freeze.json','temporal-scheduler-freeze.json','temporal-worker-timer-freeze.json')
aliases={};blobs={};metadata={}
for name in manifests:
    path=root/'configs'/name
    value=json.loads(path.read_text())
    for relative,expected in value['binaries'].items():
        data=(root/relative).read_bytes()
        actual=hashlib.sha256(data).hexdigest()
        if actual!=expected:raise RuntimeError('Frozen binary changed: '+relative)
        aliases[relative]=dict(sha256=actual,blob='sha256/'+actual+'.so',bytes=len(data))
        blobs[actual]=data
    metadata['manifests/'+name]=path.read_bytes()
for relative in ('source/native-balanced/build-worker-ep8-cap64-timeroff/CMakeCache.txt',
        'source/native-balanced/build-temporal-cap64-timer/CMakeCache.txt',
        'evidence/timer-worker-build/configure.json','evidence/timer-worker-build/binary.json',
        'evidence/timer-build-normal-binaries.json'):
    path=root/relative
    if path.is_file():metadata['build/'+relative]=path.read_bytes()
proof=dict(aliases=aliases,unique_binaries=len(blobs),
    scope='frozen normal performance libraries, selector and separate TimerV2 diagnostic build',
    diagnostic_binary_excluded_from_performance_scores=True,
    metadata={name:hashlib.sha256(data).hexdigest() for name,data in metadata.items()})
with tarfile.open(output,'w:gz') as archive:
    def add(name,data):
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
    for sha,data in sorted(blobs.items()):add('sha256/'+sha+'.so',data)
    for name,data in sorted(metadata.items()):add(name,data)
    add('BUILD-EXPORT.json',json.dumps(proof,indent=2).encode())
print(json.dumps(dict(archive=str(output),aliases=len(aliases),unique=len(blobs),
    sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size)))
