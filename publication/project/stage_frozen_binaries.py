"""Reconstruct verified aliases in a staging directory, never a live install."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile
root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,default=root/'implementation/restored-binaries')
parser.add_argument('--include-upstream',action='store_true')
args=parser.parse_args();destination=args.out.resolve()
archive=root/'implementation/frozen-build-export.tar.gz'
expected=json.loads((root/'implementation/frozen-build-export.json').read_text())
if hashlib.sha256(archive.read_bytes()).hexdigest()!=expected['sha256']:
    raise RuntimeError('Original build archive checksum mismatch')
records={}
def write(name,data,identity):
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe alias in manifest')
    target=(destination/str(relative)).resolve()
    if not target.is_relative_to(destination):raise ValueError('Alias escapes staging root')
    if hashlib.sha256(data).hexdigest()!=identity:raise RuntimeError('SO content checksum mismatch')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest()!=identity:raise RuntimeError('Refuse to overwrite a different staged file')
    else:
        with target.open('xb') as handle:handle.write(data)
    records[name]=dict(sha256=identity,bytes=len(data))
with tarfile.open(archive) as handle:
    manifest=json.load(handle.extractfile('BUILD-EXPORT.json'))
    for alias,item in manifest['aliases'].items():
        data=handle.extractfile(item['blob']).read()
        if len(data)!=item['bytes']:raise RuntimeError('SO length mismatch')
        write(alias,data,item['sha256'])
if args.include_upstream:
    name='source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so'
    write(name,(root/'implementation/upstream-selector-v4.so').read_bytes(),
          '871c167357d1d6e1575d331b3e73c7b36bcc115fb41988e02d4d8a3bb029013e')
receipt=dict(root=str(destination),files=records,live_install=False,
    includes_upstream=args.include_upstream,
    performance_worker_requires_environment='EP8_WORKER_VARIANT=ep8-cap64-aligned',
    original_abi3_and_diagnostic_worker_remain_separate=True)
(root/'implementation/binary-staging-check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(verified_staged_aliases=len(records),destination=str(destination),live_install=False)))
