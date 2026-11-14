"""Package frozen native outputs and canonical Python SHMEM for node migration.

Source archives must be transferred separately. No compiler tree, model data,
torch environment or credentials are collected by this script.
"""
import argparse
import hashlib
import json
from pathlib import Path
import site
import tarfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for root in ('control-source-a2', args.candidate):
        native = args.project/root/'ascend_deepep_hkq'
        paths += [(p, str(p.relative_to(args.project))) for p in (native/'ascend_deepep/lib').glob('*.so*')]
        paths += [(p, str(p.relative_to(args.project))) for p in (native/'ascend_deepep').glob('_C*.so')]
    for p in (args.project/'control-source-a2/MoonEP/npu_patch').glob('_C*.so'):
        paths.append((p, str(p.relative_to(args.project))))
    for p in (args.project/'control-source-a2/MoonEP/npu_patch/lib').glob('*.so*'):
        paths.append((p, str(p.relative_to(args.project))))
    worker = args.project/args.candidate/'ascend_deepep_hkq/worker-variants/s9-ported'
    paths += [(p, str(p.relative_to(args.project))) for p in worker.glob('*.so')]
    for location in site.getsitepackages():
        base = Path(location)
        for p in base.glob('cann_shmem-*.dist-info'):
            paths += [(f, 'python-runtime/'+str(f.relative_to(base))) for f in p.rglob('*') if f.is_file()]
        package = base/'shmem'
        if package.is_dir():
            paths += [(p, 'python-runtime/'+str(p.relative_to(base))) for p in package.rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts]
    assert any('worker-variants' in name for _,name in paths), 'missing candidate worker output'
    assert any('_pyshmem' in name for _,name in paths), 'missing canonical Python SHMEM binding'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    records = {}
    with tarfile.open(args.out, 'w:gz', dereference=True) as archive:
        for path, name in sorted(paths, key=lambda pair: pair[1]):
            assert name not in records, name
            records[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.add(path, arcname=name, recursive=False)
    manifest = dict(files=records, archive_sha256=hashlib.sha256(args.out.read_bytes()).hexdigest())
    args.out.with_suffix(args.out.suffix+'.json').write_text(json.dumps(manifest, indent=2))
    print('PACKAGED_NATIVE', len(records), manifest['archive_sha256'], flush=True)


if __name__ == '__main__':
    main()
