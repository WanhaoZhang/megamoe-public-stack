"""Deploy a committed composite snapshot and reuse only matching native sources.

Run inside the isolated container. Never modifies an existing source tree.
The donor is an already built, frozen checkout; binary reuse is explicit and
all copied artifacts and checked translation-unit inputs are fingerprinted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--donor', type=Path, required=True)
    parser.add_argument('--allow-source-diff', action='store_true',
                        help='record source changes while reusing donor binaries')
    args = parser.parse_args()
    manifest = json.loads(Path(str(args.archive) + '.json').read_text())
    assert sha(args.archive) == manifest['archive_sha256'], 'archive hash mismatch'
    args.destination.mkdir(parents=True, exist_ok=False)
    subprocess.run(['tar', '-xzf', str(args.archive), '-C', str(args.destination)], check=True)
    native = args.destination / 'ascend_deepep_hkq'
    donor = args.donor / 'ascend_deepep_hkq'
    sources = {}
    for component in ('csrc', 'kernels', 'include', 'cmake', '3rdparty'):
        for path in sorted((native / component).rglob('*')):
            if path.is_file():
                relative = path.relative_to(native)
                original = donor / relative
                if not (original.is_file() and sha(path) == sha(original)):
                    if not args.allow_source_diff:
                        raise AssertionError(str(relative))
                    sources[str(relative)] = dict(candidate=sha(path), donor=sha(original) if original.is_file() else None)
                    continue
                sources[str(relative)] = sha(path)
    for component in ('CMakeLists.txt', 'setup.py', 'pyproject.toml'):
        if sha(native / component) != sha(donor / component):
            if not args.allow_source_diff:
                raise AssertionError(component)
            sources[component] = dict(candidate=sha(native/component), donor=sha(donor/component))
            continue
        sources[component] = sha(native / component)
    artifacts = {}
    paths = list((donor / 'ascend_deepep').glob('*.so'))
    paths += list((donor / 'ascend_deepep' / 'lib').glob('*.so'))
    paths += list((donor / 'worker-variants' / 's9-ported').glob('*'))
    assert paths and (donor / 'worker-variants/s9-ported/libascend_deepep_megaflux_early_bf16.so').is_file()
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(donor)
        target = native / relative
        assert not target.exists(), f'snapshot unexpectedly contains a binary: {relative}'
        target.parent.mkdir(parents=True, exist_ok=True)
        # Follow symlinks into the same immutable library bytes, avoiding links
        # back into an overlay tree whose future identity could change.
        shutil.copy2(path, target)
        assert sha(path) == sha(target)
        artifacts[str(relative)] = dict(sha256=sha(target), donor=str(path.resolve()))
    record = dict(snapshot=manifest, donor=str(args.donor), checked_sources=sources,
                  reused_artifacts=artifacts, destination=str(args.destination))
    (args.destination / 'deployment.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(dict(destination=str(args.destination), checked_sources=len(sources),
                         artifacts=len(artifacts), archive_sha256=manifest['archive_sha256'])))


if __name__ == '__main__':
    main()
