"""Export pinned Wuhan sources and mandatory submodules without local build products."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def snapshot(project, output):
    base = project.parent / 'wuhan' / 'code'
    source = project / 'source'
    repos = [(name, source / name, 'HEAD') for name in
             ('ascend_deepep_hkq', 'FSDPTurbo', 'MindSpeed-LLM', 'MoonEP')]
    native = base / 'ascend_deepep_hkq'
    for name in ('shmem', 'catccos', 'catlass'):
        repos.append((f'ascend_deepep_hkq/3rdparty/{name}', native / '3rdparty' / name, 'HEAD'))
    moon_catlass = git(source / 'MoonEP', 'rev-parse', 'HEAD:npu_patch/3rdparty/catlass').decode().strip()
    repos.append(('MoonEP/npu_patch/3rdparty/catlass', native / '3rdparty/catlass', moon_catlass))
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    with tarfile.open(output, 'w:gz') as dest:
        for prefix, repo, ref in repos:
            sha = git(repo, 'rev-parse', ref).decode().strip()
            dirty = git(repo, 'status', '--porcelain', '--untracked-files=no', '--ignore-submodules=all').decode()
            if dirty:
                raise RuntimeError(f'Commit tracked source changes before exporting {repo}: {dirty}')
            payload = git(repo, 'archive', '--format=tar', sha)
            with tarfile.open(fileobj=io.BytesIO(payload)) as src:
                for item in src:
                    if prefix == 'MoonEP' and item.name.rstrip('/') == 'npu_patch/3rdparty/shmem':
                        continue  # Replace the empty gitlink directory with the shared-runtime link below.
                    item.name = prefix + '/' + item.name
                    if item.isfile():
                        content = src.extractfile(item)
                        dest.addfile(item, content)
                    else:
                        dest.addfile(item)
            records.append(dict(path=prefix, commit=sha, archive_sha256=hashlib.sha256(payload).hexdigest()))
        # MoonEP's script checks its own SHMEM source even when using the external install.
        link = tarfile.TarInfo('MoonEP/npu_patch/3rdparty/shmem')
        link.type, link.linkname = tarfile.SYMTYPE, '../../../ascend_deepep_hkq/3rdparty/shmem'
        dest.addfile(link)
    manifest = dict(repositories=records, archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(output.suffix + '.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    snapshot(args.project.resolve(), args.out.resolve())
