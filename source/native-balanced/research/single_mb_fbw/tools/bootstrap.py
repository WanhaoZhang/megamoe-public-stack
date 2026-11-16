"""Restore sealed experimental sources without touching the source project."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BRANCH = 'research/single-mb-fbw-planner'


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    original = args.baseline.resolve()
    frozen = json.loads((original / 'evidence/STEP-TIME-FREEZE.json').read_text())
    mismatch = []
    for name, expected in frozen['source'].items():
        path = original / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            mismatch.append(name)
    (ROOT / 'evidence').mkdir(exist_ok=True)
    receipt = {'baseline': str(original), 'checked': len(frozen['source']), 'mismatches': mismatch}
    (ROOT / 'evidence/source-verification.json').write_text(json.dumps(receipt, indent=2))
    if mismatch:
        raise RuntimeError(f'Frozen source mismatch: {mismatch[:8]}')
    for name in ('native-balanced', 'MindSpeed-LLM', 'FSDPTurbo'):
        destination = ROOT / 'source' / name
        if destination.exists():
            raise RuntimeError(f'Refusing to overwrite {destination}')
        source = original / 'source' / name
        base = source if (source / '.git').exists() else original / 'artifacts/git-bases' / name
        subprocess.run(['git', '-c', 'core.autocrlf=false', 'clone', '--no-hardlinks', str(base), str(destination)], check=True)
        git(destination, 'config', 'core.autocrlf', 'false')
        git(destination, 'switch', '-c', BRANCH)
        prefix = f'source/{name}/'
        for relative in frozen['source']:
            if relative.startswith(prefix):
                target = destination / relative[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original / relative, target)
        receipt[name] = {'upstream': git(destination, 'rev-parse', 'HEAD'), 'branch': BRANCH}
        # Stage only files belonging to the sealed manifest, including new files.
        paths = [r[len(prefix):] for r in frozen['source'] if r.startswith(prefix)]
        for offset in range(0, len(paths), 60):
            git(destination, 'add', '--', *paths[offset:offset + 60])
        if git(destination, 'diff', '--cached', '--name-only'):
            git(destination, 'commit', '-m', 'research: freeze verified decoupled EP8 training baseline')
        receipt[name]['snapshot'] = git(destination, 'rev-parse', 'HEAD')
    (ROOT / 'evidence/source-verification.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
