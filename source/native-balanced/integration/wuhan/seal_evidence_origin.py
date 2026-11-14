"""Seal completed run provenance on its execution host; never backdate device gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket


def node_identity():
    identity = Path('/etc/machine-id').read_text().strip()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return dict(hostname=socket.gethostname(), machine_id_sha256=hashlib.sha256(identity.encode()).hexdigest(), boot_id=boot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    a = p.parse_args()
    node = node_identity()
    for run in a.runs:
        record = json.loads((run/'acceptance.json').read_text())
        assert record['eligible'] and int((run/'run.exit').read_text()) == 0
        files = [run/x for x in ('pre.json', 'post.json', 'launch.json', 'command.json', 'run.exit', 'acceptance.json')]
        files += sorted((run/'steps').glob('*.json*'))
        assert len(list((run/'steps').glob('rank*.jsonl'))) == 8
        hashes = {str(f.relative_to(run)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
        out = run/'origin.json'
        if out.exists():
            old = json.loads(out.read_text())
            assert old['node'] == node and old['file_hashes'] == hashes, 'sealed evidence changed'
            continue
        out.write_text(json.dumps(dict(node=node, run_path=str(run.resolve()), file_hashes=hashes,
            collected_utc=datetime.now(timezone.utc).isoformat(),
            scope='Collection on known execution host after completion; original gates are not rewritten.'), indent=2)+'\n')
    print('EVIDENCE_ORIGIN_SEALED', len(a.runs), node['hostname'])


if __name__ == '__main__':
    main()
