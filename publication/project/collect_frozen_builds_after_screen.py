"""Copy immutable frozen SOs after the v1 timing screen, preserving alias hashes."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import tarfile
import time

root=Path(__file__).resolve().parent
deadline=time.monotonic()+3*3600
while True:
    jobs=json.loads((root/'screening-results.json').read_text())['jobs']
    if any(job['status']=='failed' for job in jobs):raise RuntimeError('Screen stopped; no build export started')
    if len(jobs)==18 and all(job['status']=='completed' for job in jobs):break
    if time.monotonic()>deadline:raise RuntimeError('Screen incomplete; no build export started')
    time.sleep(15)
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115'
subprocess.run(['scp',str(root/'collect_frozen_builds_remote.py'),f'pod6-1-root:{remote}/collect_frozen_builds.py'],check=True)
run=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
    '/usr/local/python3.12.13/bin/python3',remote+'/collect_frozen_builds.py'],check=True,capture_output=True,text=True)
receipt=json.loads(run.stdout.splitlines()[-1])
destination=root/'implementation/frozen-build-export.tar.gz'
subprocess.run(['scp',f'pod6-1-root:{receipt["archive"]}',str(destination)],check=True)
assert hashlib.sha256(destination.read_bytes()).hexdigest()==receipt['sha256']
with tarfile.open(destination,'r:gz') as archive:
    proof=json.load(archive.extractfile('BUILD-EXPORT.json'))
    for row in proof['aliases'].values():
        assert hashlib.sha256(archive.extractfile(row['blob']).read()).hexdigest()==row['sha256']
    for name,sha in proof['metadata'].items():
        assert hashlib.sha256(archive.extractfile(name).read()).hexdigest()==sha
receipt.update(local_archive=str(destination),all_alias_and_metadata_hashes_verified=True,
    contains_original_performance_and_separate_diagnostic_worker=True)
(root/'implementation/frozen-build-export.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
