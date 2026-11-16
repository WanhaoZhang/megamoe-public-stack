"""Index delivery source separately from the immutable per-run snapshots."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
source=root/'stack';out=root/'implementation'
head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
archive=out/('source-'+head[:12]+'.zip')
subprocess.run(['git','-C',str(source),'archive','--format=zip','--output',str(archive),'HEAD'],check=True)
patch=out/'source-diff.patch'
with patch.open('wb') as handle:subprocess.run(['git','-C',str(source),'diff','6e7dd58','HEAD'],stdout=handle,check=True)
files=[archive,patch,out/'causal-timer-worker.so',out/'worker-build.json',root/'causal-v1.json',
    root/'diagnostics/main-all-plan-equality.json',root/'analysis/causal-v1/summary.json']
index=dict(base='6e7dd58acc11b292b2b49185f6a8f082ea9d50b0',delivery_head=head,
    branch='research/step-lifetime-causal',runtime_source_commit='abaf8e9',
    note='Delivery adds reports/offline scripts and frozen configs; each evidence source-identity and runtime snapshots remain execution authority; no diagnostic timing is a performance score',
    artifacts=[dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files])
(root/'SOURCE-INDEX.json').write_text(json.dumps(index,indent=2)+'\n')
print(json.dumps(dict(delivery_head=head,archive_bytes=archive.stat().st_size)))
