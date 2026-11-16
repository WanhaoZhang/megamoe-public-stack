"""Freeze the delivery commit and index source, SO, proofs and negative data."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parent
head=subprocess.check_output(['git','-C',str(root/'stack'),'rev-parse','HEAD'],text=True).strip()
archive=root/'implementation'/('source-'+head[:12]+'.zip')
subprocess.run(['git','-C',str(root/'stack'),'archive','--format=zip','--output',str(archive),'HEAD'],check=True)
patch=root/'implementation/source-diff.patch'
with patch.open('wb') as handle:subprocess.run(['git','-C',str(root/'stack'),'diff','6e7dd58','HEAD'],stdout=handle,check=True)
files=[archive,patch,root/'implementation/worker.so',root/'implementation/worker-build.json',
 root/'raw-gates-v1.json',root/'raw-sparse-gates-v1.json',root/'raw-model-gates-v1.json',
 root/'diagnostics/raw-model-comparison-v1.json',root/'diagnostics/pilot-identity-audit.json',
 root/'schedule-pilot-v1.json',root/'schedule-pilot-v1.csv',root/'REPORT.md']
data=dict(branch='research/step-lifetime-raw-control',delivery_head=head,runtime_source_commit='1536db9',
 note='Build manifest plus per-run source snapshots are execution authority; later commits add the independent sparse oracle, frozen config and offline reporting. No new planner or stable gain.',
 files=[dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in files])
(root/'SOURCE-INDEX.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps(dict(head=head,files=len(files),archive_bytes=archive.stat().st_size)))
