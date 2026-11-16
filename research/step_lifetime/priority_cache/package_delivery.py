"""Freeze a reviewable source delivery; raw data and large local deps stay outside git."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
STACK = ROOT/'stack'
load=lambda name: json.loads((ROOT/name).read_text())
if not load('confirmation-v1.json').get('passed') or not (ROOT/'REPORT.md').exists():
    raise SystemExit('Finish and report the actual bounded schedule before delivery.')
destination = STACK/'research/step_lifetime/priority_cache'
destination.mkdir(parents=True,exist_ok=True)
small = sorted([path for path in ROOT.iterdir() if path.suffix in ('.py','.md')])
for path in small:
    shutil.copy2(path,destination/path.name)
for name in ('summary.json','pilot-v1.json','math-v1.json','model-v1.json'):
    shutil.copy2(ROOT/name,destination/name)
for name in ('build-intervention-audit.json','model-comparison.json'):
    shutil.copy2(ROOT/'diagnostics'/name,destination/name)
# The confirmation has thousands of audit fields; the full authoritative JSON
# stays in the project and is fingerprinted below instead of copied redundantly.
subprocess.run(['git','-C',str(STACK),'add','configs/lifetime-runtime-freeze.json','research/step_lifetime/priority_cache'],check=True)
staged=subprocess.check_output(['git','-C',str(STACK),'diff','--cached','--name-only'],text=True).splitlines()
if any(not (name=='configs/lifetime-runtime-freeze.json' or name.startswith('research/step_lifetime/priority_cache/')) for name in staged):
    raise SystemExit('Unexpected staged changes: '+str(staged))
if staged:
    subprocess.run(['git','-C',str(STACK),'diff','--cached','--check'],check=True)
    subprocess.run(['git','-C',str(STACK),'commit','-m','Record priority-key intervention, repeated timing and matched controls'],check=True)
head=subprocess.check_output(['git','-C',str(STACK),'rev-parse','HEAD'],text=True).strip()
base='693f7fc4f9e7617c79ffbf69cc27268609bd99e2'
with (ROOT/'implementation/source.diff').open('wb') as handle:
    subprocess.run(['git','-C',str(STACK),'diff','--binary',base,head],stdout=handle,check=True)
archive=subprocess.check_output(['git','-C',str(STACK),'archive',head])
source_zip=ROOT/'implementation/source.zip'
with zipfile.ZipFile(source_zip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isfile():
                with source.extractfile(member) as handle:
                    output.writestr('stack/'+member.name,handle.read())
    for path in small:
        output.write(path,path.name)
    for name in ('summary.json','pilot-v1.json','math-v1.json','model-v1.json'):
        output.write(ROOT/name,name)
selected = [path for path in ROOT.iterdir() if path.suffix in ('.py','.md','.json','.csv')]
selected += [path for path in (ROOT/'implementation').iterdir() if path.is_file()]
selected += [path for path in (ROOT/'figures').iterdir() if path.is_file()]
selected += [path for path in (ROOT/'diagnostics').iterdir() if path.is_file() and path.suffix=='.json']
index = dict(branch='research/step-lifetime-priority-cache',head=head,base=base,new_planner=False,
             source_zip='implementation/source.zip',compiled_worker='implementation/worker.so',
             build_compiled_source_audit='diagnostics/build-intervention-audit.json',
             raw_data_scope='Per-run evidence metadata is local; qualification full tensors remain at receipt.large_tensors_remote. No new full plan dumps in timed runs.',
             remote_project='/home/z00956592/megaflux/step-lifetime-priority-cache-20261116',
             files={str(path.relative_to(ROOT)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    for path in selected if path.name!='SOURCE-INDEX.json'})
(ROOT/'SOURCE-INDEX.json').write_text(json.dumps(index,indent=2)+'\n',newline='\n')
print(json.dumps(dict(head=head,files=len(index['files']),source_zip_sha=index['files']['implementation/source.zip']['sha256']),indent=2))
