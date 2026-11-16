"""Freeze diagnostic source, actual SO references and raw local evidence inventory."""
from pathlib import Path
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import zipfile

ROOT=Path(__file__).resolve().parent
STACK=ROOT/'stack'
for name in ('diagnosis-v4.json','diagnosis-v5.json'):
    if not json.loads((ROOT/name).read_text()).get('passed'):raise SystemExit('Pending diagnostic '+name)
if not (ROOT/'REPORT.md').exists():raise SystemExit('Report required')
proof=json.loads((ROOT/'diagnostics/main-identity-audit.json').read_text())
if not proof['passed']:raise SystemExit('Main identity audit required')
upstream=ROOT.parent/'priority-cache-20261116'
for name in ('worker.so','reference-worker.so','worker-build.json','baseline-worker-build.json'):
    shutil.copy2(upstream/'implementation'/name,ROOT/'implementation'/name)
copied={name:hashlib.sha256((ROOT/'implementation'/name).read_bytes()).hexdigest() for name in ('worker.so','reference-worker.so')}
if copied!={'worker.so':'e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed',
             'reference-worker.so':'72653abf3c22fb951624dd5dc420c488d2511d1c22e5e605a3621cb4210502bf'}:
    raise RuntimeError('Actual SO copy differs from qualified libraries')
helpers=sorted(path for path in ROOT.iterdir() if path.suffix in ('.py','.md'))
dest=STACK/'research/step_lifetime/host_boundary';dest.mkdir(parents=True,exist_ok=True)
for path in helpers:shutil.copy2(path,dest/path.name)
for name in ('summary.json','setup-provenance.json'):
    shutil.copy2(ROOT/name,dest/name)
subprocess.run(['git','-C',str(STACK),'add','research/step_lifetime/host_boundary'],check=True)
staged=subprocess.check_output(['git','-C',str(STACK),'diff','--cached','--name-only'],text=True).splitlines()
if any(not name.startswith('research/step_lifetime/host_boundary/') for name in staged):raise RuntimeError('Unexpected staged changes')
if staged:
    subprocess.run(['git','-C',str(STACK),'diff','--cached','--check'],check=True)
    subprocess.run(['git','-C',str(STACK),'commit','-m','Archive qualified host boundaries, startup failures and matched runtime diagnosis'],check=True)
head=subprocess.check_output(['git','-C',str(STACK),'rev-parse','HEAD'],text=True).strip()
base=json.loads((ROOT/'setup-provenance.json').read_text())['base_head']
with (ROOT/'implementation/source.diff').open('wb') as handle:
    subprocess.run(['git','-C',str(STACK),'diff','--binary',base,head],stdout=handle,check=True)
archive=subprocess.check_output(['git','-C',str(STACK),'archive',head])
with zipfile.ZipFile(ROOT/'implementation/source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isfile():
                with source.extractfile(member) as handle:output.writestr('stack/'+member.name,handle.read())
    for path in helpers:output.write(path,path.name)
inventory={}
for path in sorted((ROOT/'evidence').rglob('*')):
    if path.is_file() and path.suffix not in ('.tar','.gz'):
        inventory[str(path.relative_to(ROOT)).replace('\\','/')]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
(ROOT/'evidence-inventory.json').write_text(json.dumps(dict(scope='All local collected per-run files; full small-model PT remain remote, receipts record paths. Tar duplicates excluded.',files=inventory),indent=2)+'\n',newline='\n')
selected=[path for path in ROOT.iterdir() if path.suffix in ('.py','.md','.json','.csv')]
selected += [path for directory in ('implementation','figures','diagnostics') for path in (ROOT/directory).iterdir() if path.is_file() and path.suffix not in ('.log',)]
index=dict(branch='research/step-lifetime-host-boundary',head=head,base=base,diagnostic_only=True,new_planner=False,
           compiled_native_changed=False,compiled_worker_source=str(upstream/'SOURCE-INDEX.json'),
           remote_project='/home/z00956592/megaflux/step-lifetime-host-boundary-20261116',
           source_zip='implementation/source.zip',source_diff='implementation/source.diff',
           raw_local_inventory='evidence-inventory.json',
           files={str(path.relative_to(ROOT)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in selected if path.name!='SOURCE-INDEX.json'})
(ROOT/'SOURCE-INDEX.json').write_text(json.dumps(index,indent=2)+'\n',newline='\n')
print(json.dumps(dict(head=head,files=len(index['files']),local_raw_files=len(inventory)),indent=2))
