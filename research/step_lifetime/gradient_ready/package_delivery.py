"""Freeze this bounded local result; do not launch more hardware work."""
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
load=lambda path:json.loads(path.read_text())
if not load(ROOT/'study-v1.json').get('passed') or not load(ROOT/'diagnostics/pair-identity-audit.json')['passed']:
    raise SystemExit('Finish actual qualification and pair before delivery')
if not (ROOT/'REPORT.md').exists():raise SystemExit('Result report required')
upstream=ROOT.parent/'priority-cache-20261116'
for name in ('worker.so','worker-build.json'):
    shutil.copy2(upstream/'implementation'/name,ROOT/'implementation'/name)
if hashlib.sha256((ROOT/'implementation/worker.so').read_bytes()).hexdigest()!='e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed':
    raise RuntimeError('Qualified reused binary differs')
runtime_head=subprocess.check_output(['git','-C',str(STACK),'rev-parse','HEAD'],text=True).strip()
dest=STACK/'research/step_lifetime/gradient_ready';dest.mkdir(parents=True,exist_ok=True)
helpers=sorted(path for path in ROOT.iterdir() if path.suffix in ('.py','.md'))
for path in helpers:shutil.copy2(path,dest/path.name)
for name in ('summary.json','cpu-v1.json'):
    shutil.copy2(ROOT/name,dest/name)
subprocess.run(['git','-C',str(STACK),'add','research/step_lifetime/gradient_ready'],check=True)
staged=subprocess.check_output(['git','-C',str(STACK),'diff','--cached','--name-only'],text=True).splitlines()
if any(not name.startswith('research/step_lifetime/gradient_ready/') for name in staged):raise RuntimeError('Unexpected staged changes')
if staged:
    subprocess.run(['git','-C',str(STACK),'diff','--cached','--check'],check=True)
    subprocess.run(['git','-C',str(STACK),'commit','-m','Archive gradient readiness qualifications and the bounded update-time pair'],check=True)
head=subprocess.check_output(['git','-C',str(STACK),'rev-parse','HEAD'],text=True).strip()
base='68aacaeff575cb0081e7b6eef4163dcd770ad039'
with (ROOT/'implementation/source.diff').open('wb') as handle:
    subprocess.run(['git','-C',str(STACK),'diff','--binary',base,head],stdout=handle,check=True)
archive=subprocess.check_output(['git','-C',str(STACK),'archive',head])
with zipfile.ZipFile(ROOT/'implementation/source.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as output:
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source:
            if member.isfile():
                with source.extractfile(member) as handle:output.writestr('stack/'+member.name,handle.read())
    for path in helpers:output.write(path,path.name)
inventory={str(path.relative_to(ROOT)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
           for path in (ROOT/'evidence').rglob('*') if path.is_file() and path.suffix not in ('.tar','.gz')}
(ROOT/'evidence-inventory.json').write_text(json.dumps(dict(scope='Local per-run collected files; full small-model tensors and CPU source archive remain at remote receipt/task paths.',files=inventory),indent=2)+'\n',encoding='utf-8',newline='\n')
selected=[path for path in ROOT.iterdir() if path.suffix in ('.py','.md','.json','.csv')]
selected += [path for name in ('implementation','figures','diagnostics') for path in (ROOT/name).iterdir() if path.is_file()]
index=dict(branch='research/step-lifetime-grad-ready',head=head,qualified_runtime_head=runtime_head,base=base,
    persistent_goal='paused',local_control_complete=True,new_planner=False,new_native_build=False,
    reused_worker_source=str(upstream/'SOURCE-INDEX.json'),remote_project=__import__('remote').REMOTE,
    source_zip='implementation/source.zip',source_diff='implementation/source.diff',
    files={str(path.relative_to(ROOT)).replace('\\','/'):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
           for path in selected if path.name!='SOURCE-INDEX.json'})
(ROOT/'SOURCE-INDEX.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(dict(head=head,runtime_head=runtime_head,files=len(index['files']),local_raw_files=len(inventory)),indent=2))
