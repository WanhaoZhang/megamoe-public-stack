"""Refresh only verified byte-identical task files; never alter their contents."""
from pathlib import Path
import hashlib
import json
import subprocess

ROOT=Path(__file__).resolve().parent
STACK=ROOT/'stack'
if subprocess.check_output(['git','-C',str(STACK),'branch','--show-current'],text=True).strip()!='research/step-lifetime-grad-ready':
    raise SystemExit('Unexpected independent checkout')
if subprocess.check_output(['git','-C',str(STACK),'diff','--cached','--name-only'],text=True).strip():
    raise SystemExit('Unexpected staged changes')
entries=subprocess.check_output(['git','-C',str(STACK),'ls-tree','-rz','HEAD'])
head={}
for entry in entries.split(b'\0'):
    if not entry:continue
    fields,name=entry.split(b'\t',1)
    head[name.decode()]=fields.split()[2].decode()
selected=json.loads((ROOT/'diagnostics/checkout-byte-alignment.json').read_text())['files']
selected=sorted(set(selected+['tools/write_report.py','tools/write_planner_report.py']))
for name in selected:
    data=(STACK/name).read_bytes()
    digest=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
    if digest!=head[name]:raise RuntimeError('Actual content differs from HEAD; do not stage '+name)
subprocess.run(['git','-C',str(STACK),'add','--',*selected],check=True)
changed=subprocess.check_output(['git','-C',str(STACK),'diff','--cached','--name-only'],text=True).splitlines()
result=dict(files_checked=len(selected),contents_changed=False,unexpected_staged=changed)
(ROOT/'diagnostics/index-refresh.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
if changed:raise RuntimeError('Inspect unexpected index changes '+str(changed))
print(json.dumps(result))
