"""Read only process names/script basenames, never other workloads' arguments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser();ap.add_argument('hosts',nargs='+');options=ap.parse_args()
script=r'''import collections,json,pathlib,subprocess
out=[]
ids=subprocess.run(['docker','ps','--format','{{.ID}} {{.Names}}'],capture_output=True,text=True,check=True).stdout.splitlines()
for item in ids:
 identity,name=item.split(None,1)
 top=subprocess.run(['docker','top',identity,'-eo','pid,comm'],capture_output=True,text=True)
 commands=[line.split(None,1)[1].strip() for line in top.stdout.splitlines()[1:] if len(line.split(None,1))==2]
 out.append(dict(name=name,query_ok=top.returncode==0,command_counts=dict(collections.Counter(commands)),
                 nontrivial_commands=[x for x in commands if x not in ('sleep','bash','sh')]))
host=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  comm=(p/'comm').read_text().strip()
  if not (comm.startswith('python') or comm in ('pt_elastic','torchrun')):continue
  argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  names=[pathlib.Path(a).name for a in argv if a.endswith('.py')]
  if names or comm in ('pt_elastic','torchrun'):host.append(dict(pid=int(p.name),comm=comm,script_basenames=names))
 except (OSError,ValueError):continue
print(json.dumps(dict(containers=out,host_python_or_torchrun=host)))
'''
destination=root/'evidence/candidate-host-inventory';destination.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def audit(host):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3','-'],input=script,capture_output=True,text=True,timeout=35)
    record=dict(host=host,exit=result.returncode,output=result.stdout,stderr=result.stderr)
    if result.returncode==0:
        data=json.loads(result.stdout);record.update(data)
        record['quiet_process_inventory']=not data['host_python_or_torchrun'] and all(c['query_ok'] and not c['nontrivial_commands'] for c in data['containers'])
    (destination/f'{stamp}-{host}-processes.json').write_text(json.dumps(record,indent=2)+'\n')
    return {k:v for k,v in record.items() if k not in ('output','stderr')}
with ThreadPoolExecutor(max_workers=min(3,len(options.hosts))) as executor:print(json.dumps(list(executor.map(audit,options.hosts))),flush=True)
