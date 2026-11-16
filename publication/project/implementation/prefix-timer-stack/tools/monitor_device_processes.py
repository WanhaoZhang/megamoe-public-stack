"""Read-only driver ownership observation, independent of training/timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
TASK=Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
parser=argparse.ArgumentParser()
parser.add_argument('--host-tag',default='')
parser.add_argument('--quiet',action='store_true')
options=parser.parse_args()
if options.host_tag and not options.host_tag.replace('-','').isalnum():raise ValueError('invalid host tag')
suffix='-'+options.host_tag if options.host_tag else ''
hostname=subprocess.run(['hostname'],capture_output=True,text=True,check=True).stdout.strip()
cid=subprocess.run(['docker','inspect','--format','{{.Id}}','planner-executor-ep8-lab'],capture_output=True,text=True,check=True).stdout.strip()
def describe(pid):
    root=Path('/proc')/str(pid)
    try:
        args=root.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        group=root.joinpath('cgroup').read_text()
        return dict(pid=pid,script_paths=[x for x in args if x.endswith('.py')],
                    executable=os.readlink(root/'exe'),task_container=cid in group,
                    task_script=any(str(TASK) in x for x in args),cgroup=group.strip())
    except (FileNotFoundError,PermissionError,ProcessLookupError):return dict(pid=pid,process_disappeared=True)
path=TASK/f'evidence/driver-process-observations{suffix}.jsonl'
while not (TASK/f'evidence/STOP_DEVICE_MONITOR{suffix}').exists():
    result=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=25)
    processes=[]
    for line in result.stdout.splitlines():
        cells=[x.strip() for x in line.split('|')[1:-1]]
        if len(cells)>=4 and cells[0].isdigit() and cells[1].isdigit() and '/' not in cells[3]:
            item=dict(rank=int(cells[0]),name=cells[2],memory=cells[3],**describe(int(cells[1])))
            processes.append(item)
    foreign=[x for x in processes if not x.get('task_container') and not x.get('task_script') and not x.get('process_disappeared')]
    row=dict(timestamp=time.time(),host=hostname,host_tag=options.host_tag,task_container_id=cid,processes=processes,foreign_or_unidentified=foreign,
             no_process_cards=result.stdout.count('No running processes found'),exit=result.returncode)
    if foreign:row['raw_driver_info']=result.stdout
    with path.open('a') as f:f.write(json.dumps(row)+'\n')
    if foreign and not options.quiet:print(json.dumps(dict(timestamp=row['timestamp'],foreign_count=len(foreign),pids=[x['pid'] for x in foreign])),flush=True)
    time.sleep(5)
print('TASK_DEVICE_PROCESS_MONITOR_STOPPED',flush=True)
