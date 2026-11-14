"""Record conservative, time-based exclusion after a proven foreign GPU workload."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
TASK=Path('/home/z00956592/megaflux/planner-executor-ep8-lab')
cid='e4ec55cc554dd8f7ed348dfc70c33efbfb079f0c2f23eec723be7ec85df8cf56'
info=json.loads(subprocess.run(['docker','inspect',cid],capture_output=True,text=True,check=True).stdout)[0]
begin=datetime.fromisoformat(info['State']['StartedAt'].replace('Z','+00:00')).timestamp()
hold=json.loads((TASK/'evidence/OWN-QUEUE-INFRASTRUCTURE-HOLD.json').read_text());end=hold['timestamp']
observations=TASK/'evidence/driver-process-observations.jsonl';proof=None
for number,line in enumerate(observations.read_text().splitlines(),1):
    data=json.loads(line)
    if begin<=data['timestamp']<=end and any(cid in x.get('cgroup','') for x in data['foreign_or_unidentified']):
        proof=dict(file='evidence/driver-process-observations.jsonl',line=number,sha256=hashlib.sha256(line.encode()).hexdigest(),timestamp=data['timestamp']);break
if proof is None:raise RuntimeError('no recorded foreign-GPU PID proof within this window')
rows=json.loads((TASK/'evidence/campaign-primary-existing.json').read_text())
affected=[row['tag'] for row in rows if row['start']<end and row['finish']>begin]
result=dict(schema=1,intervals=[dict(begin=begin,end=end,container=cid,container_name=info['Name'],proof=proof,
    boundary='conservative entire foreign-container activity epoch up to the own between-job scheduler hold',
    exclusion_rule='all task processes overlapping this window, independent of policy, measured speed or success',
    root_cause_caution='foreign occupancy is proven; causation of individual HCCL exceptions is not established')],
    excluded_tags=affected,original_rows_unchanged=True,all_raw_measurements_retained=True,
    no_other_workload_signalled=True,own_scheduler_held_between_jobs=True)
(TASK/'evidence/EXTERNAL-CONTENTION.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(affected=len(affected),tags=affected,begin=begin,end=end,foreign_container=info['Name'])),flush=True)
