"""Use the shared SINGLE-HOST monotonic clock, never independent device zeros."""
from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parent
evidence=root.parent/'causal-diagnostics-20261116/evidence'
result={}
for mode in ('fixed_home','anchor','b0','b1','b3'):
 groups={}
 for rank in range(8):
  for line in (evidence/('causal-v1-main-'+mode)/f'framework-events-rank{rank}.jsonl').read_text().splitlines():
   row=json.loads(line)
   if row['step']!=2 or row['name']!='operator.backward':continue
   key=tuple(row['invocation'])
   if rank in groups.setdefault(key,{}):raise RuntimeError('Duplicate original operator identity')
   groups[key][rank]=row['host_begin_ns']
 rows=[]
 for key,peers in sorted(groups.items()):
  if set(peers)!=set(range(8)):raise RuntimeError('Missing matching host invocation')
  first,last=min(peers,key=peers.get),max(peers,key=peers.get)
  rows.append(dict(invocation=key,host_operator_entry_skew_ms=(peers[last]-peers[first])/1e6,
                   first_rank=first,last_rank=last))
 result[mode]=dict(records=rows,median_ms=statistics.median(row['host_operator_entry_skew_ms'] for row in rows),
                   max_ms=max(row['host_operator_entry_skew_ms'] for row in rows))
data=dict(modes=result,
 scope='existing one-node8-process diagnostic, update2 only; same-host monotonic_ns wrapper-entry times. Not device launch/completion times or avoided critical path; do not add these skews.')
(root/'diagnostics/host-operator-entry-skew.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
print(json.dumps({name:{key:row[key] for key in ('median_ms','max_ms')} for name,row in result.items()},indent=2))
