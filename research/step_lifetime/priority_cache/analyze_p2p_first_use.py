"""Check whether long HOST submission calls coincide with new peer patterns."""
from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parent
base=root.parent/'causal-diagnostics-20261116/evidence'
results={}
for mode in ('b0','b1','b3'):
 submissions=[]
 for rank in range(8):
  peers_seen=set();patterns_seen=set()
  rows=[json.loads(line) for line in (base/('causal-v1-main-'+mode)/f'lifetime-events-rank{rank}.jsonl').read_text().splitlines()]
  for row in sorted(rows,key=lambda x:x['host_begin_ns']):
   if row['name']!='p2p_submit_enqueue':continue
   peers=tuple(sorted({op['peer'] for op in row['operations']}))
   pattern=tuple((op['peer'],op['kind'],op['bytes']) for op in row['operations'])
   entry=dict(rank=rank,invocation=row['invocation'],packet=row['packet'],peers=peers,
     duration_ms=(row['host_end_ns']-row['host_begin_ns'])/1e6,
     new_peer=any(peer not in peers_seen for peer in peers),new_pattern=pattern not in patterns_seen)
   submissions.append(entry);peers_seen.update(peers);patterns_seen.add(pattern)
 long=[row for row in submissions if row['duration_ms']>10]
 results[mode]=dict(total_submissions=len(submissions),long_host_submissions=long,
     long_all_new_peer=bool(long) and all(row['new_peer'] for row in long),
     long_all_new_pattern=bool(long) and all(row['new_pattern'] for row in long),
     maximum_reused_pattern_host_ms=max((row['duration_ms'] for row in submissions if not row['new_pattern']),default=0))
data=dict(modes=results,scope='all3 existing diagnostic updates; shared-host CPU call durations, not device transfer times; first observed peer/pattern does not prove HCCL internal initialization cause')
(root/'diagnostics/p2p-first-use-observation.json').write_text(json.dumps(data,indent=2)+'\n',newline='\n')
for mode,row in results.items():
 print(mode,'submissions',row['total_submissions'],'long',len(row['long_host_submissions']),
       'all_new_peer',row['long_all_new_peer'],'all_new_pattern',row['long_all_new_pattern'],
       'max_reused_ms',row['maximum_reused_pattern_host_ms'])
 print(row['long_host_submissions'][:5])
