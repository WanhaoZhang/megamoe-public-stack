"""Actual four-stage routed backward source/re-pull/retirement intervals."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
program='''
from pathlib import Path
import json,statistics
import numpy as np
root=Path('/home/z00956592/megaflux/step-lifetime-causal-20261116/stack/evidence')
result={}
for mode in ('fixed_home','anchor','b0','b1','b3'):
 rows=[]
 for path in sorted((root/('causal-v1-main-'+mode)/'timer').glob('*backward.npz')):
  with np.load(path,allow_pickle=False) as data:
   meta=json.loads(str(data['metadata']))
   if meta['invocation'][0]!=2:continue
   raw=data['original_x_redispatch_backward'];phases={}
   for phase in range(4):
    values=[]
    for begin,end in sorted(raw[phase].tolist()):
     if begin<=0 or end<begin:continue
     if values and begin<=values[-1][1]:values[-1][1]=max(values[-1][1],end)
     else:values.append([begin,end])
    phases[str(phase)]=sum(end-begin for begin,end in values)
   rows.append(dict(rank=meta['rank'],invocation=meta['invocation'],phase_union_cycles=phases))
 per_rank={rank:{str(phase):sum(row['phase_union_cycles'][str(phase)] for row in rows if row['rank']==rank) for phase in range(4)} for rank in range(8)}
 result[mode]=dict(records=rows,per_rank=per_rank,median_over_rank={str(phase):statistics.median(row[str(phase)] for row in per_rank.values()) for phase in range(4)})
print(json.dumps(dict(modes=result,phase_names={'0':'local gather dY/X/router scores','1':'source publication, cross-rank/table rendezvous','2':'AIV1 remote X/probability pulls, excludes restore','3':'source-read retirement rendezvous'},scope='ALL last-update original V3 transport Timer arrays; per-rank union per phase; do not add overlapping phases or ranks; uncalibrated cycles')))
'''
answer=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=program.encode(),capture_output=True,check=True)
data=json.loads(answer.stdout)
(root/'diagnostics/backward-redispatch-phases.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({name:row['median_over_rank'] for name,row in data['modes'].items()},indent=2))
