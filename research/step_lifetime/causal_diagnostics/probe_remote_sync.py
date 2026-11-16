"""Break original same-rank backward waits into their actual dependency phases."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
code='''
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
   start,size=meta['layout']['extra_ranges']['28'];indices=data['stage_indices'];pairs=data['stage_pairs']
   per_phase={}
   for phase in range(8):
    first=start+phase*96;values=[]
    for begin,end in sorted(pairs[(indices>=first)&(indices<first+96),:2].tolist()):
     if values and begin<=values[-1][1]:values[-1][1]=max(values[-1][1],end)
     else:values.append([begin,end])
    per_phase[str(phase)]=sum(end-begin for begin,end in values)
   rows.append(dict(rank=meta['rank'],invocation=meta['invocation'],sync_union_cycles=per_phase))
 per_rank={rank:{str(phase):sum(row['sync_union_cycles'][str(phase)] for row in rows if row['rank']==rank) for phase in range(8)} for rank in range(8)}
 result[mode]=dict(records=rows,per_rank=per_rank,
   median_over_rank_summed_cycles={str(phase):statistics.median(row[str(phase)] for row in per_rank.values()) for phase in range(8)})
print(json.dumps(dict(modes=result,phase_names={'0':'entry cross-rank','1':'completion mark, not a wait','2':'dx local drain','3':'dx remote barrier','4':'dp storage reuse barrier','5':'dp remote-write visibility','6':'exit local','7':'exit remote'},scope='last diagnostic update; all original backward files; same-rank GetSystemCycle intervals; no cross-rank origins joined; nested intervals not additive')))
'''
answer=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=code.encode(),capture_output=True,check=True)
data=json.loads(answer.stdout)
(root/'diagnostics/backward-sync-phases.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({name:row['median_over_rank_summed_cycles'] for name,row in data['modes'].items()},indent=2))
