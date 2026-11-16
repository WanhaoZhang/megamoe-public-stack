"""Original backward dynamic slots, without summing parallel cores or stages."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
program='''
from pathlib import Path
import json,statistics
import numpy as np
root=Path('/home/z00956592/megaflux/step-lifetime-causal-20261116/stack/evidence')
# AscendTimerV2.hpp and parse_backward_timer.py: backward dynamic base11.
stages={'recompute':12,'dO_wait':13,'GMM2grad':14,'GMM1grad':16,'dW1':18,'dW2':19}
result={}
for mode in ('fixed_home','anchor','b0','b1','b3'):
 rows=[]
 for path in sorted((root/('causal-v1-main-'+mode)/'timer').glob('*backward.npz')):
  with np.load(path,allow_pickle=False) as data:
   meta=json.loads(str(data['metadata']))
   if meta['invocation'][0]!=2:continue
   raw=data['original_backward'][:96*2592].reshape(96,2592)[::3]
   values={}
   for name,index in stages.items():
    pairs=raw[:,2*(10+index*64):2*(10+(index+1)*64)].reshape(32,64,2)
    valid=(pairs[:,:,0]>0)&(pairs[:,:,1]>=pairs[:,:,0])
    durations=np.where(valid,pairs[:,:,1]-pairs[:,:,0],0)
    # First sum all experts on EACH core, then take the slowest AIC lane.
    values[name]=int(durations.sum(axis=1).max())
   rows.append(dict(rank=meta['rank'],invocation=meta['invocation'],aic_core_max_expert_sum_cycles=values))
 per_rank={rank:{name:sum(row['aic_core_max_expert_sum_cycles'][name] for row in rows if row['rank']==rank) for name in stages} for rank in range(8)}
 result[mode]=dict(records=rows,per_rank=per_rank,median_over_rank={name:statistics.median(row[name] for row in per_rank.values()) for name in stages})
print(json.dumps(dict(modes=result,scope='all original last-update backward records; each core sums experts, then max AIC core; do not add overlapped stages, kernels or ranks; cycles not converted to ms')))
'''
answer=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=program.encode(),capture_output=True,check=True)
data=json.loads(answer.stdout)
(root/'diagnostics/backward-gemm-phases.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({name:row['median_over_rank'] for name,row in data['modes'].items()},indent=2))
