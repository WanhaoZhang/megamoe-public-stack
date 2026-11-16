"""Original fixed/accumulated ready waits with their actual lane ownership."""
from pathlib import Path
import json,subprocess
root=Path(__file__).resolve().parent
program='''
from pathlib import Path
import json,statistics
import numpy as np
root=Path('/home/z00956592/megaflux/step-lifetime-causal-20261116/stack/evidence')
fixed={'KERNEL_core_max':0,'AIV_gather':1,'AIV_xsync_gather':2,'AIV_restore_table':3,
 'AIC_spin_dZ':5,'AIC_spin_Hp':7,'AIV_xsync_unpermute':8,'AIV_unpermute':9,
 'AIV_xsync_dp':10,'AIV_dp_reduce':11,'AIV_xsync_topk':12,'AIV_topk_unpermute':13,
 'AIV_table_reset':14,'AIV_recompute_pass':16,'AIV_accumulated_P1_wait':17,
 'AIC_P2_tail':18,'AIC_init':19,'AIV_init':20}
result={}
for mode in ('fixed_home','anchor','b0','b1','b3'):
 records=[]
 for path in sorted((root/('causal-v1-main-'+mode)/'timer').glob('*backward.npz')):
  with np.load(path,allow_pickle=False) as data:
   meta=json.loads(str(data['metadata']))
   if meta['invocation'][0]!=2:continue
   raw=data['original_backward'][:96*2592].reshape(96,2592)
   values={}
   for name,index in fixed.items():
    selected=raw[::3] if name.startswith('AIC') else raw[np.arange(96)%3!=0] if name.startswith('AIV') else raw
    pairs=selected[:,2*index:2*index+2];valid=(pairs[:,0]>0)&(pairs[:,1]>=pairs[:,0])
    values[name]=int(np.where(valid,pairs[:,1]-pairs[:,0],0).max())
   for name,begin,lane in (('AIC_dZ_per_expert_wait',21,0),('AIV_dX_per_expert_wait',85,1),('AIC_dO_reuse_per_expert_wait',149,0)):
    pairs=raw[lane::3,2*begin:2*(begin+64)].reshape(32,64,2)
    valid=(pairs[:,:,0]>0)&(pairs[:,:,1]>=pairs[:,:,0])
    values[name]=int(np.where(valid,pairs[:,:,1]-pairs[:,:,0],0).sum(axis=1).max())
   records.append(dict(rank=meta['rank'],invocation=meta['invocation'],cycles=values))
 names=list(records[0]['cycles'])
 per_rank={rank:{name:sum(row['cycles'][name] for row in records if row['rank']==rank) for name in names} for rank in range(8)}
 result[mode]=dict(records=records,per_rank=per_rank,median_over_rank={name:statistics.median(row[name] for row in per_rank.values()) for name in names})
print(json.dumps(dict(modes=result,scope='all original last-update backward files; per-core duration or sum of accumulated per-expert waits, then max appropriate lane; no cross-rank origins; nested/noncontiguous waits are not added to critical path')))
'''
answer=subprocess.run(['ssh','pod6-1-root','docker','exec','-i','step-lifetime-causal-20261116',
    '/usr/local/python3.12.13/bin/python3','-'],input=program.encode(),capture_output=True,check=True)
data=json.loads(answer.stdout)
(root/'diagnostics/backward-ready-phases.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({name:row['median_over_rank'] for name,row in data['modes'].items()},indent=2))
