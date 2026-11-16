"""Distinguish expert MaxVio from actual fixed-owner rank pressure."""
from pathlib import Path
import json
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'causal-diagnostics-20261116/evidence/causal-v1-main-fixed_home/timer'
rows=[]
for path in sorted(SOURCE.glob('rank0-*-forward.npz')):
    with np.load(path,allow_pickle=False) as data:
        meta=json.loads(str(data['metadata']))
        if meta['invocation'][0]!=2 or meta['recompute']:continue
        quota=data['source_quota'].astype(np.int64)
        if quota.shape!=(8,256,8):raise RuntimeError('Unexpected quota identity '+str(quota.shape))
        logical=quota.sum(axis=(0,2))
        home=logical.reshape(8,32).sum(axis=1)
        execute=quota.sum(axis=(0,1))
        if not np.array_equal(home,execute):raise RuntimeError('Fixed-home quota changed owners')
        rows.append(dict(invocation=meta['invocation'],assignment_total=int(logical.sum()),
                         expert_maxvio=float(logical.max()/logical.mean()-1),
                         home_rank_maxvio=float(home.max()/home.mean()-1),home_rank_assignments=home.tolist(),
                         perfect_linear_compute_reduction_pct=float(100*(1-home.mean()/home.max()))))
if len(rows)!=16:raise RuntimeError('Need all16 actual original forward calls of update2')
output=dict(source=str(SOURCE),scope='Old same MaxVio10 rich diagnostic, all16 original forward calls of update2; rank0 plan body contains complete EP quotas',
            rows=rows,median_expert_maxvio=statistics.median(row['expert_maxvio'] for row in rows),
            median_home_rank_maxvio=statistics.median(row['home_rank_maxvio'] for row in rows),
            max_home_rank_maxvio=max(row['home_rank_maxvio'] for row in rows),
            interpretation='An expert MaxVio target does not specify rank imbalance. Perfect linear row balancing is only a compute-load illustration, not an end-to-end speed bound: tiles, transport, barriers and non-MoE work differ.')
(ROOT/'diagnostics/actual-home-rank-pressure.json').write_text(json.dumps(output,indent=2)+'\n',newline='\n')
print(json.dumps({key:value for key,value in output.items() if key!='rows'},indent=2))
