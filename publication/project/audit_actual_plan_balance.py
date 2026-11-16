"""Inspect real captured quotas: input home skew is not execution skew."""
from pathlib import Path
import csv
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
records=[];summaries=[]
for label,tag in [('Native','timer-v2-matched-legacy-primary'),('Easy-home','timer-v2-matched-primary')]:
    directory=root/'evidence'/tag/'timer'
    for path in sorted(directory.glob('rank0-*forward.npz')):
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(data['metadata'].item())
            if meta['invocation'][0]!=2 or meta['recompute']:continue
            counts=data['logical_counts'];quota=data['source_quota'];copies=data['copies']
            if not np.array_equal(quota.sum((0,2)),counts):raise RuntimeError('Expert assignments not conserved')
            execution=quota.sum((0,1));home=counts.reshape(8,-1).sum(1)
            if int(execution.sum())!=8*16384*8:raise RuntimeError('Unexpected native-wave workload')
            active=[]
            for rank in range(8):
                other=directory/path.name.replace('rank0-',f'rank{rank}-',1)
                with np.load(other,allow_pickle=False) as rank_data:
                    rank_meta=json.loads(rank_data['metadata'].item())
                    if (rank_meta['invocation'],rank_meta['recompute'])!=(meta['invocation'],False):
                        raise RuntimeError('Captured rank invocations disagree')
                    if not np.array_equal(rank_data['source_quota'],quota) or not np.array_equal(rank_data['copies'],copies):
                        raise RuntimeError('Captured global plan differs across ranks')
                    if int(rank_data['execution_counts'].sum())!=int(execution[rank]):
                        raise RuntimeError('Actual execution counts disagree with quota')
                    active.append(int(np.count_nonzero(rank_data['execution_counts'])))
            active_replicas=sum(int(quota[:,int(expert),rank].sum())>0
                for rank in range(8) for expert in copies[rank] if expert>=0)
            records.append(dict(policy=label,tag=tag,file=path.name,invocation=json.dumps(meta['invocation']),
                home_experts_per_rank=len(counts)//8,
                input_home_rank_maxvio=float(home.max()/home.mean()-1),
                actual_execution_rank_maxvio=float(execution.max()/execution.mean()-1),
                active_replica_executions_global=active_replicas,
                active_replica_executions_per_rank=active_replicas/8,
                active_execution_slots_per_rank_mean=statistics.mean(active),
                home_rank_counts=json.dumps(home.tolist()),execution_rank_counts=json.dumps(execution.tolist()),
                all_rank_plans_and_execution_counts_checked=True))
    current=[row for row in records if row['policy']==label]
    if len(current)!=16:raise RuntimeError('Expected16 original Native waves/update at this main shape')
    summaries.append(dict(policy=label,original_waves=16,
        mean_input_home_rank_maxvio=statistics.mean(row['input_home_rank_maxvio'] for row in current),
        mean_actual_execution_rank_maxvio=statistics.mean(row['actual_execution_rank_maxvio'] for row in current),
        max_actual_execution_rank_maxvio=max(row['actual_execution_rank_maxvio'] for row in current),
        mean_active_replica_executions_per_rank=statistics.mean(row['active_replica_executions_per_rank'] for row in current),
        mean_active_execution_slots_per_rank=statistics.mean(row['active_execution_slots_per_rank_mean'] for row in current),
        mixed_layer_waves=sum(row['home_experts_per_rank']==64 for row in current)))
result=dict(role='actual captured workload, not timing attribution',update=2,
    source_tags=['timer-v2-matched-legacy-primary','timer-v2-matched-primary'],summaries=summaries,
    includes_diagnostic_worker_overhead=True,not_a_performance_score=True,
    conclusions_limited_to_this_update_and_route_bank=True)
(root/'diagnostics/actual-plan-balance.json').write_text(json.dumps(result,indent=2)+'\n')
with (root/'diagnostics/actual-plan-balance.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
x=np.arange(2)
axes[0].bar(x-.18,[row['mean_input_home_rank_maxvio'] for row in summaries],width=.36,label='Input home demand')
axes[0].bar(x+.18,[row['mean_actual_execution_rank_maxvio'] for row in summaries],width=.36,label='Actual quota destinations')
axes[0].set_xticks(x,[row['policy'] for row in summaries]);axes[0].set_ylabel('Mean rank MaxVio over16 original waves')
axes[0].legend(fontsize=8)
axes[1].bar(x,[row['mean_active_execution_slots_per_rank'] for row in summaries],color='#d58b2a')
axes[1].set_xticks(x,[row['policy'] for row in summaries]);axes[1].set_ylabel('Mean nonzero execution slots / rank / wave')
for axis in axes:axis.grid(axis='y',alpha=.2);axis.set_ylim(0,None);axis.margins(y=.15)
fig.suptitle('Actual captured plans | MaxVio10/random diagnostic update2\n'
    'Eight-rank consensus + assignment conservation checked; workload metrics are not kernel durations',fontsize=10)
fig.tight_layout()
for suffix in ('png','svg','pdf'):fig.savefig(root/f'diagnostics/actual-plan-balance.{suffix}',dpi=180,bbox_inches='tight')
print(json.dumps(result))
