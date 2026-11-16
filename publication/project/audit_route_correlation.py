"""Offline fixture audit: expert skew, rank skew and temporal complementarity differ."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_screening import task_loads

root=Path(__file__).resolve().parent
bank=root/'diagnostics/screening-maxvio-10.npy'
counts=np.bincount(np.load(bank,allow_pickle=False).reshape(-1),minlength=256)
records=[];summaries=[]
for pattern in ('same','alternating','random'):
    rank_skews=[];pair_gains=[]
    for step in range(25):
        for layer in range(4):
            loads=[task_loads(counts,pattern,(step,microbatch,layer)) for microbatch in range(4)]
            rank_skews.extend(float(load.max()/load.mean()-1) for load in loads)
            for first in (0,2):
                left,right=loads[first:first+2]
                serial=int(left.max()+right.max());combined=int((left+right).max())
                complementarity=1-combined/serial
                pair_gains.append(complementarity)
                records.append(dict(pattern=pattern,step=step,layer=layer,first_microbatch=first,
                    linear_same_layer_pair_complementarity=complementarity,
                    left_rank_maxvio=float(left.max()/left.mean()-1),
                    right_rank_maxvio=float(right.max()/right.mean()-1),
                    left_rank_loads=json.dumps(left.tolist()),right_rank_loads=json.dumps(right.tolist())))
    summaries.append(dict(pattern=pattern,logical_tasks=len(rank_skews),same_layer_pairs=len(pair_gains),
        rank_maxvio_mean=statistics.mean(rank_skews),rank_maxvio_median=statistics.median(rank_skews),
        rank_maxvio_min=min(rank_skews),rank_maxvio_max=max(rank_skews),
        rank_maxvio_p95=float(np.quantile(rank_skews,.95)),
        same_layer_pair_complementarity_mean=statistics.mean(pair_gains),
        same_layer_pair_complementarity_median=statistics.median(pair_gains),
        same_layer_pair_complementarity_max=max(pair_gains)))
result=dict(role='offline route fixture audit; no data supplied to online task selection',
    bank_sha256=hashlib.sha256(bank.read_bytes()).hexdigest(),
    expert_maxvio=float(counts.max()/counts.mean()-1),summaries=summaries,
    scope='same-layer adjacent microbatch pairs, not actual readiness or achieved grouping',
    ignores_communication_dependencies_replication_and_probe=True,training_speedup_claim=False)
(root/'diagnostics/route-correlation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
with (root/'diagnostics/route-correlation-pairs.csv').open('w',newline='') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
labels=[row['pattern'] for row in summaries]
axes[0].bar(labels,[row['rank_maxvio_mean'] for row in summaries],color='#3578a2')
axes[0].set(ylabel='Mean task rank MaxVio',title='32 home experts/rank aggregate the expert skew',ylim=(0,None))
axes[1].bar(labels,[100*row['same_layer_pair_complementarity_mean'] for row in summaries],color='#d58b2a')
axes[1].set(ylabel='Mean ideal compute-time reduction (%)',title='Adjacent same-layer pairs; no execution costs',ylim=(0,None))
for axis,values in zip(axes,([row['rank_maxvio_mean'] for row in summaries],
                           [100*row['same_layer_pair_complementarity_mean'] for row in summaries])):
    axis.grid(axis='y',alpha=.2);axis.margins(y=.2)
    for index,value in enumerate(values):axis.text(index,value,f'{value:.3f}',ha='center',va='bottom')
fig.suptitle(f'Offline fixture audit | expert MaxVio={result["expert_maxvio"]:.4f}\n'
             'These are load-vector diagnostics, not measured update-time improvements',fontsize=10)
fig.tight_layout()
for suffix in ('png','svg','pdf'):fig.savefig(root/f'diagnostics/route-correlation-audit.{suffix}',dpi=180,bbox_inches='tight')
print(json.dumps(result))
