"""Separate online controls from an offline future-derived schedule replay."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
manifest=json.loads((root/'observation-ablation.json').read_text())
jobs=manifest['jobs'];runs=[];steps=[]
frozen=json.loads((root/'stack/configs/runtime-freeze.json').read_text())['binaries']
frozen.update(json.loads((root/'stack/configs/temporal-scheduler-freeze.json').read_text())['binaries'])
for job in jobs:
    if job['status']!='completed':continue
    folder=root/'evidence'/job['tag']
    config=json.loads((folder/'effective-config.json').read_text())
    offline=job['uses_future_derived_schedule']
    if bool(config.get('uses_future_derived_schedule'))!=offline:raise RuntimeError('Experiment role mismatch')
    times=[]
    for rank in range(8):
        library=json.loads((folder/f'libraries-rank{rank}.json').read_text())['libraries']
        workers=[]
        for name,record in library.items():
            if '/source/native-balanced/' not in name:continue
            key='source/native-balanced/'+name.split('/source/native-balanced/',1)[1]
            if frozen.get(key)!=record['sha256']:raise RuntimeError('Loaded binary mismatch')
            if '/worker-variants/' in name:workers.append(record['sha256'])
        if workers!=['46772ebc9fe69442e786dd4e9bd6892bb79bfe6e3944069df05bfafaf684712e']:
            raise RuntimeError('Non-performance worker in wall-time ablation')
        rows=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        if [row['step'] for row in rows]!=list(range(25)):raise RuntimeError('Incomplete 25-update process')
        times.append(rows)
    maximum=[max(rank[step]['milliseconds'] for rank in times)/1000 for step in range(25)]
    runs.append(dict(label=job['label'],tag=job['tag'],role=job['role'],
        uses_future_derived_schedule=offline,mean_wall_seconds=statistics.mean(maximum[5:]),
        median_wall_seconds=statistics.median(maximum[5:]),sampled_peak_gib=job['receipt']['peak_driver_MiB']/1024,
        measured_updates=20,independent_repeats=1,formal=False))
    steps.extend(dict(label=job['label'],role=job['role'],step=step,warmup=step<5,
                      max_rank_wall_seconds=value) for step,value in enumerate(maximum))
baseline=next((run for run in runs if run['label']=='native-w2'),None)
identities={}
if baseline:
    reference=root/'evidence'/baseline['tag']
    for run in runs:
        folder=root/'evidence'/run['tag']
        equal={name:all(json.loads((reference/f'{name}-rank{rank}.json').read_text())==
                json.loads((folder/f'{name}-rank{rank}.json').read_text()) for rank in range(8))
            for name in ('initial','input-identity','route-identity')}
        if not all(equal.values()):raise RuntimeError('Ablation logical identities differ')
        identities[run['label']]=equal
        run['wall_ratio_to_fresh_native_w2']=run['mean_wall_seconds']/baseline['mean_wall_seconds']
        run['online_speedup_vs_native_w2']=None if run['uses_future_derived_schedule'] else 1/run['wall_ratio_to_fresh_native_w2']
offline=next((run for run in runs if run['uses_future_derived_schedule']),None)
if offline:
    expected=json.loads((root/'diagnostics/offline-primary-schedule.json').read_text())['groups']
    for rank in range(8):
        actual=[json.loads(line) for line in (root/'evidence'/offline['tag']/f'temporal-groups-rank{rank}.jsonl').read_text().splitlines()]
        if [{name:row[name] for name in ('step','group','keys','ready_keys')} for row in actual]!=expected:
            raise RuntimeError('Offline task order did not match the source schedule')
out=root/'figures';out.mkdir(exist_ok=True)
if steps:
    with (out/'observation-ablation-steps.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(steps[0]));writer.writeheader();writer.writerows(steps)
if baseline and runs:
    fig,axis=plt.subplots(figsize=(9,4.8))
    bars=axis.bar(range(len(runs)),[run['wall_ratio_to_fresh_native_w2'] for run in runs],
                  color=['#999999' if run['uses_future_derived_schedule'] else '#3976ad' for run in runs])
    for index,(bar,run) in enumerate(zip(bars,runs)):
        if run['uses_future_derived_schedule']:bar.set_hatch('//')
        axis.text(index,bar.get_height()+.02,f"{run['mean_wall_seconds']:.3f}s\n{bar.get_height():.3f}x time",ha='center',fontsize=9)
    axis.set_xticks(range(len(runs)),[run['label'] for run in runs],rotation=15,ha='right')
    axis.set(ylabel='Mean wall / fresh Native W2 mean',ylim=(0,max(bar.get_height() for bar in bars)*1.22),
        title='Observation-cost ablation | MaxVio10/random | L4 EP8 GA4\n'
              'Hatched bar uses future-derived schedule: diagnostic only, never an online score')
    axis.grid(axis='y',alpha=.2);fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'observation-ablation.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
result=dict(complete=len(runs)==4,runs=runs,identities_equal=identities,
    offline_group_order_verified=offline is not None,offline_excluded_from_online_scores=True,
    stable_speedup_claim=False,formal_matrix_completed=0)
(root/'observation-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
