"""Reduce one paired feasibility pilot without claiming independent repeats."""
from pathlib import Path
import csv
import json
import statistics
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
manifest=root/'pilot-results.json'
if not manifest.exists():
    print('No pilot admitted');raise SystemExit(0)
data=json.loads(manifest.read_text())
jobs=data['jobs']
result=dict(independent_repeats=1,formal=False,joint_strategy=False,stable_speedup_claim=False,
            complete=False,identities_equal=None,runs=[])
records=[]
frozen=json.loads((root/'stack/configs/runtime-freeze.json').read_text())['binaries']
frozen.update(json.loads((root/'stack/configs/temporal-scheduler-freeze.json').read_text())['binaries'])
for job in jobs:
    if job['status']!='completed':continue
    directory=root/'evidence'/job['tag']
    for rank in range(8):
        loaded=json.loads((directory/f'libraries-rank{rank}.json').read_text())
        workers=[]
        for path,identity in loaded['libraries'].items():
            if '/source/native-balanced/' not in path:continue
            relative='source/native-balanced/'+path.split('/source/native-balanced/',1)[1]
            if relative not in frozen or identity['sha256']!=frozen[relative]:
                raise RuntimeError(f'Actual loaded SO differs from frozen capability: {relative}')
            if '/worker-variants/' in path:workers.append(identity['sha256'])
        if workers!=['46772ebc9fe69442e786dd4e9bd6892bb79bfe6e3944069df05bfafaf684712e']:
            raise RuntimeError('Expected exactly the aligned64-slot worker')
    ranks=[[json.loads(line) for line in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
    if any([row['step'] for row in rank]!=list(range(25)) for rank in ranks):
        raise RuntimeError('Pilot lacks exactly25 updates on each rank')
    maxima=[max(rank[step]['milliseconds'] for rank in ranks)/1000 for step in range(25)]
    if any(not math.isfinite(value) or value<=0 for value in maxima):raise RuntimeError('Invalid wall time')
    for step,value in enumerate(maxima):
        records.append(dict(policy=job['policy'],tag=job['tag'],step=step,warmup=step<5,
                            max_rank_wall_seconds=value,sequence_tokens_per_second=524288/value))
    measured=maxima[5:]
    result['runs'].append(dict(policy=job['policy'],tag=job['tag'],measured_updates=20,
        mean_wall_seconds=statistics.mean(measured),median_wall_seconds=statistics.median(measured),
        min_wall_seconds=min(measured),max_wall_seconds=max(measured),
        sampled_peak_gib=job['receipt']['peak_driver_MiB']/1024,actual_loaded_binary_identities_checked=True))
if len(result['runs'])==2:
    directories=[root/'evidence'/job['tag'] for job in jobs]
    equal={name:all(json.loads((directories[0]/f'{name}-rank{rank}.json').read_text())==
                        json.loads((directories[1]/f'{name}-rank{rank}.json').read_text()) for rank in range(8))
           for name in ('initial','input-identity','route-identity')}
    result['identities_equal']=equal
    if not all(equal.values()):raise RuntimeError('Pilot identities differ; no comparative score')
    result['complete']=True
    baseline=result['runs'][0]['mean_wall_seconds']
    for run in result['runs']:
        run['normalized_mean_wall']=run['mean_wall_seconds']/baseline
        run['speedup_vs_native_mean']=baseline/run['mean_wall_seconds']
out=root/'figures';out.mkdir(exist_ok=True)
if records:
    with (out/'pilot-step-times.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
if result['complete']:
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for run in result['runs']:
        rows=[row for row in records if row['policy']==run['policy']]
        axes[0].plot([row['step'] for row in rows],[row['max_rank_wall_seconds'] for row in rows],
                     marker='.',label=run['policy'])
    axes[0].axvspan(-.5,4.5,color='gray',alpha=.15,label='5 warmup updates')
    axes[0].set(xlabel='Optimizer update',ylabel='Max over 8 ranks wall time (s)',ylim=(0,None))
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].bar([run['policy'] for run in result['runs']],[run['normalized_mean_wall'] for run in result['runs']],
                color=['#3976ad','#d88c27'])
    for index,run in enumerate(result['runs']):
        axes[1].text(index,run['normalized_mean_wall'],f"{run['mean_wall_seconds']:.3f}s\n{run['normalized_mean_wall']:.3f}x time",
                     ha='center',va='bottom',fontsize=9)
    axes[1].set(ylabel='Mean wall / common Native mean',ylim=(0,max(run['normalized_mean_wall'] for run in result['runs'])*1.25))
    axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Ascend950DT x8 feasibility pilot | MaxVio10/random | L4 H4096 F1024 S16K GA4 W2\n'
                 'One independent process per policy; 5 warmup + 20 measured; no stability claim',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'pilot-step-times.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
(root/'pilot-summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
