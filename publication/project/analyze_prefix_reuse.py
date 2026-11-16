"""Reduce runtime-v2 gates/pilot separately from the frozen v1 screening."""
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--state',type=Path)
args=parser.parse_args()
state_path=args.state or root/'prefix-reuse-transition-recovered-screen.json'
state=json.loads(state_path.read_text()) if state_path.exists() else dict(stage='not_registered',gates=[],pilot_jobs=[])
control_path=root/'prefix-fixed-home-control.json'
control=json.loads(control_path.read_text()) if control_path.exists() else dict(stage='not_registered')
frozen=json.loads((root/'stack/configs/runtime-freeze.json').read_text())['binaries']
frozen.update(json.loads((root/'stack/configs/temporal-scheduler-freeze.json').read_text())['binaries'])
rows=[];step_rows=[];identities={};gates=[];reference=None
for gate in state.get('gates',[]):
    path=root/'evidence'/(gate['tag']+'-comparison.json')
    comparison=json.loads(path.read_text()) if path.exists() else {}
    parameters=comparison.get('parameters',{})
    categories=comparison.get('categories',{})
    passed=bool(parameters and categories and all(value['passed'] for value in parameters.values()) and
        all(value['passed'] for value in categories.values()) and all(comparison.get('identities_equal',{}).values()) and
        gate['qualified'] and gate.get('comparison_exit')==0)
    gates.append(dict(tag=gate['tag'],policy=gate['policy'],passed=passed,
        maximum_parameter_relative_l2=max((value['relative_l2'] for value in parameters.values()),default=None),
        maximum_category_relative_l2=max((value['relative_l2'] for value in categories.values()),default=None)))
for job in state.get('pilot_jobs',[]):
    if not job['qualified']:continue
    folder=root/'evidence'/job['tag']
    config=json.loads((folder/'effective-config.json').read_text())
    status=json.loads((folder/'status.json').read_text())
    if not (status.get('qualified') and status.get('source_unchanged') and status.get('exclusive_devices') and
            status.get('rank_completions')==8 and status['peak_driver_MiB']<=90*1024):
        raise RuntimeError('Unqualified runtime-v2 pilot receipt')
    if (config['layers'],config['tokens'],config['microbatches'],config['forward_window'],
        config['receive_capacity'],config['grad_sync'],config['temporal_pattern'],config['steps'])!=(
        4,16384,4,2,262144,True,'random',25):raise RuntimeError('Runtime-v2 pilot recipe mismatch')
    if bool(config.get('temporal_prefix_reuse'))!=job['prefix_reuse'] or config.get('uses_future_derived_schedule'):
        raise RuntimeError('Runtime-v2 execution role mismatch')
    expected_partition='prefix+ffn_shared_finish' if job['prefix_reuse'] else 'layer'
    if config['rematerialization_partition']!=expected_partition or config['routing_prepass_counted']:
        raise RuntimeError('Unexpected rematerialization/observation contract')
    ranks=[]
    for rank in range(8):
        workers=[]
        for path,value in json.loads((folder/f'libraries-rank{rank}.json').read_text())['libraries'].items():
            if '/source/native-balanced/' not in path:continue
            relative='source/native-balanced/'+path.split('/source/native-balanced/',1)[1]
            if frozen.get(relative)!=value['sha256']:raise RuntimeError('Actual loaded SO mismatch')
            if '/worker-variants/' in path:workers.append(value['sha256'])
        if workers!=['46772ebc9fe69442e786dd4e9bd6892bb79bfe6e3944069df05bfafaf684712e']:
            raise RuntimeError('Runtime-v2 pilot expected the frozen performance worker')
        updates=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        if [row['step'] for row in updates]!=list(range(25)):raise RuntimeError('Incomplete runtime-v2 process')
        if any(not math.isfinite(row[key]) for row in updates for key in ('milliseconds','loss','grad_norm')):
            raise RuntimeError('Nonfinite runtime-v2 data')
        calls=json.loads((folder/f'route-calls-rank{rank}.json').read_text())
        if len(calls)!=400 or set(calls.values())!={2}:raise RuntimeError('Prefix was duplicated or not rematerialized')
        if job['prefix_reuse']:
            audit=json.loads((folder/f'temporal-audit-rank{rank}.json').read_text())
            if (audit.get('prepared_tasks'),audit.get('prefix_original'),audit.get('prefix_recompute'))!=(400,400,400):
                raise RuntimeError('Incomplete prefix audit')
        ranks.append(updates)
    identity={name:[json.loads((folder/f'{name}-rank{rank}.json').read_text()) for rank in range(8)]
              for name in ('initial','input-identity','route-identity')}
    if reference is None:reference=(identity,config['model'])
    equal={name:identity[name]==reference[0][name] for name in identity}
    equal['model']=config['model']==reference[1]
    if not all(equal.values()):raise RuntimeError('Runtime-v2 pilot logical identities differ')
    identities[job['tag']]=equal
    maximum=[max(rank[step]['milliseconds'] for rank in ranks)/1000 for step in range(25)]
    rows.append(dict(label=job['label'],tag=job['tag'],prefix_reuse=job['prefix_reuse'],
        rematerialization_partition=expected_partition,mean_wall_seconds=statistics.mean(maximum[5:]),
        median_wall_seconds=statistics.median(maximum[5:]),sampled_peak_gib=status['peak_driver_MiB']/1024,
        normalized_wall_to_fresh_native=None,speedup_to_fresh_native=None,
        independent_repeats=1,measured_updates=20,formal=False,new_planner=False))
    step_rows.extend(dict(label=job['label'],tag=job['tag'],step=step,warmup=step<5,
                          max_rank_wall_seconds=value) for step,value in enumerate(maximum))
baseline=next((row for row in rows if row['label']=='native'),None)
if baseline:
    for row in rows:
        row['normalized_wall_to_fresh_native']=row['mean_wall_seconds']/baseline['mean_wall_seconds']
        row['speedup_to_fresh_native']=baseline['mean_wall_seconds']/row['mean_wall_seconds']
out=root/'figures';out.mkdir(exist_ok=True)
for filename,records in (('prefix-v2-runs.csv',rows),('prefix-v2-steps.csv',step_rows)):
    if records:
        with (out/filename).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
if rows and baseline:
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for index,row in enumerate(rows):
        axes[0].bar(index,row['normalized_wall_to_fresh_native'],color='#d58b2a' if row['prefix_reuse'] else '#3578a2')
        axes[0].text(index,row['normalized_wall_to_fresh_native'],f"{row['mean_wall_seconds']:.3f}s",ha='center',va='bottom',fontsize=9)
        axes[1].bar(index,row['sampled_peak_gib'],color='#d58b2a' if row['prefix_reuse'] else '#3578a2')
    for axis in axes:
        axis.set_xticks(range(len(rows)),[row['label'].replace('_','\n') for row in rows],fontsize=9)
        axis.grid(axis='y',alpha=.2);axis.margins(y=.2);axis.set_ylim(0,None)
    axes[0].set_ylabel('Mean complete-update wall / fresh Native mean')
    axes[1].set_ylabel('Sampled whole-card HBM peak (GiB)')
    fig.suptitle('Runtime-v2 pilot | prefix + FFN/shared checkpoints | MaxVio10/random\n'
        'One independent process/point; original Native reference; no new planner or stable gain claim',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'prefix-v2-pilot.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
result=dict(stage=state['stage'],error=state.get('error'),state_path=state_path.name,
    gates=gates,main_memory_gate=state.get('main_memory_gate'),runs=rows,identities_equal=identities,
    complete=state['stage']=='pilot_complete_with_fixed_home_control' and control['stage']=='complete',
    fixed_home_control=control,new_planner=False,formal=False,
    stable_speedup_claim=False,checkpoint_partition_changed=True)
(root/'prefix-reuse-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(stage=state['stage'],qualified_small_gates=sum(gate['passed'] for gate in gates),pilot_runs=len(rows))))
