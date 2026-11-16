"""Audit common resident-lifetime runs; keep both fresh Native references."""
from pathlib import Path
import argparse,csv,json,math,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--registry',default='resident-prefix-v3.json')
parser.add_argument('--output-prefix',default='resident-v3')
parser.add_argument('--summary',default='resident-prefix-v3-summary.json')
args=parser.parse_args()
state=json.loads((root/args.registry).read_text())
frozen=json.loads((root/'stack/configs/runtime-freeze.json').read_text())['binaries']
frozen.update(json.loads((root/'stack/configs/temporal-scheduler-freeze.json').read_text())['binaries'])
rows=[];steps=[];gates=[];references={};identities={}
for gate in state['gates']:
    file=root/'evidence'/(gate['tag']+'-comparison.json')
    value=json.loads(file.read_text()) if file.exists() else {}
    parameters=value.get('parameters',{});categories=value.get('categories',{})
    gates.append(dict(label=gate['label'],tag=gate['tag'],passed=bool(parameters and categories and gate['qualified'] and
        gate['comparison_exit']==0 and all(row['passed'] for row in parameters.values()) and
        all(row['passed'] for row in categories.values()) and all(value['identities_equal'].values())),
        max_parameter_l2=max((row['relative_l2'] for row in parameters.values()),default=None),
        max_category_l2=max((row['relative_l2'] for row in categories.values()),default=None)))
for job in state['pilot_jobs']:
    if not job['qualified']:continue
    directory=root/'evidence'/job['tag']
    config=json.loads((directory/'effective-config.json').read_text())
    status=json.loads((directory/'status.json').read_text())
    if not(status.get('qualified') and status.get('source_unchanged') and status.get('exclusive_devices') and
           status['rank_completions']==8 and status['peak_driver_MiB']<=90*1024):raise RuntimeError('Unqualified v3 receipt')
    if (config['layers'],config['tokens'],config['steps'],config['microbatches'],config['forward_window'],
        config['receive_capacity'],config['temporal_pattern'],config['grad_sync'])!=(4,16384,25,4,2,262144,job.get('pattern','random'),True):
        raise RuntimeError('v3 recipe mismatch')
    if (config['temporal_prefix_reuse'],config['hold_fsdp_forward'])!=(job['prefix_reuse'],job['hold_fsdp_forward']):
        raise RuntimeError('v3 residency/partition role mismatch')
    if job.get('maxutil_rule') and config.get('maxutil_rule')!=job['maxutil_rule']:
        raise RuntimeError('Registered MaxUtil policy differs from actual configuration')
    if (config.get('uses_future_derived_schedule') or config.get('numerical_diagnostic_only') or
        config.get('routing_prepass_counted') or config.get('prefix_stage_audit') or config['prefix_diagnostic_sync']!='none'):
        raise RuntimeError('A diagnostic cannot enter v3 timing results')
    ranks=[]
    for rank in range(8):
        workers=[]
        for filename,identity in json.loads((directory/f'libraries-rank{rank}.json').read_text())['libraries'].items():
            if '/source/native-balanced/' not in filename:continue
            relative='source/native-balanced/'+filename.split('/source/native-balanced/',1)[1]
            if frozen.get(relative)!=identity['sha256']:raise RuntimeError('Loaded SO mismatch')
            if '/worker-variants/' in filename:workers.append(identity['sha256'])
        if workers!=['46772ebc9fe69442e786dd4e9bd6892bb79bfe6e3944069df05bfafaf684712e']:raise RuntimeError('Non-performance worker')
        if job.get('maxutil_rule')=='upstream_first':
            selector=json.loads((directory/f'scheduler-so-rank{rank}.json').read_text())
            if selector['abi']!=4 or selector['sha256']!='871c167357d1d6e1575d331b3e73c7b36bcc115fb41988e02d4d8a3bb029013e':
                raise RuntimeError('Author-style result did not load the frozen ABI4 selector')
        updates=[json.loads(line) for line in (directory/f'steps-rank{rank}.jsonl').read_text().splitlines()]
        if [row['step'] for row in updates]!=list(range(25)):raise RuntimeError('Incomplete v3 process')
        if any(not math.isfinite(row[key]) for row in updates for key in ('milliseconds','loss','grad_norm')):raise RuntimeError('Nonfinite v3 data')
        calls=json.loads((directory/f'route-calls-rank{rank}.json').read_text())
        if len(calls)!=400 or set(calls.values())!={2}:raise RuntimeError('v3 router/replay count mismatch')
        if job['prefix_reuse']:
            audit=json.loads((directory/f'temporal-audit-rank{rank}.json').read_text())
            if audit['stage_audit_enabled'] or audit['diagnostic_stage_windows'] or audit['diagnostic_prefix_sync']!='none':
                raise RuntimeError('Diagnostic stage instrumentation enabled in performance case')
            if (audit['prepared_tasks'],audit['prefix_original'],audit['prefix_recompute'])!=(400,400,400):raise RuntimeError('Incomplete prefix accounting')
        ranks.append(updates)
    identity={name:[json.loads((directory/f'{name}-rank{rank}.json').read_text()) for rank in range(8)]
              for name in ('initial','input-identity','route-identity')}
    pattern=job.get('pattern','random')
    reference=references.setdefault(pattern,(identity,config['model']))
    equal={name:identity[name]==reference[0][name] for name in identity};equal['model']=config['model']==reference[1]
    if not all(equal.values()):raise RuntimeError('v3 logical identities differ')
    identities[job['tag']]=equal
    maximum=[max(rank[step]['milliseconds'] for rank in ranks)/1000 for step in range(25)]
    rows.append(dict(label=job['label'],tag=job['tag'],pattern=pattern,repeat=job.get('repeat',1),hold_fsdp_forward=job['hold_fsdp_forward'],prefix_reuse=job['prefix_reuse'],
        mean_wall_seconds=statistics.mean(maximum[5:]),median_wall_seconds=statistics.median(maximum[5:]),
        peak_sampled_gib=status['peak_driver_MiB']/1024,speedup_vs_fresh_original=None,speedup_vs_common_resident=None,
        independent_repeats=1,measured_updates=20,formal=False,new_planner=False))
    steps.extend(dict(label=job['label'],tag=job['tag'],pattern=pattern,repeat=job.get('repeat',1),step=step,warmup=step<5,max_rank_wall_seconds=value)
                 for step,value in enumerate(maximum))
baselines={label:next((row for row in rows if row['label']==label),None) for label in ('native_original','native_resident')}
for row in rows:
    for label,field in (('native_original','speedup_vs_fresh_original'),('native_resident','speedup_vs_common_resident')):
        baseline=next((item for item in rows if (item['label'],item['repeat'],item['pattern'])==(label,row['repeat'],row['pattern'])),None)
        if baseline:row[field]=baseline['mean_wall_seconds']/row['mean_wall_seconds']
out=root/'figures';out.mkdir(exist_ok=True)
for name,records in ((args.output_prefix+'-runs.csv',rows),(args.output_prefix+'-steps.csv',steps)):
    if records:
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
if rows and all(baselines.values()):
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.4))
    names={'native_original':'Native\noriginal','native_resident':'Native\nresident',
           'fixed_native':'Fixed group\nNative','fifo':'FIFO\nNative','easy_home':'Easy*\nhome',
           'easy_independent':'Easy*\nindependent','fixed_home':'Fixed group\nhome'}
    for index,row in enumerate(rows):
        color='#999999' if row['label']=='native_original' else '#3578a2'
        for axis,key in zip(axes,('mean_wall_seconds','speedup_vs_common_resident','peak_sampled_gib')):
            axis.bar(index,row[key],color=color)
            axis.text(index,row[key],f'{row[key]:.3f}',ha='center',va='bottom',rotation=25,fontsize=8)
    for axis,ylabel in zip(axes,('Mean complete-update wall (s)','Speedup vs common resident Native','Sampled whole-card HBM (GiB)')):
        axis.set_xticks(range(len(rows)),[names.get(row['label'],row['label']) for row in rows],fontsize=8)
        axis.set_ylabel(ylabel);axis.grid(axis='y',alpha=.2);axis.set_ylim(0,None);axis.margins(y=.22)
    axes[1].axhline(1,color='gray',linestyle='--');axes[2].axhline(90,color='crimson',linestyle='--',label='90GiB common ceiling')
    axes[2].set_ylim(0,95)
    axes[2].legend(fontsize=7)
    fig.suptitle('Common resident-lifetime v3 pilot | MaxVio10/random | L4 EP8 GA4 W2\n'
        'One process per point;5 warmup+20 measured; gray: original-runtime reference\n'
        '*Easy uses the work-tie training adapter; author-first-mask variant is a separate experiment',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','svg','pdf'):fig.savefig(out/f'{args.output_prefix}-pilot.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)
result=dict(stage=state['stage'],error=state.get('error'),gates=gates,runs=rows,identities_equal=identities,
    main_memory_gate=state.get('main_memory_gate'),complete=state['stage']=='pilot_complete',
    formal=False,new_planner=False,afore_cache_implemented=False,stable_speedup_claim=False,
    original_and_common_resident_references_separate=True)
(root/args.summary).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(stage=state['stage'],passed_gates=sum(gate['passed'] for gate in gates),pilot_points=len(rows))))
