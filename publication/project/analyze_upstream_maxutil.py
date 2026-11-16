"""Audit author-style selection and fresh, same-pattern common-runtime runs."""
from pathlib import Path
import json,subprocess,sys,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'stack/tools'))
from temporal_scheduler import ReadyTask,TaskKey,select_window
subprocess.run([sys.executable,'-X','utf8',str(root/'analyze_resident_prefix_v3.py'),
    '--registry','upstream-maxutil-results.json','--output-prefix','upstream-maxutil',
    '--summary','upstream-maxutil-audit.json'],check=True)
state=json.loads((root/'upstream-maxutil-results.json').read_text())
audit=json.loads((root/'upstream-maxutil-audit.json').read_text())
group_checks=[]
for row in audit['runs']:
    if row['label']!='upstream_home':continue
    directory=root/'evidence'/row['tag']
    groups=[json.loads(line) for line in (directory/'temporal-groups-rank0.jsonl').read_text().splitlines()]
    ages={};mixed=0;waves=0
    for group in groups:
        ready=tuple(ReadyTask(TaskKey(*key),tuple(cost),ages.get(tuple(key),0))
                    for key,cost in zip(group['ready_keys'],group['ready_loads']))
        expected=select_window(ready,'easybalance_upstream',2)
        actual=tuple(TaskKey(*key) for key in group['keys'])
        if expected.selected!=actual:raise RuntimeError('Actual online selected set disagrees with author-style rule')
        for task in ready:
            if task.key not in actual:ages[(task.key.step,task.key.microbatch,task.key.layer)]=task.deferred+1
        waves+=len(group['native_waves'])
        mixed+=sum(len(wave['domain_layers'])>1 for wave in group['native_waves'])
    selected=lambda items:[{key:row[key] for key in ('step','keys','ready_keys','ready_loads','native_waves')} for row in items]
    for rank in range(1,8):
        peers=[json.loads(line) for line in (directory/f'temporal-groups-rank{rank}.jsonl').read_text().splitlines()]
        if selected(peers)!=selected(groups):raise RuntimeError('Actual online rank schedule/load consensus differs')
    group_checks.append(dict(tag=row['tag'],pattern=row['pattern'],groups=len(groups),waves=waves,mixed_layer_waves=mixed,
        all_rank_schedule_consensus=True,author_style_selected_sets_checked=True))
paired=[]
for pattern in ('same','alternating','random'):
    rows={row['label']:row for row in audit['runs'] if row['pattern']==pattern}
    if set(rows)!={'native_resident','fixed_home','upstream_home'}:continue
    native,fixed,upstream=(rows[label]['mean_wall_seconds'] for label in ('native_resident','fixed_home','upstream_home'))
    paired.append(dict(pattern=pattern,native_seconds=native,fixed_home_seconds=fixed,upstream_seconds=upstream,
        upstream_vs_fixed_time_reduction=1-upstream/fixed,upstream_vs_native_time_reduction=1-upstream/native,
        independent_processes_per_method=1))
result=dict(stage=state['stage'],error=state.get('error'),complete=state['stage']=='pilot_complete',
    runs=audit['runs'],paired=paired,gates=audit['gates'],device_gate=state.get('device_gate'),
    main_memory_gate=state.get('main_memory_gate'),group_checks=group_checks,
    new_planner=False,full_author_cuda_runtime_reproduced=False,formal_matrix_completed=0,
    old_work_tie_results_separate=True,stable_gain_claim=False,afore_device_baseline=False)
(root/'upstream-maxutil-summary.json').write_text(json.dumps(result,indent=2)+'\n')
if paired:
    with (root/'figures/upstream-maxutil-pairs.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
if audit['runs']:
    patterns=('same','alternating','random');labels=('native_resident','fixed_home','upstream_home')
    fig,axes=plt.subplots(1,2,figsize=(11.5,5.0));colors=('#3578a2','#21956c','#db8c29')
    names={'native_resident':'Native (legacy checkpoint)','fixed_home':'Fixed grouping + home','upstream_home':'Author MaxUtil + home'}
    for index,(label,color) in enumerate(zip(labels,colors)):
        rows=[row for row in audit['runs'] if row['label']==label]
        x=[patterns.index(row['pattern'])+(index-1)*.24 for row in rows]
        for axis,key in zip(axes,('mean_wall_seconds','peak_sampled_gib')):
            axis.bar(x,[row[key] for row in rows],width=.24,label=names[label],color=color)
            for pos,row in zip(x,rows):axis.text(pos,row[key],f'{row[key]:.3f}',ha='center',va='bottom',fontsize=8,rotation=20)
    for axis in axes:axis.set_xticks(range(3),patterns);axis.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean complete GA4 update wall (s)');axes[0].set_ylim(0,None);axes[0].margins(y=.2)
    axes[1].set_ylabel('Sampled whole-card HBM (GiB)');axes[1].set_ylim(0,95)
    axes[1].axhline(90,color='crimson',linestyle='--',label='90GiB common ceiling')
    for axis in axes:axis.legend(fontsize=8,loc='lower left')
    fig.suptitle('Author core MaxUtil | L4 EP8 E256 Top8 H4096 F1024 S16384/rank\n'
        'MaxVio10; GA4 W2; each method/pattern: one process,5 warmup+20 measured',fontsize=10)
    fig.tight_layout()
    for extension in ('png','svg','pdf'):fig.savefig(root/f'figures/upstream-maxutil-screen.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
print(json.dumps(dict(stage=state['stage'],audited=len(audit['runs']),complete_pattern_pairs=len(paired))))
