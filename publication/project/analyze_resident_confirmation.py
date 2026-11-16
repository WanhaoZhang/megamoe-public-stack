"""Audit all process runs first, then compare independent paired means."""
from pathlib import Path
import json,statistics,subprocess,sys,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,'-X','utf8',str(root/'analyze_resident_prefix_v3.py'),
    '--registry','resident-v3-confirmation.json','--output-prefix','resident-v3-confirmation',
    '--summary','resident-v3-confirmation-audit.json'],check=True)
state=json.loads((root/'resident-v3-confirmation.json').read_text())
audit=json.loads((root/'resident-v3-confirmation-audit.json').read_text())
original=json.loads((root/'resident-prefix-v3.json').read_text())
labels=('native_resident','easy_home','fixed_home')
for row in audit['runs']:
    tag=row['tag'];first=next(job for job in original['pilot_jobs'] if job['label']==row['label'])
    left=root/'evidence'/tag;right=root/'evidence'/first['tag']
    for filename in ('source-identity.json',)+tuple(f'{name}-rank{rank}.json' for rank in range(8)
            for name in ('initial','input-identity','route-identity')):
        if json.loads((left/filename).read_text())!=json.loads((right/filename).read_text()):
            raise RuntimeError('Confirmation logical/source identity differs: '+tag+'/'+filename)
paired=[]
for repeat in (1,2,3):
    cases={row['label']:row for row in audit['runs'] if row['repeat']==repeat}
    if set(cases)!=set(labels):continue
    native,easy,fixed=(cases[label]['mean_wall_seconds'] for label in labels)
    paired.append(dict(repeat=repeat,native_seconds=native,easy_seconds=easy,fixed_home_seconds=fixed,
        easy_vs_native_time_reduction=1-easy/native,fixed_vs_native_time_reduction=1-fixed/native,
        easy_vs_fixed_time_reduction=1-easy/fixed,
        easy_vs_fixed_speedup=fixed/easy,
        easy_peak_gib=cases['easy_home']['peak_sampled_gib'],fixed_peak_gib=cases['fixed_home']['peak_sampled_gib']))
aggregates=[]
for label in labels:
    runs=[row for row in audit['runs'] if row['label']==label]
    means=[row['mean_wall_seconds'] for row in runs]
    aggregates.append(dict(label=label,independent_processes=len(runs),mean_of_process_means=statistics.mean(means),
        median_of_process_means=statistics.median(means),max_sampled_hbm_gib=max(row['peak_sampled_gib'] for row in runs),
        process_means_seconds=means))
complete=state['stage']=='pilot_complete' and len(paired)==3
result=dict(stage=state['stage'],complete=complete,paired=paired,aggregates=aggregates,
    all_runs_audited=True,independent_unit='fresh process,20 measured optimizer updates after5 warmup',
    source_and_initial_input_route_identities_equal=True,not_a_new_planner=True,
    no_afore_device_baseline=True,formal_matrix_completed=0,
    stable_selection_gain_established=bool(complete and
        all(row['easy_vs_fixed_time_reduction']>0 for row in paired) and
        statistics.median(row['easy_vs_fixed_time_reduction'] for row in paired)>=.03),
    evidence_qualified_only_for_maxvio10_random=True)
(root/'resident-v3-confirmation-summary.json').write_text(json.dumps(result,indent=2)+'\n')
if paired:
    with (root/'figures/resident-v3-confirmation-pairs.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.6))
    colors={'native_resident':'#3578a2','easy_home':'#db8c29','fixed_home':'#21956c'}
    for label in labels:
        rows=sorted((row for row in audit['runs'] if row['label']==label),key=lambda row:row['repeat'])
        axes[0].plot([row['repeat'] for row in rows],[row['mean_wall_seconds'] for row in rows],
            '-o',label=label,color=colors[label])
    axes[0].set(xlabel='Independent process round',ylabel='Mean complete-update wall (s)',xticks=(1,2,3));axes[0].legend(fontsize=8)
    axes[1].plot([row['repeat'] for row in paired],[100*row['easy_vs_fixed_time_reduction'] for row in paired],'-o',color=colors['easy_home'])
    axes[1].axhline(0,color='gray',linestyle='--')
    axes[1].axhline(3,color='#21956c',linestyle=':',label='Preregistered3% screening threshold')
    axes[1].set(xlabel='Independent paired round',ylabel='Easy time reduction vs fixed-home control (%)',xticks=(1,2,3));axes[1].legend(fontsize=8)
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle('MaxVio10/random | L4 EP8 GA4 W2 | common public forward residency\n'
        'Round1 retained from seven-arm screen; two fresh rotated rounds; no new planner',fontsize=10)
    fig.tight_layout()
    for extension in ('png','svg','pdf'):fig.savefig(root/f'figures/resident-v3-confirmation.{extension}',dpi=180,bbox_inches='tight')
    plt.close(fig)
print(json.dumps(dict(stage=state['stage'],audited=len(audit['runs']),paired_rounds=len(paired),
    stable_selection_gain=result['stable_selection_gain_established'])))
