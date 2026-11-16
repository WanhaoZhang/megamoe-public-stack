"""Preregister a no-probe offline control; never label it an online score."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
subprocess.run(['ssh','pod6-1-root','mkdir','-p',f'{remote}/diagnostic-inputs'],check=True)
for size in ('small','primary'):
    subprocess.run(['scp',str(root/f'diagnostics/offline-{size}-schedule.json'),
                    f'pod6-1-root:{remote}/diagnostic-inputs/'],check=True)
with (root/'evidence/cpu-contract-current.log').open('w') as handle:
    subprocess.run(['ssh','pod6-1-root','docker','exec','-w',remote,
        'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
        'tools/test_temporal_contract.py'],stdout=handle,stderr=subprocess.STDOUT,check=True)

def gate(tag,arguments):
    return subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
                           '--metadata-only','--tag',tag,*arguments]).returncode

def compare(tag):
    destination=f'{remote}/evidence/{tag}-comparison.json'
    code=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
        '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
        f'{remote}/evidence/gradient-w2-legacy',f'{remote}/evidence/{tag}',
        '--step0-only','--out',destination]).returncode
    subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    return code

small=['--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
    '--microbatches','4','--recv-capacity','4096','--maxvio','10','--planner','native',
    '--temporal-scheduler','ascendc','--grad-sync','--dump-grad-step0','--route-bank',
    '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes']
proofs=[]
for label,policy,window,extra in [('legacy-w1','legacy',1,[]),('offline-no-probe','easy_home',2,
        ['--diagnostic-only','--temporal-schedule-replay',f'{remote}/diagnostic-inputs/offline-small-schedule.json'])]:
    tag=f'observation-gate-{label}'
    execution=gate(tag,small+['--temporal-policy',policy,'--forward-window',str(window)]+extra)
    comparison=compare(tag) if execution==0 else None
    proofs.append(dict(label=label,tag=tag,execution_exit=execution,comparison_exit=comparison))
    (root/'observation-gates.json').write_text(json.dumps(proofs,indent=2))
    if label=='offline-no-probe' and (execution or comparison):
        raise RuntimeError('No-probe offline correctness failed; no large ablation admitted')

arms=[('native-w1','legacy',1,False),('native-w2','legacy',2,False),
      ('online-easy-home','easy_home',2,False),('offline-no-probe','easy_home',2,True)]
jobs=[dict(tag=f'observation-v1-{label}',label=label,policy=policy,window=window,
    role='offline_task_selection_diagnostic' if offline else 'online_reference',
    uses_future_derived_schedule=offline,status='planned',optimizer_updates=25,
    warmup_updates=5,measured_updates=20,independent_repeats=1,formal=False)
    for label,policy,window,offline in arms]
def save():
    (root/'observation-ablation.json').write_text(json.dumps(dict(jobs=jobs,formal_matrix=False,
        offline_rows_excluded_from_online_scores=True,correctness_gates=proofs),indent=2))
save()
for job in jobs:
    if job['label']=='native-w1' and any(proof['execution_exit'] or proof['comparison_exit'] for proof in proofs if proof['label']=='legacy-w1'):
        job['status']='failed_correctness_not_timed';save();continue
    job['status']='running';save()
    extra=['--diagnostic-only','--temporal-schedule-replay',f'{remote}/diagnostic-inputs/offline-primary-schedule.json'] if job['uses_future_derived_schedule'] else []
    code=gate(job['tag'],['--layers','4','--steps','25','--tokens','16384','--hidden','4096',
        '--ffn','1024','--microbatches','4','--forward-window',str(job['window']),
        '--recv-capacity','262144','--maxvio','10','--planner','native','--temporal-pattern','random',
        '--temporal-policy',job['policy'],'--temporal-scheduler','ascendc','--grad-sync']+extra)
    path=root/'evidence'/job['tag']/'status.json'
    receipt=json.loads(path.read_text()) if path.exists() else {}
    job.update(status='completed' if not code and receipt.get('qualified') and
        receipt.get('peak_driver_MiB',float('inf'))<=90*1024 else 'failed',receipt=receipt)
    save()
    if job['status']!='completed':raise RuntimeError('Ablation stopped at failure/common-memory gate: '+job['tag'])
