"""Admit the first paired pilot only after all original-order W2 checks pass."""
from pathlib import Path
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
tag='ordered-shared-easy_independent-w2'
receipt=json.loads((root/f'{tag}-receipt.json').read_text())
if receipt['exit'] or receipt['collection_exit']:raise RuntimeError('Final W2 gate not collected')
destination=f'{remote}/evidence/{tag}-comparison.json'
comparison=subprocess.run(['ssh','pod6-1-root','docker','exec',
    'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
    f'{remote}/tools/compare_temporal_gradients.py',f'{remote}/evidence/gradient-w2-legacy',
    f'{remote}/evidence/{tag}','--step0-only','--out',destination])
subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
comparison.check_returncode()
gates=[]
for policy in ('fixed_native','easy_home','fifo_native','easy_independent'):
    name=f'ordered-shared-{policy}-w2'
    status=json.loads((root/'evidence'/name/'status.json').read_text())
    gradient=json.loads((root/'evidence'/f'{name}-comparison.json').read_text())
    if not status['qualified'] or not status['source_unchanged'] or not gradient['passed']:
        raise RuntimeError(f'Unqualified W2 policy: {policy}')
    gates.append(dict(tag=name,execution_qualified=True,full_gradient_passed=True))
(root/'ordered-w2-complete.json').write_text(json.dumps(gates,indent=2))
# The only post-gate model-tool edits move JSON persistence outside the timer
# and rename a diagnostic backend field. No tensor math or update changes.
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
jobs=[dict(tag=f'pilot-clean-timing-random-{policy}-w2',policy=policy,pattern='random',maxvio=10,
    layers=4,ep=8,experts=256,topk=8,hidden=4096,ffn=1024,tokens=16384,
    microbatches=4,window=2,receive_capacity=262144,optimizer_updates=25,
    warmup_updates=5,measured_updates=20,independent_repeats=1,status='planned',formal=False)
    for policy in ('legacy','easy_home')]
def save():
    (root/'pilot-results.json').write_text(json.dumps(dict(jobs=jobs,formal_matrix=False,
        common_memory_gib=90,afore_cache_included=False,joint_strategy_included=False,
        correctness_gates=gates,diagnostic_persistence_outside_timing=True),indent=2))
save()
for job in jobs:
    job['status']='running';save()
    run=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
        '--metadata-only','--tag',job['tag'],'--layers','4','--steps','25','--tokens','16384',
        '--hidden','4096','--ffn','1024','--microbatches','4','--forward-window','2',
        '--recv-capacity','262144','--maxvio','10','--planner','native',
        '--temporal-pattern','random','--temporal-policy',job['policy'],
        '--temporal-scheduler','ascendc','--grad-sync'])
    path=root/'evidence'/job['tag']/'status.json'
    status=json.loads(path.read_text()) if path.exists() else {}
    job.update(status='completed' if run.returncode==0 and status.get('qualified') and
        status.get('peak_driver_MiB',float('inf'))<=90*1024 else 'failed',receipt=status)
    save()
    if job['status']!='completed':raise RuntimeError(f'Pilot failed: {job["tag"]}')
