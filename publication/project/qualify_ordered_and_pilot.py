"""Gate remaining W2 paths, then collect one preregistered 25-update pair.

This is a feasibility pilot, not the three-repeat screening/formal matrix.
The W4 diagnosis is recorded but cannot override the common W2 memory rule.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+3600
while True:
    path=root/'ordered-shared-diagnosis.json'
    earlier=json.loads(path.read_text()) if path.exists() else []
    if len(earlier)==3 or any(record['execution_exit'] for record in earlier):break
    if time.monotonic()>deadline:raise RuntimeError('Order diagnosis did not finish')
    time.sleep(5)
if len(earlier)!=3 or any(record['execution_exit'] or record['comparison_exit']
                         for record in earlier if record['window']==2):
    raise RuntimeError('Ordered W2 correctness failed; pilot not admitted')
records=[]
for policy in ('fifo_native','easy_independent'):
    tag=f'ordered-shared-{policy}-w2'
    result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
        '--metadata-only','--tag',tag,'--layers','2','--steps','3','--tokens','256',
        '--hidden','512','--ffn','512','--microbatches','4','--forward-window','2',
        '--recv-capacity','4096','--maxvio','10','--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0','--grad-sync'])
    comparison=None
    if result.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        comparison=subprocess.run(['ssh','pod6-1-root','docker','exec',
            'temporal-replica-frontier-20261115','/usr/local/python3.12.13/bin/python3',
            f'{remote}/tools/compare_temporal_gradients.py',f'{remote}/evidence/gradient-w2-legacy',
            f'{remote}/evidence/{tag}','--step0-only','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    records.append(dict(tag=tag,policy=policy,execution_exit=result.returncode,
                        comparison_exit=comparison,formal_performance=False))
    (root/'ordered-w2-qualification.json').write_text(json.dumps(records,indent=2))
    if result.returncode or comparison:raise RuntimeError(f'W2 gate failed: {tag}')

# Freeze order and the single pilot point before any large-shape measurement.
jobs=[dict(tag=f'pilot-ordered-random-{policy}-w2',policy=policy,pattern='random',maxvio=10,
    layers=4,ep=8,experts=256,topk=8,hidden=4096,ffn=1024,tokens=16384,
    microbatches=4,window=2,receive_capacity=262144,optimizer_updates=25,
    warmup_updates=5,measured_updates=20,independent_repeats=1,status='planned',formal=False)
    for policy in ('legacy','easy_home')]
def save():
    (root/'pilot-results.json').write_text(json.dumps(dict(jobs=jobs,formal_matrix=False,
        common_memory_gib=90,afore_cache_included=False,joint_strategy_included=False),indent=2))
save()
for job in jobs:
    job['status']='running';save()
    result=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate',
        '--metadata-only','--tag',job['tag'],'--layers','4','--steps','25','--tokens','16384',
        '--hidden','4096','--ffn','1024','--microbatches','4','--forward-window','2',
        '--recv-capacity','262144','--maxvio','10','--planner','native',
        '--temporal-pattern','random','--temporal-policy',job['policy'],
        '--temporal-scheduler','ascendc','--grad-sync'])
    status_path=root/'evidence'/job['tag']/'status.json'
    receipt=json.loads(status_path.read_text()) if status_path.exists() else {}
    job.update(status='completed' if result.returncode==0 and receipt.get('qualified') and
        receipt.get('peak_driver_MiB',float('inf'))<=90*1024 else 'failed',receipt=receipt)
    save()
    if job['status']!='completed':raise RuntimeError(f'Pilot failed: {job["tag"]}')
