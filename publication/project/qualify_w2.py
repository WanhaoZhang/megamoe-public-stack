"""Qualify the admitted W2 family without erasing failed W4 evidence."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+4200
while True:
    path=root/'gradient-mismatch-diagnosis.json'
    if path.exists() and len(json.loads(path.read_text()))==3:break
    if time.monotonic()>=deadline:raise RuntimeError('Fenced diagnosis did not finish')
    time.sleep(5)
results=[]
for policy in ('fixed_native','easy_home','fifo_native','easy_independent'):
    tag=f'gradient-w2-{policy}'
    command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','1','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0','--grad-sync']
    run=subprocess.run(command)
    comparison=None
    if run.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        comparison=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
            f'{remote}/evidence/gradient-fenced-legacy',f'{remote}/evidence/{tag}','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    results.append(dict(policy=policy,tag=tag,execution_exit=run.returncode,
                        gradient_comparison_exit=comparison,window=2,formal_performance=False))
    (root/'w2-qualification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
raise SystemExit(0 if all(record['execution_exit']==0 and record['gradient_comparison_exit']==0
                         for record in results) else 1)
