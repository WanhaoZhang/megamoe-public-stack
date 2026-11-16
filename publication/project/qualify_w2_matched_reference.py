"""Compare the W2 family to its same-window Native baseline, without new tolerances.

The earlier W4-baseline cross-window comparisons remain separate diagnostics.
This reference choice is recorded before collecting the new baseline.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+5400
gate=root/'evidence/runset-math-v1/status.json'
while not gate.exists():
    if time.monotonic()>=deadline:raise RuntimeError('Independent runset gate did not finish')
    time.sleep(5)
tag='gradient-w2-legacy'
command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
    '--tag',tag,'--layers','2','--steps','1','--tokens','256','--hidden','512','--ffn','512',
    '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
    '--planner','native','--route-bank',
    '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
    '--temporal-policy','legacy','--temporal-scheduler','ascendc','--dump-grad-step0','--grad-sync']
subprocess.run(command,check=True)
results=[]
for policy in ('fixed_native','easy_home','fifo_native','easy_independent'):
    candidate=f'gradient-w2-{policy}'
    destination=f'{remote}/evidence/{candidate}-matched-comparison.json'
    code=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
        '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
        f'{remote}/evidence/{tag}',f'{remote}/evidence/{candidate}','--out',destination]).returncode
    subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    results.append(dict(policy=policy,comparison_exit=code,reference=tag,window=2,
                        relative_l2_threshold=.02,all_parameters_checked=True))
    (root/'w2-matched-qualification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
raise SystemExit(0 if all(record['comparison_exit']==0 for record in results) else 1)
