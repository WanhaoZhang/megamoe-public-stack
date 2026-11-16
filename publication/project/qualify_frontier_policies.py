"""Finite, dependent small-model gradient and three-Adam policy qualification."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
gate=root/'evidence/frontier-primary-easy-w2-r4/status.json'
deadline=time.monotonic()+4200
while not gate.exists():
    if time.monotonic()>=deadline:raise RuntimeError('Primary receipt not collected')
    time.sleep(5)
if not json.loads(gate.read_text()).get('qualified'):
    raise RuntimeError('Primary common-resource gate failed; no policy screening launched')
subprocess.run(['scp',str(root/'stack/tools/compare_temporal_gradients.py'),
                f'pod6-1-root:{remote}/tools/'],check=True)
results=[]
for policy in ('easy_home','fixed_native','fifo_native','easy_independent'):
    tag=f'frontier-gradient-v3-{policy}'
    command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','4','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0']
    run=subprocess.run(command)
    comparison=None
    if run.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        reference=f'{remote}/evidence/frontier-gradient-legacy'
        compare=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
            reference,f'{remote}/evidence/{tag}','--out',destination,'--step0-only'])
        comparison=compare.returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    results.append(dict(policy=policy,tag=tag,execution_exit=run.returncode,
                        gradient_comparison_exit=comparison,formal_performance=False))
    (root/'policy-qualification.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
raise SystemExit(0 if all(record['execution_exit']==0 and record['gradient_comparison_exit']==0
                         for record in results) else 1)
