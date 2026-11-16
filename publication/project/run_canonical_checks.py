"""Validate canonical within-task row order after the independent FFN oracle."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
gate=root/'evidence/runset-math-v2/status.json'
deadline=time.monotonic()+2400
while not gate.exists():
    if time.monotonic()>=deadline:raise RuntimeError('FFN oracle did not finish')
    time.sleep(5)
if not json.loads(gate.read_text()).get('qualified'):
    raise RuntimeError('Independent FFN oracle failed; canonical model validation not launched')
records=[]
for policy,window in [('fixed_native',2),('fixed_home',4)]:
    tag=f'canonical-gradient-{policy}-w{window}'
    arguments=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','1','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window',str(window),'--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0','--grad-sync']
    run=subprocess.run(arguments)
    code=None
    if run.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        reference='gradient-w2-legacy' if window==2 else 'gradient-fenced-legacy'
        code=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
            f'{remote}/evidence/{reference}',f'{remote}/evidence/{tag}','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    records.append(dict(tag=tag,policy=policy,window=window,execution_exit=run.returncode,
                        comparison_exit=code,formal_performance=False))
    (root/'canonical-qualification.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
