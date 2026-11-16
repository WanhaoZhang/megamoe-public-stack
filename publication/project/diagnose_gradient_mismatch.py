"""Do not enter performance with a failed gradient gate; isolate its cause."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+4200
while True:
    path=root/'policy-qualification.json'
    if path.exists() and len(json.loads(path.read_text()))==4:break
    if time.monotonic()>=deadline:raise RuntimeError('Policy diagnostics did not finish')
    time.sleep(5)
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
records=[]
for policy in ('legacy','fixed_native','fixed_home'):
    tag=f'gradient-fenced-{policy}'
    command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','1','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','4','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0','--grad-sync']
    run=subprocess.run(command)
    comparison=None
    if run.returncode==0 and policy!='legacy':
        destination=f'{remote}/evidence/{tag}-comparison.json'
        compare=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
            f'{remote}/evidence/gradient-fenced-legacy',f'{remote}/evidence/{tag}',
            '--out',destination])
        comparison=compare.returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    records.append(dict(policy=policy,tag=tag,execution_exit=run.returncode,
                        gradient_comparison_exit=comparison,performance_enabled=False))
    (root/'gradient-mismatch-diagnosis.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records[-1]),flush=True)
