"""If canonical four-task replay still fails, test public FSDP forward residency."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
deadline=time.monotonic()+3600
while True:
    path=root/'canonical-qualification.json'
    if path.exists() and len(json.loads(path.read_text()))==2:break
    if time.monotonic()>=deadline:raise RuntimeError('Canonical runset qualification did not finish')
    time.sleep(5)
records=json.loads(path.read_text())
if records[-1]['comparison_exit']==0:
    print('Canonical four-task case passed; no FSDP lease experiment needed',flush=True)
    raise SystemExit(0)
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
results=[]
for policy in ('legacy','fixed_home'):
    tag=f'lease-gradient-{policy}-w4'
    command=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only',
        '--tag',tag,'--layers','2','--steps','1','--tokens','256','--hidden','512','--ffn','512',
        '--microbatches','4','--forward-window','4','--recv-capacity','4096','--maxvio','10',
        '--planner','native','--route-bank',
        '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy',policy,'--temporal-scheduler','ascendc','--dump-grad-step0',
        '--grad-sync','--hold-fsdp-forward']
    run=subprocess.run(command)
    code=None
    if policy!='legacy' and run.returncode==0:
        destination=f'{remote}/evidence/{tag}-comparison.json'
        code=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',f'{remote}/tools/compare_temporal_gradients.py',
            f'{remote}/evidence/lease-gradient-legacy-w4',f'{remote}/evidence/{tag}','--out',destination]).returncode
        subprocess.run(['scp',f'pod6-1-root:{destination}',str(root/'evidence')],check=True)
    results.append(dict(policy=policy,tag=tag,execution_exit=run.returncode,comparison_exit=code,
                        public_reshard_after_forward=False,formal_performance=False))
    (root/'fsdp-lease-diagnosis.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results[-1]),flush=True)
