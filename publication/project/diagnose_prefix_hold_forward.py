"""Public FSDP lifetime isolation after the five registered numeric cases."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
state_path=root/'prefix-hold-forward-control.json'
if state_path.exists():raise RuntimeError('Do not overwrite the registered lifetime control')
state=dict(stage='waiting_for_five_cases',tag='prefix-diagnostic-hold-forward',
    hypothesis='public reshard_after_forward=False retains compute images across suspended prefixes',
    root_cause_not_established=True,steps=3,dump_full_step0_gradient=True,
    performance_score=False,formal=False,source_change_required=False)
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
deadline=time.monotonic()+3600
while True:
    cases=json.loads((root/'prefix-failure-diagnostics.json').read_text())['cases']
    if any(case['status']=='execution_failed' for case in cases):raise RuntimeError('Earlier protocol/infrastructure failure; lifetime control unrun')
    if all(case['status'] in ('passed_numeric','failed_numeric') for case in cases):break
    if time.monotonic()>deadline:raise RuntimeError('Earlier diagnosis incomplete; lifetime control unrun')
    time.sleep(10)
state['stage']='running';save()
tag=state['tag']
code=subprocess.run([sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only','--tag',tag,
    '--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512','--microbatches','4','--forward-window','2',
    '--recv-capacity','4096','--maxvio','10','--planner','native','--route-bank',
    '/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
    '--temporal-policy','easy_home','--temporal-scheduler','ascendc','--grad-sync','--dump-grad-step0',
    '--temporal-prefix-reuse','--hold-fsdp-forward']).returncode
state['execution_exit']=code
if code==0:
    remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
    destination=remote+'/evidence/'+tag+'-comparison.json'
    state['comparison_exit']=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
        '/usr/local/python3.12.13/bin/python3',remote+'/tools/compare_temporal_gradients.py',remote+'/evidence/gradient-w2-legacy',
        remote+'/evidence/'+tag,'--step0-only','--out',destination]).returncode
    subprocess.run(['scp','pod6-1-root:'+destination,str(root/'evidence')],check=True)
state['stage']='passed_numeric' if state.get('comparison_exit')==0 else 'failed_numeric' if code==0 else 'execution_failed'
save();print(json.dumps(state))
