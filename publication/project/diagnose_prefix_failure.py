"""Preregistered small numerical isolations; all timings excluded from scores."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
source='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
state_path=root/'prefix-failure-diagnostics.json'
if state_path.exists():raise RuntimeError('Existing diagnostic registration; never reuse its tags')
cases=[dict(label=label,sync=sync,offline=offline,tag='prefix-diagnostic-'+label,status='planned')
    for label,sync,offline in (
        ('repeat-control','none',False),('stage-chain','stage_chain',False),
        ('prefix-device-sync','prefix_device',False),('backward-global','backward_global',False),
        ('deferred-same-order','none',True))]
state=dict(role='numerical diagnosis only; never performance scores',cases=cases,
    observations='v2 Easy-home failed despite identical v1 task order and logical identities',
    steps=3,output_gradient_limit_unchanged=.02,new_planner=False,formal=False,
    module_sha256=hashlib.sha256((root/'stack/tools/temporal_prefix_reuse.py').read_bytes()).hexdigest())

def save():state_path.write_text(json.dumps(state,indent=2)+'\n')

save()
subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
for case in cases:
    case['status']='running';save()
    args=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only','--tag',case['tag'],
        '--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512','--microbatches','4',
        '--forward-window','2','--recv-capacity','4096','--maxvio','10','--planner','native',
        '--route-bank','/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes',
        '--temporal-policy','easy_home','--temporal-scheduler','ascendc','--grad-sync','--dump-grad-step0',
        '--temporal-prefix-reuse','--prefix-diagnostic-sync',case['sync']]
    if case['offline']:args+=['--temporal-schedule-replay',source+'/diagnostic-inputs/offline-small-schedule.json','--diagnostic-only']
    code=subprocess.run(args).returncode
    directory=root/'evidence'/case['tag']
    receipt=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
    case.update(execution_exit=code,receipt=receipt,comparison_exit=None)
    if code==0 and receipt.get('qualified'):
        out=source+'/evidence/'+case['tag']+'-comparison.json'
        case['comparison_exit']=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',source+'/tools/compare_temporal_gradients.py',
            source+'/evidence/gradient-w2-legacy',source+'/evidence/'+case['tag'],'--step0-only','--out',out]).returncode
        subprocess.run(['scp','pod6-1-root:'+out,str(root/'evidence')],check=True)
    case['status']='passed_numeric' if case['comparison_exit']==0 else 'failed_numeric' if case['comparison_exit'] is not None else 'execution_failed'
    save()
    if case['status']=='execution_failed':raise RuntimeError('Infrastructure/protocol failed; no further cases '+case['tag'])
