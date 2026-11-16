"""Common public FSDP compute-image lifetime: qualify every path, then pilot."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import subprocess
import sys

root=Path(__file__).resolve().parent
path=root/'resident-prefix-v3.json'
if path.exists():raise RuntimeError('Existing v3 registration: do not reuse tags')
arms=(('native_resident','legacy',False),('fixed_native','fixed_native',True),
      ('fifo','fifo_native',True),('easy_home','easy_home',True),
      ('easy_independent','easy_independent',True),('fixed_home','fixed_home',True))
pilot=(('native_original','legacy',False,False),)+tuple((label,policy,prefix,True) for label,policy,prefix in arms)
state=dict(stage='registered',created=datetime.now(timezone.utc).isoformat(),new_planner=False,formal=False,
    recipe_change='public reshard_after_forward=False for common resident arms; no parameter/optimizer owner migration',
    model='L4/EP8/E256/Top8/H4096/F1024/S16384/GA4/W2/receive262144',
    stage_audit=False,diagnostic_sync='none',independent_repeats=1,gates=[],pilot_jobs=[],
    preregistered_small=[dict(label=label,policy=policy,prefix=prefix,hold=True) for label,policy,prefix in arms],
    preregistered_pilot=[dict(label=label,policy=policy,prefix=prefix,hold=hold) for label,policy,prefix,hold in pilot],
    source_module_sha256=hashlib.sha256((root/'stack/tools/temporal_prefix_reuse.py').read_bytes()).hexdigest(),
    failure_history=['prefix-reuse-summary.json','prefix-failure-diagnostics-summary.json','prefix-hold-forward-control.json'])

def save():
    state['updated']=datetime.now(timezone.utc).isoformat();path.write_text(json.dumps(state,indent=2)+'\n')

def run(tag,policy,prefix,hold,*,small=False,steps=3):
    args=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only','--tag',tag,
        '--layers','2' if small else '4','--steps',str(steps),'--tokens','256' if small else '16384',
        '--hidden','512' if small else '4096','--ffn','512' if small else '1024',
        '--microbatches','4','--forward-window','2','--recv-capacity','4096' if small else '262144',
        '--maxvio','10','--planner','native','--temporal-pattern','random','--temporal-policy',policy,
        '--temporal-scheduler','ascendc','--grad-sync']
    if prefix:args+=['--temporal-prefix-reuse']
    if hold:args+=['--hold-fsdp-forward']
    if small:args+=['--route-bank','/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes','--dump-grad-step0']
    code=subprocess.run(args).returncode
    folder=root/'evidence'/tag
    receipt=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
    return dict(tag=tag,policy=policy,prefix_reuse=prefix,hold_fsdp_forward=hold,execution_exit=code,receipt=receipt,
        qualified=bool(code==0 and receipt.get('qualified') and receipt.get('peak_driver_MiB',float('inf'))<=90*1024))

save()
try:
    subprocess.run([sys.executable,'-X','utf8',str(root/'sync_frontier.py')],check=True)
    for label,policy,prefix in arms:
        tag='resident-v3-gate-'+label;state.update(stage='small_full_gradient',running_tag=tag);save()
        gate=run(tag,policy,prefix,True,small=True)
        gate['label']=label;gate['comparison_exit']=None
        if gate['qualified']:
            remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
            destination=remote+'/evidence/'+tag+'-comparison.json'
            gate['comparison_exit']=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
                '/usr/local/python3.12.13/bin/python3',remote+'/tools/compare_temporal_gradients.py',
                remote+'/evidence/gradient-w2-legacy',remote+'/evidence/'+tag,'--step0-only','--out',destination]).returncode
            subprocess.run(['scp','pod6-1-root:'+destination,str(root/'evidence')],check=True)
        state['gates'].append(gate);save()
        if not gate['qualified'] or gate['comparison_exit']!=0:raise RuntimeError('v3 numeric/source/memory gate failed: '+tag)
    state.update(stage='main_memory_gate',running_tag='resident-v3-primary-adam3');save()
    memory=run('resident-v3-primary-adam3','easy_home',True,True)
    state['main_memory_gate']=memory;save()
    if not memory['qualified']:raise RuntimeError('v3 common shape memory/lifecycle gate failed; no model shrink')
    for label,policy,prefix,hold in pilot:
        tag='resident-v3-pilot-'+label;state.update(stage='pilot',running_tag=tag);save()
        job=run(tag,policy,prefix,hold,steps=25)
        job.update(label=label,status='completed' if job['qualified'] else 'failed',
                   independent_repeats=1,warmup=5,measured=20,formal=False)
        state['pilot_jobs'].append(job);save()
        if not job['qualified']:raise RuntimeError('v3 pilot stopped at '+tag)
    state.update(stage='pilot_complete');state.pop('running_tag',None);save()
except BaseException as error:
    state.update(stage='stopped_at_gate',error=f'{type(error).__name__}: {error}');save();raise
