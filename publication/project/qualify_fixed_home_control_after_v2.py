"""Preregistered fixed-group/home control, after runtime-v2 core gates/pilot."""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
main_path=root/'prefix-reuse-transition-recovered-screen.json'
control_path=root/'prefix-fixed-home-control.json'
if control_path.exists():raise RuntimeError('Inspect the already registered control; do not reuse its tags')
control=dict(stage='waiting_for_core',formal=False,new_planner=False,
    preregistration='prefix-v2-preregistration.json',registered=datetime.now(timezone.utc).isoformat())

def save_control():
    control_path.write_text(json.dumps(control,indent=2)+'\n')

def save_main(value):
    value['updated']=datetime.now(timezone.utc).isoformat()
    main_path.write_text(json.dumps(value,indent=2)+'\n')

def report():
    for script in ('analyze_prefix_reuse.py','report_current.py'):
        subprocess.run([sys.executable,'-X','utf8',str(root/script)],check=True)

def run(tag,*,small):
    args=[sys.executable,'-X','utf8',str(root/'frontier_remote.py'),'gate','--metadata-only','--tag',tag,
        '--layers','2' if small else '4','--steps','3' if small else '25',
        '--tokens','256' if small else '16384','--hidden','512' if small else '4096',
        '--ffn','512' if small else '1024','--microbatches','4','--forward-window','2',
        '--recv-capacity','4096' if small else '262144','--maxvio','10','--planner','native',
        '--temporal-pattern','random','--temporal-policy','fixed_home','--temporal-scheduler','ascendc',
        '--grad-sync','--temporal-prefix-reuse']
    if small:args+=['--route-bank','/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes','--dump-grad-step0']
    code=subprocess.run(args).returncode
    path=root/'evidence'/tag/'status.json'
    status=json.loads(path.read_text()) if path.exists() else {}
    return dict(tag=tag,policy='fixed_home',prefix_reuse=True,execution_exit=code,receipt=status,
        qualified=bool(code==0 and status.get('qualified') and status.get('peak_driver_MiB',float('inf'))<=90*1024))

save_control()
try:
    deadline=time.monotonic()+5*3600
    while True:
        state=json.loads(main_path.read_text())
        if state['stage']=='stopped_at_gate':raise RuntimeError('Core runtime-v2 gate failed; control stays unrun')
        if state['stage']=='pilot_complete':break
        if time.monotonic()>deadline:raise RuntimeError('Core did not complete; control stays unrun')
        time.sleep(10)
    state['stage']='additional_fixed_home_gradient_gate';state['running_tag']='prefix-v2-fixed_home-adam3'
    save_main(state);control['stage']='small_gradient_gate';save_control()
    gate=run('prefix-v2-fixed_home-adam3',small=True)
    if gate['qualified']:
        remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
        destination=remote+'/evidence/'+gate['tag']+'-comparison.json'
        result=subprocess.run(['ssh','pod6-1-root','docker','exec','temporal-replica-frontier-20261115',
            '/usr/local/python3.12.13/bin/python3',remote+'/tools/compare_temporal_gradients.py',
            remote+'/evidence/gradient-w2-legacy',remote+'/evidence/'+gate['tag'],'--step0-only','--out',destination])
        gate['comparison_exit']=result.returncode
        subprocess.run(['scp','pod6-1-root:'+destination,str(root/'evidence')],check=True)
    else:gate['comparison_exit']=None
    state['gates'].append(gate);save_main(state);control['gate']=gate;save_control();report()
    if not gate['qualified'] or gate['comparison_exit']!=0:raise RuntimeError('Fixed-home control numeric gate failed')
    control['stage']='pilot';save_control()
    state['stage']='additional_fixed_home_pilot';state['running_tag']='prefix-v2-pilot-random-fixed_home';save_main(state)
    job=run('prefix-v2-pilot-random-fixed_home',small=False)
    job.update(label='fixed_home',status='completed' if job['qualified'] else 'failed',
               measured_updates=20,warmup_updates=5,independent_repeats=1,formal=False)
    state['pilot_jobs'].append(job);save_main(state);control['pilot']=job;save_control();report()
    if not job['qualified']:raise RuntimeError('Fixed-home pilot failed; no score assigned')
    control['stage']='complete';save_control()
    state['stage']='pilot_complete_with_fixed_home_control';state.pop('running_tag',None);save_main(state);report()
except BaseException as error:
    control.update(stage='stopped_at_gate',error=f'{type(error).__name__}: {error}');save_control()
    report()
    raise
