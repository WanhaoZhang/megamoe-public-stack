"""Queue the explicit runtime-v2 prototype only after the frozen v1 screen ends.

This is an execution refinement, not a new planner or a formal matrix. Each
failed gate stops subsequent stages; no foreign process is stopped.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
remote='/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack'
python='/usr/local/python3.12.13/bin/python3'
container='temporal-replica-frontier-20261115'
parser=argparse.ArgumentParser()
parser.add_argument('--state-name',default='prefix-reuse-transition.json')
args=parser.parse_args()
if Path(args.state_name).name!=args.state_name or not args.state_name.endswith('.json'):
    parser.error('state-name must be one JSON basename in the project directory')
state_path=root/args.state_name
if state_path.exists():
    raise RuntimeError('Transition already registered; inspect its existing evidence instead of restarting tags')
state=dict(stage='waiting_for_v1_screen',updated=datetime.now(timezone.utc).isoformat(),
    formal=False,new_planner=False,afore_cache=False,source_modified=False,gates=[],pilot_jobs=[],
    rematerialization_partition='prefix+ffn_shared_finish',
    source_v1_commit=subprocess.check_output(['git','-C',str(root/'stack'),'rev-parse','HEAD'],text=True).strip())

def save():
    state['updated']=datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')

def command(*args):
    return subprocess.run([str(value) for value in args],check=True)

def local(script,*args):
    return command(sys.executable,'-X','utf8',root/script,*args)

def run(tag,policy,*,steps,tokens,hidden,ffn,layers,capacity,prefix,dump=False):
    args=[sys.executable,'-X','utf8',root/'frontier_remote.py','gate','--metadata-only','--tag',tag,
        '--layers',layers,'--steps',steps,'--tokens',tokens,'--hidden',hidden,'--ffn',ffn,
        '--microbatches','4','--forward-window','2','--recv-capacity',capacity,'--maxvio','10',
        '--planner','native','--temporal-pattern','random','--temporal-policy',policy,
        '--temporal-scheduler','ascendc','--grad-sync']
    if tokens==256:
        args+=['--route-bank','/home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes']
    if prefix:args+=['--temporal-prefix-reuse']
    if dump:args+=['--dump-grad-step0']
    code=subprocess.run([str(value) for value in args]).returncode
    path=root/'evidence'/tag/'status.json'
    receipt=json.loads(path.read_text()) if path.exists() else {}
    qualified=code==0 and receipt.get('qualified') and receipt.get('peak_driver_MiB',float('inf'))<=90*1024
    return dict(tag=tag,policy=policy,prefix_reuse=prefix,execution_exit=code,receipt=receipt,qualified=bool(qualified))

save()
try:
    deadline=time.monotonic()+3*3600
    while True:
        manifest=json.loads((root/'screening-results.json').read_text())
        jobs=manifest['jobs']
        if any(job['status']=='failed' for job in jobs):
            raise RuntimeError('v1 screen failed; preserve its source and diagnose before any runtime transition')
        if len(jobs)==18 and all(job['status']=='completed' for job in jobs):
            break
        if time.monotonic()>deadline:
            raise RuntimeError('v1 screen incomplete; no source transition performed')
        time.sleep(10)
    local('analyze_screening.py')
    local('report_current.py')
    local('archive_implementation.py')
    # Immutable v1 report and code artifact; all actual per-run sources remain archived too.
    snapshot=root/'implementation/v1-screening-code'
    snapshot.mkdir()
    for name in ('source-changes.tar.gz','source-diff.patch','source-manifest.json'):
        shutil.copyfile(root/'implementation'/name,snapshot/name)
    shutil.copyfile(root/'REPORT.md',root/'REPORT-v1-screening.md')
    shutil.copyfile(root/'screening-summary.json',snapshot/'screening-summary.json')
    state['v1_report_sha256']=hashlib.sha256((root/'REPORT-v1-screening.md').read_bytes()).hexdigest()
    state['stage']='applying_explicit_runtime_v2';save()
    local('prototypes/stage_prefix_reuse.py','--apply')
    state['source_modified']=True;save()
    local('sync_frontier.py')
    for policy in ('fixed_native','fifo_native','easy_home','easy_independent'):
        tag=f'prefix-v2-{policy}-adam3'
        state['stage']='small_full_gradient';state['running_tag']=tag;save()
        record=run(tag,policy,steps=3,tokens=256,hidden=512,ffn=512,layers=2,capacity=4096,prefix=True,dump=True)
        if record['qualified']:
            destination=f'{remote}/evidence/{tag}-comparison.json'
            result=subprocess.run(['ssh','pod6-1-root','docker','exec',container,python,
                f'{remote}/tools/compare_temporal_gradients.py',f'{remote}/evidence/gradient-w2-legacy',
                f'{remote}/evidence/{tag}','--step0-only','--out',destination])
            record['comparison_exit']=result.returncode
            command('scp',f'pod6-1-root:{destination}',root/'evidence')
        else:record['comparison_exit']=None
        state['gates'].append(record);save()
        if not record['qualified'] or record['comparison_exit']!=0:
            raise RuntimeError('Runtime-v2 numeric/source/exclusivity gate failed: '+tag)
    state['stage']='main_shape_memory_gate';state['running_tag']='prefix-v2-primary-adam3';save()
    record=run('prefix-v2-primary-adam3','easy_home',steps=3,tokens=16384,hidden=4096,ffn=1024,
               layers=4,capacity=262144,prefix=True)
    state['main_memory_gate']=record;save()
    if not record['qualified']:
        raise RuntimeError('Runtime-v2 main shape memory/lifecycle failed; no pilot or model shrink')
    # All available v2 variants receive the same refinement; ordinary Native stays a strong original-path reference.
    for label,policy,prefix in (('native','legacy',False),('fixed','fixed_native',True),
            ('fifo','fifo_native',True),('easy_home','easy_home',True),('easy_independent','easy_independent',True)):
        tag=f'prefix-v2-pilot-random-{label}'
        state['stage']='common_runtime_pilot';state['running_tag']=tag;save()
        record=run(tag,policy,steps=25,tokens=16384,hidden=4096,ffn=1024,layers=4,capacity=262144,prefix=prefix)
        record.update(label=label,status='completed' if record['qualified'] else 'failed',
                      measured_updates=20,warmup_updates=5,independent_repeats=1,formal=False)
        state['pilot_jobs'].append(record);save()
        if not record['qualified']:
            raise RuntimeError('Runtime-v2 pilot failed; subsequent points remain unrun')
    state['stage']='pilot_complete';state.pop('running_tag',None);save()
except BaseException as error:
    state.update(stage='stopped_at_gate',error=f'{type(error).__name__}: {error}')
    save()
    raise
