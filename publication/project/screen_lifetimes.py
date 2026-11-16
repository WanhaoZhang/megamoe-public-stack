"""Repeat a bounded B0/B2/B3 screen on the now-qualified common executor.

This is neither a new policy nor a large matrix. All three points are retained.
"""
from pathlib import Path
import json
import math
import subprocess
import remote
from audit_runs import audit, load, WORKERS

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'common-lifetime-screen-v1.json'
if STATE.exists():raise SystemExit('Screen state exists; inspect instead of replacing it.')
if not load(ROOT/'diagnostics/pilot-identity-audit.json')['passed']:
    raise SystemExit('Initial pair source and identity audit required')
variant=list(WORKERS)[1]
state=dict(stage='small',runs=[],predeclared_modes=['b0','b2','b3'],independent_main_repeats=1,
           new_planner=False,note='Old B controls used an older compute runtime. Requalify and rescreen them after the common key-cache intervention; no post-hoc point deletion or large matrix.')
def save():STATE.write_text(json.dumps(state,indent=2)+'\n',newline='\n')
save()
try:
    for small in (True,False):
        if not small and not load(ROOT/'confirmation-v1.json').get('passed'):
            raise RuntimeError('Common-runtime confirmation/anchor/home must complete before main screen')
        state['stage']='small' if small else 'main';save()
        for mode in state['predeclared_modes']:
            tag='priority-'+('small' if small else 'screen')+'-'+mode+'-v1'
            state['current']=tag;save()
            args=['--layers','2' if small else '4','--steps','3' if small else '25',
                  '--tokens','256' if small else '16384','--hidden','512' if small else '4096',
                  '--ffn','512' if small else '1024','--microbatches','4','--forward-window','2',
                  '--recv-capacity','4096' if small else '262144','--maxvio','10',
                  '--planner','moonep' if small else 'native','--temporal-pattern','same',
                  '--temporal-policy','fixed_native','--worker-variant',variant,'--grad-sync',
                  '--hold-fsdp-forward','--temporal-prefix-reuse','--lifetime-mode',mode,
                  '--lifetime-raw-schedule','preserve']
            if small:args+=['--dump-grad-step0','--finite-diagnostics','--route-bank',remote.REMOTE+'/stack/small-routes']
            else:args+=['--temporal-scheduler','ascendc','--temporal-scheduler-library',remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
            try:remote.gate(tag,args,metadata_only=True)
            except SystemExit as done:code=done.code
            folder=ROOT/'evidence'/tag;status=load(folder/'status.json')
            passed=code==0 and all(status.get(key) for key in ('qualified','source_unchanged','exclusive_devices')) and status['peak_driver_MiB']<=90*1024
            if not passed:
                state['runs'].append(dict(tag=tag,small=small,mode=mode,passed=False,status=status));save()
                raise RuntimeError('Unqualified screen run '+tag)
            if small:
                for rank in range(8):
                    updates=load(folder/f'updates-rank{rank}.json')
                    steps=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
                    passed &= len(updates)==len(steps)==3 and all(row['changed'] for row in updates)
                    passed &= all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in steps)
                path=remote.REMOTE+'/stack/evidence/'+tag+'-gradient-comparison.json'
                command=['ssh',remote.HOST,'docker','exec',remote.CONTAINER,'/usr/local/python3.12.13/bin/python3',
                         remote.REMOTE+'/stack/tools/compare_temporal_gradients.py',
                         '/home/z00956592/megaflux/step-lifetime-raw-control-20261116/stack/evidence/raw-model-preserve-v1',
                         remote.REMOTE+'/stack/evidence/'+tag,'--out',path]
                with (ROOT/'diagnostics'/(tag+'-comparison.log')).open('wb') as handle:
                    comparison=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
                local=ROOT/'diagnostics'/(tag+'-gradient-comparison.json')
                subprocess.run(['scp',remote.HOST+':'+path,str(local)],check=True)
                passed &= comparison.returncode==0 and load(local)['passed']
                row=dict(tag=tag,small=True,mode=mode,passed=bool(passed),peak_driver_MiB=status['peak_driver_MiB'])
            else:
                row=audit(tag,variant,reference='priority-pilot-reference-v1',lifetime=mode)
                row.update(small=False,mode=mode)
            state['runs'].append(row);save()
            if not row['passed']:raise RuntimeError('Screen verification failed '+tag)
    state.update(stage='complete',passed=True);state.pop('current',None);save()
except BaseException as error:
    state.update(stage='failed',error=str(error));save();raise
