"""Bounded user-authorized finish: numeric gates and one B3 pair, then stop."""
from pathlib import Path
import hashlib
import json
import math
import statistics
import subprocess
import remote

ROOT=Path(__file__).resolve().parent
STATE=ROOT/'study-v1.json'
WORKER='ep8-cap64-lifetime-priority-cache-v1'
SO='e2868fd2fef49e089513191c9ccbb0ca3c39c857ac9a07d8f6bce255d6f946ed'
if STATE.exists():raise SystemExit('Study already started; inspect its live handle/state instead of restarting')
load=lambda path:json.loads(path.read_text())
if not load(ROOT/'cpu-v1.json')['passed']:raise SystemExit('CPU qualification required')
cpu_sources=load(ROOT/'evidence/gradient-ready-cpu-v1/source-identity.json')
cpu_core=('gradient_bank_ready.py','step_lifetime.py','lifetime_device.py','test_gradient_bank_ready.py',
          'test_lifetime_delivery.py','test_gradient_ready_capacity.py')
if not all(hashlib.sha256((ROOT/'stack/tools'/name).read_bytes()).hexdigest()==cpu_sources['tools/'+name] for name in cpu_core):
    raise SystemExit('CPU qualification core changed')
state=dict(stage='math',runs=[],persistent_goal='paused',bounded_finish=True,new_planner=False,
           formal_repeats=1,auto_expand=False)
def save():STATE.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8',newline='\n')

def gate(tag,args):
    state['current']=tag;save()
    try:remote.gate(tag,args,metadata_only=True)
    except SystemExit as done:code=done.code
    folder=ROOT/'evidence'/tag;status=load(folder/'status.json')
    passed=code==0 and all(status.get(key) for key in ('qualified','source_unchanged','exclusive_devices')) and status['peak_driver_MiB']<=90*1024
    if not passed:raise RuntimeError('Unqualified run '+tag+' actual exit '+str(status.get('exit')))
    for rank in range(8):
        libraries=load(folder/f'libraries-rank{rank}.json')
        workers=[value['sha256'] for path,value in libraries['libraries'].items() if '/worker-variants/' in path and path.endswith('.so')]
        if workers!=[SO] or len(libraries['shmem'])!=1:raise RuntimeError('Actual worker/SHMEM mismatch '+tag)
    return folder,status

save()
try:
    math_cases=[('reference-bank','call_stream','bank',8),('bank-bank','bank_write','bank',8),
                ('bank-sparse','bank_write','sparse_remote',8),('bank-capacity','bank_write','sparse_remote',1)]
    for name,scope,route,entries in math_cases:
        tag='grad-ready-math-'+name+'-v1'
        folder,status=gate(tag,['--mode','lifetime_math','--worker-variant',WORKER,
            '--lifetime-mode','all','--lifetime-raw-schedule','preserve','--gradient-ready-scope',scope,
            '--lifetime-entries',str(entries),'--route-case',route,'--route-bank',remote.REMOTE+'/stack/small-routes'])
        rows=[row for rank in range(8) for row in load(folder/f'rank{rank}.json')['records']]
        passed=len(rows)==64 and all(row['passed'] and row['gradient_ready_scope']==scope and row['lifetime_entries']==entries for row in rows)
        if route=='sparse_remote':passed &= all(row['empty_home_experts']==32 for row in rows)
        if scope=='bank_write':passed &= all(row['lifetime_stats']['ready_bank_writes']>0 and row['lifetime_stats']['ready_bank_waits']>0 for row in rows)
        state['runs'].append(dict(tag=tag,phase='math',passed=bool(passed),scope=scope,route=route,entries=entries,
            records=len(rows),max_relative_l2={key:max(row['comparisons'][key]['relative_l2'] for row in rows) for key in ('output','dx','dp','dw1','dw2')}));save()
        if not passed:raise RuntimeError('Independent NPU math failed '+tag)
    state['stage']='model';save()
    for scope in ('call_stream','bank_write'):
        tag='grad-ready-model-'+scope+'-v1'
        folder,status=gate(tag,['--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
            '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10','--planner','moonep',
            '--temporal-pattern','same','--temporal-policy','fixed_native','--worker-variant',WORKER,'--grad-sync',
            '--hold-fsdp-forward','--temporal-prefix-reuse','--lifetime-mode','b3','--lifetime-raw-schedule','preserve',
            '--gradient-ready-scope',scope,'--dump-grad-step0','--finite-diagnostics',
            '--route-bank',remote.REMOTE+'/stack/small-routes'])
        passed=True
        for rank in range(8):
            updates=load(folder/f'updates-rank{rank}.json')
            steps=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
            passed &= len(updates)==len(steps)==3 and all(row['changed'] for row in updates)
            passed &= all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in steps)
            config=load(folder/f'lifetime-config-rank{rank}.json')
            passed &= config['gradient_ready_scope']==scope and config['raw_schedule']=='preserve' and config['worker_capability']==2
            passed &= load(folder/f'lifetime-audit-rank{rank}.json')['pending']==0
            for step in range(3):
                finite=load(folder/f'gradient-finite-step{step}-rank{rank}.json')
                passed &= bool(finite) and all(row.get('finite',True) for row in finite)
        result_path=remote.REMOTE+'/stack/evidence/'+tag+'-comparison.json'
        command=['ssh',remote.HOST,'docker','exec',remote.CONTAINER,'/usr/local/python3.12.13/bin/python3',
            remote.REMOTE+'/stack/tools/compare_temporal_gradients.py',
            '/home/z00956592/megaflux/step-lifetime-raw-control-20261116/stack/evidence/raw-model-preserve-v1',
            remote.REMOTE+'/stack/evidence/'+tag,'--out',result_path]
        with (ROOT/'diagnostics'/(tag+'-comparison.log')).open('wb') as handle:
            comparison=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
        local=ROOT/'diagnostics'/(tag+'-comparison.json')
        subprocess.run(['scp',remote.HOST+':'+result_path,str(local)],check=True)
        passed &= comparison.returncode==0 and load(local)['passed']
        state['runs'].append(dict(tag=tag,phase='model',scope=scope,passed=bool(passed),peak_driver_MiB=status['peak_driver_MiB']));save()
        if not passed:raise RuntimeError('Full small-model qualification failed '+tag)
    state['stage']='pilot';save()
    for scope in ('call_stream','bank_write'):
        tag='grad-ready-pilot-'+scope+'-v1'
        folder,status=gate(tag,['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
            '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10','--planner','native',
            '--temporal-pattern','same','--temporal-policy','fixed_native','--worker-variant',WORKER,'--grad-sync',
            '--hold-fsdp-forward','--temporal-prefix-reuse','--lifetime-mode','b3','--lifetime-raw-schedule','preserve',
            '--gradient-ready-scope',scope,'--temporal-scheduler','ascendc',
            '--temporal-scheduler-library',remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so'])
        all_steps=[[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
        passed=all(len(rows)==25 and all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in rows) for rows in all_steps)
        for rank in range(8):
            config=load(folder/f'lifetime-config-rank{rank}.json');audit=load(folder/f'lifetime-audit-rank{rank}.json')
            passed &= config['gradient_ready_scope']==scope and audit['pending']==0
        times=[max(rows[step]['milliseconds'] for rows in all_steps) for step in range(5,25)]
        state['runs'].append(dict(tag=tag,phase='pilot',scope=scope,passed=bool(passed),median_step_ms=statistics.median(times),
            measured_rank_max_step_ms=times,peak_driver_MiB=status['peak_driver_MiB']));save()
        if not passed:raise RuntimeError('Formal pair verification failed '+tag)
    pair=[row for row in state['runs'] if row['phase']=='pilot']
    state.update(stage='complete',passed=True,reduction_pct=100*(1-pair[1]['median_step_ms']/pair[0]['median_step_ms']))
    state.pop('current',None);save()
except BaseException as error:
    state.update(stage='failed',error=str(error));save();raise
