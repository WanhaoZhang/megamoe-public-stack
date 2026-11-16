"""Finite qualified intervention; incomplete jobs are never silently restarted."""
from pathlib import Path
import argparse,hashlib,json,math,statistics,subprocess
import remote
root=Path(__file__).resolve().parent
candidate='ep8-cap64-lifetime-priority-cache-v1'
reference='ep8-cap64-lifetime-raw-control-v2'
p=argparse.ArgumentParser();p.add_argument('phase',choices=('math','model','pilot'));a=p.parse_args()
state_path=root/(a.phase+'-v1.json')
if state_path.exists():raise RuntimeError('Registered cohort exists; inspect its live/terminal state before making any new version')
state=dict(stage=a.phase,runs=[],new_planner=False)
def save():state_path.write_text(json.dumps(state,indent=2)+'\n',newline='\n')
def gate(tag,args):
 try:remote.gate(tag,args,metadata_only=True)
 except SystemExit as done:code=done.code
 folder=root/'evidence'/tag;status=json.loads((folder/'status.json').read_text())
 passed=code==0 and all(status.get(k) for k in ('qualified','source_unchanged','exclusive_devices')) and status['peak_driver_MiB']<=90*1024
 return folder,status,bool(passed)
def require(phase):
 proof=json.loads((root/(phase+'-v1.json')).read_text())
 if not proof.get('passed'):raise RuntimeError('Prerequisite incomplete: '+phase)
 for run in proof['runs']:
  if 'runtime_sources' not in run:continue
  for name,digest in run['runtime_sources'].items():
   if hashlib.sha256((root/'stack'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Stale correctness proof: '+name)
 return proof
core=('tools/lifetime_device.py','tools/step_lifetime.py','tools/train.py','tools/grouped_experts.py',
 'tools/temporal_model.py','tools/temporal_prefix_reuse.py','source/native-balanced/ascend_deepep/fsdp2_bridge.py',
 'source/native-balanced/ascend_deepep/megaflux_worker_pipeline.py',
 'source/native-balanced/kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h')
def runtime(folder):
 source=json.loads((folder/'source-identity.json').read_text())
 result={name:source[name] for name in core}
 if not all(hashlib.sha256((root/'stack'/name).read_bytes()).hexdigest()==digest for name,digest in result.items()):raise RuntimeError('Local runtime differs from actual qualification')
 return result
save()
try:
 if a.phase=='math':
  for route_case in ('bank','sparse_remote'):
   tag='priority-math-'+route_case+'-v1'
   folder,status,passed=gate(tag,['--mode','lifetime_math','--worker-variant',candidate,
     '--lifetime-mode','all','--lifetime-raw-schedule','preserve','--route-case',route_case,
     '--route-bank',remote.REMOTE+'/stack/small-routes'])
   rows=[]
   for rank in range(8):
    path=folder/f'rank{rank}.json'
    if path.exists():rows.extend(json.loads(path.read_text())['records'])
   passed &= len(rows)==64 and all(row['passed'] and row['raw_schedule']=='preserve' and row['route_case']==route_case for row in rows)
   if route_case=='sparse_remote':passed &= all(row['empty_home_experts']==32 for row in rows)
   passed &= max((r['lifetime_stats']['gradient_remote_bytes'] for r in rows),default=0)>0
   state['runs'].append(dict(tag=tag,passed=bool(passed),records=len(rows),runtime_sources=runtime(folder),
     max_relative_l2={key:max((row['comparisons'][key]['relative_l2'] for row in rows),default=None) for key in ('output','dx','dp','dw1','dw2')}));save()
   if not passed:raise RuntimeError('Priority math failed: '+route_case)
 elif a.phase=='model':
  require('math')
  tag='priority-model-v1'
  args=['--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
    '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
    '--planner','moonep','--temporal-pattern','same','--temporal-policy','fixed_native',
    '--worker-variant',candidate,'--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse',
    '--lifetime-mode','b1','--lifetime-raw-schedule','preserve','--dump-grad-step0','--finite-diagnostics',
    '--route-bank',remote.REMOTE+'/stack/small-routes']
  folder,status,passed=gate(tag,args)
  for rank in range(8):
   updates=json.loads((folder/f'updates-rank{rank}.json').read_text())
   steps=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
   passed &= len(updates)==len(steps)==3 and all(row['changed'] for row in updates)
   passed &= all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in steps)
   for step in range(3):
    finite=json.loads((folder/f'gradient-finite-step{step}-rank{rank}.json').read_text())
    passed &= all(row.get('finite',True) for row in finite)
   libraries=json.loads((folder/f'libraries-rank{rank}.json').read_text())
   expected=set(json.loads((root/'stack/configs/lifetime-runtime-freeze.json').read_text())['binaries'].values())
   loaded=[value['sha256'] for name,value in libraries['libraries'].items() if '/worker-variants/'+candidate+'/' in name]
   passed &= bool(loaded) and set(loaded)<=expected and len(libraries['shmem'])==1
  result_path=remote.REMOTE+'/stack/evidence/priority-model-comparison-v1.json'
  command=['ssh',remote.HOST,'docker','exec',remote.CONTAINER,'/usr/local/python3.12.13/bin/python3',
    remote.REMOTE+'/stack/tools/compare_temporal_gradients.py',
    '/home/z00956592/megaflux/step-lifetime-raw-control-20261116/stack/evidence/raw-model-preserve-v1',
    remote.REMOTE+'/stack/evidence/'+tag,'--out',result_path]
  with (root/'diagnostics/model-comparison-driver.log').open('wb') as log:
   comparison=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  subprocess.run(['scp',remote.HOST+':'+result_path,str(root/'diagnostics/model-comparison.json')],check=True)
  passed &= comparison.returncode==0 and json.loads((root/'diagnostics/model-comparison.json').read_text())['passed']
  state['runs'].append(dict(tag=tag,passed=bool(passed),runtime_sources=runtime(folder)));save()
  if not passed:raise RuntimeError('Three-update full model qualification failed')
 else:
  require('math');require('model')
  if not json.loads((root/'diagnostics/build-intervention-audit.json').read_text()).get('passed'):
   raise RuntimeError('Compiled source/options intervention audit is required before timing')
  for name,variant in (('reference',reference),('cached_keys',candidate)):
   tag='priority-pilot-'+name+'-v1'
   args=['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
     '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
     '--planner','native','--temporal-pattern','same','--temporal-policy','fixed_native',
     '--worker-variant',variant,'--grad-sync','--hold-fsdp-forward','--temporal-prefix-reuse',
     '--lifetime-mode','b1','--lifetime-raw-schedule','preserve','--temporal-scheduler','ascendc',
     '--temporal-scheduler-library',remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
   folder,status,passed=gate(tag,args)
   records=[[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
   passed &= all(len(rows)==25 for rows in records)
   times=[max(rows[step]['milliseconds'] for rows in records) for step in range(5,25)]
   state['runs'].append(dict(tag=tag,name=name,variant=variant,passed=bool(passed),
      median_step_ms=statistics.median(times),measured_rank_max_step_ms=times,peak_driver_MiB=status['peak_driver_MiB']));save()
   if not passed:raise RuntimeError('Unqualified pilot: '+name)
  state['reduction_pct']=100*(1-state['runs'][1]['median_step_ms']/state['runs'][0]['median_step_ms'])
  state['independent_processes_per_point']=1
 state.update(stage='complete',passed=True);save()
except BaseException as error:
 state.update(stage='failed',error=str(error));save();raise
