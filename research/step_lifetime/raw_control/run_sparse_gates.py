"""Independent math qualification of BOTH schedule modes in one ABI2 binary."""
from pathlib import Path
import hashlib,json,subprocess,sys
import remote
root=Path(__file__).resolve().parent
variant='ep8-cap64-lifetime-raw-control-v2'
state_path=root/'raw-sparse-gates-v1.json'
if state_path.exists():raise RuntimeError('Keep gate registration; choose a versioned new script/tag after any source fix')
state=dict(stage='math',runs=[],performance=False)
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for schedule in ('legacy','preserve'):
  tag='raw-control-sparse-'+schedule+'-v1'
  try:
   remote.gate(tag,['--mode','lifetime_math','--worker-variant',variant,'--lifetime-mode','all',
       '--lifetime-raw-schedule',schedule,'--route-case','sparse_remote','--route-bank',remote.REMOTE+'/stack/small-routes'],metadata_only=True)
  except SystemExit as done:code=done.code
  folder=root/'evidence'/tag
  status=json.loads((folder/'status.json').read_text())
  records=[]
  for rank in range(8):
   path=folder/f'rank{rank}.json'
   if path.exists():records.extend(json.loads(path.read_text())['records'])
  active_hits=max((r['lifetime_stats']['weight_hit_bytes'] for r in records),default=0)
  gradient_bytes=max((r['lifetime_stats']['gradient_remote_bytes'] for r in records),default=0)
  sources=json.loads((folder/'source-identity.json').read_text())
  checked=('tools/gate_lifetime_math.py','tools/lifetime_device.py','tools/step_lifetime.py',
    'source/native-balanced/kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h',
    'source/native-balanced/kernels/mega_moe_bf16/megaflux_worker_bf16.cpp',
    'source/native-balanced/ascend_deepep/megaflux_worker_pipeline.py')
  matches=all(hashlib.sha256((root/'stack'/name).read_bytes()).hexdigest()==sources[name] for name in checked)
  passed=(code==0 and all(status.get(k) for k in ('qualified','source_unchanged','exclusive_devices'))
      and status['peak_driver_MiB']<=90*1024 and len(records)==64 and all(r['passed'] for r in records)
      and active_hits>0 and gradient_bytes>0 and matches and all(r['route_case']=='sparse_remote' and r['raw_schedule']==schedule and r['empty_home_experts']==32 for r in records))
  item=dict(tag=tag,schedule=schedule,passed=bool(passed),exit=code,records=len(records),
    active_cache_hit_bytes=active_hits,gradient_remote_bytes=gradient_bytes,
    max_relative_l2={name:max((r['comparisons'][name]['relative_l2'] for r in records),default=None)
      for name in ('output','dx','dp','dw1','dw2')},
    runtime_sources={name:sources[name] for name in checked},peak_driver_MiB=status['peak_driver_MiB'])
  state['runs'].append(item);save()
  if not passed:raise RuntimeError('Independent raw schedule math gate failed: '+schedule)
 state.update(stage='complete',passed=True);save()
except BaseException as error:
 state.update(stage='failed',error=str(error));save();raise
