"""Three Adam updates and complete step0 gradients; never a performance score."""
from pathlib import Path
import hashlib,json,math,subprocess
import remote
root=Path(__file__).resolve().parent
path=root/'raw-model-gates-v1.json'
if path.exists():raise RuntimeError('Keep prior registration and use new versioned tags')
for name in ('raw-gates-v1.json','raw-sparse-gates-v1.json'):
 proof=json.loads((root/name).read_text())
 if not proof.get('passed'):raise RuntimeError('Both independent math qualifications must pass first')
 for row in proof['runs']:
  for source,digest in row['runtime_sources'].items():
   if source=='tools/gate_lifetime_math.py':continue  # immutable oracle snapshots retained per gate
   if hashlib.sha256((root/'stack'/source).read_bytes()).hexdigest()!=digest:
    raise RuntimeError('Independent math proof is stale for runtime source: '+source)
state=dict(stage='model',runs=[],performance=False)
def save():path.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for schedule in ('legacy','preserve'):
  tag='raw-model-'+schedule+'-v1'
  args=['--layers','2','--steps','3','--tokens','256','--hidden','512','--ffn','512',
    '--microbatches','4','--forward-window','2','--recv-capacity','4096','--maxvio','10',
    '--planner','moonep','--temporal-pattern','same','--temporal-policy','fixed_native',
    '--worker-variant','ep8-cap64-lifetime-raw-control-v2','--grad-sync','--hold-fsdp-forward',
    '--temporal-prefix-reuse','--lifetime-mode','b1','--lifetime-raw-schedule',schedule,
    '--route-bank',remote.REMOTE+'/stack/small-routes','--dump-grad-step0','--finite-diagnostics']
  try:remote.gate(tag,args,metadata_only=True)
  except SystemExit as done:code=done.code
  folder=root/'evidence'/tag;status=json.loads((folder/'status.json').read_text())
  passed=code==0 and all(status.get(k) for k in ('qualified','source_unchanged','exclusive_devices')) and status['peak_driver_MiB']<=90*1024
  for rank in range(8):
   updates=json.loads((folder/f'updates-rank{rank}.json').read_text())
   steps=[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()]
   passed &= len(updates)==len(steps)==3 and all(row['changed'] for row in updates)
   passed &= all(math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) for row in steps)
   passed &= all(any('.mlp.experts.' in name for name in row['changed']) for row in updates)
   for step in range(3):
    finite=json.loads((folder/f'gradient-finite-step{step}-rank{rank}.json').read_text())
    passed &= all(row.get('finite',True) for row in finite)
  state['runs'].append(dict(tag=tag,schedule=schedule,exit=code,passed=bool(passed)));save()
  if not passed:raise RuntimeError('Three-update gate failed: '+schedule)
 output=remote.REMOTE+'/stack/evidence/raw-model-comparison-v1.json'
 command=['ssh',remote.HOST,'docker','exec',remote.CONTAINER,'/usr/local/python3.12.13/bin/python3',
  remote.REMOTE+'/stack/tools/compare_temporal_gradients.py',
  remote.REMOTE+'/stack/evidence/raw-model-legacy-v1',remote.REMOTE+'/stack/evidence/raw-model-preserve-v1',
  '--out',output]
 with (root/'diagnostics/raw-model-comparison-v1.log').open('wb') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 subprocess.run(['scp',remote.HOST+':'+output,str(root/'diagnostics/raw-model-comparison-v1.json')],check=True)
 compared=json.loads((root/'diagnostics/raw-model-comparison-v1.json').read_text())
 if result.returncode or not compared['passed']:raise RuntimeError('Full-model per-parameter FP32 gradient gate failed')
 state.update(stage='complete',passed=True,comparison='diagnostics/raw-model-comparison-v1.json');save()
except BaseException as error:
 state.update(stage='failed',error=str(error));save();raise
