"""One matched TimerOFF pair to test the executor correction, not a new planner."""
from pathlib import Path
import json,statistics
import remote
root=Path(__file__).resolve().parent
state_path=root/'schedule-pilot-v1.json'
if state_path.exists():raise RuntimeError('Keep previous pilot registration')
for name in ('raw-gates-v1.json','raw-sparse-gates-v1.json','raw-model-gates-v1.json'):
 if not json.loads((root/name).read_text()).get('passed'):raise RuntimeError('Missing correctness gate: '+name)
state=dict(stage='screening',runs=[],independent_repeats_per_point=1,new_planner=False,
           scope='same ABI2 SO, only raw compute-schedule flag differs; common executor correction, not policy gain')
def save():state_path.write_text(json.dumps(state,indent=2)+'\n')
save()
try:
 for schedule in ('legacy','preserve'):
  tag='raw-schedule-'+schedule+'-same-v1'
  args=['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
   '--microbatches','4','--forward-window','2','--recv-capacity','262144','--maxvio','10',
   '--planner','native','--temporal-pattern','same','--temporal-policy','fixed_native',
   '--worker-variant','ep8-cap64-lifetime-raw-control-v2','--grad-sync','--hold-fsdp-forward',
   '--temporal-prefix-reuse','--lifetime-mode','b1','--lifetime-raw-schedule',schedule,
   '--temporal-scheduler','ascendc','--temporal-scheduler-library',
   remote.REMOTE+'/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so']
  try:remote.gate(tag,args,metadata_only=True)
  except SystemExit as done:code=done.code
  folder=root/'evidence'/tag;status=json.loads((folder/'status.json').read_text())
  rank_steps=[[json.loads(line) for line in (folder/f'steps-rank{rank}.jsonl').read_text().splitlines()] for rank in range(8)]
  passed=code==0 and all(status.get(k) for k in ('qualified','source_unchanged','exclusive_devices'))
  passed &= status['peak_driver_MiB']<=90*1024 and all(len(rows)==25 for rows in rank_steps)
  times=[max(rows[step]['milliseconds'] for rows in rank_steps) for step in range(5,25)]
  state['runs'].append(dict(tag=tag,schedule=schedule,passed=bool(passed),median_step_ms=statistics.median(times),
      measured_rank_max_step_ms=times,peak_driver_MiB=status['peak_driver_MiB']));save()
  if not passed:raise RuntimeError('Unqualified schedule pilot: '+schedule)
 reference,candidate=state['runs']
 state.update(stage='complete',passed=True,
    reduction_vs_legacy_pct=100*(1-candidate['median_step_ms']/reference['median_step_ms']),
    conclusion='single-process-pair screening only; no stable gain or stronger-baseline win established')
 save()
except BaseException as error:
 state.update(stage='failed',error=str(error));save();raise
