"""One explicitly linked infrastructure replacement; no successful slow retry."""
import json
from controller import TASK, run_job, source_identity
from campaign_one import foreign_observation, write_json, initialize
from campaign_records import rows_for, init_failure, verified_contention, resolved_existing

initialize('pod6-1-root')
original=rows_for(TASK,['primary-existing'])
assert len(original)==90,'finish all original scheduled cells first'
recoveries=rows_for(TASK,['primary-existing-recovery'])
needed=[]
for row in original:
    contaminated=verified_contention(TASK,row)
    if row['state']=='pass' and not contaminated:
        assert not any(x.get('retry_of')==row['tag'] and x['qualified'] for x in recoveries)
        continue
    assert contaminated or init_failure(TASK,row),'not an infrastructure-replaceable failure: '+row['tag']
    prior=[x for x in recoveries if x.get('retry_of')==row['tag']]
    good=[x for x in prior if x['state']=='pass' and not verified_contention(TASK,x)]
    assert len(good)<=1
    if not good:needed.append((row,prior,contaminated))
if not needed:
    resolved=resolved_existing(TASK)
    write_json(TASK/'evidence/RESOLVED-primary-existing-CELLS.json',dict(planned_cells=90,
        qualified_cells=len(resolved),original_attempts=90,recovery_attempts=len(recoveries),
        all_failures_retained=True,no_valid_successful_or_slow_run_replaced=True,tags=[x['tag'] for x in resolved]))
    print(json.dumps(dict(status='complete',resolved_cells=90,recovery_attempts=len(recoveries))),flush=True)
    raise SystemExit(0)
row,prior,contaminated=needed[0]
assert len(prior)<3,'three explicit infrastructure retries exhausted'
assert not prior or init_failure(TASK,prior[-1]) or verified_contention(TASK,prior[-1]),'retry failure needs investigation'
attempt=len(prior)+1
tag=row['tag']+f'-contention-retry{attempt}' if contaminated else row['tag']+f'-init-retry{attempt}'
reference=json.loads((TASK/f'evidence/{row["tag"]}/source-identity.json').read_text())
assert source_identity()==reference,'recovery must use exact original measurement sources'
command=row['command'];extra=command[command.index('tools/launch.sh')+4:]
directory=TASK/'evidence'/tag
if directory.exists():
    status=json.loads((directory/'status.json').read_text())
    assert json.loads((directory/'source-identity.json').read_text())==reference
    assert status['command'][-len(extra):]==extra
else:
    status=run_job('balanced','train',tag,extra,1800)
state='pass' if status['qualified'] else 'failed'
text=(directory/'stdout.log').read_text(errors='replace')
if not status['qualified']:
    if 'out of memory' in text.lower():state='oom'
    elif status['exit'] in (124,137):state='timeout'
    elif 'device plan failed' in text:state='invalid_plan'
if status['peak_driver_MiB']>90*1024:state='memory_gate_exceeded'
retry=dict(tag=tag,planner=row['planner'],maxvio=row['maxvio'],repeat=row['repeat'],state=state,
    retry_of=row['tag'],attempt=attempt+1,retry_reason='verified foreign-GPU activity' if contaminated else 'zero-step HCCL initializer failure',**status)
observation=foreign_observation(retry)
if observation:retry['foreign_gpu_observation']=observation
recoveries.append(retry)
write_json(TASK/'evidence/campaign-primary-existing-recovery.json',recoveries)
clean=sum(x['state']=='pass' and not verified_contention(TASK,x) for x in recoveries)
print(json.dumps(dict(status=state,linked_clean_recoveries=clean,remaining_before_attempt=len(needed),last=retry)),flush=True)
if state!='pass' or observation:raise RuntimeError('recovery needs review; next cell not queued')
