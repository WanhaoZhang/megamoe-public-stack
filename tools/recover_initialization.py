"""Recover only explicitly diagnosed pre-training HCCL initializer failures."""
import json
from controller import TASK,run_job
from campaign_records import rows_for,init_failure,resolved_campaign,verified_contention
from campaign_one import foreign_observation,write_json
from resource_admission import wait_if_occupied

def recover_campaign(name,expected):
    original=rows_for(TASK,[name])
    if len(original)!=expected:raise RuntimeError('finish the original scheduled matrix first')
    recoveries=rows_for(TASK,[name+'-recovery'])
    results=TASK/f'evidence/campaign-{name}-recovery.json'
    if not results.exists():results.write_text('[]')
    for row in original:
        contention=verified_contention(TASK,row)
        if row['state']=='pass' and not contention:continue
        if not contention and not init_failure(TASK,row):raise RuntimeError('training/policy failure needs investigation, not automatic retry')
        prior=[x for x in recoveries if x.get('retry_of')==row['tag']]
        if any(x['state']=='pass' and not verified_contention(TASK,x) for x in prior):continue
        for attempt in range(len(prior)+1,4):
            if attempt>1 and not init_failure(TASK,prior[-1]) and not verified_contention(TASK,prior[-1]):raise RuntimeError('retry failed beyond initialization')
            tag=row['tag']+(f'-contention-retry{attempt}' if contention else f'-init-retry{attempt}')
            command=row['command'];extra=command[command.index('tools/launch.sh')+4:]
            wait_if_occupied(tag)
            directory=TASK/'evidence'/tag
            if directory.exists():
                if not (directory/'status.json').exists():raise RuntimeError('partial recovery requires review: '+tag)
                status=json.loads((directory/'status.json').read_text())
                if status['command'][-len(extra):]!=extra:raise RuntimeError('recovery receipt request mismatch')
            else:status=run_job('balanced','train',tag,extra)
            retry=dict(tag=tag,planner=row['planner'],maxvio=row['maxvio'],repeat=row['repeat'],state='pass' if status['qualified'] else 'failed',
                retry_of=row['tag'],attempt=attempt+1,retry_reason='verified foreign-GPU contention window; all original data retained' if contention else 'pre-training HCCL Broadcast/RunAicpuIndOpThreadInit exception; zero training steps',**status)
            observation=foreign_observation(retry)
            if observation:retry['foreign_gpu_observation']=observation
            recoveries.append(retry);prior.append(retry)
            write_json(results,recoveries)
            if status['peak_driver_MiB']>90*1024:raise RuntimeError('recovery exceeded common memory gate')
            if observation:raise RuntimeError('foreign activity in linked recovery; retain and wait')
            if status['qualified']:break
        else:raise RuntimeError('three recovery attempts failed; preserve hardware/init blocking status')
    resolved=resolved_campaign(TASK,name,expected)
    (TASK/f'evidence/RESOLVED-{name}-CELLS.json').write_text(json.dumps(dict(planned_cells=expected,qualified_cells=len(resolved),
        original_attempts=len(original),recovery_attempts=len(recoveries),all_failures_retained=True,
        no_successful_run_replaced=True,tags=[r['tag'] for r in resolved]),indent=2))
    print('INITIALIZATION_RECOVERY_COMPLETE',name,len(resolved),len(recoveries),flush=True)
    return resolved

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--campaign',default='primary-existing');p.add_argument('--expected',type=int,default=90);a=p.parse_args()
    recover_campaign(a.campaign,a.expected)
