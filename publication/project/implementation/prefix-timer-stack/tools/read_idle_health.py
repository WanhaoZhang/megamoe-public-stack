"""One between-job idle observation; no reset, process stop or cleanup."""
import datetime
import fcntl
import json
import time
from controller import TASK,idle,hbm_sample

tag='idle-health-after-init-errors-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
with (TASK/'evidence/device-jobs.lock').open('a') as lock:
    print('WAITING_BETWEEN_TASK_JOBS',tag,flush=True);fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
    admission_wait=[]
    for attempt in range(30):
        try:first=idle();break
        except RuntimeError as error:
            admission_wait.append(dict(attempt=attempt,reason=str(error)));time.sleep(1)
    else:
        (TASK/f'evidence/{tag}.json').write_text(json.dumps(dict(no_reset=True,no_cleanup=True,admission_wait=admission_wait,admitted=False),indent=2))
        raise RuntimeError('idle admission did not settle within30 checks')
    samples=[first];print('IDLE_HEALTH_LOCK_ACQUIRED',flush=True)
    for index in range(7):
        time.sleep(5);sample=hbm_sample();samples.append(sample)
        print(json.dumps(dict(sample=index,no_process_cards=sample['no_process_cards'],hbm_mib=[x['used_MiB'] for x in sample['cards']])),flush=True)
    result=dict(no_reset=True,no_cleanup=True,no_training_process_interrupted=True,samples=samples,admission_wait=admission_wait,
        all_idle=all(x['no_process_cards']==8 and all(c['utilization']==0 for c in x['cards']) for x in samples))
    (TASK/f'evidence/{tag}.json').write_text(json.dumps(result,indent=2))
    if not result['all_idle']:raise RuntimeError('unexpected device owner/activity during idle health observation')
print('IDLE_HEALTH_COMPLETE',tag,flush=True)
