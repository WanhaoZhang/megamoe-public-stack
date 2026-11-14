"""Task-local read-only admission checks; foreign jobs are never signalled."""
from datetime import datetime,timezone
import json
import subprocess
import time
from controller import TASK,idle


def foreign_coordinators():
    result=subprocess.run(['docker','top','agy-live-a5','-eo','pid,comm'],capture_output=True,text=True)
    absent=result.returncode!=0 and 'No such container' in result.stderr
    if absent:return []
    if result.returncode:
        inspect=subprocess.run(['docker','inspect','--format','{{json .State}}','agy-live-a5'],capture_output=True,text=True)
        if inspect.returncode==0 and json.loads(inspect.stdout).get('Running') is False:return []
        raise RuntimeError('foreign coordinator inventory failed')
    commands=[x.split(None,1)[1].strip() for x in result.stdout.splitlines()[1:] if len(x.split(None,1))==2]
    return [x for x in commands if x not in ('sleep','bash','sh','tini','docker-init')]


def wait_if_occupied(label,quiet_seconds=600):
    """Proceed immediately if idle; after a collision require a full quiet gap."""
    began=time.time();quiet_start=None;was_blocked=False;last_reason=None
    path=TASK/'evidence/resource-admission.jsonl'
    while True:
        if (TASK/'evidence/USER-CLOSEOUT-REQUEST.json').exists():
            raise RuntimeError('task closed at user request; automatic device admission disabled')
        try:
            active=foreign_coordinators()
            if active:raise RuntimeError('active foreign coordinators: '+','.join(active))
            sample=idle();reason=None
        except (RuntimeError,subprocess.TimeoutExpired) as exc:
            reason=str(exc);sample=None
        now=time.time()
        if reason is None and not was_blocked:return
        if reason is not None:
            was_blocked=True;quiet_start=None
        elif quiet_start is None:quiet_start=now
        done=reason is None and now-quiet_start>=quiet_seconds
        if reason!=last_reason or done:
            row=dict(label=label,time_utc=datetime.now(timezone.utc).isoformat(),reason=reason,
                action='admit_after_sustained_idle' if done else 'read_only_wait',elapsed=now-began,
                quiet_seconds=0 if quiet_start is None else now-quiet_start,driver=sample,
                no_foreign_task_interruption=True)
            with path.open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True);last_reason=reason
        if done:return
        time.sleep(10)
