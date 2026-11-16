"""Start one task-owned supervisor with durable files, independent of SSH I/O."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
from controller import TASK

name='new-host-supervisor-v1'
log=TASK/f'evidence/{name}.log';receipt=TASK/f'evidence/{name}.json'
assert not log.exists() and not receipt.exists(),'refuse to duplicate an existing supervisor'
command=[sys.executable,str(TASK/'tools/resume_new_host_qualification.py')]
with log.open('wb') as output:
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
record=dict(time_utc=datetime.now(timezone.utc).isoformat(),pid=process.pid,command=command,
    stdout=str(log),task_owned_only=True,ssh_session_not_required=True,not_an_automation=True)
receipt.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
