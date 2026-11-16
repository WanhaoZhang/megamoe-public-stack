import json,pathlib,time
from controller import TASK,run_job,idle

def wait_path(path):
    for attempt in range(1800):
        if path.exists():return
        time.sleep(1)
    raise RuntimeError('Missing prerequisite '+str(path))

prerequisite=TASK/'evidence'/'gate-full-moonep-L2-budget'/'status.json'
wait_path(prerequisite)
status=json.loads(prerequisite.read_text())
assert status['qualified'] and status['peak_driver_MiB']<=90*1024,status
for backend in ['balanced','shared']:
    tag=f'gate-full-{backend}-L2-budget-hot20-r2'
    status=run_job(backend,'train',tag,['--layers','2','--maxvio','20','--steps','8','--diagnose-memory'])
    assert status['qualified'] and status['peak_driver_MiB']<=90*1024,(tag,status)
print('ALL_FULL_MODEL_GATES_PASS',flush=True)
