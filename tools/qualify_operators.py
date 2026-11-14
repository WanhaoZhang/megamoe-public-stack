import json,pathlib,time
from controller import TASK,run_job,idle

def wait_idle():
    for attempt in range(90):
        try:
            idle()
            return
        except RuntimeError:
            time.sleep(2)
    raise RuntimeError('Task GPU teardown did not complete')

def checked(backend,mode,tag,arguments,capacity=64):
    wait_idle()
    status=run_job(backend,mode,tag,arguments,1800,None,capacity)
    if not status['qualified']:
        raise RuntimeError('Gate failed: '+tag)
    return status

pending=TASK/'evidence'/'memory-moon-L2-v32'/'status.json'
for attempt in range(900):
    if pending.exists():
        break
    time.sleep(1)
else:
    raise RuntimeError('Prior memory probe did not finalize')

for name, cap in [('balanced',64),('shared',64),('balanced32',32),('shared32',32)]:
    marker=TASK/'evidence'/f'build-aligned-{name}-{cap}-r3.exit'
    for attempt in range(900):
        if marker.exists():
            assert marker.read_text().strip()=='0',str(marker)
            break
        time.sleep(1)
    else:raise RuntimeError('Worker build did not complete')

jobs=[
    ('balanced','validate','gate-balanced64-aligned-r4',[] ,64),
    ('shared','validate','gate-shared64-aligned-r4',[],64),
    ('balanced','validate','gate-balanced32-aligned-r4',['--experts','128'],32),
    ('shared','validate','gate-shared32-aligned-r4',['--experts','128'],32),
    ('moonep','validate','gate-moonep-async-aligned-r4',[],64),
    ('balanced','pool','gate-pool-balanced64',['--experts','256','--slots','32','--tokens','128','--ffn','256','--limit','10','--restore','panel','--checkpoint','--parameter-fp32','--require-replicas'],64),
    ('shared','pool','gate-pool-shared64',['--experts','256','--slots','32','--tokens','128','--ffn','256','--limit','10','--restore','panel','--checkpoint','--parameter-fp32','--require-replicas'],64),
]
for backend,mode,tag,arguments,capacity in jobs:
    checked(backend,mode,tag,arguments,capacity)
print('ALL_OPERATOR_GATES_PASS',flush=True)
