"""Full L2/EP8 three-update memory gate with only the authorized common fallback."""
import json
import re
from controller import TASK,run_job

policies=['moonep','native','ultra_ep','echo','eplb_fixed']
attempts=[]
for capacity in (163840,147456):
    rows=[];memory_failure=False
    for planner in policies:
        tag=f'gate-full-cap{capacity}-{planner}-v1'
        status=run_job('balanced','train',tag,['--planner',planner,'--recv-capacity',str(capacity),'--maxvio','20','--steps','3'],1200)
        text=(TASK/'evidence'/tag/'stdout.log').read_text(errors='replace')
        memory_failure=(status['peak_driver_MiB']>90*1024 or 'out of memory' in text.lower())
        rows.append(dict(planner=planner,capacity=capacity,memory_failure=memory_failure,**status))
        (TASK/'evidence'/f'capacity-gate-{capacity}.json').write_text(json.dumps(rows,indent=2))
        if memory_failure:break
        if not status['qualified']:
            raise RuntimeError(f'Non-memory failure in full-model gate: {tag}; investigate before any timing')
    attempts.append(dict(capacity=capacity,rows=rows,memory_failure=memory_failure))
    (TASK/'evidence/capacity-attempts.json').write_text(json.dumps(attempts,indent=2))
    if not memory_failure:
        record=dict(status='pass',capacity=capacity,threshold_GiB=90,policies=policies,
                    layers=2,shape=dict(ep=8,E=256,K=8,H=7168,F=2048,S=16384),
                    model_shape_changed=False,attempts=attempts)
        (TASK/'configs/selected-capacity.json').write_text(json.dumps(record,indent=2))
        print('COMMON_CAPACITY_GATE_PASS',capacity,flush=True)
        break
else:
    (TASK/'configs/selected-capacity.json').write_text(json.dumps(dict(status='memory_blocked',attempts=attempts),indent=2))
    raise RuntimeError('Both common capacities exceed memory limits; formal matrix stopped without cropping model')
