"""Record a real admission failure and cancel only unstarted old-style jobs."""
from pathlib import Path
import json
root=Path(__file__).resolve().parent
path=root/'resident-v3-confirmation.json';state=json.loads(path.read_text())
if state['stage']!='stopped_at_gate' or state.get('running_tag')!='resident-v3-confirm-native_resident-r2':
    raise RuntimeError('Unexpected state; do not change a live controller')
log=(root/'resident-v3-confirm-native_resident-r2-remote.log').read_text()
if 'Driver reports GPU processes before admission' not in log:
    raise RuntimeError('Admission classification requires exact evidence')
snapshot=root/'resident-v3-confirmation-before-upstream-cancel.json'
if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
state['pilot_jobs'][-1]['classification']='foreign_device_job_before_final_admission; training_not_launched'
attempted={job['tag'] for job in state['pilot_jobs']}
state['cancelled_before_admission']=[dict(tag=f'resident-v3-confirm-{label}-r{repeat}',
    reason='Prioritize canonical upstream MaxUtil conformance audit; no data-dependent route changes')
    for repeat,labels in state['preregistered_order'].items() for label in labels
    if f'resident-v3-confirm-{label}-r{repeat}' not in attempted]
state.update(stop_reason='Automatic admission guard stopped at external NPU usage; after upstream audit, cancel four unstarted old-tie confirmations',
    controller_processes_signalled=0,device_jobs_signalled=0,original_failure_retained=True)
path.write_text(json.dumps(state,indent=2)+'\n')
print(json.dumps(dict(cancelled_unstarted=len(state['cancelled_before_admission']),signalled=0)))
