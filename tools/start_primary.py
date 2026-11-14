"""Start the authorized90-run baseline matrix only after complete qualification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from controller import TASK,source_identity

selection=TASK/'configs/selected-capacity.json'
deadline=time.time()+5400
while not selection.exists():
    if time.time()>deadline:raise RuntimeError('No completed common-capacity selection; matrix not started')
    time.sleep(5)
selected=json.loads(selection.read_text())
if selected['status']!='pass':raise RuntimeError('Common-capacity gate blocked formal timing')
final=json.loads((TASK/'evidence/final-gates-v4.json').read_text())
if len(final)!=7 or any(not x['qualified'] for x in final):raise RuntimeError('Incomplete final correctness gates')
if any(not row['qualified'] for row in selected['attempts'][-1]['rows']):raise RuntimeError('Incomplete full-model gate')
files={}
root=TASK/'source/native-balanced'
for p in [*sorted((root/'ascend_deepep').rglob('*.so')),*sorted((root/'worker-variants/ep8-cap64-aligned').glob('*.so'))]:
    files[str(p.relative_to(TASK))]=hashlib.sha256(p.read_bytes()).hexdigest()
freeze=dict(timestamp=time.time(),capacity=selected['capacity'],source=source_identity(),binaries=files,
            fixed_owners=True,ordinary_executor=True,framework_shared_expert=True,
            original_forward_plans_once=True,primary_runs=90)
(TASK/'evidence/BASELINE-FREEZE.json').write_text(json.dumps(freeze,indent=2))
status=dict(status='running',stage='primary_existing_90',formal_completed=0,formal_planned_initial=108,
            receive_capacity=selected['capacity'],ordinary_only=True,step_time_implemented=False,
            calibration_completed=False,source_frozen=True)
(TASK/'STATUS.json').write_text(json.dumps(status,indent=2))
command=[sys.executable,str(TASK/'tools/campaign_planners.py'),'--name','primary-existing','--capacity',str(selected['capacity'])]
print('START_PRIMARY_EXISTING',json.dumps(command),flush=True)
result=subprocess.run(command)
rows=json.loads((TASK/'evidence/campaign-primary-existing.json').read_text())
status.update(formal_completed=sum(x['qualified'] for x in rows),formal_attempted=len(rows),
              status='stage_complete' if result.returncode==0 else 'needs_investigation',
              stage='existing_planner_comparison_complete' if result.returncode==0 else 'primary_existing_failed')
(TASK/'STATUS.json').write_text(json.dumps(status,indent=2))
raise SystemExit(result.returncode)
