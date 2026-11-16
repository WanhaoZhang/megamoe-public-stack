set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 - <<'PY'
import json,sys
sys.path.insert(0,'tools')
from controller import TASK,run_job,source_identity
from resource_admission import wait_if_occupied
from campaign_one import foreign_observation
last=json.loads((TASK/'evidence/step-time-stage-driver.jsonl').read_text().splitlines()[-1])
assert last['state']=='all_measurements_complete','finish all formal stages first'
freeze=json.loads((TASK/'evidence/STEP-TIME-FREEZE.json').read_text());assert source_identity()==freeze['source']
tag='planner-primary-plan-bank';directory=TASK/'evidence'/tag
if directory.exists():
 status=json.loads((directory/'status.json').read_text());assert status['qualified'] and status['source_unchanged']
else:
 wait_if_occupied(tag)
 status=run_job('balanced','plan',tag,['--dump-plan-bank','--policies','moonep,native,ultra_ep,echo,eplb_fixed,step_time',
     '--cost-model',str(TASK/'configs/step-time-full.json')],seconds=1800)
assert status['qualified'] and status['peak_driver_MiB']<=90*1024,status
assert not foreign_observation(dict(tag=tag,**status)),'diagnostic foreign GPU overlap'
PY
python3 tools/analyze_plan_bank.py
python3 tools/audit_final_identities.py
