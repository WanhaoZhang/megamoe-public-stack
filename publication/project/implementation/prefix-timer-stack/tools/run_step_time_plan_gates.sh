set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 - <<'PY'
import sys
sys.path.insert(0,'tools')
from controller import TASK,run_job
from resource_admission import wait_if_occupied
tag='step-time-plan-semantic-v1'
wait_if_occupied(tag)
result=run_job('balanced','plan',tag,['--step-time-gate','--model',str(TASK/'calibration/step-time-guarded-provisional.json')])
assert result['qualified'] and result['peak_driver_MiB']<=90*1024,result
tag='step-time-plan-main-shape-v1'
wait_if_occupied(tag)
result=run_job('balanced','plan',tag,['--step-time-gate','--main-shape','--model',str(TASK/'calibration/step-time-guarded-provisional.json')])
assert result['qualified'] and result['peak_driver_MiB']<=90*1024,result
PY
