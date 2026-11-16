set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 - <<'PY'
import json,sys
sys.path.insert(0,'tools')
from controller import TASK,run_job
from resource_admission import wait_if_occupied
tag='diagnostic-fixed-state-pair-aligned-vio5'
wait_if_occupied(tag)
status=run_job('balanced','validate',tag,['--fixed-state-pair','--root',str(TASK),'--planner','step_time','--maxvio','5','--steps','25',
    '--cost-model',str(TASK/'configs/step-time-full.json')],seconds=7200)
assert status['qualified'] and status['peak_driver_MiB']<=90*1024,status
PY
