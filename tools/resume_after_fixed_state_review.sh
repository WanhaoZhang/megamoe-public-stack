set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 -u - <<'PY'
import json,pathlib,subprocess,time
root=pathlib.Path.cwd();started=time.monotonic()
status=root/'evidence/diagnostic-fixed-state-pair-aligned-vio5/status.json'
print('WAIT_FOR_IDENTICAL_WEIGHT_DIAGNOSTIC',flush=True)
while not status.exists():
    if time.monotonic()-started>7500:raise RuntimeError('diagnostic controller did not finish; no formal resume')
    time.sleep(10)
data=json.loads(status.read_text())
assert data['qualified'] and data['source_unchanged'] and data['peak_driver_MiB']<=90*1024,data
subprocess.run(['python3','tools/audit_fixed_state_pair.py'],check=True)
subprocess.run(['python3','tools/diagnose_trajectory.py'],check=True)
subprocess.run(['python3','tools/analyze_all_stages.py'],check=True)
print('IDENTICAL_WEIGHT_REVIEW_PASSED_RETAINING_ORIGINAL_TRAJECTORY_WARNINGS',flush=True)
subprocess.run(['python3','-u','tools/execute_step_time_campaign.py'],check=True)
subprocess.run(['bash','tools/run_final_plan_diagnostics.sh'],check=True)
PY
