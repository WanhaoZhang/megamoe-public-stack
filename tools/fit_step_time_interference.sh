set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 tools/fit_step_time_interference.py
python3 tools/calibrate_step_time_guard.py --model calibration/step-time-interference-provisional.json
