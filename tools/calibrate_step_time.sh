set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
python3 tools/test_step_time_cpu.py
python3 tools/calibrate_step_time_guard.py
