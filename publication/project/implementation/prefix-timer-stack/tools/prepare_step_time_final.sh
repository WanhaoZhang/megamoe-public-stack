set -euo pipefail
cd /home/z00956592/megaflux/planner-executor-ep8-lab
if [ ! -f evidence/step-time-models-frozen.json ]; then python3 tools/freeze_step_time_models.py; fi
docker exec -w /home/z00956592/megaflux/planner-executor-ep8-lab planner-executor-ep8-lab bash tools/build_step_time.sh
python3 tools/test_step_time_cpu.py --model configs/step-time-full.json
