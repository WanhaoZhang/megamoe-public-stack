set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
export PATH=/usr/local/python3.12.13/bin:$PATH
python3 "$TASK/tools/prepare_routes.py" --out "$TASK/routes/calibration" --seed 20261114 --targets 0.3,2,7,12,18
python3 "$TASK/tools/prepare_routes.py" --out "$TASK/routes/heldout-a" --seed 31415
python3 "$TASK/tools/prepare_routes.py" --out "$TASK/routes/heldout-b" --seed 27182
