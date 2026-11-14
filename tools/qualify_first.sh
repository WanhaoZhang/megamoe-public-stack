set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
python3 "$TASK/tools/qualification.py" --phase plan --revision v2
python3 "$TASK/tools/qualification.py" --phase accuracy --revision v2
python3 "$TASK/tools/qualification.py" --phase pool --revision v2
