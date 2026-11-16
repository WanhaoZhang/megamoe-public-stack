#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
/usr/local/python3.12.13/bin/python3.12 -m venv "$TASK/dependencies/plot-env"
"$TASK/dependencies/plot-env/bin/python" -m pip install matplotlib==3.10.7
"$TASK/dependencies/plot-env/bin/python" -m pip freeze > "$TASK/dependencies/plot-env-requirements.txt"
"$TASK/dependencies/plot-env/bin/python" -c "import matplotlib,numpy; print(matplotlib.__version__,numpy.__version__)"
