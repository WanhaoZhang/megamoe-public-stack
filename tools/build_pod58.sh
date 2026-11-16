#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
cd "$task"
tar -xf artifacts/native-a812203.tar -C source/native-balanced
python3 - "$task" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);t=p/'tools/build_unified.sh'
s=t.read_text().replace('build-unified-cap','build-study-unified-cap')
s=s.replace('python3 setup.py build_ext --inplace','python3 setup.py build_ext --inplace --build-temp build-study-host/temp --build-lib build-study-host/lib')
t.write_text(s)
PY
bash tools/build_unified.sh
