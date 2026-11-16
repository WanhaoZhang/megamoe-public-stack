#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
cd "$task"
python3 - <<'PY'
from pathlib import Path
import hashlib
p=Path('/home/z00956592/megaflux/single-mb-fbw-planner-study')
for name,expected in [('configs/step-time-full.json','ac66dfb82a49cf0bb453a9ebca8b8b43fd824077de546d96e96fb20881dc92a4'),
    ('source/native-balanced/ascend_deepep/lib/libascend_deepep_step_time_planner.so','664c65d62c82b8ac7a62a6cef1053cab7f63a00e1599b3acad5e4d63c8c0d66b')]:
    assert hashlib.sha256((p/name).read_bytes()).hexdigest()==expected,name
print('FIXED_STATE_PREREQUISITES_VERIFIED')
PY
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- \
  python3 tools/controller.py --backend balanced --mode validate \
    --tag diagnostic-fixed-state-pair-vio5-restored-v2 --seconds 7200 \
    --fixed-state-pair --root "$task" --planner step_time --maxvio 5 --steps 25 \
    --cost-model "$task/configs/step-time-full.json"
