#!/usr/bin/env bash
set -euo pipefail
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
python3 - <<'PY'
import torch_npu
value="expandable_segments:True,garbage_collection_threshold:0.7"
torch_npu.npu.memory._set_allocator_settings(value)
print("ALLOCATOR_CONFIG_ACCEPTED",value,flush=True)
PY
