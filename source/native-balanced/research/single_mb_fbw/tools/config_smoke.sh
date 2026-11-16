#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH="$task/source/FSDPTurbo:$task/source/MindSpeed-LLM"
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
python3 - <<'PY'
from fsdp_turbo.fsdp_turbo_config import MegaMoeBalancedConfig
names=['native','moonep','ultra_ep','echo','eplb_fixed','step_time']
names += [f'eplb_budget_{b}' for b in (0,1,2,4,8,16,32,64,128,256)]
for name in names:
    assert MegaMoeBalancedConfig(planner=name).planner==name
for name in ('eplb_budget_-1','eplb_budget_3','eplb_budget_257','unknown'):
    try: MegaMoeBalancedConfig(planner=name)
    except ValueError: pass
    else: raise AssertionError(name)
print('CONFIG_SMOKE_PASS',len(names),4)
PY
