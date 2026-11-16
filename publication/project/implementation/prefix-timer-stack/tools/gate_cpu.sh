#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export PYTHONPATH="$TASK/source/FSDPTurbo:$TASK/source/MindSpeed-LLM:$TASK/tools"
export GLOO_SOCKET_IFNAME=lo OMP_NUM_THREADS=1
cd "$TASK/source/FSDPTurbo"
python3 -m compileall -q fsdp_turbo/distributed/expert_parallel fsdp_turbo/distributed/fully_shard_parallel fsdp_turbo/ops/npu/moonep_grouped_matmul.py "$TASK/tools" "$TASK/source/MindSpeed-LLM/mindspeed_llm/fsdp2/features/ep8_imbalance.py"
python3 -m pytest tests/unit_tests/distributed/expert_parallel/moonep/test_grad_norm_agreement.py -q
