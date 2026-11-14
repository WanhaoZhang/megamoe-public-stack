#!/usr/bin/env bash
set -euo pipefail
set +u;source /usr/local/Ascend/cann-9.2.0/set_env.sh;set -u
cd /project/source
export SHMEM_HOME_PATH=/project/dependencies/sdk MAX_JOBS=2
export PYTHONPATH=/project/python-runtime:/project/source
export LD_LIBRARY_PATH=/project/dependencies/sdk/lib:/project/dependencies/sdk/memfabric_hybrid/lib:/project/source/ascend_deepep/lib:${LD_LIBRARY_PATH:-}
python3 integration/wuhan/test_early_dw1_identity.py
python3 integration/wuhan/test_timer_capture_cpu.py
python3 integration/wuhan/test_timer_trace_cpu.py
python3 integration/wuhan/test_restore_operand_cpu.py
python3 integration/wuhan/test_bridge_cpu.py
# Existing colleague dispatch/vector paths use the current adaptive/auto defaults.
# C1 is explicitly 1; worker role experiments remain separate mutually exclusive designs.
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1
bash scripts/build_megaflux_timer_capture.sh rho63-base-a2
cmake --build build-worker-rho63-base-a2-off --target ascend_deepep_mega_moe_bf16 ascend_deepep_kernels ascend_deepep_replica ascend_deepep_replica_control ascend_deepep_replica_device ascend_deepep_replica_optimized --parallel 2
cp build-worker-rho63-base-a2-off/lib/*.so ascend_deepep/lib/
python3 setup.py build_ext --inplace
