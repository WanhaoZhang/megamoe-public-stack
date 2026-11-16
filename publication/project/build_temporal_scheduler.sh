#!/bin/bash
set -euo pipefail
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
task=/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack
native="$task/source/native-balanced"
sdk=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_LIBRARY_PATH="$sdk/shmem/lib:$sdk/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
cmake -S "$native" -B "$native/build-temporal-scheduler" \
  -DASCEND_HOME_PATH=/home/pkg/b070/cann-9.2.0 \
  -DSHMEM_HOME_PATH="$sdk" -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
  -DMEGAMOE_MAX_LOCAL_EXPERTS=64 \
  -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR=/home/pkg/b070/cann-9.2.0/pkg_inc/runtime/runtime
cmake --build "$native/build-temporal-scheduler" --target ascend_deepep_temporal_scheduler -j2
cp "$native/build-temporal-scheduler/lib/libascend_deepep_temporal_scheduler.so" "$native/ascend_deepep/lib/"
sha256sum "$native/ascend_deepep/lib/libascend_deepep_temporal_scheduler.so"
