#!/bin/bash
set -euo pipefail
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
task=/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack
sdk=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_LIBRARY_PATH="$sdk/shmem/lib:$sdk/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
/usr/local/python3.12.13/bin/python3 "$task/build_timer_worker.py"
