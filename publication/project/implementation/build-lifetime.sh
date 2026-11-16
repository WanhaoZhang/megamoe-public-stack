set -euo pipefail
cd /home/z00956592/megaflux/step-lifetime-kernel-20261115/stack
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS=64 MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MAX_JOBS=2
bash source/native-balanced/scripts/build_megaflux_worker_pipeline.sh ep8-cap64-lifetime-v1 0 0 0
