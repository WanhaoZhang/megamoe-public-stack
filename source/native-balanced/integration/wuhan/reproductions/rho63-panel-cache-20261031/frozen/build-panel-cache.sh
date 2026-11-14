#!/usr/bin/env bash
set -euo pipefail
set +u;source /usr/local/Ascend/cann-9.2.0/set_env.sh;set -u
cd /project/panel-cache-source
export SHMEM_HOME_PATH=/project/dependencies/sdk MAX_JOBS=2
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1
bash scripts/build_megaflux_timer_capture.sh rho63-panel-cache-a2
