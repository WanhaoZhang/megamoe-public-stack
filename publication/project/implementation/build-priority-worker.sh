set -euo pipefail
cd /home/z00956592/megaflux/step-lifetime-priority-cache-20261116/stack
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS=64 MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MAX_JOBS=2 MEGAFLUX_CALIBRATION_TIMER=OFF
bash source/native-balanced/scripts/build_megaflux_worker_pipeline.sh ep8-cap64-lifetime-priority-cache-v1 0 0 0

python3 - <<'PYFREEZE'
import hashlib,json,pathlib,shutil
root=pathlib.Path.cwd()
variant='ep8-cap64-lifetime-priority-cache-v1'
folder=root/'source/native-balanced/worker-variants'/variant
first=folder/'libascend_deepep_megaflux_early_bf16.so'
shutil.copy2(first,folder/'libascend_deepep_megaflux_worker_bf16.so')
sha=hashlib.sha256(first.read_bytes()).hexdigest()
manifest=dict(scope='private ABI2 TimerOFF raw-output schedule control',binaries={str(p.relative_to(root)):sha for p in folder.glob('*.so')})
for name in ('lifetime-runtime-freeze.json',):
 (root/'configs'/name).write_text(json.dumps(manifest,indent=2)+'\n')
print('PRIORITY_WORKER',sha)
PYFREEZE
