#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
root="$task/source/native-priority"
if [ ! -e "$root" ]; then
  cp -a --reflink=auto "$task/source/native-balanced" "$root"
  printf '%s\n' "$task" > "$root/.single-mb-study-owner"
  tar -xf "$task/artifacts/native-priority-source.tar" -C "$root"
else
  test "$(cat "$root/.single-mb-study-owner")" = "$task"
fi
for priority in 0 1; do
  dir="$root/build-matched-priority$priority"
  cmake -S "$root" -B "$dir" \
    -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DMEGAMOE_MAX_LOCAL_EXPERTS=64 \
    -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    -DMEGAFLUX_WORKER_FORWARD=0 -DMEGAFLUX_WORKER_BACKWARD=0 \
    -DMEGAFLUX_SHARED_EXPERT_FORWARD=1 -DMEGAFLUX_SHARED_EXPERT_BACKWARD=1 \
    -DMEGAFLUX_BACKWARD_RESTORE=2 -DMEGAFLUX_RESTORE_OPERAND_READY=1 \
    -DMEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 -DMEGAFLUX_S9_SWIGLU_OVERRIDE=2 \
    -DMEGAFLUX_GRADIENT_PRIORITY="$priority"
  cmake --build "$dir" --target ascend_deepep_megaflux_worker_bf16 --parallel 2
  destination="$task/source/native-balanced/worker-variants/matched-priority$priority"
  mkdir -p "$destination"
  cp "$dir/lib/libascend_deepep_megaflux_worker_bf16.so" "$destination/libascend_deepep_megaflux_early_bf16.so"
  python3 - "$task" "$destination" "$priority" <<'PY'
import pathlib,json,hashlib,sys
task,dest,priority=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),int(sys.argv[3])
so=dest/'libascend_deepep_megaflux_early_bf16.so'
(dest/'build.json').write_text(json.dumps(dict(priority=priority,timer=False,
    source_archive_sha256=hashlib.sha256((task/'artifacts/native-priority-source.tar').read_bytes()).hexdigest(),
    so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),qualification='pending'),indent=2))
PY
done
echo MATCHED_PRIORITY_BUILD_COMPLETE
