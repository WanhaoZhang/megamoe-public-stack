#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
root="$task/source/native-owner-window"
archive="$task/artifacts/native-owner-window-source.tar"
archive_sha=$(sha256sum "$archive" | cut -d' ' -f1)
if [ ! -e "$root" ]; then
  cp -a --reflink=auto "$task/source/native-balanced" "$root"
  printf '%s\n' "$task" > "$root/.single-mb-study-owner"
  tar -xf "$archive" -C "$root"
  printf '%s\n' "$archive_sha" > "$root/.source-archive.sha"
else
  test "$(cat "$root/.single-mb-study-owner")" = "$task"
  test "$(cat "$root/.source-archive.sha")" = "$archive_sha"
fi
mkdir -p "$root/3rdparty/catlass"
tar -xf "$task/artifacts/catlass-d7d0bc2-include.tar" -C "$root/3rdparty/catlass"

build_mode() {
  local mode="$1"
  local build_dir="$root/build-owner-window$mode"
  cmake -S "$root" -B "$build_dir" \
    -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DMEGAMOE_MAX_LOCAL_EXPERTS=64 \
    -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto \
    -DMEGAFLUX_WORKER_FORWARD=0 -DMEGAFLUX_WORKER_BACKWARD=0 \
    -DMEGAFLUX_WORKER_SHARED_PREFIX=0 -DMEGAFLUX_WORKER_EXTERNAL_REFRESH=0 \
    -DMEGAFLUX_SHARED_EXPERT_FORWARD=1 -DMEGAFLUX_SHARED_EXPERT_BACKWARD=1 \
    -DMEGAFLUX_BACKWARD_RESTORE=2 -DMEGAFLUX_RESTORE_OPERAND_READY=1 \
    -DMEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 -DMEGAFLUX_S9_SWIGLU_OVERRIDE=2 \
    -DMEGAFLUX_OWNER_ROWS=16 -DMEGAFLUX_OWNER_COMPACT=0 \
    -DMEGAFLUX_OWNER_EARLY_RATIO=1 -DMEGAFLUX_OWNER_AIV1_TAIL=0 \
    -DMEGAFLUX_OWNER_DOUBLE_BUFFER=OFF -DMEGAFLUX_GRADIENT_PRIORITY=0 \
    -DMEGAFLUX_EARLY_DW1_WIDE=0 -DMEGAFLUX_OWNER_LATE_ONLY="$mode"
  cmake --build "$build_dir" --target ascend_deepep_megaflux_worker_bf16 --parallel 2
  local destination="$task/source/native-balanced/worker-variants/owner-window$mode"
  mkdir -p "$destination"
  cp "$build_dir/lib/libascend_deepep_megaflux_worker_bf16.so" "$destination/libascend_deepep_megaflux_early_bf16.so"
  python3 - "$archive" "$destination" "$build_dir" "$mode" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys
archive,destination,build_dir=map(Path,sys.argv[1:4]);mode=int(sys.argv[4])
cache=build_dir/'CMakeCache.txt';flags={}
for line in cache.read_text().splitlines():
 if ':' in line and '=' in line:
  key,value=line.split('=',1);key=key.split(':',1)[0]
  if key.startswith(('MEGAFLUX_','ASCEND_DEEPEP_','MEGAMOE_')):flags[key]=value
so=destination/'libascend_deepep_megaflux_early_bf16.so'
(destination/'build.json').write_text(json.dumps(dict(owner_late_only=mode,timer=False,
    source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    source_commit=subprocess.check_output(['git','get-tar-commit-id'],input=archive.read_bytes()).decode().strip(),
    so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
    cmake_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),compile_flags=flags,
    qualification='build_only_math_pending'),indent=2))
PY
}
build_mode 0 > "$task/evidence/owner-window0-build.log" 2>&1 & pid0=$!
build_mode 1 > "$task/evidence/owner-window1-build.log" 2>&1 & pid1=$!
code0=0;wait "$pid0" || code0=$?
code1=0;wait "$pid1" || code1=$?
test "$code0" = 0 && test "$code1" = 0
echo OWNER_WINDOW_BUILD_COMPLETE
