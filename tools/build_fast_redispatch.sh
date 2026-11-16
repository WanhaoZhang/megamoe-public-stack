#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
root="$task/source/native-fast-redispatch"
archive="$task/artifacts/native-fast-redispatch-source.tar"
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
  local name="$1"; local runahead="$2"; local batch="$3"; local cache="$4"; local timer="$5"
  local build_dir="$root/build-fast-$name"
  cmake -S "$root" -B "$build_dir" \
    -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER="$timer" -DMEGAFLUX_COORDINATE_TIMER=OFF -DMEGAMOE_MAX_LOCAL_EXPERTS=64 \
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
    -DMEGAFLUX_EARLY_DW1_WIDE=0 -DMEGAFLUX_OWNER_LATE_ONLY=0 -DMEGAFLUX_REDISPATCH_P1_RUNAHEAD="$runahead" -DMEGAFLUX_REDISPATCH_TRANSPORT_BATCH="$batch" -DMEGAFLUX_GRADIENT_KEY_CACHE="$cache"
  cmake --build "$build_dir" --target ascend_deepep_megaflux_worker_bf16 --parallel 2
  local destination="$task/source/native-balanced/worker-variants/$name"
  mkdir -p "$destination"
  cp "$build_dir/lib/libascend_deepep_megaflux_worker_bf16.so" "$destination/libascend_deepep_megaflux_early_bf16.so"
  python3 - "$archive" "$destination" "$build_dir" "$name" "$runahead" "$batch" "$cache" "$timer" <<'PY'
from pathlib import Path
import hashlib,json,subprocess,sys
archive,destination,build_dir=map(Path,sys.argv[1:4]);name=sys.argv[4];runahead,batch,cache=map(int,sys.argv[5:8]);timer=sys.argv[8]=="ON"
cache_path=build_dir/'CMakeCache.txt';flags={}
for line in cache_path.read_text().splitlines():
 if ':' in line and '=' in line:
  key,value=line.split('=',1);key=key.split(':',1)[0]
  if key.startswith(('MEGAFLUX_','ASCEND_DEEPEP_','MEGAMOE_')):flags[key]=value
so=destination/'libascend_deepep_megaflux_early_bf16.so'
(destination/'build.json').write_text(json.dumps(dict(variant=name,runahead=runahead,transport_batch=batch,gradient_key_cache=cache,timer=timer,coordinate_timer=False,
    source_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    source_commit=subprocess.check_output(['git','get-tar-commit-id'],input=archive.read_bytes()).decode().strip(),
    so_sha256=hashlib.sha256(so.read_bytes()).hexdigest(),
    cmake_cache_sha256=hashlib.sha256(cache_path.read_bytes()).hexdigest(),compile_flags=flags,
    qualification='build_only_math_pending'),indent=2))
PY
}
pids=()
for spec in "fastpipe-r0-t0 0 0 1 OFF" "fastpipe-r3-t0 3 0 1 OFF" "fastpipe-r4-t0 4 0 1 OFF" "fastpipe-r5-t0 5 0 1 OFF" "fastpipe-r5-t1 5 1 1 OFF" "fastpipe-r5-t2 5 2 1 OFF" "fastlite-cache0 0 0 0 ON" "fastlite-cache1 0 0 1 ON"; do
  read -r name runahead batch cache timer <<< "$spec"
  build_mode "$name" "$runahead" "$batch" "$cache" "$timer" > "$task/evidence/$name-build.log" 2>&1 & pids+=("$!")
done
failed=0
for job in "${pids[@]}"; do wait "$job" || failed=1; done
test "$failed" = 0
echo FAST_REDISPATCH_BUILD_COMPLETE
