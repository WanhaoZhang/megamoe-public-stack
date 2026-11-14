#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export MAX_JOBS=${MAX_JOBS:-4} OMP_NUM_THREADS=1
export CPLUS_INCLUDE_PATH="$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include:${CPLUS_INCLUDE_PATH:-}"
# Reuse the baseline worker built by build_benchmark.sh; only planner SOs differ.
test -f "$ROOT/worker-variants/planner-bench/build.json"
: "${ASCEND_HOME_PATH:?Source matching CANN first}"
: "${SHMEM_HOME_PATH:?Pinned SHMEM installation}"
for threshold in 0 1050; do
  name=strict; if [[ "$threshold" == 1050 ]]; then name=direct1050; fi
  build="$ROOT/build-planner-$name"; native="$ROOT/planner-variants/$name"
  test ! -e "$native/build.json"
  cmake -S "$ROOT" -B "$build" \
    -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DMEGAFLUX_MOONEP_DIRECT_PERMILLE="$threshold" \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
    -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH"
  cmake --build "$build" --target ascend_deepep_moonep_planner --parallel "$MAX_JOBS"
  mkdir -p "$native"
  cp "$build/lib/libascend_deepep_moonep_planner.so" "$native/"
  python - "$ROOT" "$name" "$threshold" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);n=r/'planner-variants'/sys.argv[2]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[r/'CMakeLists.txt',r/'test/megamoe_planner_test/build_routing_variants.sh',
       *sorted((r/'include').rglob('*.h')),*sorted((r/'kernels').rglob('*.h')),
       *sorted((r/'kernels').rglob('*.hpp')),*sorted((r/'kernels/mega_moe_bf16').glob('*.cpp'))]
(n/'build.json').write_text(json.dumps(dict(direct_permille=int(sys.argv[3]),timer=0,
  native_sha256=sha(n/'libascend_deepep_moonep_planner.so'),
  sources={str(p.relative_to(r)):sha(p) for p in files}),indent=2))
print('ROUTING_PLANNER_BUILT',sys.argv[2])
PY
done
