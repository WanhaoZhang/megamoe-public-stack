#!/usr/bin/env bash
# CPU compilation only, inside a task-owned container and fresh source directory.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${ASCEND_HOME_PATH:?Source the matching CANN environment}"
: "${SHMEM_HOME_PATH:?Use the pinned task-owned SHMEM installation}"
BUILD="$ROOT/build-planner-benchmark"
test ! -e "$ROOT/worker-variants/planner-bench/build.json"
export MAX_JOBS=${MAX_JOBS:-4} OMP_NUM_THREADS=1
# Windows-origin archives can contain CRLF. Normalize shell text in this fresh
# task-owned build tree; kernel and planner sources are unchanged.
python - "$ROOT" <<'PY'
from pathlib import Path
import sys
for p in (Path(sys.argv[1])/'scripts').glob('*.sh'):
    data=p.read_bytes()
    if b'\r\n' in data:p.write_bytes(data.replace(b'\r\n',b'\n'))
PY
export ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
export CPLUS_INCLUDE_PATH="$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include:${CPLUS_INCLUDE_PATH:-}"
cmake -S "$ROOT" -B "$BUILD" \
 -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
 -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
 -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto \
 -DASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE=auto \
 -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR" \
 -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH"
cmake --build "$BUILD" --parallel "$MAX_JOBS" --target \
 ascend_deepep_kernels ascend_deepep_mega_moe_bf16 ascend_deepep_replica \
 ascend_deepep_replica_control ascend_deepep_replica_device ascend_deepep_replica_optimized \
 ascend_deepep_moonep_planner ascend_deepep_large_ep_planner ascend_deepep_plan_validator
mkdir -p "$ROOT/ascend_deepep/lib"
cp "$BUILD"/lib/*.so "$ROOT/ascend_deepep/lib/"
export MEGAFLUX_WORKER_SHARED_PREFIX=0 MEGAFLUX_WORKER_EXTERNAL_REFRESH=0
export MEGAFLUX_OWNER_ROWS=16 MEGAFLUX_OWNER_COMPACT=0 MEGAFLUX_DISPATCH_READY_LARGEST=0
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1 MEGAFLUX_RESTORE_DELAY_CYCLES=0
bash "$ROOT/scripts/build_megaflux_worker_pipeline.sh" planner-bench 0 0 0
cp "$ROOT/worker-variants/planner-bench/libascend_deepep_megaflux_early_bf16.so" \
   "$ROOT/ascend_deepep/lib/libascend_deepep_megaflux_worker_bf16.so"
export ASCEND_DEEPEP_KERNEL_LIB_DIR="$ROOT/ascend_deepep/lib"
export LD_LIBRARY_PATH="$ROOT/ascend_deepep/lib:$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
# No pip installs or environment replacements: the caller selects this tree via PYTHONPATH.
python setup.py build_ext --inplace

python - "$ROOT" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1])
paths=sorted((r/'ascend_deepep/lib').glob('*.so'))+sorted((r/'ascend_deepep').glob('*.so'))
if not paths:raise RuntimeError('no native libraries built')
manifest={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(r/'build-planner-benchmark/native-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
