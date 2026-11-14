#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?candidate composite source root}
CONTROL=${2:?completed control composite source root}
OUT=${3:?evidence directory}
mkdir -p "$OUT"
trap 'rc=$?; printf "%s\n" "$rc" > "$OUT/build.exit"' EXIT
set +u
source /usr/local/Ascend/cann-9.2.0/set_env.sh
set -u
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0
export CPLUS_INCLUDE_PATH="$ASCEND_HOME_PATH/python/site-packages/pypto/lib/framework/3rd/include:${CPLUS_INCLUDE_PATH:-}"
export SHMEM_HOME_PATH="$CONTROL/ascend_deepep_hkq/3rdparty/shmem/install"
export MAX_JOBS=8
BUILD="$ROOT/ascend_deepep_hkq/build-integration"
NATIVE="$ROOT/ascend_deepep_hkq/ascend_deepep/lib"
WORKER="$ROOT/ascend_deepep_hkq/worker-variants/s9-ported"
test -f "$SHMEM_HOME_PATH/set_env.sh"
cmake -S "$ROOT/ascend_deepep_hkq" -B "$BUILD" \
 -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" \
 -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
 -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto \
 -DASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE=auto \
 -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
 -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" \
 -DMEGAFLUX_WORKER_FORWARD=0 -DMEGAFLUX_WORKER_BACKWARD=0 \
 -DMEGAFLUX_WORKER_SHARED_PREFIX=0 -DMEGAFLUX_WORKER_EXTERNAL_REFRESH=0 \
 -DMEGAFLUX_OWNER_ROWS=16 -DMEGAFLUX_OWNER_COMPACT=0
cmake --build "$BUILD" --parallel "$MAX_JOBS" --target \
 ascend_deepep_kernels ascend_deepep_mega_moe_bf16 ascend_deepep_replica \
 ascend_deepep_replica_control ascend_deepep_replica_device \
 ascend_deepep_replica_optimized ascend_deepep_moonep_planner ascend_deepep_plan_validator \
 ascend_deepep_megaflux_worker_bf16
mkdir -p "$NATIVE" "$WORKER"
cp "$BUILD"/lib/*.so "$NATIVE/"
cp -P "$SHMEM_HOME_PATH"/shmem/lib/*.so* "$NATIVE/"
cp "$BUILD/lib/libascend_deepep_megaflux_worker_bf16.so" "$WORKER/libascend_deepep_megaflux_early_bf16.so"
export LD_LIBRARY_PATH="$WORKER:$NATIVE:$SHMEM_HOME_PATH/shmem/lib:${LD_LIBRARY_PATH:-}"
export WORKER_NATIVE="$WORKER"
cd "$ROOT/ascend_deepep_hkq"
python -m pip install -e . --no-deps --no-build-isolation
python -m pip install -e "$ROOT/FSDPTurbo" --no-deps --no-build-isolation
python - "$ROOT" "$OUT" <<'PY'
import hashlib, json, pathlib, sys
import torch, torch_npu
import ascend_deepep
root, out = map(pathlib.Path, sys.argv[1:])
paths = [*root.glob('ascend_deepep_hkq/ascend_deepep/lib/*.so*'),
         *root.glob('ascend_deepep_hkq/ascend_deepep/_C*.so'),
         *root.glob('ascend_deepep_hkq/worker-variants/s9-ported/*.so')]
record = {'package': ascend_deepep.__file__, 'native': {
    str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}}
(out/'native.json').write_text(json.dumps(record, indent=2))
print('CANDIDATE_NATIVE_BUILT', record['package'], flush=True)
PY
