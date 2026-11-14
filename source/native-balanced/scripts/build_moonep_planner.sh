#!/usr/bin/env bash
# Only planner and binding change; frozen S9/original execution SOs are inputs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Source matching CANN first}"
: "${SHMEM_HOME_PATH:?Set the frozen SHMEM dependency}"
BUILD="$ROOT/build_moonep_planner"
cmake -S "$ROOT" -B "$BUILD" \
 -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
 -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
 -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto \
 -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
 -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH"
cmake --build "$BUILD" --target ascend_deepep_moonep_planner --parallel "${MAX_JOBS:-2}"
cp "$BUILD/lib/libascend_deepep_moonep_planner.so" "$ROOT/ascend_deepep/lib/"
cd "$ROOT"
MAX_JOBS=${MAX_JOBS:-2} python3 setup.py build_ext --inplace
