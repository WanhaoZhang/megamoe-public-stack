#!/usr/bin/env bash
# Independent candidate build. Original/S5/v2 native kernels remain unchanged.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Source the matching CANN environment first}"
BUILD_DIR=${MEGAFLUX_FULL_BUILD_DIR:-"${ROOT}/build_megaflux_full"}
SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-"${ROOT}/3rdparty/shmem/install"}
TIMER_MODE=${MEGAFLUX_FULL_TIMER:-OFF}
case "$TIMER_MODE" in ON|OFF) ;; *) echo 'MEGAFLUX_FULL_TIMER must be ON or OFF' >&2; exit 2;; esac
cmake -S "${ROOT}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER="${TIMER_MODE}" \
    -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" --target ascend_deepep_megaflux_full_bf16 --parallel "${MAX_JOBS:-4}"
if [[ "${MEGAFLUX_INSTALL:-1}" == "0" ]]; then exit 0; fi
cp "${BUILD_DIR}/lib/libascend_deepep_megaflux_full_bf16.so" "${ROOT}/ascend_deepep/lib/"
cd "${ROOT}"
SHMEM_HOME_PATH="${SHMEM_HOME_PATH}" "${MEGAFLUX_PYTHON:-python3}" -m pip install -e . --no-build-isolation --no-deps
