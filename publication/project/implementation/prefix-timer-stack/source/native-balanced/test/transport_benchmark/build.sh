#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../.." && pwd)
BUILD_DIR=${DEEPEP_TRANSPORT_BENCHMARK_BUILD_DIR:-"${SCRIPT_DIR}/build"}
INSTALL_DIR=${DEEPEP_TRANSPORT_BENCHMARK_INSTALL_DIR:-"${SCRIPT_DIR}/install"}
RUNTIME_INCLUDE_DIR=${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-}

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set; source the CANN set_env.sh first" >&2
    exit 1
fi

if [[ -z "${RUNTIME_INCLUDE_DIR}" ]]; then
    ASCEND_HOME_REAL=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    ASCEND_PARENT=$(dirname "${ASCEND_HOME_REAL}")
    RUNTIME_HEADER=$(find "${ASCEND_HOME_REAL}" -maxdepth 8 -type f -name rt.h -print -quit 2>/dev/null || true)
    if [[ -z "${RUNTIME_HEADER}" ]]; then
        RUNTIME_HEADER=$(find "${ASCEND_PARENT}" -maxdepth 8 -type f -name rt.h -print -quit 2>/dev/null || true)
    fi
    if [[ -z "${RUNTIME_HEADER}" ]]; then
        echo "[ERROR] Cannot find rt.h under ${ASCEND_PARENT}" >&2
        exit 1
    fi
    RUNTIME_INCLUDE_DIR=$(dirname "${RUNTIME_HEADER}")
fi

bash "${ROOT_DIR}/scripts/build_shmem.sh" -soc_type Ascend950
SHMEM_HOME_PATH="${ROOT_DIR}/3rdparty/shmem/install"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
    -DASCEND_DEEPEP_BUILD_TRANSPORT_BENCHMARK=ON \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${RUNTIME_INCLUDE_DIR}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" --target deepep_transport_benchmark --parallel

mkdir -p "${INSTALL_DIR}/bin" "${INSTALL_DIR}/lib"
cp "${BUILD_DIR}/bin/deepep_transport_benchmark" "${INSTALL_DIR}/bin/"
cp "${BUILD_DIR}/lib/libdeepep_transport_benchmark_kernel.so" "${INSTALL_DIR}/lib/"
find "${SHMEM_HOME_PATH}/shmem/lib" -maxdepth 1 \
    \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) \
    -exec cp -P {} "${INSTALL_DIR}/lib/" \;

echo "[SUCCESS] Built ${INSTALL_DIR}/bin/deepep_transport_benchmark"
