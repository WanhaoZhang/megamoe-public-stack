#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/../../.." && pwd)
BUILD_DIR=${MEGA_MOE_W4A8_INDEPENDENT_BUILD_DIR:-"${SCRIPT_DIR}/build"}
INSTALL_DIR=${MEGA_MOE_W4A8_INDEPENDENT_INSTALL_DIR:-"${SCRIPT_DIR}/install"}
RUNTIME_INCLUDE_DIR=${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-}
SOC_TYPE=Ascend950

while [[ $# -gt 0 ]]; do
    case "$1" in
        -soc_type)
            SOC_TYPE=${2:?"-soc_type requires Ascend950"}
            shift 2
            ;;
        --runtime-include-dir)
            RUNTIME_INCLUDE_DIR=${2:?"--runtime-include-dir requires a path"}
            shift 2
            ;;
        --build-dir)
            BUILD_DIR=${2:?"--build-dir requires a path"}
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR=${2:?"--install-dir requires a path"}
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ "${SOC_TYPE}" != "Ascend950" ]]; then
    echo "[ERROR] MegaMoe W4A8 only supports Ascend950" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set; source the CANN set_env.sh first" >&2
    exit 1
fi
if [[ ! -x "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" ]]; then
    echo "[ERROR] Bisheng compiler not found under ASCEND_HOME_PATH=${ASCEND_HOME_PATH}" >&2
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
        echo "[ERROR] Pass --runtime-include-dir <directory-containing-rt.h>" >&2
        exit 1
    fi
    RUNTIME_INCLUDE_DIR=$(dirname "${RUNTIME_HEADER}")
fi
if [[ ! -f "${RUNTIME_INCLUDE_DIR}/rt.h" ]]; then
    echo "[ERROR] rt.h not found in ${RUNTIME_INCLUDE_DIR}" >&2
    exit 1
fi

echo "[INFO] Building independent MegaMoe W4A8 C++ runner"
echo "[INFO] CANN: ${ASCEND_HOME_PATH}"
echo "[INFO] Runtime headers: ${RUNTIME_INCLUDE_DIR}"
echo "[INFO] Build directory: ${BUILD_DIR}"

bash "${ROOT_DIR}/scripts/build_shmem.sh" -soc_type Ascend950
SHMEM_HOME_PATH="${ROOT_DIR}/3rdparty/shmem/install"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DCMAKE_BUILD_TYPE=Release \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
    -DASCEND_DEEPEP_BUILD_MEGA_MOE_W4A8=ON \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${RUNTIME_INCLUDE_DIR}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" \
    --target ascend_deepep_mega_moe_w4a8 mega_moe_w4a8_stub \
    --parallel

mkdir -p "${INSTALL_DIR}/bin" "${INSTALL_DIR}/lib"
cp "${BUILD_DIR}/bin/mega_moe_w4a8_stub" "${INSTALL_DIR}/bin/"
cp "${BUILD_DIR}/lib/libascend_deepep_mega_moe_w4a8.so" "${INSTALL_DIR}/lib/"
find "${SHMEM_HOME_PATH}/shmem/lib" -maxdepth 1 \
    \( -type f -o -type l \) \( -name '*.so' -o -name '*.so.*' \) \
    -exec cp -P {} "${INSTALL_DIR}/lib/" \;

echo "[SUCCESS] Independent MegaMoe build completed"
echo "[INFO] Runner: ${INSTALL_DIR}/bin/mega_moe_w4a8_stub"
echo "[INFO] Kernel: ${INSTALL_DIR}/lib/libascend_deepep_mega_moe_w4a8.so"
