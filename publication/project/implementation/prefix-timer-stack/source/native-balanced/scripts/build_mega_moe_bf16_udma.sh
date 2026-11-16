#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${MEGA_MOE_BF16_UDMA_BUILD_DIR:-"${ROOT_DIR}/build_mega_moe_bf16_udma"}
SOC_TYPE="Ascend950"
RUNTIME_INCLUDE_DIR=${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-}
PYTHON_BIN=${MEGA_MOE_BF16_UDMA_PYTHON:-}
ENABLE_TIMER=OFF

while [[ $# -gt 0 ]]; do
    case "$1" in
        -soc_type)
            SOC_TYPE="$2"
            shift 2
            ;;
        --runtime-include-dir|-runtime_include_dir)
            RUNTIME_INCLUDE_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --enable-timer)
            ENABLE_TIMER=ON
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "${SOC_TYPE}" != "Ascend950" ]]; then
    echo "[ERROR] MegaMoe BF16 UDMA only supports -soc_type Ascend950"
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first."
    exit 1
fi
if [[ ! -f "${ROOT_DIR}/3rdparty/catlass/include/catlass/catlass.hpp" ]]; then
    echo "[ERROR] DeepEP CATLASS submodule is not initialized."
    exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    else
        PYTHON_BIN=$(command -v python)
    fi
fi

if [[ -z "${RUNTIME_INCLUDE_DIR}" ]]; then
    ASCEND_HOME_REAL_PATH=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    ASCEND_PACKAGE_ROOT=$(dirname "${ASCEND_HOME_REAL_PATH}")
    RUNTIME_HEADER=$(find "${ASCEND_HOME_REAL_PATH}" -maxdepth 8 -type f -name rt.h -print -quit 2>/dev/null || true)
    if [[ -z "${RUNTIME_HEADER}" ]]; then
        RUNTIME_HEADER=$(find "${ASCEND_PACKAGE_ROOT}" -maxdepth 8 -type f -name rt.h -print -quit 2>/dev/null || true)
    fi
    if [[ -z "${RUNTIME_HEADER}" ]]; then
        echo "[ERROR] Cannot find rt.h under ${ASCEND_PACKAGE_ROOT}"
        exit 1
    fi
    RUNTIME_INCLUDE_DIR=$(dirname "${RUNTIME_HEADER}")
fi

bash "${ROOT_DIR}/scripts/build_shmem.sh" -soc_type Ascend950
SHMEM_HOME_PATH="${ROOT_DIR}/3rdparty/shmem/install"

echo "[INFO] Timer instrumentation: ${ENABLE_TIMER}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
    -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16_UDMA=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER="${ENABLE_TIMER}" \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${RUNTIME_INCLUDE_DIR}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" \
    --target ascend_deepep_mega_moe_bf16_udma mega_moe_bf16_udma_stub \
    --parallel

mkdir -p "${ROOT_DIR}/ascend_deepep/lib"
cp "${BUILD_DIR}/lib/libascend_deepep_mega_moe_bf16_udma.so" "${ROOT_DIR}/ascend_deepep/lib/"
find "${SHMEM_HOME_PATH}/shmem/lib" -maxdepth 1 \( -type f -o -type l \) \
    \( -name "*.so" -o -name "*.so.*" \) \
    -exec cp -P {} "${ROOT_DIR}/ascend_deepep/lib/" \;

if [[ ! -f "${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_kernels.so" ]]; then
    echo "[ERROR] Base kernel library is missing. Run scripts/build.sh first."
    exit 1
fi

export ASCEND_DEEPEP_KERNEL_LIB_DIR="${ROOT_DIR}/ascend_deepep/lib"
SHMEM_HOME_PATH="${SHMEM_HOME_PATH}" \
    "${PYTHON_BIN}" -m pip install -e "${ROOT_DIR}" --no-build-isolation --no-deps

echo "[INFO] MegaMoe BF16 UDMA build completed."
echo "[INFO] Python class: ascend_deepep.MegaMoeBF16UDMA"
