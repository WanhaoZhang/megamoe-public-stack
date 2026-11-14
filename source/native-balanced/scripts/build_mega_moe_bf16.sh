#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${MEGA_MOE_BF16_BUILD_DIR:-"${ROOT_DIR}/build_mega_moe_bf16"}
SOC_TYPE="Ascend950"
RUNTIME_INCLUDE_DIR=${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-}
PYTHON_BIN=${MEGA_MOE_BF16_PYTHON:-}
ENABLE_TIMER=OFF
GMM1_SCHEDULE=adaptive-auto
SWIGLU_DECOUPLE=auto
ENABLE_DEBUG_PRINT=OFF

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            echo "Usage: bash scripts/build_mega_moe_bf16.sh [options]"
            echo "  -soc_type Ascend950"
            echo "  --runtime-include-dir DIR    Directory containing rt.h"
            echo "  --python EXECUTABLE         Python with torch and torch_npu"
            echo "  --enable-debug-print        Enable legacy backward phase diagnostics"
            echo "  --swiglu-decouple off|on|auto  Default: auto (H<3584; H=3584 only mxfp8)"
            echo "  --enable-timer              Enable normal stage timers"
            echo "  Backward: none=UB; bf16/mxfp8=GM with workspace reuse (fixed)"
            echo "  --gmm1-schedule adaptive-auto|adaptive-ready-first|adaptive-expert|legacy"
            echo "                               Default: adaptive-auto"
            exit 0
            ;;
        --swiglu-decouple)
            if [[ $# -lt 2 || ( "$2" != "off" && "$2" != "on" && "$2" != "auto" ) ]]; then
                echo "[ERROR] --swiglu-decouple requires off, on or auto"
                exit 1
            fi
            SWIGLU_DECOUPLE="$2"
            shift 2
            ;;
        --gmm1-schedule)
            if [[ $# -lt 2 || ( "$2" != "adaptive-auto" && "$2" != "adaptive-ready-first" && "$2" != "adaptive-expert" && "$2" != "legacy" ) ]]; then
                echo "[ERROR] --gmm1-schedule requires adaptive-auto, adaptive-ready-first, adaptive-expert or legacy"
                exit 1
            fi
            GMM1_SCHEDULE="$2"
            shift 2
            ;;
        -soc_type)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] -soc_type requires Ascend950"
                exit 1
            fi
            SOC_TYPE="$2"
            shift 2
            ;;
        --runtime-include-dir|-runtime_include_dir)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] $1 requires the directory that contains rt.h"
                exit 1
            fi
            RUNTIME_INCLUDE_DIR="$2"
            shift 2
            ;;
        --python)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --python requires a Python executable"
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        --enable-timer)
            ENABLE_TIMER=ON
            shift
            ;;
        --enable-debug-print)
            ENABLE_DEBUG_PRINT=ON
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "${SOC_TYPE}" != "Ascend950" ]]; then
    echo "[ERROR] MegaMoe BF16 only supports -soc_type Ascend950"
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first."
    exit 1
fi
if [[ ! -f "${ROOT_DIR}/3rdparty/catlass/include/catlass/catlass.hpp" ]]; then
    echo "[ERROR] DeepEP CATLASS submodule is not initialized."
    echo "[ERROR] Run git submodule update --init --recursive first."
    exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON_BIN="${CONDA_PREFIX}/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python3)
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=$(command -v python)
    else
        echo "[ERROR] Cannot find a Python executable."
        exit 1
    fi
fi

PYTHON_REAL_PATH=$("${PYTHON_BIN}" -c \
    'import os, sys; print(os.path.realpath(sys.executable))')
if ! "${PYTHON_BIN}" -c 'import torch, torch_npu' >/dev/null 2>&1; then
    echo "[ERROR] Python cannot import torch and torch_npu: ${PYTHON_REAL_PATH}"
    exit 1
fi

if [[ -n "${RUNTIME_INCLUDE_DIR}" ]]; then
    if [[ ! -f "${RUNTIME_INCLUDE_DIR}/rt.h" ]]; then
        echo "[ERROR] rt.h not found under ${RUNTIME_INCLUDE_DIR}"
        exit 1
    fi
else
    ASCEND_HOME_REAL_PATH=$(cd "${ASCEND_HOME_PATH}" && pwd -P)
    ASCEND_PACKAGE_ROOT=$(dirname "${ASCEND_HOME_REAL_PATH}")
    RUNTIME_HEADER=$(find "${ASCEND_HOME_REAL_PATH}" \
        -maxdepth 8 -type f -name rt.h -print -quit 2>/dev/null || true)
    if [[ -z "${RUNTIME_HEADER}" ]]; then
        RUNTIME_HEADER=$(find "${ASCEND_PACKAGE_ROOT}" \
            -maxdepth 8 -type f -name rt.h -print -quit 2>/dev/null || true)
    fi
    if [[ -z "${RUNTIME_HEADER}" ]]; then
        echo "[ERROR] Cannot find rt.h under ${ASCEND_PACKAGE_ROOT}"
        exit 1
    fi
    RUNTIME_INCLUDE_DIR=$(dirname "${RUNTIME_HEADER}")
fi

echo "[INFO] Building standalone MegaMoe BF16 target"
echo "[INFO] ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
echo "[INFO] Runtime include root: ${RUNTIME_INCLUDE_DIR}"
echo "[INFO] BUILD_DIR: ${BUILD_DIR}"
echo "[INFO] Python: ${PYTHON_REAL_PATH}"
echo "[INFO] SwiGLU decoupling: ${SWIGLU_DECOUPLE} (build-time policy)"
echo "[INFO] Timer instrumentation: ${ENABLE_TIMER}"
echo "[INFO] Backward: none=UB; bf16/mxfp8=GM with Hp/dO reuse"
echo "[INFO] GMM1 schedule: ${GMM1_SCHEDULE} (build-time selection)"
echo "[INFO] Backward phase debug print: ${ENABLE_DEBUG_PRINT}"

bash "${ROOT_DIR}/scripts/build_shmem.sh" -soc_type Ascend950
SHMEM_HOME_PATH="${ROOT_DIR}/3rdparty/shmem/install"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
    -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER="${ENABLE_TIMER}" \
    -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE="${GMM1_SCHEDULE}" \
    -DASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE="${SWIGLU_DECOUPLE}" \
    -DASCEND_DEEPEP_MEGA_MOE_BF16_DEBUG_PRINT="${ENABLE_DEBUG_PRINT}" \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${RUNTIME_INCLUDE_DIR}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" \
    --target ascend_deepep_mega_moe_bf16 \
    --clean-first \
    --parallel

mkdir -p "${ROOT_DIR}/ascend_deepep/lib"
cp "${BUILD_DIR}/lib/libascend_deepep_mega_moe_bf16.so" \
    "${ROOT_DIR}/ascend_deepep/lib/"
find "${SHMEM_HOME_PATH}/shmem/lib" \
    -maxdepth 1 \( -type f -o -type l \) \( -name "*.so" -o -name "*.so.*" \) \
    -exec cp -P {} "${ROOT_DIR}/ascend_deepep/lib/" \;

if [[ ! -f "${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_kernels.so" ]]; then
    echo "[ERROR] Base kernel library is missing."
    echo "[ERROR] Run bash scripts/build.sh -soc_type Ascend950 first."
    exit 1
fi

export ASCEND_DEEPEP_KERNEL_LIB_DIR="${ROOT_DIR}/ascend_deepep/lib"
pushd "${ROOT_DIR}" >/dev/null
SHMEM_HOME_PATH="${SHMEM_HOME_PATH}" \
    "${PYTHON_BIN}" -m pip install -e . --no-build-isolation --no-deps
popd >/dev/null

echo "[INFO] MegaMoe BF16 build completed."
echo "[INFO] Kernel library: ${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_mega_moe_bf16.so"
echo "[INFO] Python class: ascend_deepep.MegaMoeBF16"
