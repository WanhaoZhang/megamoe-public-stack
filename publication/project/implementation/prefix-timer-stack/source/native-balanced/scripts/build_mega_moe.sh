#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${MEGA_MOE_BUILD_DIR:-"${ROOT_DIR}/build_mega_moe"}
SOC_TYPE="Ascend950"
RUNTIME_INCLUDE_DIR=${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-}
PYTHON_BIN=${MEGA_MOE_PYTHON:-}
ENABLE_TIMER=OFF

while [[ $# -gt 0 ]]; do
    case "$1" in
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
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "${SOC_TYPE}" != "Ascend950" ]]; then
    echo "[ERROR] MegaMoe W8A8 only supports -soc_type Ascend950"
    exit 1
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first."
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
        echo "[ERROR] Activate the PyTorch environment or pass --python <path>."
        exit 1
    fi
fi

if ! PYTHON_REAL_PATH=$("${PYTHON_BIN}" -c \
    'import os, sys; print(os.path.realpath(sys.executable))' 2>/dev/null); then
    echo "[ERROR] Cannot execute Python: ${PYTHON_BIN}"
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import torch, torch_npu' >/dev/null 2>&1; then
    echo "[ERROR] The selected Python cannot import torch and torch_npu:"
    echo "        ${PYTHON_REAL_PATH}"
    echo "[ERROR] Activate the correct environment or rerun with:"
    echo "        bash scripts/build_mega_moe.sh --python <python-with-torch-and-torch_npu>"
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
        echo "[ERROR] Locate it with:"
        echo "        find \"${ASCEND_PACKAGE_ROOT}\" -type f -name rt.h"
        echo "[ERROR] Then rerun with:"
        echo "        bash scripts/build_mega_moe.sh --runtime-include-dir <directory-containing-rt.h>"
        exit 1
    fi
    RUNTIME_INCLUDE_DIR=$(dirname "${RUNTIME_HEADER}")
fi

echo "[INFO] Building standalone MegaMoe W8A8 targets"
echo "[INFO] ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
echo "[INFO] Runtime include root: ${RUNTIME_INCLUDE_DIR}"
echo "[INFO] BUILD_DIR: ${BUILD_DIR}"
echo "[INFO] Python: ${PYTHON_REAL_PATH}"
echo "[INFO] Timer instrumentation: ${ENABLE_TIMER}"

bash "${ROOT_DIR}/scripts/build_shmem.sh" -soc_type Ascend950
SHMEM_HOME_PATH="${ROOT_DIR}/3rdparty/shmem/install"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH=a5 \
    -DASCEND_DEEPEP_BUILD_MEGA_MOE=ON \
    -DASCEND_DEEPEP_ENABLE_TIMER="${ENABLE_TIMER}" \
    -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${RUNTIME_INCLUDE_DIR}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" \
    --target ascend_deepep_mega_moe mega_moe_stub \
    --parallel

mkdir -p "${ROOT_DIR}/ascend_deepep/lib"
mkdir -p "${ROOT_DIR}/ascend_deepep/bin"
cp "${BUILD_DIR}/lib/libascend_deepep_mega_moe.so" \
    "${ROOT_DIR}/ascend_deepep/lib/"
cp "${BUILD_DIR}/bin/mega_moe_stub" \
    "${ROOT_DIR}/ascend_deepep/bin/"
find "${SHMEM_HOME_PATH}/shmem/lib" \
    -maxdepth 1 \( -type f -o -type l \) \( -name "*.so" -o -name "*.so.*" \) \
    -exec cp -P {} "${ROOT_DIR}/ascend_deepep/lib/" \;

if [[ ! -f "${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_kernels.so" ]]; then
    echo "[ERROR] Base kernel library is missing:"
    echo "        ${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_kernels.so"
    echo "[ERROR] Run bash scripts/build.sh -soc_type Ascend950 once before"
    echo "        building the MegaMoe Python binding."
    exit 1
fi

export ASCEND_DEEPEP_KERNEL_LIB_DIR="${ROOT_DIR}/ascend_deepep/lib"
pushd "${ROOT_DIR}" >/dev/null
SHMEM_HOME_PATH="${SHMEM_HOME_PATH}" \
    "${PYTHON_BIN}" -m pip install -e . --no-build-isolation --no-deps
popd >/dev/null

echo "[INFO] MegaMoe build completed."
echo "[INFO] Kernel library: ${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_mega_moe.so"
echo "[INFO] CSV runner: ${ROOT_DIR}/ascend_deepep/bin/mega_moe_stub"
echo "[INFO] Python class: ascend_deepep.MegaMoe"
