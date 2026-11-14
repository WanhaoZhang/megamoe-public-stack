#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR=${BUILD_DIR:-"${ROOT_DIR}/build"}
SOC_TYPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -soc_type)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] -soc_type requires a value, for example: -soc_type Ascend950"
                exit 1
            fi
            SOC_TYPE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ "${SOC_TYPE}" == "Ascend950" ]]; then
    ARCH=a5
    SHMEM_BUILD_ARGS=(-soc_type Ascend950)
else
    ARCH=a2
    SHMEM_BUILD_ARGS=()
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[ERROR] ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first."
    exit 1
fi

echo "[INFO] ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
echo "[INFO] SOC_TYPE: ${SOC_TYPE:-default A2/910B}"
echo "[INFO] ASCEND_DEEPEP_BISHENG_ARCH: ${ARCH}"

# git -C "${ROOT_DIR}" submodule update --init --recursive
bash "${ROOT_DIR}/scripts/build_shmem.sh" "${SHMEM_BUILD_ARGS[@]}"

SHMEM_HOME_PATH="${ROOT_DIR}/3rdparty/shmem/install"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
    -DASCEND_DEEPEP_BISHENG_ARCH="${ARCH}" \
    -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH}"
cmake --build "${BUILD_DIR}" --target ascend_deepep_kernels --parallel

mkdir -p "${ROOT_DIR}/ascend_deepep/lib"
cp "${BUILD_DIR}/lib/libascend_deepep_kernels.so" "${ROOT_DIR}/ascend_deepep/lib/"
find "${SHMEM_HOME_PATH}/shmem/lib" \
    -maxdepth 1 \( -type f -o -type l \) \( -name "*.so" -o -name "*.so.*" \) \
    -exec cp -P {} "${ROOT_DIR}/ascend_deepep/lib/" \;

export ASCEND_DEEPEP_KERNEL_LIB_DIR="${ROOT_DIR}/ascend_deepep/lib"
pushd "${ROOT_DIR}" >/dev/null
SHMEM_HOME_PATH="${SHMEM_HOME_PATH}" python -m pip install -e . --no-build-isolation --no-deps
popd >/dev/null

echo "[INFO] Build completed."
echo "[INFO] Kernel library: ${ROOT_DIR}/ascend_deepep/lib/libascend_deepep_kernels.so"
