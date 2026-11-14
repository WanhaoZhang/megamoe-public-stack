#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SHMEM_SOURCE_DIR="${ROOT_DIR}/3rdparty/shmem"
SHMEM_INSTALL_DIR="${SHMEM_SOURCE_DIR}/install"
SHMEM_SOURCE_STAMP="${SHMEM_INSTALL_DIR}/.deepep_shmem_source_commit"

if [[ ! -f "${SHMEM_SOURCE_DIR}/scripts/build.sh" ]]; then
    echo "[ERROR] cann/shmem submodule is not initialized."
    echo "        Run: git submodule update --init --recursive"
    exit 1
fi

# --- in-script source fixup（幂等 sed）---
# TCP 连接重试窗口：上游 20000 次 × 1ms ≈ 20s，大 EP 多机下太紧（rank 0 的
# store server 可能晚于最快的 client rank 开始连接，20s 内连不上直接失败）。
# 是 20000 就改成 80000（≈ 80s）；已是 80000 则跳过；两者都不是则报错
# （说明上游改了这一行，需要人工更新下面的 sed）。
STORE_TCP_CPP="${SHMEM_SOURCE_DIR}/src/host/bootstrap/config_store/store_tcp_config.cpp"
TCP_RETRY_TAG=""
if [[ ! -f "${STORE_TCP_CPP}" ]]; then
    echo "[ERROR] ${STORE_TCP_CPP} not found; shmem source layout changed."
    echo "        Update the sed fixup in scripts/build_shmem.sh."
    exit 1
elif grep -q "CONNECT_RETRY_MAX_TIMES = 20000;" "${STORE_TCP_CPP}"; then
    echo "[INFO] Widening SHMEM TCP connect retry window: 20000 -> 80000 (20s -> 80s)"
    sed -i \
        -e "s/CONNECT_RETRY_MAX_TIMES = 20000;/CONNECT_RETRY_MAX_TIMES = 80000;/" \
        -e "s/20000 retries/80000 retries/" \
        -e "s/ 20s total patience/ 80s total patience/" \
        "${STORE_TCP_CPP}"
    TCP_RETRY_TAG="-tcp80"
elif grep -q "CONNECT_RETRY_MAX_TIMES = 80000;" "${STORE_TCP_CPP}"; then
    TCP_RETRY_TAG="-tcp80"
else
    echo "[ERROR] ${STORE_TCP_CPP}: CONNECT_RETRY_MAX_TIMES is neither 20000 nor 80000."
    echo "        Upstream changed the line; update the sed fixup in scripts/build_shmem.sh."
    exit 1
fi

# 增量构建：已存在产物、源码 commit（含 fixup 标记）一致且未要求强制重建时直接复用。
# 通过环境变量 MEGA_MOE_FORCE_SHMEM_REBUILD=1 或传 --force-shmem-rebuild 强制重建。
FORCE_REBUILD=${MEGA_MOE_FORCE_SHMEM_REBUILD:-0}
PASSTHRU_ARGS=()
for arg in "$@"; do
    if [[ "${arg}" == "--force-shmem-rebuild" ]]; then
        FORCE_REBUILD=1
    else
        PASSTHRU_ARGS+=("${arg}")
    fi
done

SHMEM_SOURCE_COMMIT=$(git -C "${SHMEM_SOURCE_DIR}" rev-parse HEAD 2>/dev/null || true)
if [[ -n "${TCP_RETRY_TAG}" ]]; then
    # sed fixup 状态参与戳：fixup 生效前后编译的产物不会互相复用
    SHMEM_SOURCE_COMMIT="${SHMEM_SOURCE_COMMIT}${TCP_RETRY_TAG}"
fi
SHMEM_INSTALLED_COMMIT=""
if [[ -f "${SHMEM_SOURCE_STAMP}" ]]; then
    SHMEM_INSTALLED_COMMIT=$(<"${SHMEM_SOURCE_STAMP}")
fi

if [[ "${FORCE_REBUILD}" == "0" && \
      -n "${SHMEM_SOURCE_COMMIT}" && \
      "${SHMEM_INSTALLED_COMMIT}" == "${SHMEM_SOURCE_COMMIT}" && \
      -f "${SHMEM_INSTALL_DIR}/set_env.sh" ]]; then
    echo "[INFO] Reusing SHMEM installation: ${SHMEM_INSTALL_DIR}"
    echo "[INFO] SHMEM source commit: ${SHMEM_SOURCE_COMMIT}"
    echo "[INFO] Set MEGA_MOE_FORCE_SHMEM_REBUILD=1 to force a rebuild."
    exit 0
fi

if [[ -f "${SHMEM_INSTALL_DIR}/set_env.sh" && \
      "${SHMEM_INSTALLED_COMMIT}" != "${SHMEM_SOURCE_COMMIT}" ]]; then
    echo "[INFO] SHMEM source changed: ${SHMEM_INSTALLED_COMMIT:-unknown} -> ${SHMEM_SOURCE_COMMIT:-unknown}"
    echo "[INFO] Rebuilding SHMEM installation."
fi

pushd "${SHMEM_SOURCE_DIR}" >/dev/null
echo "[INFO] Building SHMEM... bash scripts/build.sh ${PASSTHRU_ARGS[*]}"
bash scripts/build.sh "${PASSTHRU_ARGS[@]}"
popd >/dev/null

if [[ ! -f "${SHMEM_INSTALL_DIR}/set_env.sh" ]]; then
    echo "[ERROR] SHMEM build finished but install/set_env.sh was not generated."
    exit 1
fi

if [[ -n "${SHMEM_SOURCE_COMMIT}" ]]; then
    printf '%s\n' "${SHMEM_SOURCE_COMMIT}" > "${SHMEM_SOURCE_STAMP}"
fi
