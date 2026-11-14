#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Source matching CANN first}"
SHMEM_HOME_PATH=${SHMEM_HOME_PATH:-"$ROOT/3rdparty/shmem/install"}
test -f "$ROOT/ascend_deepep/lib/libascend_deepep_megaflux_full_bf16.so"
test -f "$ROOT/ascend_deepep/lib/libascend_deepep_replica_optimized.so"
cd "$ROOT"
SHMEM_HOME_PATH="$SHMEM_HOME_PATH" "${MEGAFLUX_PYTHON:-python3}" -m pip install -e . --no-build-isolation --no-deps
