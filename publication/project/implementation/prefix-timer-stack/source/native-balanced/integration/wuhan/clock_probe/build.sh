#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${ASCEND_HOME_PATH:?source the existing CANN set_env.sh}"
: "${CLOCK_PROBE_BUILD_DIR:?set a fresh private build directory}"
cmake -S "$HERE" -B "$CLOCK_PROBE_BUILD_DIR" \
  -DCMAKE_CXX_COMPILER="${BISHENG:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}" \
  -DASCEND_HOME_PATH="$ASCEND_HOME_PATH" \
  -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR:-}"
cmake --build "$CLOCK_PROBE_BUILD_DIR" --parallel --target deepep_clock_probe
