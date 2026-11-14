#!/usr/bin/env bash
set -euo pipefail
TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${MEGA_MOE_W4A8_PYTHON:-python3}
exec "${PYTHON_BIN}" "${TEST_DIR}/test_mega_moe_w4a8.py" "$@"
