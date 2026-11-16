#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEGA_MOE_BF16_MGROUP=1 bash "${SCRIPT_DIR}/test_mega_moe.sh" "$@"
