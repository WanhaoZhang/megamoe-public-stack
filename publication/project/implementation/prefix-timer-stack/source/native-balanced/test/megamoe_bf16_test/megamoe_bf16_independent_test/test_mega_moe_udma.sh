#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MEGA_MOE_BF16_UDMA=1 \
MEGA_MOE_BF16_INDEPENDENT_RUNNER="${SCRIPT_DIR}/install_udma/bin/mega_moe_bf16_udma_stub" \
    bash "${SCRIPT_DIR}/test_mega_moe.sh" "$@"
