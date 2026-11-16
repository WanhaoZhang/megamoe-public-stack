#!/usr/bin/env bash
set -euo pipefail

TEST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN="python3"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            if [[ $# -lt 2 ]]; then
                echo "[ERROR] --python requires an executable"
                exit 1
            fi
            PYTHON_BIN="$2"
            shift 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
    ARGS+=(0,1)
fi

exec "${PYTHON_BIN}" "${TEST_DIR}/test_mega_moe_backward.py" "${ARGS[@]}"
