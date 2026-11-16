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
        --performance-only)
            ARGS+=(--mode performance)
            shift
            ;;
        --keep-data)
            # Python tests retain their data directory by default.
            shift
            ;;
        --binary)
            echo "[ERROR] --binary belongs to the legacy C++ stub runner."
            echo "[ERROR] The project test now loads ascend_deepep.MegaMoe from Python."
            exit 1
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done

exec "${PYTHON_BIN}" "${TEST_DIR}/test_mega_moe.py" "${ARGS[@]}"
