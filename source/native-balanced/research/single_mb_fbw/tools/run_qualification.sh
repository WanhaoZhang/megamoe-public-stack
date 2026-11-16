#!/usr/bin/env bash
set -euo pipefail
cd /home/z00956592/megaflux/single-mb-fbw-planner-study
python3 tools/host_guard.py --container single-mb-fbw-planner-study -- python3 tools/qualification_stage.py
