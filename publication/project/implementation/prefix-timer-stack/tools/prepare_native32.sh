#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
for arm in balanced shared; do
 DEST="$TASK/source/native-${arm}32"
 test ! -e "$DEST"
 mkdir "$DEST"
 tar -C "$TASK/source/native-$arm" --exclude='build-*' --exclude='build' --exclude='worker-variants' --exclude='ascend_deepep/lib' --exclude='*.so' --exclude='__pycache__' --exclude='3rdparty' -cf - . | tar -C "$DEST" -xf -
 mkdir -p "$DEST/3rdparty"
done
