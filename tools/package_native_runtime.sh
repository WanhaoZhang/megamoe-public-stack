#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
SDK=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
for arm in balanced shared; do
 DEST="$TASK/source/native-$arm/ascend_deepep/lib"
 for subdir in "$SDK/shmem/lib" "$SDK/memfabric_hybrid/lib"; do
  if [ -d "$subdir" ]; then
   find "$subdir" -maxdepth 1 \( -type f -o -type l \) \( -name 'libshmem*.so*' -o -name 'aclshmem_bootstrap*.so*' \) -exec cp -P {} "$DEST/" \;
  fi
 done
done
for arm in balanced shared; do
 cp "$TASK/source/native-$arm/build-cap64/lib/"*.so "$TASK/source/native-$arm/ascend_deepep/lib/"
done

for arm in balanced shared; do
 cp -P /home/z00956592/v4pro-lab/operator/a8e809a/ascend_deepep/lib/libprofapi.so /home/z00956592/v4pro-lab/operator/a8e809a/ascend_deepep/lib/libprofimpl.so "$TASK/source/native-$arm/ascend_deepep/lib/"
done
