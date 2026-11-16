#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
ROOT="$TASK/source/MoonEP"
SDK="$TASK/dependencies/moonep-shmem/install"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export SHMEM_HOME_PATH="$SDK"
export LD_LIBRARY_PATH="$SDK/shmem/lib:$SDK/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$ROOT/npu_patch/3rdparty"
if [ ! -e "$ROOT/npu_patch/3rdparty/catlass" ]; then
 ln -s "$TASK/dependencies/moonep-catlass" "$ROOT/npu_patch/3rdparty/catlass"
fi
RT_INCLUDE="$ASCEND_HOME_PATH/aarch64-linux/pkg_inc/runtime/runtime"
if [ ! -f "$RT_INCLUDE/rt.h" ]; then RT_INCLUDE="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"; fi
cmake -DASCEND_MOONEP_SHMEM_SOURCE_DIR="$TASK/dependencies/moonep-shmem" -S "$ROOT/npu_patch/native" -B "$ROOT/build-ep8" -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" -DASCEND_MOONEP_BISHENG_ARCH=a5 -DSHMEM_HOME_PATH="$SDK" -DASCEND_MOONEP_RUNTIME_INCLUDE_DIR="$RT_INCLUDE"
cmake --build "$ROOT/build-ep8" --target ascend_moonep_kernels ascend_moonep_kernels_server_dedup --parallel 4
mkdir -p "$ROOT/npu_patch/lib"
cp "$ROOT/build-ep8/lib/"*.so "$ROOT/npu_patch/lib/"
for libdir in "$SDK/shmem/lib" "$SDK/memfabric_hybrid/lib"; do
 if [ -d "$libdir" ]; then find "$libdir" -maxdepth 1 \( -type f -o -type l \) \( -name "*.so" -o -name "*.so.*" \) -exec cp -P {} "$ROOT/npu_patch/lib/" \;; fi
done
cp -P /home/z00956592/v4pro-lab/operator/a8e809a/ascend_deepep/lib/libprofapi.so /home/z00956592/v4pro-lab/operator/a8e809a/ascend_deepep/lib/libprofimpl.so "$ROOT/npu_patch/lib/"
export MOONEP_BUILD_BACKEND=npu ASCEND_MOONEP_KERNEL_LIB_DIR="$ROOT/npu_patch/lib" MAX_JOBS=4
cd "$ROOT/npu_patch"
python3 setup.py build_ext --inplace
echo MOONEP_BUILD_COMPLETE
