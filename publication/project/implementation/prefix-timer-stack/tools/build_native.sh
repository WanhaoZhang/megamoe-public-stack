#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
NAME=$1
CAP=${2:-64}
ROOT="$TASK/source/native-$NAME"
DEPS="${EP8_NATIVE_DEPS:-/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies}"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export SHMEM_HOME_PATH="$DEPS/sdk"
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS="$CAP"
export MAX_JOBS=4
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$TASK/evidence/build-$NAME-$CAP" "$ROOT/ascend_deepep/lib"
mkdir -p "$ROOT/3rdparty"
for dep in catlass catccos; do
 if [ -d "$ROOT/3rdparty/$dep" ] && [ -z "$(ls -A "$ROOT/3rdparty/$dep")" ]; then rmdir "$ROOT/3rdparty/$dep"; fi
 if [ ! -e "$ROOT/3rdparty/$dep" ]; then ln -s "$DEPS/$dep" "$ROOT/3rdparty/$dep"; fi
done
cd "$ROOT"
cmake -S . -B "build-cap$CAP" -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DMEGAMOE_MAX_LOCAL_EXPERTS="$CAP" -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
cmake --build "build-cap$CAP" --target ascend_deepep_kernels ascend_deepep_mega_moe_bf16 ascend_deepep_replica ascend_deepep_replica_control ascend_deepep_plan_validator ascend_deepep_moonep_planner ascend_deepep_large_ep_planner --parallel 4
cp build-cap"$CAP"/lib/*.so ascend_deepep/lib/
for libdir in "$SHMEM_HOME_PATH/shmem/lib" "$SHMEM_HOME_PATH/memfabric_hybrid/lib"; do
 if [ -d "$libdir" ]; then find "$libdir" -maxdepth 1 \( -type f -o -type l \) \( -name 'libshmem*.so*' -o -name 'aclshmem_bootstrap*.so*' \) -exec cp -P {} ascend_deepep/lib/ \;; fi
done
cp -P /home/z00956592/v4pro-lab/operator/a8e809a/ascend_deepep/lib/libprofapi.so /home/z00956592/v4pro-lab/operator/a8e809a/ascend_deepep/lib/libprofimpl.so "$ROOT/ascend_deepep/lib/"
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1
export MEGAFLUX_SHARED_EXPERT_FORWARD=1 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 MEGAFLUX_OWNER_DOUBLE_BUFFER=OFF
if [[ "$NAME" == shared* ]]; then export MEGAFLUX_OWNER_DOUBLE_BUFFER=ON; fi
bash scripts/build_megaflux_worker_pipeline.sh "ep8-cap$CAP-timeroff" 0 0 0
export ASCEND_DEEPEP_KERNEL_LIB_DIR="$ROOT/ascend_deepep/lib"
python3 setup.py build_ext --inplace
python3 - "$TASK" "$NAME" "$CAP" <<'PY'
import hashlib,json,pathlib,sys
task,name,cap=sys.argv[1:];r=pathlib.Path(task)/'source'/('native-'+name)
files=list((r/'ascend_deepep').rglob('*.so'))+list((r/'worker-variants'/('ep8-cap'+cap+'-timeroff')).glob('*.so'))
out={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(pathlib.Path(task)/'evidence'/('build-'+name+'-'+cap)/'binaries.json').write_text(json.dumps(out,indent=2))
print('NATIVE_BUILD_COMPLETE',name,cap,len(out),flush=True)
PY
