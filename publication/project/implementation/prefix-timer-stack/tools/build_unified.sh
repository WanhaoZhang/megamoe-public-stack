set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
CAP=${EP8_BUILD_CAP:-64}
ROOT="$TASK/source/native-balanced${EP8_BUILD_SUFFIX:-}"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS="$CAP" MAX_JOBS=4
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
cmake -S . -B "build-unified-cap$CAP" -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DMEGAMOE_MAX_LOCAL_EXPERTS="$CAP" -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
cmake --build "build-unified-cap$CAP" --target ascend_deepep_unified_planner --parallel 4
cp "build-unified-cap$CAP/lib/libascend_deepep_unified_planner.so" "ascend_deepep/lib/libascend_deepep_unified_planner.so.stage-$$"
mv "ascend_deepep/lib/libascend_deepep_unified_planner.so.stage-$$" ascend_deepep/lib/libascend_deepep_unified_planner.so
export ASCEND_DEEPEP_KERNEL_LIB_DIR="$ROOT/ascend_deepep/lib"
python3 setup.py build_ext --inplace
python3 - "$TASK" "$ROOT" "$CAP" <<'PY'
import hashlib,json,pathlib,sys
r,n,cap=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
files=list((n/'ascend_deepep').rglob('*.so'))+list((n/('worker-variants/ep8-cap'+cap+'-aligned')).glob('*.so'))
out={str(p.relative_to(n)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(r/('evidence/unified-build-cap'+cap+'-binaries.json')).write_text(json.dumps(out,indent=2))
print('UNIFIED_BUILD_COMPLETE',len(out),flush=True)
PY
