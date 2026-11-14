set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
ROOT="$TASK/source/native-balanced"
export TASK ROOT
python3 - <<'PY'
import hashlib,json,os,pathlib
r=pathlib.Path(os.environ['TASK']);n=pathlib.Path(os.environ['ROOT'])
assert json.loads((r/'evidence/CALIBRATION-ACCEPTANCE.json').read_text())['passed'],'finish independent calibration first'
assert (n/'include/mega_moe_step_time_policy.h').is_file(),'no implemented cost/search policy'
freeze=json.loads((r/'evidence/BASELINE-FREEZE.json').read_text())
core={p:s for p,s in freeze['binaries'].items() if any(x in p for x in
    ('_C.cpython','libascend_deepep_unified_planner.so','worker-variants/ep8-cap64-aligned/libascend_deepep_megaflux_early_bf16.so'))}
assert len(core)==3
for path,digest in core.items():assert hashlib.sha256((r/path).read_bytes()).hexdigest()==digest,path
(r/'evidence/step-time-build-baseline-core.json').write_text(json.dumps(core,indent=2))
PY
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export SHMEM_HOME_PATH=/home/z00956592/megaflux/shared-host-progress-20261110/inputs-c15/frozen-runtime/dependencies/sdk
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
export LD_LIBRARY_PATH="$SHMEM_HOME_PATH/shmem/lib:$SHMEM_HOME_PATH/memfabric_hybrid/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_DEEPEP_MAX_LOCAL_EXPERTS=64
cd "$ROOT"
cmake -S . -B build-step-time-cap64 -DCMAKE_CXX_COMPILER="$ASCEND_HOME_PATH/tools/bisheng_compiler/bin/bisheng" -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_ENABLE_TIMER=OFF -DMEGAMOE_MAX_LOCAL_EXPERTS=64 -DSHMEM_HOME_PATH="$SHMEM_HOME_PATH" -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="$ASCEND_HOME_PATH/pkg_inc/runtime/runtime"
cmake --build build-step-time-cap64 --target ascend_deepep_step_time_planner --parallel 2
cp build-step-time-cap64/lib/libascend_deepep_step_time_planner.so ascend_deepep/lib/libascend_deepep_step_time_planner.so.stage
mv ascend_deepep/lib/libascend_deepep_step_time_planner.so.stage ascend_deepep/lib/libascend_deepep_step_time_planner.so
python3 - <<'PY'
from datetime import datetime,timezone
import hashlib,json,os,pathlib
r=pathlib.Path(os.environ['TASK']);n=pathlib.Path(os.environ['ROOT'])
for path,digest in json.loads((r/'evidence/step-time-build-baseline-core.json').read_text()).items():
    assert hashlib.sha256((r/path).read_bytes()).hexdigest()==digest,'frozen Host/planner/worker changed'
paths=[n/'include/mega_moe_step_time_policy.h',n/'kernels/mega_moe_bf16/mega_moe_step_time_planner.cpp',
       n/'ascend_deepep/lib/libascend_deepep_step_time_planner.so',n/'build-step-time-cap64/CMakeCache.txt']
record=dict(time_utc=datetime.now(timezone.utc).isoformat(),timer=False,base_core_unchanged=True,
    files={p.relative_to(r).as_posix():dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in paths})
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
(r/f'evidence/step-time-build-{stamp}.json').write_text(json.dumps(record,indent=2))
print('STEP_TIME_SEPARATE_PLUGIN_BUILT',record,flush=True)
PY
