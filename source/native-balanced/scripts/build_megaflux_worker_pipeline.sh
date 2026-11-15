#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Source matching CANN first}"
NAME=${1:?variant name}; FWD=${2:?0/1}; BWD=${3:?0/1}; TIMER=${4:-0}
PREFIX=${MEGAFLUX_WORKER_SHARED_PREFIX:-0}
EXTERNAL=${MEGAFLUX_WORKER_EXTERNAL_REFRESH:-0}
OWNER_ROWS=${MEGAFLUX_OWNER_ROWS:-16}
OWNER_COMPACT=${MEGAFLUX_OWNER_COMPACT:-0}
OWNER_EARLY_RATIO=${MEGAFLUX_OWNER_EARLY_RATIO:-1}
OWNER_AIV1_TAIL=${MEGAFLUX_OWNER_AIV1_TAIL:-0}
INPUT_READY_LARGEST=${MEGAFLUX_DISPATCH_READY_LARGEST:-0}
RESTORE=${MEGAFLUX_BACKWARD_RESTORE:-0}
OPERAND_READY=${MEGAFLUX_RESTORE_OPERAND_READY:-1}
RESTORE_DELAY=${MEGAFLUX_RESTORE_DELAY_CYCLES:-0}
SWIGLU_DECOUPLE=${MEGAFLUX_SWIGLU_DECOUPLE:-auto}
GM_TILE_READY=${MEGAFLUX_GM_SWIGLU_TILE_READY:-0}
NONE_GM=${MEGAFLUX_BACKWARD_NONE_GM:-0}
[[ "$SWIGLU_DECOUPLE" =~ ^(auto|on|off)$ ]] || exit 2
[[ "$GM_TILE_READY" =~ ^[01]$ ]] || exit 2
[[ "$NONE_GM" =~ ^[01]$ ]] || exit 2
X_DELAY_SOURCE=${MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES:-0}
X_DELAY_PULL=${MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES:-0}
[[ "$X_DELAY_SOURCE" =~ ^[0-9]+$ && "$X_DELAY_PULL" =~ ^[0-9]+$ ]] || { echo 'X redispatch diagnostic delays must be nonnegative integers'; exit 2; }
# Match CMake's S9 panel default; explicit 0 retains the previous schedule.
REPLICATED_DW1_DEFAULT=0
if [[ "$RESTORE" == 2 ]]; then REPLICATED_DW1_DEFAULT=1; fi
REPLICATED_DW1_EARLY=${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY:-$REPLICATED_DW1_DEFAULT}
SHARED_BACKWARD=${MEGAFLUX_SHARED_EXPERT_BACKWARD:-0}
[[ "$NAME" =~ ^[a-z0-9-]+$ ]] || exit 2
BUILD="$ROOT/build-worker-$NAME"; NATIVE="$ROOT/worker-variants/$NAME"
[[ ! -e "$NATIVE/build.json" ]] || { echo 'Frozen variant exists; use a new name'; exit 2; }
mkdir -p "$NATIVE"
cmake -S "$ROOT" -B "$BUILD" \
 -DCMAKE_CXX_COMPILER="${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" \
 -DMEGAMOE_MAX_LOCAL_EXPERTS="${ASCEND_DEEPEP_MAX_LOCAL_EXPERTS:-32}" -DASCEND_DEEPEP_BISHENG_ARCH=a5 -DASCEND_DEEPEP_BUILD_MEGA_MOE_BF16=ON \
 -DASCEND_DEEPEP_ENABLE_TIMER=$([[ "$TIMER" == 1 ]] && echo ON || echo OFF) \
 -DMEGAFLUX_CALIBRATION_TIMER="${MEGAFLUX_CALIBRATION_TIMER:-OFF}" \
 -DMEGAFLUX_MOONEP_TIMER="${MEGAFLUX_MOONEP_TIMER:-OFF}" \
 -DASCEND_DEEPEP_BF16_GMM1_SCHEDULE=adaptive-auto \
 -DASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE="$SWIGLU_DECOUPLE" \
 -DASCEND_DEEPEP_BF16_GM_SWIGLU_TILE_READY=$([[ "$GM_TILE_READY" == 1 ]] && echo ON || echo OFF) \
 -DASCEND_DEEPEP_RUNTIME_INCLUDE_DIR="${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
 -DSHMEM_HOME_PATH="${SHMEM_HOME_PATH:-$ROOT/3rdparty/shmem/install}" \
 -DMEGAFLUX_WORKER_FORWARD="$FWD" -DMEGAFLUX_WORKER_BACKWARD="$BWD" \
 -DMEGAFLUX_BACKWARD_NONE_GM="$NONE_GM" \
 -DMEGAFLUX_SHARED_EXPERT_FORWARD="${MEGAFLUX_SHARED_EXPERT_FORWARD:-0}" \
 -DMEGAFLUX_BACKWARD_RESTORE="$RESTORE" \
 -DMEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY="$REPLICATED_DW1_EARLY" \
 -DMEGAFLUX_SHARED_EXPERT_BACKWARD="$SHARED_BACKWARD" \
 -DMEGAFLUX_RESTORE_OPERAND_READY="$OPERAND_READY" -DMEGAFLUX_RESTORE_DELAY_CYCLES="$RESTORE_DELAY" \
 -DMEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES="$X_DELAY_SOURCE" \
 -DMEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES="$X_DELAY_PULL" \
 -DMEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE="${MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE:-0}" \
 -DMEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE="${MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE:-0}" \
 -DMEGAFLUX_WORKER_SHARED_PREFIX="$PREFIX" \
 -DMEGAFLUX_WORKER_EXTERNAL_REFRESH="$EXTERNAL" -DMEGAFLUX_OWNER_ROWS="$OWNER_ROWS" \
 -DMEGAFLUX_OWNER_COMPACT="$OWNER_COMPACT" -DMEGAFLUX_OWNER_EARLY_RATIO="$OWNER_EARLY_RATIO" \
 -DMEGAFLUX_OWNER_DOUBLE_BUFFER="${MEGAFLUX_OWNER_DOUBLE_BUFFER:-OFF}" \
 -DMEGAFLUX_OWNER_AIV1_TAIL="$OWNER_AIV1_TAIL" -DMEGAFLUX_DISPATCH_READY_LARGEST="$INPUT_READY_LARGEST"
cmake --build "$BUILD" --target ascend_deepep_megaflux_worker_bf16 --parallel "${MAX_JOBS:-4}"
cp "$BUILD/lib/libascend_deepep_megaflux_worker_bf16.so" "$NATIVE/libascend_deepep_megaflux_early_bf16.so"
python3 - "$ROOT" "$NAME" "$FWD" "$BWD" "$TIMER" "$PREFIX" "$EXTERNAL" "$OWNER_ROWS" "$OWNER_COMPACT" "$RESTORE" "$OPERAND_READY" "$RESTORE_DELAY" "$SWIGLU_DECOUPLE" "$GM_TILE_READY" "$NONE_GM" "$OWNER_AIV1_TAIL" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); name=sys.argv[2];fwd,bwd,timer,prefix,external,owner_rows,owner_compact,restore,operand_ready,restore_delay=map(int,sys.argv[3:13])
swiglu_decouple=sys.argv[13];gm_tile_ready=int(sys.argv[14]);none_gm=int(sys.argv[15])
x_delay_source=int(__import__('os').environ.get('MEGAFLUX_X_REDISPATCH_DELAY_SOURCE_CYCLES','0'))
x_delay_pull=int(__import__('os').environ.get('MEGAFLUX_X_REDISPATCH_DELAY_PULL_CYCLES','0'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[r/'CMakeLists.txt',r/'scripts/build_megaflux_worker_pipeline.sh',*sorted((r/'cmake').glob('*.cmake')),
       *sorted((r/'include').rglob('*.h')),*sorted((r/'kernels').rglob('*.h')),
       *sorted((r/'kernels').rglob('*.hpp')),*sorted((r/'kernels/mega_moe_bf16').glob('*.cpp'))]
b=r/('build-worker-'+name);n=r/'worker-variants'/name
effective_swiglu=next(int(line.split('=',1)[1]) for line in (b/'CMakeCache.txt').read_text().splitlines()
    if line.startswith('MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE:INTERNAL='))
cache_values=dict(line.split('=',1) for line in (b/'CMakeCache.txt').read_text().splitlines()
    if line.startswith(('MEGAFLUX_OWNER_DOUBLE_BUFFER:BOOL=', 'MEGAFLUX_SHARED_DEFER_GATE_DW:BOOL=',
                        'MEGAFLUX_SHARED_REUSE_GEMM:BOOL=')))
def enabled(key):
    return cache_values.get(key + ':BOOL', 'OFF').upper() not in ('0', 'OFF', 'NO', 'FALSE', '')
d=dict(name=name,forward=fwd,backward=bwd,timer=timer,shared_prefix=prefix,external_refresh=external,owner_rows=owner_rows,owner_compact=owner_compact,backward_restore=restore,
 swiglu_decouple=swiglu_decouple,swiglu_decouple_mode={'off':0,'on':1,'auto':2}[swiglu_decouple],
 s9_effective_swiglu_decouple_mode=effective_swiglu,gm_swiglu_tile_ready=gm_tile_ready,
 backward_none_gm_version=none_gm,
 shared_expert_forward=int(__import__('os').environ.get('MEGAFLUX_SHARED_EXPERT_FORWARD','0')),
 owner_early_ratio=int(__import__('os').environ.get('MEGAFLUX_OWNER_EARLY_RATIO','1')),
 owner_aiv1_tail=int(sys.argv[16]),
 owner_double_buffer=enabled('MEGAFLUX_OWNER_DOUBLE_BUFFER'),
 shared_defer_gate_dw=enabled('MEGAFLUX_SHARED_DEFER_GATE_DW'),
 shared_reuse_gemm=enabled('MEGAFLUX_SHARED_REUSE_GEMM'),
 restore_operand_ready=operand_ready,restore_delay_cycles=restore_delay,
 x_redispatch_delay_source_cycles=x_delay_source,x_redispatch_delay_pull_cycles=x_delay_pull,
 x_redispatch_diagnostic_only=bool(x_delay_source or x_delay_pull),
 restore_panel_acquire_cache=int(__import__('os').environ.get('MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE','0')),
 restore_p1_panel_acquire_cache=int(__import__('os').environ.get('MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE','0')),
 expected_flags=fwd|(bwd<<1),
 native_sha256=sha(n/'libascend_deepep_megaflux_early_bf16.so'),
 sources={str(p.relative_to(r)):sha(p) for p in files},
 compile_files={str(p.relative_to(b)):sha(p) for p in b.rglob('flags.make')},
 cmake_cache_sha256=sha(b/'CMakeCache.txt'))
(n/'build.json').write_text(json.dumps(d,indent=2))
print('WORKER_PIPELINE_BUILT',name,d['native_sha256'],d['expected_flags'])
PY

python3 - "$NATIVE" "$INPUT_READY_LARGEST" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);(p/"input-ready-selection.json").write_text(json.dumps(dict(largest=int(sys.argv[2]),new_counter_bytes=0,expert_order_changed=False)))
PY

python3 - "$NATIVE" "$REPLICATED_DW1_EARLY" "$SHARED_BACKWARD" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'build.json'
d=json.loads(p.read_text());d['backward_replicated_dw1_early']=int(sys.argv[2])
d['shared_expert_backward']=int(sys.argv[3])
d['shared_expert_backward_abi']=int(sys.argv[3])
d['shared_expert_backward_v2_abi']=2 if int(sys.argv[3]) else 0
d['shared_expert_backward_v3_abi']=3 if int(sys.argv[3]) else 0
d['shared_expert_backward_v3_bytes']=200 if int(sys.argv[3]) else 0
d['swiglu_decouple']=__import__('os').environ.get('MEGAFLUX_SWIGLU_DECOUPLE','auto')
d['swiglu_decouple_mode']={'off':0,'on':1,'auto':2}[d['swiglu_decouple']]
cache=(p.parents[2]/('build-worker-'+p.parent.name)/'CMakeCache.txt').read_text()
# _MODE records the request; the effective alias preserves legacy S9 auto.
effective=[line.split('=',1)[1] for line in cache.splitlines()
           if line.startswith('MEGAFLUX_EFFECTIVE_SWIGLU_DECOUPLE_MODE:INTERNAL=')]
if len(effective)!=1 or effective[0] not in ('0','1','2'):
    raise RuntimeError('missing S9 effective SwiGLU mode in CMake cache')
d['s9_effective_swiglu_decouple_mode']=int(effective[0])
d['gm_swiglu_tile_ready']=int(__import__('os').environ.get('MEGAFLUX_GM_SWIGLU_TILE_READY','0'))
d['backward_none_gm_version']=int(__import__('os').environ.get('MEGAFLUX_BACKWARD_NONE_GM','0'))
p.write_text(json.dumps(d,indent=2))
PY
