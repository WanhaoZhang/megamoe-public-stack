#!/usr/bin/env bash
# Build two isolated, matched S9/panel variants; never overwrite installed SOs.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PREFIX=${1:?usage: build_megaflux_timer_capture.sh unique-prefix}
[[ "$PREFIX" =~ ^[a-z0-9-]+$ ]] || exit 2
: "${ASCEND_HOME_PATH:?Source matching CANN first}"
: "${SHMEM_HOME_PATH:?Set the SHMEM SDK root used by the installed binding}"
test -f "$ROOT/3rdparty/catlass/include/catlass/gemm_coord.hpp" || {
  echo 'Missing CATLASS dependency; run git submodule update --init --recursive first';exit 2;
}
for mode in off on; do
  test ! -e "$ROOT/worker-variants/$PREFIX-$mode" || { echo 'Use a new prefix'; exit 2; }
done
export MEGAFLUX_WORKER_SHARED_PREFIX=0 MEGAFLUX_WORKER_EXTERNAL_REFRESH=0
export MEGAFLUX_OWNER_ROWS=16 MEGAFLUX_OWNER_COMPACT=0 MEGAFLUX_DISPATCH_READY_LARGEST=0
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1 MEGAFLUX_RESTORE_DELAY_CYCLES=0
# Export the resolved value so worker and paired-Timer receipts agree.
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=${MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY:-1}
for mode in off on; do
  timer=0; if [[ "$mode" == on ]]; then timer=1; fi
  export MEGAFLUX_MOONEP_TIMER=$timer
  name=$PREFIX-$mode
  bash "$ROOT/scripts/build_megaflux_worker_pipeline.sh" "$name" 0 0 "$timer"
  build="$ROOT/build-worker-$name"; native="$ROOT/worker-variants/$name"
  # Keep the colleague's direct policy configurable elsewhere; this paired
  # diagnostic build deliberately uses the original strict MoonEP policy.
  cmake -S "$ROOT" -B "$build" -DMEGAFLUX_MOONEP_DIRECT_PERMILLE=0
  cmake --build "$build" --target ascend_deepep_moonep_planner ascend_deepep_large_ep_planner ascend_deepep_plan_validator --parallel "${MAX_JOBS:-2}"
  cp "$build/lib/libascend_deepep_moonep_planner.so" "$native/"
  cp "$build/lib/libascend_deepep_plan_validator.so" "$native/"
  cp "$build/lib/libascend_deepep_large_ep_planner.so" "$native/"
  python3 - "$ROOT" "$name" "$timer" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);name=sys.argv[2];timer=int(sys.argv[3])
native=root/'worker-variants'/name;build=root/('build-worker-'+name)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files=[root/'CMakeLists.txt',*sorted((root/'cmake').glob('*.cmake')),
       *sorted((root/'kernels').rglob('*.h')),*sorted((root/'kernels').rglob('*.hpp')),
       *sorted((root/'kernels/mega_moe_bf16').glob('*.cpp')),
       *sorted((root/'include').rglob('*.h')),
       root/'scripts/build_megaflux_worker_pipeline.sh']
files=[p for p in files if p.is_file()]
files.append(root/'scripts/build_megaflux_timer_capture.sh')
p=native/'build.json';old=json.loads(p.read_text())
record=dict(schema=1,timer=timer,worker_stage_schema=(8 if old['owner_aiv1_tail'] else 7) if timer else 0,
            planner_timer_schema=timer,flags=0,restore=2,owner_rows=16,
            direct_permille=0,diagnostic_only=True,
            owner_aiv1_tail=old['owner_aiv1_tail'],
            owner_double_buffer=old['owner_double_buffer'],
            shared_defer_gate_dw=old['shared_defer_gate_dw'],
            shared_reuse_gemm=old['shared_reuse_gemm'],
            backward_replicated_dw1_early=int(__import__('os').environ['MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY']),
            restore_panel_acquire_cache=int(__import__('os').environ.get('MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE','0')),
            shared_expert_forward=int(__import__('os').environ.get('MEGAFLUX_SHARED_EXPERT_FORWARD','0')),
            libraries={p.name:sha(p) for p in sorted(native.glob('*.so'))},
            sources={str(p.relative_to(root)):sha(p) for p in files},
            cmake_cache_sha256=sha(build/'CMakeCache.txt'))
(native/'timer-capture-build.json').write_text(json.dumps(record,indent=2))
# The second configure fixed direct=0; refresh the original worker cache receipt.
old['cmake_cache_sha256']=record['cmake_cache_sha256']
p.write_text(json.dumps(old,indent=2))
print('TIMER_CAPTURE_BUILT',name,json.dumps(record['libraries']))
PY
done
printf 'Run: bash integration/wuhan/run_timer_capture.sh --variants %q --out NEW_DIRECTORY\n' "$ROOT/worker-variants/$PREFIX"
