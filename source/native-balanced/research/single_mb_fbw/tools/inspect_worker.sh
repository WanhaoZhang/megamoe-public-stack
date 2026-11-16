#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
export LD_LIBRARY_PATH="$task/source/native-balanced/ascend_deepep/lib:$task/dependencies/sdk/shmem/lib:${LD_LIBRARY_PATH:-}"
python3 - "$task" <<'PY'
import torch,ctypes,json,sys,pathlib,hashlib
p=pathlib.Path(sys.argv[1]);so=p/'source/native-balanced/worker-variants/ep8-cap64-aligned/libascend_deepep_megaflux_early_bf16.so'
lib=ctypes.CDLL(str(so));names=['megaflux_worker_pipeline_flags','megaflux_worker_shared_prefix','megaflux_worker_owner_rows','megaflux_worker_owner_compact','megaflux_worker_owner_early_ratio','megaflux_worker_owner_aiv1_tail','megaflux_backward_restore_mode','megaflux_restore_operand_ready','megaflux_worker_swiglu_decoupling_mode','megaflux_worker_gm_swiglu_tile_ready','megaflux_restore_panel_acquire_cache','megaflux_restore_p1_panel_acquire_cache','megaflux_worker_external_refresh','megaflux_dispatch_ready_largest','megaflux_backward_replicated_dw1_early']
r={n:int(getattr(lib,n)()) for n in names if hasattr(lib,n)}
for n in ('megaflux_shared_expert_forward_abi','megaflux_shared_expert_backward_version',
          'megaflux_shared_expert_backward_v2_version','megaflux_shared_expert_backward_v3_version',
          'megaflux_routed_redispatch_version'):
    r[n]=int(getattr(lib,n)())
r['so_sha256']=hashlib.sha256(so.read_bytes()).hexdigest()
(p/'evidence/baseline-worker-flags.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
PY
