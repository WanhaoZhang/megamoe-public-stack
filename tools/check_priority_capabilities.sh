#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
export LD_LIBRARY_PATH="$task/source/native-balanced/ascend_deepep/lib:$task/dependencies/sdk/shmem/lib:${LD_LIBRARY_PATH:-}"
python3 - "$task" "${PRIORITY_VARIANT_PREFIX:-matched-priority}" "${EARLY_DW1_WIDE:-0}" <<'PY'
import torch,ctypes,json,sys,pathlib,hashlib
p=pathlib.Path(sys.argv[1]);prefix=sys.argv[2];wide=int(sys.argv[3]);baseline=json.loads((p/'evidence/baseline-worker-flags.json').read_text());result=[]
for priority in (0,1):
    so=p/f'source/native-balanced/worker-variants/{prefix}{priority}/libascend_deepep_megaflux_early_bf16.so'
    lib=ctypes.CDLL(str(so))
    actual={n:int(getattr(lib,n)()) for n in baseline if n!='so_sha256'}
    assert actual=={k:v for k,v in baseline.items() if k!='so_sha256'},(priority,actual,baseline)
    assert int(lib.megaflux_gradient_priority())==priority
    assert (int(lib.megaflux_early_dw1_wide()) if hasattr(lib,'megaflux_early_dw1_wide') else 0)==wide
    result.append(dict(priority=priority,capabilities=actual,so_sha256=hashlib.sha256(so.read_bytes()).hexdigest()))
(p/f'evidence/{prefix}-capabilities.json').write_text(json.dumps(result,indent=2));print('PRIORITY_CAPABILITIES_PASS',json.dumps(result))
PY
