#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD="$task/dependencies/runtime/libstdc++.so.6.0.35"
export LD_LIBRARY_PATH="$task/source/native-balanced/ascend_deepep/lib:$task/dependencies/sdk/shmem/lib:${LD_LIBRARY_PATH:-}"
python3 - "$task" <<'PY'
from pathlib import Path
import ctypes,hashlib,json,sys,torch
task=Path(sys.argv[1]);baseline=json.loads((task/'evidence/baseline-worker-flags.json').read_text());rows=[]
archive_sha=hashlib.sha256((task/'artifacts/native-gradient-key-cache-source.tar').read_bytes()).hexdigest()
for mode in (0,1):
 variant=task/f'source/native-balanced/worker-variants/gradient-key-cache{mode}'
 so=variant/'libascend_deepep_megaflux_early_bf16.so';manifest=json.loads((variant/'build.json').read_text())
 assert hashlib.sha256(so.read_bytes()).hexdigest()==manifest['so_sha256']
 assert manifest['source_archive_sha256']==archive_sha
 cache=task/f'source/native-gradient-key-cache/build-gradient-key-cache{mode}/CMakeCache.txt'
 assert hashlib.sha256(cache.read_bytes()).hexdigest()==manifest['cmake_cache_sha256']
 lib=ctypes.CDLL(str(so));actual={name:int(getattr(lib,name)()) for name in baseline if name!='so_sha256'}
 assert actual=={name:value for name,value in baseline.items() if name!='so_sha256'}
 assert int(lib.megaflux_owner_late_only())==0
 assert int(lib.megaflux_gradient_key_cache())==mode
 assert int(lib.megaflux_redispatch_p1_runahead())==0
 assert int(lib.megaflux_gradient_priority())==int(lib.megaflux_early_dw1_wide())==0
 assert int(lib.megaflux_early_bf16_timer_enabled())==int(lib.megaflux_early_bf16_backward_timer_enabled())==0
 assert manifest['compile_flags']['MEGAFLUX_GRADIENT_KEY_CACHE']==str(mode)
 rows.append(dict(mode=mode,capabilities=actual,**manifest))
(task/'evidence/gradient-key-cache-capabilities.json').write_text(json.dumps(rows,indent=2))
print('GRADIENT_KEY_CACHE_CAPABILITIES_PASS',flush=True)
PY
