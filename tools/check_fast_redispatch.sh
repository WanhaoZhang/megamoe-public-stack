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
archive_sha=hashlib.sha256((task/'artifacts/native-fast-redispatch-source.tar').read_bytes()).hexdigest()
specs=[(f'fastpipe-r{r}-t{b}',r,b,1,False) for r,b in ((0,0),(3,0),(4,0),(5,0),(5,1),(5,2))]
specs += [(f'fastlite-cache{c}',0,0,c,True) for c in (0,1)]
for name,runahead,batch,cache,timer in specs:
 variant=task/'source/native-balanced/worker-variants'/name
 so=variant/'libascend_deepep_megaflux_early_bf16.so';manifest=json.loads((variant/'build.json').read_text())
 assert manifest['source_archive_sha256']==archive_sha
 assert hashlib.sha256(so.read_bytes()).hexdigest()==manifest['so_sha256']
 cmake=task/f'source/native-fast-redispatch/build-fast-{name}/CMakeCache.txt'
 assert hashlib.sha256(cmake.read_bytes()).hexdigest()==manifest['cmake_cache_sha256']
 lib=ctypes.CDLL(str(so));actual={n:int(getattr(lib,n)()) for n in baseline if n!='so_sha256'}
 assert actual=={n:v for n,v in baseline.items() if n!='so_sha256'}
 assert int(lib.megaflux_redispatch_p1_runahead())==runahead
 assert int(lib.megaflux_redispatch_transport_batch())==batch
 assert int(lib.megaflux_gradient_key_cache())==cache
 assert int(lib.megaflux_dispatch_stage_schema())==(10 if timer else 0)
 assert bool(lib.megaflux_early_bf16_timer_enabled())==bool(lib.megaflux_early_bf16_backward_timer_enabled())==timer
 assert int(lib.megaflux_owner_late_only())==int(lib.megaflux_gradient_priority())==int(lib.megaflux_early_dw1_wide())==0
 for field,value in [('MEGAFLUX_REDISPATCH_P1_RUNAHEAD',str(runahead)),('MEGAFLUX_REDISPATCH_TRANSPORT_BATCH',str(batch)),
                      ('MEGAFLUX_GRADIENT_KEY_CACHE',str(cache)),('MEGAFLUX_COORDINATE_TIMER','OFF')]:
  assert manifest['compile_flags'][field]==value,(name,field)
 rows.append(dict(capabilities=actual,**manifest))
assert len({r['source_archive_sha256'] for r in rows})==1
(task/'evidence/fast-redispatch-capabilities-v1.json').write_text(json.dumps(rows,indent=2))
print('FAST_REDISPATCH_CAPABILITIES_PASS',len(rows),flush=True)
PY
