#!/usr/bin/env bash
set -euo pipefail
task=/home/z00956592/megaflux/single-mb-fbw-planner-study
python3 - "$task" "${PRIORITY_VARIANT_PREFIX:-matched-priority}" <<'PY'
from pathlib import Path
import hashlib,json,sys
task=Path(sys.argv[1]);prefix=sys.argv[2]
root=task/'source'/('native-wide' if prefix=='wide-priority' else 'native-priority')
for priority in (0,1):
    variant=task/f'source/native-balanced/worker-variants/{prefix}{priority}'
    cache=root/f'build-{prefix}{priority}/CMakeCache.txt'
    flags={}
    for line in cache.read_text().splitlines():
        if '=' not in line or ':' not in line:continue
        key,value=line.split('=',1);key=key.split(':',1)[0]
        if key.startswith(('MEGAFLUX_','ASCEND_DEEPEP_','MEGAMOE_')):flags[key]=value
    assert flags['MEGAFLUX_SHARED_EXPERT_FORWARD']=='1'
    assert flags['MEGAFLUX_SHARED_EXPERT_BACKWARD']=='1'
    assert flags['MEGAFLUX_GRADIENT_PRIORITY']==str(priority)
    assert flags.get('MEGAFLUX_EARLY_DW1_WIDE','0')==str(int(prefix=='wide-priority'))
    manifest=json.loads((variant/'build.json').read_text())
    so=variant/'libascend_deepep_megaflux_early_bf16.so'
    assert manifest['so_sha256']==hashlib.sha256(so.read_bytes()).hexdigest()
    manifest.update(cmake_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),compile_flags=flags,
                    qualification='capability_checked_math_pending')
    (variant/'build.json').write_text(json.dumps(manifest,indent=2))
print('WORKER_BUILD_RECIPES_SEALED',prefix)
PY
