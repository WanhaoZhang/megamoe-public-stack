set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
export TASK
python3 - <<'PY'
import json,os,pathlib,sys
r=pathlib.Path(os.environ['TASK'])
sys.path.insert(0,str(r/'tools'))
from campaign_records import resolved_existing
rows=resolved_existing(r)
assert len(rows)==90
p=r/'source/native-balanced/kernels/mega_moe_bf16/mega_moe_link_calibration.cpp'
s=p.read_text()
needle='    output.SetGlobalBuffer((__gm__ uint8_t*)base+targetoff);'
if 'input.SetL2CacheHint' not in s:
 assert s.count(needle)==1
 p.write_text(s.replace(needle,needle+'\n    input.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);\n    output.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);'))
PY
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
cd "$TASK/source/native-balanced"
cmake --build build-worker-unified-cap64-calibration-timer --target ascend_deepep_link_calibration --parallel 2
cp build-worker-unified-cap64-calibration-timer/lib/libascend_deepep_link_calibration.so ascend_deepep/lib/libascend_deepep_link_calibration.so
mkdir -p worker-variants/unified-cap64-calibration-timer
cp ascend_deepep/lib/libascend_deepep_link_calibration.so worker-variants/unified-cap64-calibration-timer/libascend_deepep_link_calibration.so
sha256sum ascend_deepep/lib/libascend_deepep_link_calibration.so
python3 - <<'PY'
from datetime import datetime,timezone
import hashlib,json,os,pathlib
r=pathlib.Path(os.environ['TASK']);n=r/'source/native-balanced'
paths=[n/'kernels/mega_moe_bf16/mega_moe_link_calibration.cpp',n/'ascend_deepep/lib/libascend_deepep_link_calibration.so',
       n/'build-worker-unified-cap64-calibration-timer/CMakeCache.txt']
receipt=dict(time_utc=datetime.now(timezone.utc).isoformat(),diagnostic_only=True,
    files={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    baseline_formal_host_planner_worker_unchanged=True,
    provenance='source/build products at diagnostic admission; formal processes never call the link-calibration entry')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
(r/f'evidence/link-build-{stamp}.json').write_text(json.dumps(receipt,indent=2))
PY
