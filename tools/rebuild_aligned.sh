#!/usr/bin/env bash
set -euo pipefail
TASK=/home/z00956592/megaflux/planner-executor-ep8-lab
NAME=$1
CAP=$2
ROOT="$TASK/source/native-$NAME"
source /home/pkg/b070/ascend-toolkit/set_env.sh
source /home/pkg/b070/cann-9.2.0/set_env.sh
export PATH=/usr/local/python3.12.13/bin:$PATH
export LD_PRELOAD=/home/z00956592/v4pro-lab/operator/a8e809a/libstdc++/libstdc++.so.6.0.35
cd "$ROOT"
cmake --build "build-worker-ep8-cap$CAP-timeroff" --target ascend_deepep_megaflux_worker_bf16 --parallel 2
NEW="worker-variants/ep8-cap$CAP-aligned"
mkdir -p "$NEW"
cp "build-worker-ep8-cap$CAP-timeroff/lib/libascend_deepep_megaflux_worker_bf16.so" "$NEW/libascend_deepep_megaflux_early_bf16.so"
python3 - "$NEW" "$NAME" "$CAP" <<'PY'
import hashlib,json,pathlib,sys
new,name,cap=sys.argv[1:];r=pathlib.Path(new);so=r/'libascend_deepep_megaflux_early_bf16.so'
(r/'build.json').write_text(json.dumps(dict(name=name,capacity=int(cap),timer=False,change='private shared backward kernel splitExperts u64 alignment',sha256=hashlib.sha256(so.read_bytes()).hexdigest()),indent=2))
PY
printf 'ALIGNED_WORKER_BUILD_COMPLETE %s %s\n' "$NAME" "$CAP"
