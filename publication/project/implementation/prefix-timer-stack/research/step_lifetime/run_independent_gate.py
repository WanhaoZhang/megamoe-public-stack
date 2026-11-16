"""Admitted independent FFN output/dx/dp/dW gate for all four lifetime modes."""
from pathlib import Path
import json
import remote
root=Path(__file__).resolve().parents[3]
remote.REMOTE='/home/z00956592/megaflux/step-lifetime-kernel-20261115'
remote.CONTAINER='step-lifetime-kernel-20261115'
tag='lifetime-independent-ffn-v3'
try:
 remote.gate(tag,['--mode','lifetime_math','--worker-variant','ep8-cap64-lifetime-v1',
   '--lifetime-mode','all','--route-bank',remote.REMOTE+'/stack/small-routes'],metadata_only=True)
except SystemExit as result:
 code=result.code
records=[]
for rank in range(8):
 path=root/'evidence'/tag/f'rank{rank}.json'
 if path.exists():records.extend(json.loads(path.read_text())['records'])
passed=(code==0 and len(records)==64 and all(item['passed'] for item in records))
hits=max((item['lifetime_stats']['weight_hit_bytes'] for item in records),default=0)
traffic=max((item['lifetime_stats']['gradient_remote_bytes'] for item in records),default=0)
passed=passed and hits>0 and traffic>0
proof=dict(actual_cache_hit_bytes=hits,actual_gradient_remote_bytes=traffic,passed=passed,exit=code,records=len(records),expected_records=64,tag=tag,
 scope='eight ranks, b0-b3, W1/W2, checkpoint replay, independent output/dx/dp/owner-dW with EP scaling',
 output_threshold=.004,gradient_threshold=.02)
(root/'diagnostics/independent-lifetime-math.json').write_text(json.dumps(proof,indent=2))
if passed:(root/'HOLD-B').unlink(missing_ok=True)
raise SystemExit(0 if passed else 1)
