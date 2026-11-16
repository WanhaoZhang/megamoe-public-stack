"""Run or collect the admitted, forced-replica independent numerical gate."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import remote
ROOT=Path(__file__).resolve().parent
CORE=('tools/lifetime_device.py','tools/step_lifetime.py','tools/grouped_experts.py','tools/train.py',
      'tools/temporal_model.py','tools/temporal_prefix_reuse.py',
      'source/native-balanced/ascend_deepep/fsdp2_bridge.py',
      'source/native-balanced/ascend_deepep/megaflux_worker_pipeline.py')

def collect(tag,code):
 folder=ROOT/'evidence'/tag
 records=[]
 for rank in range(8):
  path=folder/f'rank{rank}.json'
  if path.exists():records.extend(json.loads(path.read_text())['records'])
 status=json.loads((folder/'status.json').read_text())
 sources=json.loads((folder/'source-identity.json').read_text())
 hits=max((item['lifetime_stats']['weight_hit_bytes'] for item in records),default=0)
 traffic=max((item['lifetime_stats']['gradient_remote_bytes'] for item in records),default=0)
 identities={name:sources[name] for name in CORE}
 matches=all(hashlib.sha256((ROOT/'stack'/name).read_bytes()).hexdigest()==value for name,value in identities.items())
 passed=(code==0 and status.get('qualified') and status.get('source_unchanged') and status.get('exclusive_devices')
  and len(records)==64 and all(item['passed'] for item in records) and hits>0 and traffic>0 and matches)
 manifest=ROOT/'stack/configs/lifetime-runtime-freeze.json'
 if hashlib.sha256(manifest.read_bytes()).hexdigest()!=sources['configs/lifetime-runtime-freeze.json']:
  raise RuntimeError('Worker freeze manifest differs from qualified snapshot')
 proof=dict(passed=passed,exit=code,tag=tag,records=len(records),expected_records=64,
  actual_cache_hit_bytes=hits,actual_gradient_remote_bytes=traffic,payload_scope='logical payload, not hardware link counters',
  runtime_sources=identities,local_sources_match=matches,
  worker_sha256=next(iter(json.loads(manifest.read_text())['binaries'].values())),
  scope='eight ranks, b0-b3, W1/W2, checkpoint replay, independent output/dx/dp/owner-dW with EP scaling',
  output_threshold=.004,gradient_threshold=.02)
 if records:
  proof['max_relative_l2']={key:max(item['comparisons'][key]['relative_l2'] for item in records) for key in ('output','dx','dp','dw1','dw2')}
 for path in (ROOT/'diagnostics'/(tag+'.json'),ROOT/'diagnostics/independent-lifetime-math.json'):
  path.write_text(json.dumps(proof,indent=2)+'\n')
 if passed:(ROOT/'HOLD-B').unlink(missing_ok=True)
 return passed

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 parser.add_argument('--tag',default='lifetime-independent-ffn-v4')
 parser.add_argument('--collect-only',action='store_true')
 args=parser.parse_args()
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.tag):parser.error('Safe tag required')
 remote.REMOTE='/home/z00956592/megaflux/step-lifetime-kernel-20261115'
 remote.CONTAINER='step-lifetime-kernel-20261115'
 if args.collect_only:code=json.loads((ROOT/'evidence'/args.tag/'status.json').read_text())['exit']
 else:
  try:
   remote.gate(args.tag,['--mode','lifetime_math','--worker-variant','ep8-cap64-lifetime-v1',
    '--lifetime-mode','all','--route-bank',remote.REMOTE+'/stack/small-routes'],metadata_only=True)
  except SystemExit as result:code=result.code
 raise SystemExit(0 if collect(args.tag,code) else 1)
