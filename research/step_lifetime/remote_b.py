"""Private ABI1 lane, sharing only the global hardware admission lock."""
from pathlib import Path
import argparse
import time
import remote
parser=argparse.ArgumentParser()
parser.add_argument('action',choices=('deploy','gate'))
parser.add_argument('--tag')
parser.add_argument('--metadata-only',action='store_true')
args,extra=parser.parse_known_args()
remote.REMOTE='/home/z00956592/megaflux/step-lifetime-kernel-20261115'
remote.CONTAINER='step-lifetime-kernel-20261115'
if args.action=='deploy':
    remote.ROOT=Path(__file__).resolve().parents[3]/'implementation/kernel-lane'
    remote.ROOT.mkdir(parents=True,exist_ok=True)
    (remote.ROOT/'evidence').mkdir(exist_ok=True)
    remote.deploy()
else:
    if not args.tag or not all(value.isalnum() or value in '-_.' for value in args.tag):parser.error('safe tag required')
    if any(not all(value.isalnum() or value in '-_./:' for value in item) for item in extra):parser.error('unexpected shell character')
    while (remote.ROOT/'HOLD-B').exists():time.sleep(1)
    if '--steps' in extra and int(extra[extra.index('--steps')+1])==25:
        proof=remote.ROOT/'diagnostics/independent-lifetime-math.json'
        import json
        if not proof.exists() or not json.loads(proof.read_text()).get('passed'):
            raise RuntimeError('Independent lifetime output/dx/dp/dW gate is required before B timings')
        import hashlib
        checked=json.loads(proof.read_text())
        if not checked.get('runtime_sources') or any(hashlib.sha256((remote.STACK/name).read_bytes()).hexdigest()!=value
            for name,value in checked['runtime_sources'].items()):
            raise RuntimeError('Lifetime math proof is stale for the local runtime sources')
    remote.gate(args.tag,extra,metadata_only=args.metadata_only)
