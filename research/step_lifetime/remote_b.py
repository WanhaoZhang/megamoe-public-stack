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
    remote.gate(args.tag,extra,metadata_only=args.metadata_only)
