"""Isolated original-Timer prefix qualification lane, never performance."""
from pathlib import Path
import argparse
import remote
parser=argparse.ArgumentParser()
parser.add_argument('action',choices=('deploy','gate'))
parser.add_argument('--tag')
args,extra=parser.parse_known_args()
root=Path(__file__).resolve().parent
remote.REMOTE='/home/z00956592/megaflux/step-lifetime-prefix-timer-20261116'
remote.CONTAINER='step-lifetime-prefix-timer-20261116'
remote.STACK=root/'implementation/prefix-timer-stack'
if args.action=='deploy':
 remote.ROOT=root/'implementation/timer-lane';remote.ROOT.mkdir(exist_ok=True)
 (remote.ROOT/'evidence').mkdir(exist_ok=True)
 remote.deploy()
else:
 if not args.tag or not all(value.isalnum() or value in '-_.' for value in args.tag):parser.error('safe tag required')
 if '--diagnostic-timer' not in extra and '--plan-hash-qualification' not in extra:parser.error('This lane accepts diagnostics only')
 if any(not all(value.isalnum() or value in '-_./:' for value in item) for item in extra):parser.error('unexpected shell character')
 remote.gate(args.tag,extra,metadata_only=True)
