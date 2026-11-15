from pathlib import Path
import argparse
import remote
p=argparse.ArgumentParser()
p.add_argument('action',choices=('deploy','gate'))
p.add_argument('--tag')
a,extra=p.parse_known_args()
remote.REMOTE='/home/z00956592/megaflux/step-lifetime-causal-20261116'
remote.CONTAINER='step-lifetime-causal-20261116'
if a.action=='deploy':remote.deploy()
else:
 if not a.tag or not all(c.isalnum() or c in '-_.' for c in a.tag):p.error('safe tag required')
 if any(not all(c.isalnum() or c in '-_./:' for c in item) for item in extra):p.error('invalid shell character')
 if '--diagnostic-timer' not in extra:p.error('causal lane is diagnostics only')
 remote.gate(a.tag,extra,metadata_only=True)
