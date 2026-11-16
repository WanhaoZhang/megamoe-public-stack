"""Qualify final interfaces and the common capacity before freezing timings."""
import json
import argparse
from controller import TASK,run_job

parser=argparse.ArgumentParser();parser.add_argument('--revision',default='v4');options=parser.parse_args()

jobs=[
 ('plan','gate-final-plan64',[],64),
 ('validate','gate-final-accuracy-eplb',['--planner','eplb_fixed','--recv-capacity','8192'],64),
 ('pool','gate-final-ticket-replay',['--planner','moonep','--restore','panel','--experts','256','--slots','32','--capacity','4224','--checkpoint','--parameter-fp32','--limit','10'],64),
 ('pool','gate-final-eplb-replay',['--planner','eplb_fixed','--restore','panel','--experts','256','--slots','32','--capacity','4224','--checkpoint','--parameter-fp32','--limit','10'],64),
 ('validate','gate-final-cap32',['--planner','moonep','--experts','128','--recv-capacity','8192'],32),
 ('scaling','gate-final-fsdp-scale',['--singleton-bypass'],64),
 ('norm','gate-final-global-norm',[],64),
]
rows=[]
for mode,tag,args,cap in jobs:
    tag+='-'+options.revision
    status=run_job('balanced',mode,tag,args,900,capacity=cap)
    rows.append(dict(tag=tag,**status));(TASK/'evidence'/f'final-gates-{options.revision}.json').write_text(json.dumps(rows,indent=2))
    if not status['qualified']:raise RuntimeError(f'Final gate failed: {tag}')
print('FINAL_CORRECTNESS_GATES_PASS',flush=True)
