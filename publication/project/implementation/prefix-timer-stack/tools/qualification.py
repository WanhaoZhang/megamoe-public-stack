"""Serial device correctness gates; every failed process keeps its receipt."""
import argparse
import json
from pathlib import Path
from controller import TASK,run_job

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['plan','accuracy','pool','training'],required=True);ap.add_argument('--revision',default='v2');a=ap.parse_args()
    rows=[]
    if a.phase=='plan':jobs=[('plan','common',[])]
    elif a.phase=='accuracy':jobs=[('validate',name,['--planner',name,'--recv-capacity','8192']) for name in ['moonep','native','ultra_ep','echo','eplb_fixed']]
    elif a.phase=='pool':jobs=[('pool',name,['--planner',name,'--restore','panel','--experts','256','--slots','32','--capacity','4224','--tokens','257','--checkpoint','--parameter-fp32','--limit','10']) for name in ['moonep','native','ultra_ep','echo','eplb_fixed']]
    else:jobs=[('train',name,['--planner',name,'--recv-capacity','163840','--maxvio','20','--steps','3']) for name in ['moonep','native','ultra_ep','echo','eplb_fixed']]
    for mode,name,extra in jobs:
        tag=f'gate-{a.phase}-{name}-{a.revision}'
        status=run_job('balanced',mode,tag,extra,900)
        rows.append(dict(tag=tag,**status))
        (TASK/'evidence'/f'qualification-{a.phase}-{a.revision}.json').write_text(json.dumps(rows,indent=2))
        if not status['qualified']:raise RuntimeError(f'Gate failed: {tag}')
        if status['peak_driver_MiB']>90*1024:raise RuntimeError(f'Common memory gate exceeded: {tag}')
    print('QUALIFICATION_PHASE_PASS',a.phase,flush=True)
if __name__=='__main__':main()
