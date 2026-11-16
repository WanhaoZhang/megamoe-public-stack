"""Qualify two fresh workers, controlling every exported baseline capability."""
import json
import argparse
from controller import TASK,run_job


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wide',action='store_true');parser.add_argument('--revision',default='v2');args=parser.parse_args()
    prefix='wide-priority' if args.wide else 'matched-priority'
    receipt=f'priority-qualification-wide-{args.revision}.json' if args.wide else f'priority-qualification-{args.revision}.json'
    results=[]
    for priority in (0,1):
        worker=f'{prefix}{priority}'
        for planner in (('native','eplb_budget_32','eplb_fixed') if args.wide else ('native','eplb_budget_32')):
            tag=f'{worker}-accuracy-{planner}-{args.revision}'
            result=run_job('balanced','validate',tag,['--planner',planner,'--recv-capacity','8192'],900,worker)
            results.append(dict(tag=tag,**result))
            (TASK/'evidence'/receipt).write_text(json.dumps(results,indent=2))
            if not result['qualified']:raise RuntimeError(tag)
        tag=f'{worker}-checkpoint-{args.revision}'
        result=run_job('balanced','pool',tag,['--planner','eplb_budget_32','--restore','panel',
            '--experts','256','--slots','32','--capacity','4224','--tokens','257',
            '--checkpoint','--parameter-fp32','--limit','10'],900,worker)
        results.append(dict(tag=tag,**result))
        (TASK/'evidence'/receipt).write_text(json.dumps(results,indent=2))
        if not result['qualified']:raise RuntimeError(tag)
        tag=f'{worker}-full-shape-{args.revision}'
        result=run_job('balanced','train',tag,['--planner','native','--recv-capacity','163840',
            '--maxvio','20','--steps','3'],900,worker)
        results.append(dict(tag=tag,**result))
        (TASK/'evidence'/receipt).write_text(json.dumps(results,indent=2))
        if not result['qualified'] or result['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    print('PRIORITY_QUALIFICATION_PASS',len(results),flush=True)


if __name__=='__main__':main()
