"""Frozen online-planning experiment matrix; failures are explicit rows."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from controller import TASK,run_job

POLICIES=['moonep','native','ultra_ep','echo','eplb_fixed']
VIO=[.2,1,5,10,15,20]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--planners',default=','.join(POLICIES));ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--first-repeat',type=int,default=1);ap.add_argument('--capacity',type=int,default=163840);ap.add_argument('--route-bank',type=Path);ap.add_argument('--steps',type=int,default=25);ap.add_argument('--targets',default=','.join(map(str,VIO)));ap.add_argument('--timer',action='store_true');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    planners=args.planners.split(',');targets=list(map(float,args.targets.split(',')))
    manifest=TASK/'configs'/f'campaign-{args.name}.json';results=TASK/'evidence'/f'campaign-{args.name}.json'
    contract=dict(name=args.name,planners=planners,targets=targets,repeats=args.repeats,first_repeat=args.first_repeat,
        receive_capacity=args.capacity,route_bank=str(args.route_bank or TASK/'routes'),steps=args.steps,
        warmup=5,timer=args.timer,ordinary_only=True,online_planning=True,device_complete_wall_max8=True)
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())!=contract:raise RuntimeError('campaign contract already exists or changed')
    else:manifest.write_text(json.dumps(contract,indent=2))
    rows=json.loads(results.read_text()) if results.exists() else []
    completed={x['tag'] for x in rows}
    for repeat in range(args.first_repeat,args.first_repeat+args.repeats):
        rotation=(repeat-1)%len(planners);order=planners[rotation:]+planners[:rotation]
        for maxvio in targets:
            for planner in order:
                tag=f'{args.name}-vio{maxvio:g}-{planner}-r{repeat}'
                if tag in completed:continue
                extra=['--planner',planner,'--recv-capacity',str(args.capacity),'--maxvio',str(maxvio),'--steps',str(args.steps)]
                if args.route_bank:extra+=['--route-bank',str(args.route_bank)]
                if args.timer:extra+=['--diagnostic-timer']
                status=run_job('balanced','train',tag,extra,1800,worker='unified-cap64-calibration-timer' if args.timer else None)
                state='pass' if status['qualified'] else 'failed'
                text=(TASK/'evidence'/tag/'stdout.log').read_text(errors='replace')
                if not status['qualified']:
                    if 'out of memory' in text.lower():state='oom'
                    elif status['exit'] in (124,137):state='timeout'
                    elif 'device plan failed' in text or 'Plan' in text and 'capacity' in text:state='invalid_plan'
                if status['peak_driver_MiB']>90*1024:state='memory_gate_exceeded'
                row=dict(tag=tag,planner=planner,maxvio=maxvio,repeat=repeat,state=state,**status)
                rows.append(row);results.write_text(json.dumps(rows,indent=2))
                print(json.dumps(dict(campaign=args.name,completed=len(rows),total=args.repeats*len(planners)*len(targets),last=row)),flush=True)
                # Runtime/protocol faults require investigation before queuing
                # more collective jobs. An infeasible plan stays a failed point.
                if state in ('memory_gate_exceeded','oom','timeout'):
                    raise RuntimeError(f'Campaign safety gate: {tag}: {state}')
    print('PLANNER_CAMPAIGN_FINISHED',args.name,len(rows),flush=True)
if __name__=='__main__':main()
