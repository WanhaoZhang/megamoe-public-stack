"""Preregister and execute gated temporal-policy screening on one EP8 node.

No joint policy or Afore cache is invented by this runner. The formal paired
matrix requires a separately frozen, qualified policy and selected baseline.
"""
import argparse
import json
from pathlib import Path
import time


POLICIES=(('legacy_w1','legacy',1),('legacy_w2','legacy',2),
          ('fixed_native','fixed_native',2),('fifo_native','fifo_native',2),
          ('easy_home','easy_home',2),('easy_independent','easy_independent',2))
PATTERNS=('same','alternating','random')


def manifest(repeats=1):
    """Rotate policy order; every job owns exactly 25 fixed-batch updates."""
    jobs=[]
    for repeat in range(repeats):
        ordered=POLICIES[repeat%len(POLICIES):]+POLICIES[:repeat%len(POLICIES)]
        for pattern in PATTERNS:
            for label,policy,window in ordered:
                jobs.append(dict(tag=f'screen-vio10-{pattern}-{label}-r{repeat+1}',
                    policy=policy,pattern=pattern,maxvio=10,repeat=repeat+1,
                    forward_window=window,
                    optimizer_updates=25,warmup_updates=5,measured_updates=20,
                    status='planned',formal=False))
    return jobs


def require_gates(root):
    """Refuse a performance launch until the relevant evidence exists."""
    gates=json.loads((root/'configs/temporal-gates.json').read_text())
    required=('cpu_contract','same_weight_full_gradient','device_scheduler_reference',
              'primary_online_adam3','common_memory_budget','group_protocol_recovery',
              'legacy_w1_full_gradient','timer_v2_diagnostic','source_archive_integrity')
    failed=[name for name in required if gates.get(name,{}).get('passed') is not True]
    if failed:
        raise RuntimeError(f'Unqualified temporal screening gates: {failed}')
    if gates.get('formal_matrix_enabled'):
        raise RuntimeError('This runner screens existing methods; it does not run the joint-policy formal matrix')
    return gates


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=1,choices=(1,3))
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--container',default='temporal-replica-frontier-20261115')
    args=parser.parse_args()
    jobs=manifest(args.repeats)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    def save():args.out.write_text(json.dumps(dict(stage='screening',jobs=jobs,
        full_plan_complete=False,afore_cache_included=False,joint_policy_included=False,
        updated=time.time()),indent=2))
    save()
    if not args.execute:
        print(json.dumps(dict(planned=len(jobs),executed=0,out=str(args.out))))
        return
    require_gates(args.root)
    import controller
    controller.TASK=args.root.resolve()
    controller.CONTAINER=args.container
    for job in jobs:
        job['status']='running';save()
        arguments=['--layers','4','--steps','25','--tokens','16384','--hidden','4096','--ffn','1024',
            '--microbatches','4','--forward-window',str(job['forward_window']),'--recv-capacity','262144',
            '--maxvio','10','--planner','native','--temporal-policy',job['policy'],
            '--temporal-pattern',job['pattern'],'--temporal-scheduler','ascendc','--grad-sync']
        status=controller.run_job('balanced','train',job['tag'],arguments,1800,'ep8-cap64-aligned',64)
        job['receipt']=status
        job['status']='completed' if status['qualified'] else 'failed'
        save()
        if not status['qualified'] or status.get('peak_driver_MiB',0)>90*1024:
            raise RuntimeError(f'Screening stopped at failed/common-memory gate: {job["tag"]}')


if __name__=='__main__':main()
