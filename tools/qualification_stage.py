"""Run native reference, checkpoint and full-model qualification serially."""
import json
from controller import TASK,run_job


def main():
    jobs=[]
    for planner in ('native','eplb_budget_0','eplb_budget_8','eplb_budget_32'):
        jobs.append(('validate',f'accuracy-{planner}-v2',
            ['--planner',planner,'--recv-capacity','8192']))
    for planner in ('eplb_budget_8','eplb_budget_32'):
        jobs.append(('pool',f'checkpoint-{planner}-v2',
            ['--planner',planner,'--restore','panel','--experts','256','--slots','32',
             '--capacity','4224','--tokens','257','--checkpoint','--parameter-fp32','--limit','10']))
    for planner in ('native','ultra_ep','eplb_budget_8','eplb_budget_32'):
        jobs.append(('train',f'full-shape-{planner}-v2',
            ['--planner',planner,'--recv-capacity','163840','--maxvio','20','--steps','3']))
    results=[]
    for mode,tag,extra in jobs:
        status=run_job('balanced',mode,tag,extra,900)
        results.append(dict(tag=tag,mode=mode,**status))
        (TASK/'evidence/qualification-stage-v2.json').write_text(json.dumps(results,indent=2))
        if not status['qualified']:raise RuntimeError(f'Qualification failed: {tag}')
        if status['peak_driver_MiB']>90*1024:raise RuntimeError(f'Memory budget exceeded: {tag}')
    print('QUALIFICATION_STAGE_COMPLETE',len(results),flush=True)


if __name__=='__main__':main()
