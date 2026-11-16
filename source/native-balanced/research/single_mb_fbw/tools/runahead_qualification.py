"""Qualify the redispatch dependency cut before any timing claim."""
import json
from controller import TASK,run_job


def main():
    result=[]
    for mode in (0,1,2,3):
        worker=f'redispatch-runahead{mode}'
        cases=[('validate',f'accuracy-{policy}',['--planner',policy,'--recv-capacity','8192'])
               for policy in ('native','eplb_budget_32','eplb_fixed')]
        cases.append(('pool','checkpoint',['--planner','eplb_budget_32','--restore','panel',
            '--experts','256','--slots','32','--capacity','4224','--tokens','257',
            '--checkpoint','--parameter-fp32','--limit','10']))
        cases.extend(('train',f'full-shape-{policy}',['--planner',policy,'--maxvio','20',
            '--recv-capacity','163840','--steps','3']) for policy in ('native','ultra_ep','eplb_fixed'))
        if mode==3:
            cases.append(('validate','saved-input-fallback',['--planner','eplb_fixed',
                '--recv-capacity','8192','--routed-input-mode','saved']))
        for kind,label,extra in cases:
            tag=f'{worker}-{label}-v1';status=run_job('balanced',kind,tag,extra,900,worker)
            result.append(dict(mode=mode,tag=tag,**status))
            (TASK/'evidence/redispatch-runahead-qualification-v1.json').write_text(json.dumps(result,indent=2))
            if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    print('REDISPATCH_RUNAHEAD_QUALIFICATION_PASS',len(result),flush=True)


if __name__=='__main__':main()
