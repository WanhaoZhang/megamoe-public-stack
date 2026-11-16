"""Causal diagnostics only: planner generation, common compiler and assignment."""
import json
from controller import TASK,run_job


def main():
    gates=json.loads((TASK/'evidence/prepared-plan-gates-v3.json').read_text())
    if len(gates)!=6 or not all(g['numerical_gate_passed'] for g in gates):
        raise RuntimeError('Prepared diagnostics have not passed their paired gates')
    result=[]
    for vio in (20,.2):
        for planner in ('native','eplb_budget_32','eplb_fixed'):
            for kind in ('online','quota','body'):
                tag=f'intervention-vio{vio:g}-{planner}-{kind}-v2'
                extra=['--planner',planner,'--maxvio',str(vio),'--steps','25','--recv-capacity','163840']
                extra+=['--prepared-kind',kind,'--proposal-mode','original']
                status=run_job('balanced','prepared',tag,extra,1800)
                result.append(dict(tag=tag,maxvio=vio,planner=planner,kind=kind,proposal='original',
                    diagnostic_only=True,online_score=False,**status))
                (TASK/'evidence/prepared-plan-pilot-v2.json').write_text(json.dumps(result,indent=2))
                if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
        for proposal in ('local-source','reverse-slots'):
            tag=f'intervention-vio{vio:g}-eplb_budget_32-body-{proposal}-v2'
            status=run_job('balanced','prepared',tag,['--planner','eplb_budget_32',
                '--maxvio',str(vio),'--steps','25','--recv-capacity','163840',
                '--prepared-kind','body','--proposal-mode',proposal],1800)
            result.append(dict(tag=tag,maxvio=vio,planner='eplb_budget_32',kind='body',proposal=proposal,
                diagnostic_only=True,online_score=False,**status))
            (TASK/'evidence/prepared-plan-pilot-v2.json').write_text(json.dumps(result,indent=2))
            if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    print('PREPARED_PLAN_PILOT_COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
