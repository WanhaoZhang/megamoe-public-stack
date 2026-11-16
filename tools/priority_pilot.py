"""One-process screening only; full-update comparison in one native runtime."""
import json
from controller import TASK,run_job


def main():
    gates=json.loads((TASK/'evidence/priority-qualification-v2.json').read_text())
    if len(gates)!=8 or not all(g['qualified'] for g in gates):raise RuntimeError('Priority qualification incomplete')
    arms=[('native',None),('ultra_ep','matched-priority0'),('native','matched-priority1'),
        ('ultra_ep',None),('native','matched-priority0'),('ultra_ep','matched-priority1')]
    result=[]
    for vio in (.2,20):
        for planner,worker in arms:
            label=worker or 'legacy'
            tag=f'pilot-vio{vio:g}-{planner}-{label}-v1'
            status=run_job('balanced','train',tag,['--planner',planner,'--recv-capacity','163840',
                '--maxvio',str(vio),'--steps','25'],900,worker)
            result.append(dict(tag=tag,maxvio=vio,planner=planner,worker=worker,repeat=1,**status))
            (TASK/'evidence/priority-pilot-v1.json').write_text(json.dumps(result,indent=2))
            if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    print('PRIORITY_PILOT_COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
