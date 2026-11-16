"""Screen sparse/dense replica extremes; expand only at the engineering gate."""
import json
import statistics
from controller import TASK,run_job
from verify_training_run import verify


def main():
    gates=json.loads((TASK/'evidence/owner-window-qualification-v1.json').read_text())
    if len(gates)!=12 or not all(g['qualified'] for g in gates):raise RuntimeError('Window qualification incomplete')
    result=[]
    def run(vio,policy,mode):
        worker=f'owner-window{mode}';tag=f'owner-window-screen-vio{vio:g}-{policy}-{mode}-v1'
        status=run_job('balanced','train',tag,['--planner',policy,'--maxvio',str(vio),
            '--recv-capacity','163840','--steps','25'],900,worker)
        row=dict(tag=tag,maxvio=vio,planner=policy,mode=mode,independent_repeats=1,**status)
        result.append(row)
        if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
        directory=TASK/'evidence'/tag;verify(directory)
        ranks=[[json.loads(l) for l in (directory/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        row['mean_ms']=statistics.mean(max(x[s]['milliseconds'] for x in ranks) for s in range(5,25))
        (TASK/'evidence/owner-window-screen-v1.json').write_text(json.dumps(result,indent=2))
    for policy,mode in [('native',0),('eplb_fixed',1),('eplb_fixed',0),('native',1)]:run(20,policy,mode)
    means={(r['planner'],r['mode']):r['mean_ms'] for r in result}
    gains={p:1-means[p,1]/means[p,0] for p in ('native','eplb_fixed')}
    expand=any(g>.02 for g in gains.values())
    (TASK/'evidence/owner-window-screen-decision-v1.json').write_text(json.dumps(dict(
        gains=gains,expanded=expand,threshold=.02,independent_repeats=1,
        scope='engineering screen only; an EPLB-only win is not a competitive planner claim'),indent=2))
    if expand:
        for policy in ('ultra_ep','eplb_budget_32'):
            for mode in (1,0):run(20,policy,mode)
        for policy in ('native','ultra_ep','eplb_budget_32','eplb_fixed'):
            for mode in (0,1):run(.2,policy,mode)
    print('OWNER_WINDOW_SCREEN_COMPLETE',len(result),gains,flush=True)


if __name__=='__main__':main()
