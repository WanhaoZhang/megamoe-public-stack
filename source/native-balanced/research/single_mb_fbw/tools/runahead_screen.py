"""Screen the real full-update boundary against fixed strong planners."""
import json
import statistics
from controller import TASK,run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    gates=json.loads((TASK/'evidence/redispatch-runahead-qualification-v1.json').read_text())
    if len(gates)!=29 or not all(g['qualified'] for g in gates):raise RuntimeError('Runahead qualification incomplete')
    capabilities=json.loads((TASK/'evidence/redispatch-runahead-capabilities.json').read_text())
    hashes={row['mode']:row['so_sha256'] for row in capabilities}
    identities={};result=[]
    def run(vio,policy,mode):
        if compiling():raise RuntimeError('CPU build activity invalidates timing admission')
        worker=f'redispatch-runahead{mode}';tag=f'redispatch-runahead-screen-vio{vio:g}-{policy}-{mode}-v1'
        status=run_job('balanced','train',tag,['--planner',policy,'--maxvio',str(vio),
            '--recv-capacity','163840','--steps','25'],900,worker)
        if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
        directory=TASK/'evidence'/tag;checked=verify(directory)
        current=[r['identities'] for r in checked]
        if current!=identities.setdefault(vio,current):raise RuntimeError('Unmatched initial conditions')
        verify_loaded_worker(checked,worker,hashes[mode])
        ranks=[[json.loads(l) for l in (directory/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        mean=statistics.mean(max(x[s]['milliseconds'] for x in ranks) for s in range(5,25))
        result.append(dict(tag=tag,maxvio=vio,planner=policy,mode=mode,mean_ms=mean,independent_repeats=1,**status))
        (TASK/'evidence/redispatch-runahead-screen-v1.json').write_text(json.dumps(result,indent=2))
    for policy,mode in [('native',0),('ultra_ep',3),('ultra_ep',0),('native',3)]:run(20,policy,mode)
    for policy,mode in [('native',3),('ultra_ep',0),('ultra_ep',3),('native',0)]:run(.2,policy,mode)
    means={(r['maxvio'],r['planner'],r['mode']):r['mean_ms'] for r in result}
    gains={f'{vio:g}/{p}':1-means[vio,p,3]/means[vio,p,0] for vio in (20,.2) for p in ('native','ultra_ep')}
    expand=any(g>.02 for g in gains.values())
    (TASK/'evidence/redispatch-runahead-screen-decision-v1.json').write_text(json.dumps(dict(
        gains=gains,expanded=expand,threshold=.02,independent_repeats=1,
        scope='Engineering screen only; no new online planner or significance claim'),indent=2))
    if expand:
        for vio in (20,.2):
            for policy in ('native','ultra_ep'):
                for mode in (1,2):run(vio,policy,mode)
        for policy in ('eplb_budget_32','eplb_fixed'):
            for mode in (0,3):run(20,policy,mode)
    print('REDISPATCH_RUNAHEAD_SCREEN_COMPLETE',len(result),gains,flush=True)


if __name__=='__main__':main()
