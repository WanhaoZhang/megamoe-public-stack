"""Online matched executor comparisons; identical plans and numerical contract."""
import json
import statistics
from controller import TASK,run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    gates=json.loads((TASK/'evidence/gradient-key-cache-qualification-v1.json').read_text())
    if len(gates)!=14 or not all(g['qualified'] for g in gates):raise RuntimeError('Key cache qualification incomplete')
    caps=json.loads((TASK/'evidence/gradient-key-cache-capabilities.json').read_text())
    hashes={r['mode']:r['so_sha256'] for r in caps};identities={};result=[]
    order=[('native',0),('eplb_fixed',1),('ultra_ep',1),('eplb_budget_32',0),
           ('eplb_budget_32',1),('ultra_ep',0),('eplb_fixed',0),('native',1)]
    for vio in (20,.2):
        for policy,mode in (order if vio==20 else [(p,1-m) for p,m in order]):
            if compiling():raise RuntimeError('CPU compiler activity before online timing')
            worker=f'gradient-key-cache{mode}';tag=f'gradient-key-cache-screen-vio{vio:g}-{policy}-{mode}-v1'
            status=run_job('balanced','train',tag,['--planner',policy,'--maxvio',str(vio),
                '--recv-capacity','163840','--steps','25'],900,worker)
            if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
            directory=TASK/'evidence'/tag;checked=verify(directory)
            current=[r['identities'] for r in checked]
            if current!=identities.setdefault(vio,current):raise RuntimeError('Unmatched initial conditions')
            verify_loaded_worker(checked,worker,hashes[mode])
            ranks=[[json.loads(l) for l in (directory/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
            mean=statistics.mean(max(x[s]['milliseconds'] for x in ranks) for s in range(5,25))
            result.append(dict(tag=tag,maxvio=vio,planner=policy,mode=mode,mean_ms=mean,
                independent_repeats=1,scope='executor screening, not a new planner',**status))
            (TASK/'evidence/gradient-key-cache-screen-v1.json').write_text(json.dumps(result,indent=2))
    means={(r['maxvio'],r['planner'],r['mode']):r['mean_ms'] for r in result}
    gains={f'{vio:g}/{p}':1-means[vio,p,1]/means[vio,p,0] for vio in (20,.2)
        for p in ('native','ultra_ep','eplb_budget_32','eplb_fixed')}
    (TASK/'evidence/gradient-key-cache-screen-summary-v1.json').write_text(json.dumps(dict(
        gains=gains,independent_repeats=1,scope='Matched source metadata intervention; no causality from source read counts alone'),indent=2))
    print('GRADIENT_KEY_CACHE_SCREEN_COMPLETE',len(result),gains,flush=True)


if __name__=='__main__':main()
