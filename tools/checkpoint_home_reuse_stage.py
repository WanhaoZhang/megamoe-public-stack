"""Default-off immediate recompute/home-mirror probe. Not a new planner."""
import json
import statistics
from controller import TASK, run_job
from remaining_causal_stages import compiling
from verify_training_run import verify, verify_loaded_worker


def main():
    caps=json.loads((TASK/'evidence/fast-redispatch-capabilities-v1.json').read_text())
    hashes={r['variant']:r['so_sha256'] for r in caps}
    qualified=json.loads((TASK/'evidence/fast-redispatch-stage-v1.json').read_text())['eligible']
    workers=['fastpipe-r0-t0']
    screen=json.loads((TASK/'evidence/fast-redispatch-screen-v1.json').read_text())
    # Data-dependent single-process selection is exploratory, never confirmation.
    scores={w:statistics.mean(r['mean_ms'] for r in screen if r['worker']==w and r['maxvio']==20)
            for w in qualified if w.startswith('fastpipe')}
    winner=min(scores,key=scores.get)
    if winner not in workers:workers.append(winner)
    receipt=dict(workers=workers,pipeline_selection='Exploratory mean over Native/Ultra high20 endpoint screens',
                 gates=[],screens=[],complete=False,new_planner=False)
    output=TASK/'evidence/checkpoint-home-reuse-stage-v1.json'
    def save():output.write_text(json.dumps(receipt,indent=2))
    def run(kind,tag,args,w):
        status=run_job('balanced',kind,tag,args,900,w)
        if not status['qualified'] or not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:
            raise RuntimeError(tag)
        return status
    for w in workers:
        cases=[(p,True,False) for p in ('native','eplb_budget_32','eplb_fixed')]
        cases += [('eplb_budget_32',True,True),('eplb_budget_32',False,False)]
        for p,checkpoint,poison in cases:
            tag=f'home-reuse-{w}-{p}-ck{int(checkpoint)}-poison{int(poison)}-v1'
            args=['--planner',p,'--restore','panel','--experts','256','--slots','32',
                  '--capacity','4224','--tokens','257','--parameter-fp32','--limit','10',
                  '--checkpoint-home-reuse']
            if checkpoint:args.append('--checkpoint')
            if poison:args.append('--poison-recompute-home')
            status=run('pool',tag,args,w)
            audits=[]
            for rank in range(8):
                audits.append(json.loads((TASK/'evidence'/tag/f'rank{rank}.json').read_text())['home_reuse_audit'])
            receipt['gates'].append(dict(tag=tag,home_audit=audits,**status));save()
        for p in ('native','ultra_ep','eplb_fixed'):
            tag=f'home-reuse-{w}-main-{p}-v1'
            status=run('train',tag,['--planner',p,'--maxvio','20','--steps','3','--checkpoint-home-reuse'],w)
            receipt['gates'].append(dict(tag=tag,**status));save()
    identities={}
    for vio in (20,.2):
        for p in ('native','ultra_ep'):
            for w in workers:
                for reuse in ((0,1) if vio==20 else (1,0)):
                    if compiling():raise RuntimeError('Compiler active before full-update screen')
                    tag=f'home-reuse-screen-vio{vio:g}-{p}-{w}-h{reuse}-v1'
                    args=['--planner',p,'--maxvio',str(vio),'--steps','25']
                    if reuse:args.append('--checkpoint-home-reuse')
                    status=run('train',tag,args,w);root=TASK/'evidence'/tag
                    checked=verify(root);verify_loaded_worker(checked,w,hashes[w])
                    actual=[r['identities'] for r in checked]
                    if actual!=identities.setdefault(vio,actual):raise RuntimeError('Initial state mismatch')
                    ranks=[[json.loads(line) for line in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
                    audits=[json.loads((root/f'planning-audit-rank{r}.json').read_text())[0]['home_reuse'] for r in range(8)]
                    if any(a['hits']!=(50 if reuse else 0) for a in audits):raise RuntimeError('Home proof usage mismatch')
                    mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
                    receipt['screens'].append(dict(tag=tag,maxvio=vio,planner=p,worker=w,reuse=reuse,
                        mean_ms=mean,independent_processes=1,screening_only=True,home_audit=audits,**status));save()
    receipt['complete']=True;save()
    print('CHECKPOINT_HOME_REUSE_STAGE_COMPLETE',len(receipt['gates']),len(receipt['screens']),flush=True)


if __name__=='__main__':main()
