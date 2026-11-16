"""Fresh paired process replications; screening runs are excluded from inference."""
import json
import statistics
from controller import TASK,run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    if not json.loads((TASK/'evidence/forced-subset-pilot-stage-v1.json').read_text()).get('complete'):
        raise RuntimeError('Forced diagnostic queue is not retired')
    gates=json.loads((TASK/'evidence/gradient-key-cache-qualification-v1.json').read_text())
    if len(gates)!=14 or not all(g['qualified'] for g in gates):raise RuntimeError('Qualification incomplete')
    caps=json.loads((TASK/'evidence/gradient-key-cache-capabilities.json').read_text())
    hashes={r['mode']:r['so_sha256'] for r in caps};identities=None;rows=[]
    policies=('native','ultra_ep','eplb_fixed')
    for repeat in range(5):
        order=policies if repeat%2==0 else policies[::-1]
        for offset,policy in enumerate(order):
            for mode in ((0,1) if (repeat+offset)%2==0 else (1,0)):
                if compiling():raise RuntimeError('Compiler active before confirmation timing')
                worker=f'gradient-key-cache{mode}';tag=f'key-cache-confirm-vio20-{policy}-r{repeat}-m{mode}-v1'
                status=run_job('balanced','train',tag,['--planner',policy,'--maxvio','20',
                    '--recv-capacity','163840','--steps','25'],900,worker)
                if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
                root=TASK/'evidence'/tag;checked=verify(root);current=[r['identities'] for r in checked]
                if identities is None:identities=current
                if current!=identities:raise RuntimeError('Initial identity mismatch')
                verify_loaded_worker(checked,worker,hashes[mode])
                ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
                mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
                rows.append(dict(tag=tag,repeat=repeat,maxvio=20,planner=policy,mode=mode,
                    mean_ms=mean,independent_process_repetitions=5,new_planner=False,**status))
                (TASK/'evidence/key-cache-confirmation-v1.json').write_text(json.dumps(rows,indent=2))
    (TASK/'evidence/key-cache-confirmation-stage-v1.json').write_text(json.dumps(dict(complete=True,
        cells=len(rows),inference_unit='paired independent process; postwarmup steps are not replicates'),indent=2))
    print('KEY_CACHE_CONFIRMATION_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
