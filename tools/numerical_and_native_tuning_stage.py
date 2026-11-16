"""After home probe: numerical provenance controls and old Native tuning pilot."""
import hashlib
import json
import statistics
from controller import TASK,run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    prior=json.loads((TASK/'evidence/checkpoint-home-reuse-stage-v1.json').read_text())
    if not prior['complete']:raise RuntimeError('Home probe queue not retired')
    receipt=dict(complete=False,determinism=[],native_tuning_gates=[],native_tuning_screens=[])
    inputs=[TASK/'tools/gradient_incoming.py',TASK/'tools/numerical_and_native_tuning_stage.py',
            *sorted((TASK/'artifacts').glob('native-tune-vio20-*'))]
    def identity():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    frozen=identity();receipt['immutable_input_hashes']=frozen
    output=TASK/'evidence/numerical-and-native-tuning-stage-v1.json'
    def save():output.write_text(json.dumps(receipt,indent=2))
    for deterministic in (False,True):
        tag=f'fast-metadata-native-self-vio5-deterministic{int(deterministic)}-v1'
        args=['--incoming-gradient-diagnostic','--root',str(TASK),'--planner','native','--maxvio','5','--steps','3']
        if deterministic:args.append('--deterministic-diagnostic')
        status=run_job('balanced','validate',tag,args,1800,'fastpipe-r0-t0')
        if identity()!=frozen:raise RuntimeError('Immutable diagnostic input/code changed')
        receipt['determinism'].append(dict(tag=tag,training_performance_score=False,**status));save()
        if not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
        # A deterministic operator error is a diagnostic outcome, not a reason
        # to relax any gate or disable the switch and label it passed.
    caps=json.loads((TASK/'evidence/fast-redispatch-capabilities-v1.json').read_text())
    hashes={r['variant']:r['so_sha256'] for r in caps};identities=None
    for worker in prior['workers']:
        eligible=[]
        for family in ('tile384','tile1536','penalty4'):
            bank=TASK/f'artifacts/native-tune-vio20-{family}.npz'
            tag=f'native-tune-gate-{family}-{worker}-v1'
            args=['--prepared-kind','body','--proposal-mode','bank','--candidate-bank',str(bank),
                  '--paired-diagnostic','--planner','native','--maxvio','20','--steps','3']
            status=run_job('balanced','prepared_validate',tag,args,1800,worker)
            if identity()!=frozen:raise RuntimeError('Immutable tuning bank/code changed')
            receipt['native_tuning_gates'].append(dict(family=family,worker=worker,tag=tag,**status));save()
            if not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
            if status['qualified']:eligible.append(family)
        for family in ('native','ultra_ep',*eligible):
            if compiling():raise RuntimeError('Compiler active before prepared diagnostic')
            tag=f'native-tune-pilot-vio20-{family}-{worker}-v1'
            args=['--prepared-kind','body','--planner',family if family in ('native','ultra_ep') else 'native',
                  '--maxvio','20','--steps','25']
            if family in ('native','ultra_ep'):args+=['--proposal-mode','original']
            else:args+=['--proposal-mode','bank','--candidate-bank',str(TASK/f'artifacts/native-tune-vio20-{family}.npz')]
            status=run_job('balanced','prepared',tag,args,1800,worker)
            if identity()!=frozen:raise RuntimeError('Immutable tuning bank/code changed')
            if not status['qualified'] or not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
            root=TASK/'evidence'/tag;checked=verify(root);verify_loaded_worker(checked,worker,hashes[worker])
            actual=[r['identities'] for r in checked]
            if identities is None:identities=actual
            if identities!=actual:raise RuntimeError('Initial condition mismatch')
            ranks=[[json.loads(line) for line in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
            mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
            receipt['native_tuning_screens'].append(dict(family=family,worker=worker,tag=tag,mean_ms=mean,
                independent_processes=1,prepared_diagnostic_only=True,online_score=False,**status));save()
    receipt['complete']=True;save()
    print('NUMERICAL_NATIVE_TUNING_STAGE_COMPLETE',flush=True)


if __name__=='__main__':main()
