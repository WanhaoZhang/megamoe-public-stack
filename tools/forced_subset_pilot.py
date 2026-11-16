"""Forced native plans under matched old/fast metadata executors.

These instrumented body-replay scores only diagnose plan headroom/ranking.
They cannot be reported as online planner results.
"""
import json
import statistics
from controller import TASK,run_job
from remaining_causal_stages import compiling
from verify_training_run import verify,verify_loaded_worker


def main():
    prior=json.loads((TASK/'evidence/gradient-key-cache-stage-v1.json').read_text())
    if not prior.get('complete'):raise RuntimeError('Metadata-cache queue has not retired')
    gates=json.loads((TASK/'evidence/forced-subset-gates-v1.json').read_text())
    proof=json.loads((TASK/'analysis/forced-subset-physical-audit.json').read_text())
    physical={r['tag'] for r in proof['records'] if r['passed'] and r['physical_rank_call_digests_checked']==48 and r['changed_global_invocations']>=2}
    eligible={(r['maxvio'],r['candidate']) for r in gates if r['numerical_gate_passed'] and r['tag'] in physical}
    caps=json.loads((TASK/'evidence/gradient-key-cache-capabilities.json').read_text())
    hashes={r['mode']:r['so_sha256'] for r in caps};coupled=[];excluded=[]
    # Cache1 changes only exact integer ordering metadata, but also check its
    # composition with both forced candidate families at the active high skew.
    for name in ('proxy_best','sparse_best'):
        if (20,name) not in eligible:continue
        tag=f'forced-subset-cache1-gate-vio20-{name}-v1'
        status=run_job('balanced','prepared_validate',tag,['--prepared-kind','body','--proposal-mode','bank',
            '--candidate-bank',str(TASK/f'artifacts/native-subsets-vio20-{name}.npz'),
            '--paired-diagnostic','--planner','native','--maxvio','20','--steps','3'],1800,'gradient-key-cache1')
        out=TASK/'evidence'/tag
        summaries=[json.loads((out/f'rank{r}.json').read_text()) for r in range(8)] if status['exit']==0 else []
        passed=len(summaries)==8 and status['source_unchanged'] and all(s['passed'] and s['steps']==3 and s['training_performance_score'] is False for s in summaries)
        coupled.append(dict(tag=tag,candidate=name,numerical_gate_passed=passed,**status))
        (TASK/'evidence/forced-subset-cache1-gates-v1.json').write_text(json.dumps(coupled,indent=2))
        if not status['source_unchanged'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
    cache1_eligible={r['candidate'] for r in coupled if r['numerical_gate_passed']}
    result=[];identities={};plans={}
    for vio in (1,5,20):
        for mode in (0,1):
            order=['native','sparse_best','ultra_ep','proxy_best']
            if mode==1:order=order[::-1]
            for name in order:
                forced=name in ('proxy_best','sparse_best')
                if forced and ((vio,name) not in eligible or (mode==1 and name not in cache1_eligible)):
                    excluded.append(dict(maxvio=vio,mode=mode,candidate=name,reason='numerical or captured-physical-plan gate not passed'))
                    continue
                if compiling():raise RuntimeError('CPU compiler activity before diagnostic timing')
                worker=f'gradient-key-cache{mode}';tag=f'forced-subset-pilot-vio{vio:g}-{name}-cache{mode}-v1'
                extra=['--planner','native' if forced else name,'--maxvio',str(vio),'--steps','25',
                    '--recv-capacity','163840','--prepared-kind','body','--proposal-mode','bank' if forced else 'original']
                if forced:extra+=['--candidate-bank',str(TASK/f'artifacts/native-subsets-vio{vio:g}-{name}.npz')]
                status=run_job('balanced','prepared',tag,extra,1800,worker)
                if not status['qualified'] or status['peak_driver_MiB']>90*1024:raise RuntimeError(tag)
                directory=TASK/'evidence'/tag;checked=verify(directory)
                actual=[r['identities'] for r in checked]
                if actual!=identities.setdefault(vio,actual):raise RuntimeError('Unmatched model/input/route identity')
                verify_loaded_worker(checked,worker,hashes[mode])
                changed=[]
                for rank in range(8):
                    manifest=json.loads((directory/f'prepared-plan-manifest-rank{rank}.json').read_text())
                    if manifest['online_score'] or len(manifest['records'])!=50:raise RuntimeError('Replay scope/coverage')
                    records=manifest['records'];ref=plans.setdefault((vio,name,rank),records)
                    if records!=ref:raise RuntimeError('The executor intervention changed the physical plan')
                    changed.append(sum(r['changed_from_original'] for r in records))
                ranks=[[json.loads(l) for l in (directory/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
                mean=statistics.mean(max(x[s]['milliseconds'] for x in ranks) for s in range(5,25))
                result.append(dict(tag=tag,maxvio=vio,planner=name,mode=mode,mean_ms=mean,
                    changed_prepared_plans_by_rank=changed,diagnostic_only=True,online_score=False,
                    independent_repeats=1,**status))
                (TASK/'evidence/forced-subset-pilot-v1.json').write_text(json.dumps(dict(rows=result,excluded=excluded),indent=2))
    (TASK/'evidence/forced-subset-pilot-stage-v1.json').write_text(json.dumps(dict(complete=True,cells=len(result),
        scope='Replay headroom only; sparse_best at vio1 is the known no-replication baseline, not a new algorithm'),indent=2))
    print('FORCED_SUBSET_PILOT_COMPLETE',len(result),flush=True)


if __name__=='__main__':main()
