"""Paired independent-process statistics; never treat updates as replicates."""
import json
import math
import statistics
from pathlib import Path
from verify_training_run import verify,verify_loaded_worker

ROOT=Path(__file__).resolve().parents[1]


def t_interval(values):
    # Prespecified five pairs, two-sided Student t, df4. This parametric CI
    # assumes independent approximately normal paired log ratios/differences.
    if len(values)!=5:raise ValueError('Expected five independent process pairs')
    mean=statistics.mean(values);half=2.7764451051977987*statistics.stdev(values)/math.sqrt(5)
    return [mean-half,mean+half]


def main():
    rows=json.loads((ROOT/'evidence/key-cache-confirmation-v1.json').read_text())
    stage=json.loads((ROOT/'evidence/key-cache-confirmation-stage-v1.json').read_text())
    if len(rows)!=30 or not stage['complete']:raise RuntimeError('Confirmation incomplete')
    caps=json.loads((ROOT/'evidence/gradient-key-cache-capabilities.json').read_text())
    hashes={r['mode']:r['so_sha256'] for r in caps};identity=None;source=None;index={}
    for row in rows:
        root=ROOT/'evidence'/row['tag'];checked=verify(root)
        actual=[r['identities'] for r in checked]
        if identity is None:identity=actual
        if actual!=identity:raise RuntimeError('Mismatched initial states')
        verify_loaded_worker(checked,f"gradient-key-cache{row['mode']}",hashes[row['mode']])
        before=json.loads((root/'source-identity.json').read_text())
        common={k:v for k,v in before.items() if '/worker-variants/' not in k}
        if source is None:source=common
        if common!=source:raise RuntimeError('Different common source within confirmation')
        ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
        if abs(mean-row['mean_ms'])>1e-8:raise RuntimeError('Summary/raw timing mismatch')
        if row['peak_driver_MiB']>90*1024:raise RuntimeError('Memory budget mismatch')
        key=row['planner'],row['repeat'],row['mode']
        if key in index:raise RuntimeError('Duplicate independent cell')
        index[key]=mean
    summaries=[]
    for policy in ('native','ultra_ep','eplb_fixed'):
        pairs=[dict(repeat=r,original_ms=index[policy,r,0],cache_ms=index[policy,r,1],
                    gain=1-index[policy,r,1]/index[policy,r,0]) for r in range(5)]
        differences=[r['original_ms']-r['cache_ms'] for r in pairs]
        logs=[math.log(r['cache_ms']/r['original_ms']) for r in pairs]
        lo,hi=t_interval(logs);gain_ci=[1-math.exp(hi),1-math.exp(lo)]
        summaries.append(dict(planner=policy,pairs=pairs,mean_original_ms=statistics.mean(r['original_ms'] for r in pairs),
            mean_cache_ms=statistics.mean(r['cache_ms'] for r in pairs),paired_mean_difference_ms=statistics.mean(differences),
            difference_t95_ms=t_interval(differences),geometric_mean_step_reduction=1-math.exp(statistics.mean(logs)),
            reduction_log_t95=gain_ci,all_five_faster=all(v>0 for v in differences),
            lower_t95_exceeds_2_percent=gain_ci[0]>.02,
            multiplicity_adjusted=False))
    result=dict(passed=True,maxvio=20,measured_source_commit='cb4c5df',new_planner=False,
        repeats=5,unit='paired independent process; screening excluded',warmup=5,updates_per_process=25,
        scope='fixed synthetic route bank and shape; no natural training quality inference',
        inference='Unadjusted parametric paired t95 on log ratios; df4; no per-step pseudo-replication',summaries=summaries)
    (ROOT/'analysis/key-cache-confirmation-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
