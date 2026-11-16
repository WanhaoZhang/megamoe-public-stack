"""Audit paired replay headroom, keeping it separate from online performance."""
import json
import statistics
from pathlib import Path
from verify_training_run import verify,verify_loaded_worker

ROOT=Path(__file__).resolve().parents[1]


def main():
    receipt=json.loads((ROOT/'evidence/forced-subset-pilot-v1.json').read_text())
    if len(receipt['rows'])!=24 or receipt['excluded']:raise RuntimeError('Pilot incomplete or excluded cells need review')
    caps=json.loads((ROOT/'evidence/gradient-key-cache-capabilities.json').read_text())
    hashes={r['mode']:r['so_sha256'] for r in caps};identities={};plans={};index={};records=[]
    for row in receipt['rows']:
        root=ROOT/'evidence'/row['tag'];checked=verify(root);actual=[r['identities'] for r in checked]
        if actual!=identities.setdefault(row['maxvio'],actual):raise RuntimeError('Unmatched initial conditions')
        verify_loaded_worker(checked,f"gradient-key-cache{row['mode']}",hashes[row['mode']])
        for rank in range(8):
            manifest=json.loads((root/f'prepared-plan-manifest-rank{rank}.json').read_text())
            if manifest['online_score'] or len(manifest['records'])!=50:raise RuntimeError('Replay boundary/coverage')
            key=row['maxvio'],row['planner'],rank
            if manifest['records']!=plans.setdefault(key,manifest['records']):raise RuntimeError('Metadata intervention changed the plan')
        ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
        if abs(mean-row['mean_ms'])>1e-8:raise RuntimeError('Raw/summary disagreement')
        if row['peak_driver_MiB']>90*1024:raise RuntimeError('Memory ceiling exceeded')
        index[row['maxvio'],row['planner'],row['mode']]=mean
        records.append({k:row[k] for k in ('tag','maxvio','planner','mode','mean_ms','peak_driver_MiB')})
    comparisons=[]
    for vio in (1,5,20):
        for mode in (0,1):
            native=index[vio,'native',mode];ultra=index[vio,'ultra_ep',mode];strong=min(native,ultra)
            for name in ('proxy_best','sparse_best'):
                candidate=index[vio,name,mode]
                comparisons.append(dict(maxvio=vio,mode=mode,candidate=name,mean_ms=candidate,
                    gain_over_native=1-candidate/native,gain_over_ultra=1-candidate/ultra,
                    gain_over_best_existing_in_cell=1-candidate/strong,
                    engineering_2_percent_gate=candidate<.98*strong))
    result=dict(passed=True,cells=24,records=records,comparisons=comparisons,
        scope='Identical prepared-body full-update replay diagnostic; not online planner performance',
        independent_processes_per_cell=1,model='Inherited unqualified proxy; forced proposals, no model-guard fallback',
        candidate_domain='Delete Native copies and return their complete source quotas to fixed owner',
        numerical_and_physical_gate='6 original plus2 cache1 composition gates; captured physical hashes independently audited',
        caveats=['Not exhaustive placement, quota or pipeline optimization','Sparse-best at MaxVio1 equals the known no-replication baseline',
                 'No repeat uncertainty from this single-process screen','Prepared guard/storage overhead paid by all arms'])
    (ROOT/'analysis/forced-subset-pilot-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
