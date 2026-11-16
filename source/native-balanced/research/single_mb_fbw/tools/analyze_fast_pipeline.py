"""Independent audit of sealed fast-pipeline screens; processes are N=1."""
import json
from pathlib import Path
import statistics
from verify_training_run import verify,verify_loaded_worker

ROOT=Path(__file__).resolve().parents[1]

def main():
    stage=json.loads((ROOT/'evidence/fast-redispatch-stage-v1.json').read_text())
    if not stage['complete']:raise RuntimeError('Fast pipeline stage incomplete')
    caps=json.loads((ROOT/'evidence/fast-redispatch-capabilities-v1.json').read_text())
    by_worker={r['variant']:r for r in caps}
    if len(by_worker)!=8 or len({r['source_archive_sha256'] for r in caps})!=1:
        raise RuntimeError('Expected eight same-source workers')
    rows=json.loads((ROOT/'evidence/fast-redispatch-screen-v1.json').read_text())
    identities={};measurements=[]
    for row in rows:
        root=ROOT/'evidence'/row['tag'];checked=verify(root);worker=row['worker']
        verify_loaded_worker(checked,worker,by_worker[worker]['so_sha256'])
        actual=[r['identities'] for r in checked]
        if actual!=identities.setdefault(row['maxvio'],actual):raise RuntimeError('Different initial parameters/inputs/routes')
        # The policy and physical assignment remain the same for all workers.
        for rank in range(8):
            route=json.loads((root/f'planning-audit-rank{rank}.json').read_text())
            if any(r['replay']!=dict(original=50,recompute=50,backward=50) for r in route):
                raise RuntimeError('Forward/recompute/backward lifetime mismatch')
        ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        ms=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
        if abs(ms-row['mean_ms'])>1e-8:raise RuntimeError('Recomputed process mean mismatch')
        measurements.append(dict(tag=row['tag'],maxvio=row['maxvio'],planner=row['planner'],worker=worker,
            mean_ms=ms,peak_driver_MiB=row['peak_driver_MiB'],independent_processes=1,
            loaded_so_sha256=by_worker[worker]['so_sha256']))
    expected={(v,p,w) for v in (20,.2) for p in ('native','ultra_ep')
              for w in stage['eligible'] if w.startswith('fastpipe')}
    if {(r['maxvio'],r['planner'],r['worker']) for r in measurements}!=expected:raise RuntimeError('Missing/extra cells')
    lookup={(r['maxvio'],r['planner'],r['worker']):r['mean_ms'] for r in measurements}
    comparisons=[]
    for r in measurements:
        baseline=lookup[(r['maxvio'],r['planner'],'fastpipe-r0-t0')]
        comparisons.append(dict(**r,reduction_vs_same_planner_r0=1-r['mean_ms']/baseline))
    result=dict(passed=True,source_archive_sha256=caps[0]['source_archive_sha256'],
        native_source_commit=caps[0]['source_commit'],eligible=stage['eligible'],rows=comparisons,
        scope='Full online stats/planning/forward/recompute/backward/optimizer, final20 of25 updates, rank maximum per update',
        caveats=['One independent process per cell; exploratory screening, no confidence interval or stable win',
                 'Unchanged logical input identity is verified; this analyzer does not capture every physical plan',
                 'Numerical qualification uses the original gates; strict all-parameter and trajectory caveats remain',
                 'Executor improvement alone is not new planner novelty'])
    (ROOT/'analysis/fast-pipeline-screen-summary.json').write_text(json.dumps(result,indent=2))
    print('FAST_PIPELINE_SCREEN_AUDIT_PASS',len(comparisons))
    for r in comparisons:print(r['maxvio'],r['planner'],r['worker'],round(r['mean_ms'],3),round(r['reduction_vs_same_planner_r0']*100,3))

if __name__=='__main__':main()
