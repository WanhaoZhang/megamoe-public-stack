"""Re-audit downloaded full-update evidence independently of the remote summary."""
import json
import statistics
from pathlib import Path
from verify_training_run import verify, verify_loaded_worker

ROOT=Path(__file__).resolve().parents[1]


def main():
    rows=json.loads((ROOT/'evidence/gradient-key-cache-screen-v1.json').read_text())
    caps=json.loads((ROOT/'evidence/gradient-key-cache-capabilities.json').read_text())
    hashes={r['mode']:r['so_sha256'] for r in caps}
    identities={}; records=[]
    assert len(rows)==16
    for row in rows:
        root=ROOT/'evidence'/row['tag']; checked=verify(root)
        actual=[r['identities'] for r in checked]
        assert actual==identities.setdefault(row['maxvio'],actual)
        verify_loaded_worker(checked,f"gradient-key-cache{row['mode']}",hashes[row['mode']])
        fetch=json.loads((root/'fetch.json').read_text())
        assert len(fetch['files'])>=80
        ranks=[[json.loads(l) for l in (root/f'steps-rank{r}.jsonl').read_text().splitlines()] for r in range(8)]
        mean=statistics.mean(max(r[s]['milliseconds'] for r in ranks) for s in range(5,25))
        assert abs(mean-row['mean_ms'])<1e-8
        records.append({k:row[k] for k in ('tag','maxvio','planner','mode','mean_ms','peak_driver_MiB')})
    index={(r['maxvio'],r['planner'],r['mode']):r['mean_ms'] for r in records}
    comparisons=[dict(maxvio=v,planner=p,old_ms=index[v,p,0],cache_ms=index[v,p,1],
                      gain=1-index[v,p,1]/index[v,p,0]) for v in (20,.2)
                 for p in ('native','ultra_ep','eplb_budget_32','eplb_fixed')]
    out=dict(passed=True,independent_processes_per_cell=1,warmup=5,updates=25,
             comparisons=comparisons,records=records,
             measured_source_commit='cb4c5df',new_planner=False,
             limitation='Screening only; five fresh paired process repetitions pending')
    (ROOT/'analysis/gradient-key-cache-screen-audit.json').write_text(json.dumps(out,indent=2))
    print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))


if __name__=='__main__':main()
