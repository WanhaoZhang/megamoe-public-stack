"""Isolated diagnostic costs/plan summaries; never substituted for training time."""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TARGETS=(.2,1,5,10,15,20);POLICIES=('moonep','native','ultra_ep','echo','eplb_fixed','step_time')
def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',default='planner-primary-plan-bank');a=p.parse_args();directory=ROOT/'evidence'/a.tag
    status=json.loads((directory/'status.json').read_text());assert status['qualified'] and status['source_unchanged']
    streams=[]
    for rank in range(8):
        rows=[json.loads(x) for x in (directory/f'plan-summary-rank{rank}.jsonl').read_text().splitlines()]
        mapping={(x['policy'],x['maxvio'],x['step'],x['layer']):x for x in rows}
        assert len(mapping)==len(rows)==1800;streams.append(mapping)
    calls=[];points=[]
    for policy in POLICIES:
        for vio in TARGETS:
            for step in range(25):
                for layer in range(2):
                    current=[r[policy,vio,step,layer] for r in streams]
                    assert len({r['copies_sha256'] for r in current})==len({r['quota_sha256'] for r in current})==1
                    assert all(math.isfinite(r[k]) and r[k]>=0 for r in current for k in ('isolated_plan_wall_ms','isolated_plan_stream_ms'))
                    x=current[0]
                    calls.append(dict(policy=policy,maxvio=vio,step=step,layer=layer,wall_ms=max(r['isolated_plan_wall_ms'] for r in current),
                        stream_ms=max(r['isolated_plan_stream_ms'] for r in current),replicas=x['replica_count'],local_fraction=x['local_assignment_fraction'],
                        maximum_rank_rows=max(x['execution_rows_by_rank']),minimum_rank_rows=min(x['execution_rows_by_rank']),
                        full_remote_gradient_bytes=x['replica_count']*x['full_gradient_bytes_per_nonzero_replica'],copies_sha256=x['copies_sha256'],quota_sha256=x['quota_sha256']))
            warm=[x for x in calls if x['policy']==policy and x['maxvio']==vio and x['step']>=5]
            assert len(warm)==40;times=sorted(x['wall_ms'] for x in warm)
            points.append(dict(policy=policy,maxvio=vio,calls=40,mean_wall_ms=statistics.mean(times),p50_wall_ms=statistics.median(times),p90_wall_ms=times[35],
                min_wall_ms=times[0],max_wall_ms=times[-1],mean_stream_ms=statistics.mean(x['stream_ms'] for x in warm),
                mean_global_replicas=statistics.mean(x['replicas'] for x in warm),mean_source_local_fraction=statistics.mean(x['local_fraction'] for x in warm),
                maximum_rank_rows=max(x['maximum_rank_rows'] for x in warm),mean_global_full_gradient_bytes=statistics.mean(x['full_remote_gradient_bytes'] for x in warm),
                diagnostic_only=True,sequential_processes=1,training_score=False))
    for name,data in [('planning-overhead.csv',points),('isolated-plan-calls.csv',calls)]:
        with (ROOT/'analysis'/name).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,list(data[0]));w.writeheader();w.writerows(data)
    record=dict(passed=True,tag=a.tag,global_plan_calls=len(calls),points=points,
        semantics='one sequential plan-only process;5 warmup steps+20 measured;40 current-route calls/point; max8-rank wall per invocation; not formal training performance',
        plan_snapshots='evidence/'+a.tag+'/plans',actual_formal_decisions_are_separate=True,
        source_files={str((directory/f'plan-summary-rank{r}.jsonl').relative_to(ROOT)):hashlib.sha256((directory/f'plan-summary-rank{r}.jsonl').read_bytes()).hexdigest() for r in range(8)})
    (ROOT/'analysis/planning-overhead.json').write_text(json.dumps(record,indent=2));print('ISOLATED_PLAN_BANK_PASS',len(calls),len(points),flush=True)
if __name__=='__main__':main()
