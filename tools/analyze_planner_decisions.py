"""Audit actual formal decisions, de-duplicating the eight identical rank votes."""
import csv,json
from pathlib import Path
import numpy as np
from campaign_records import resolved_campaign
ROOT=Path(__file__).resolve().parents[1]
def main():
    stages=['primary-step-time','fourth-paired','heldout-a','heldout-b','ablation-no_gradient_service','ablation-linear_kernel','ablation-uniform_topology']
    rows=[];summary=[]
    for stage in stages:
        ledger=ROOT/f'evidence/campaign-{stage}.json'
        if not ledger.exists():continue
        count=len(json.loads(ledger.read_text()));attempts=resolved_campaign(ROOT,stage,count)
        for run in attempts:
            if run['planner']!='step_time':continue
            directory=ROOT/'evidence'/run['tag'];ranks=[np.load(directory/f'planner-decisions-rank{r}.npy',allow_pickle=False) for r in range(8)]
            if any(x.shape!=(50,8) or x.dtype!=np.int32 for x in ranks):raise ValueError('missing actual 50-call decision stream')
            if any(not np.array_equal(ranks[0],x) for x in ranks[1:]):raise ValueError('ranks disagree on a policy decision')
            for index,values in enumerate(ranks[0]):
                evaluations,rounds,adopted,policy,before,after,flags,capacity=map(int,values)
                before,after=values[4:6].view(np.float32).tolist()
                assert np.isfinite(before) and np.isfinite(after) and after<=before and capacity==163840
                assert evaluations<=4 and rounds<=1
                assert (adopted==0 and policy==1 and before==after) if flags&2 else (adopted>0 and policy==5 and after<before)
                rows.append(dict(stage=stage,tag=run['tag'],maxvio=run['maxvio'],repeat=run['repeat'],step=index//2,layer=index%2,
                    evaluations=evaluations,rounds=rounds,adopted=adopted,source_policy=policy,predicted_before_ms=before,predicted_after_ms=after,
                    uncertainty_fallback=bool(flags&2),initial_outside_domain=bool(flags&1 and rounds==0),any_candidate_outside_domain=bool(flags&1),actual_capacity=capacity))
        for vio in (.2,1,5,10,15,20):
            subset=[x for x in rows if x['stage']==stage and x['maxvio']==vio]
            if subset:summary.append(dict(stage=stage,maxvio=vio,global_invocations=len(subset),adopted=sum(x['adopted']>0 for x in subset),
                uncertainty_fallback=sum(x['uncertainty_fallback'] for x in subset),initial_outside_domain=sum(x['initial_outside_domain'] for x in subset),
                mean_evaluations=float(np.mean([x['evaluations'] for x in subset]))))
    out=ROOT/'analysis';out.mkdir(exist_ok=True)
    (out/'planner-decisions-summary.json').write_text(json.dumps(dict(actual_execution=True,eight_ranks_identical=True,global_invocations=len(rows),points=summary),indent=2))
    if rows:
        with (out/'planner-decisions.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(actual_global_invocations=len(rows),adopted=sum(x['adopted']>0 for x in rows),points=len(summary))),flush=True)
if __name__=='__main__':main()
