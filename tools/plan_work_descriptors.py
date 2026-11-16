"""Current-route work descriptors; never turn tile counts into latency bounds."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from budget_scan import ROOT,contract


def describe(q):
    dest=q.sum(axis=0,dtype=np.int64);rows=dest.sum(axis=0)
    active=(dest>0).sum(axis=0);m128=((dest+127)//128).sum(axis=0);m192=((dest+191)//192).sum(axis=0)
    jobs=dict(real_rows=rows,forward_m192=m192,backward_m128=m128,
              active_expert_instances=active,dw1_output_tile_jobs=active*896,dw2_output_tile_jobs=active*448)
    return {name:dict(per_rank=[int(v) for v in values],total=int(values.sum()),maximum=int(values.max()),
        max_over_mean=float(values.max()/values.mean()) if values.any() else 1.) for name,values in jobs.items()}


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',required=True,type=Path);a=p.parse_args()
    records=[];inputs={}
    for vio in (.2,1,5,10,15,20):
        route=a.routes/f'maxvio-{vio:g}.npy';payload=route.read_bytes();inputs[str(route)]=hashlib.sha256(payload).hexdigest()
        ids=np.load(route,allow_pickle=False);initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        for name in ('native','ultra_ep','moonep','echo','eplb_fixed'):
            cases=[]
            for call in range(50):
                perm=np.random.default_rng(42+100003*call).permutation(256);source=np.empty_like(initial);source[:,perm]=initial
                proposal=getattr(contract,name)(source,32,32)
                if not np.array_equal(proposal.quota.sum(2),source):raise RuntimeError('Assignment changed')
                cases.append(dict(key=[call//2,call%2],**describe(proposal.quota)))
            keys=tuple(describe(proposal.quota))
            summary={key:dict(mean_max_over_mean=float(np.mean([r[key]['max_over_mean'] for r in cases])),
                              worst_max_over_mean=max(r[key]['max_over_mean'] for r in cases),
                              mean_total=float(np.mean([r[key]['total'] for r in cases]))) for key in keys}
            records.append(dict(maxvio=vio,planner=name,summary=summary,cases=cases))
        print('WORK_DESCRIPTOR_BANK_COMPLETE',vio,flush=True)
    result=dict(scope='Analytic descriptors of1500 original fixed-owner policy plans; not a calibrated execution or latency model',
        shape=dict(ep=8,experts=256,hidden=7168,ffn=2048,home=32,slots=32),
        input_sha256=inputs,records=records,
        caveats=['Output tile jobs do not count dW arithmetic FLOPs: reduction K depends on token rows',
                 'Partial row tiles and cyclic core assignment have unequal execution costs',
                 'Max/mean descriptor ratios do not bound removable whole-step latency',
                 'Replica weight traffic, owner service dependencies, startup, barriers and framework work are absent'])
    (ROOT/'analysis/plan-work-descriptors.json').write_text(json.dumps(result,indent=2))
    compact=dict(scope=result['scope'],shape=result['shape'],input_sha256=inputs,caveats=result['caveats'],
                 rows=[{k:v for k,v in row.items() if k!='cases'} for row in records])
    (ROOT/'analysis/plan-work-descriptors-summary.json').write_text(json.dumps(compact,indent=2))
    print('PLAN_WORK_DESCRIPTOR_AUDIT_PASS',len(records)*50)


if __name__=='__main__':main()
