"""Reconstruct exact timed copies/quotas before extracting structural features.

No feature here is a time estimate or a calibrated planner cost. Full dW
payload is charged once per nonzero contributor, never scaled by token count.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from budget_scan import contract,ROOT

MIB=1024**2


def describe(quota,copies):
    dest=quota.sum(axis=0,dtype=np.int64);active=dest>0
    home=32;ranks=8
    remote=active.copy();remote[np.arange(256),np.arange(256)//home]=False
    links=quota.sum(axis=1,dtype=np.int64);np.fill_diagonal(links,0)
    fanin=remote.sum(axis=1)
    owner_partial=np.zeros(ranks,dtype=np.int64)
    gradient_links=np.zeros((ranks,ranks),dtype=np.int64)
    for expert,rank in np.argwhere(remote):gradient_links[rank,expert//home]+=84
    for expert in np.flatnonzero(fanin):owner_partial[expert//home]+=int(active[expert].sum())*84
    home_rows=[int(dest[r*home:(r+1)*home,r].sum()) for r in range(ranks)]
    return dict(rows_by_rank=dest.sum(axis=0).tolist(),home_dHp_rows_by_rank=home_rows,
        declared_copies_by_rank=(copies>=0).sum(axis=1).tolist(),
        host_physical_allocation_extent=64,
        physical_expert_extent=64 if np.any(copies>=0) else 32,
        gradient_pipe_active=bool(np.any(copies>=0)),
        extent_note='Actual kernel overrides to homeTiling32 and skips EnableGradientPipe when header active-copy flag is zero; host allocation remains64',
        active_physical_by_rank=active.sum(axis=0).tolist(),active_remote_by_rank=remote.sum(axis=0).tolist(),
        active_remote_total=int(remote.sum()),owner_remote_fanin_max=int(fanin.max()),
        gradient_partial_MiB_total=int(remote.sum())*84,
        gradient_partial_MiB_links=gradient_links.tolist(),owner_partial_read_MiB=owner_partial.tolist(),
        W2_restore_MiB_by_rank=((copies>=0).sum(axis=1)*28).tolist(),
        source_remote_rows=int(links.sum()),source_link_max_rows=int(links.max()),
        nonempty_source_expert_destination_fragments=int(np.count_nonzero(quota)),
        row_tiles128_by_rank=((dest+127)//128).sum(axis=0).tolist(),
        row_tiles192_by_rank=((dest+191)//192).sum(axis=0).tolist(),
        padded_rows128_total=int((((dest+127)//128)*128-dest).sum()),
        dW_output_tile_tasks_total=int(active.sum())*(896+448),
        home_dHp_rows_note='Rows only, not predicted runahead time; per-source/operand readiness and contention still matter')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--routes',type=Path,required=True)
    parser.add_argument('--receipt',default='prepared-plan-pilot-v2.json');args=parser.parse_args()
    runs=json.loads((ROOT/'evidence'/args.receipt).read_text());seen=set();cases=[]
    for run in runs:
        key=run['maxvio'],run['planner'],run['proposal']
        if key in seen:continue
        seen.add(key);vio,policy,proposal_mode=key;directory=ROOT/'evidence'/run['tag']
        route=args.routes/f'maxvio-{vio:g}.npy';ids=np.load(route,allow_pickle=False)
        initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        manifests=[json.loads((directory/f'prepared-plan-manifest-rank{r}.json').read_text())['records'] for r in range(8)]
        features=[]
        for record in manifests[0]:
            step,layer=record['key'];perm=np.random.default_rng(42+100003*(step*2+layer)).permutation(256)
            source=np.empty_like(initial);source[:,perm]=initial
            if policy.startswith('eplb_budget_'):plan=contract.eplb_fixed(source,32,32,int(policy.split('_')[-1]))
            else:plan=getattr(contract,policy)(source,32,32)
            cp=plan.copies.copy();q=plan.quota.copy()
            if proposal_mode=='local-source':q=contract.local_quota(source,q.sum(axis=0))
            elif proposal_mode=='reverse-slots':
                for d in range(8):
                    used=np.flatnonzero(cp[d]>=0);cp[d,used]=cp[d,used[::-1]]
            elif proposal_mode!='original':raise ValueError('Unsupported intervention')
            hashes={name+'_sha256':hashlib.sha256(a.astype(np.int32).tobytes()).hexdigest() for name,a in [('copies',cp),('quota',q)]}
            for rank_records in manifests:
                match=next(r for r in rank_records if r['key']==[step,layer])
                if any(match[k]!=value for k,value in hashes.items()):raise ValueError(f'Actual plan mismatch {key,step,layer}')
            if not np.array_equal(q.sum(axis=2),source):raise ValueError('Source conservation')
            features.append(dict(step=step,layer=layer,**hashes,**describe(q,cp)))
        cases.append(dict(maxvio=vio,planner=policy,proposal=proposal_mode,run=run['tag'],
            route_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),features=features))
    result=dict(schema=1,shape=dict(ep=8,experts=256,hidden=7168,ffn=2048,tokens=16384,topk=8),
        plan_count=sum(len(c['features']) for c in cases),
        scope='Every CPU copy/quota hash checked against all eight ranks of actual timed preparations; structural features only',
        limitations=['No latency fit or causal attribution from feature correlation',
            'Row tiles do not model per-expert waves: the AIC start cursor rotates across experts',
            'Source fragments are not the number of dispatch handshakes, which also publish zero-row arrivals'],cases=cases)
    (ROOT/'analysis/prepared-plan-features.json').write_text(json.dumps(result,indent=2))
    summary=[]
    for c in cases:
        fs=[f for f in c['features'] if f['step']>=5]
        fields=('active_remote_total','gradient_partial_MiB_total','source_remote_rows','dW_output_tile_tasks_total')
        summary.append(dict(maxvio=c['maxvio'],planner=c['planner'],proposal=c['proposal'],
            **{f:float(np.mean([row[f] for row in fs])) for f in fields},
            home_dHp_rows_min_rank_mean=float(np.mean([min(row['home_dHp_rows_by_rank']) for row in fs]))))
    (ROOT/'analysis/prepared-plan-features-summary.json').write_text(json.dumps(dict(scope=result['scope'],summary=summary),indent=2))
    print(json.dumps(dict(plan_count=result['plan_count'],summary=summary),indent=2))


if __name__=='__main__':main()
