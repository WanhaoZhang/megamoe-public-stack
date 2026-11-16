"""Fixed-placement min-max-link relaxation, with integer feasible repair.

This is an offline network-row diagnostic/strong quota oracle, NOT a latency
model, an online planner, or a proof of whole-step optimality.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from budget_scan import contract,ROOT


def links(q):
    result=q.sum(axis=1,dtype=np.int64);np.fill_diagonal(result,0);return result


def optimize(source,dest):
    p,e=source.shape
    if dest.shape!=(e,p) or np.any(source<0) or np.any(dest<0) or not np.array_equal(source.sum(0),dest.sum(1)):
        raise ValueError('Inconsistent expert marginals')
    q=np.zeros((p,e,p),np.int32);variables=[];shared=[]
    for x in range(e):
        ds=np.flatnonzero(dest[x]>0)
        if len(ds)==1:q[:,x,int(ds[0])]=source[:,x]
        elif len(ds)>1:
            shared.append(x)
            variables.extend((int(s),x,int(d)) for s in np.flatnonzero(source[:,x]) for d in ds)
    fixed=links(q);local=np.minimum(source,dest.T).sum(axis=1)
    # Certified combinatorial ROW lower bound, independent of LP numerics.
    analytic=max(int(fixed.max(initial=0)),int(np.max((source.sum(1)-local+p-2)//max(p-1,1),initial=0)),
                 int(np.max((dest.sum(0)-local+p-2)//max(p-1,1),initial=0)))
    if not variables:return q,dict(lp_solver_optimum=float(fixed.max(initial=0)),analytic_row_lower_bound=analytic,
        integer_peak_rows=int(fixed.max(initial=0)),integer_gap_over_lp=0.,variables=0,lp_numerical_certificate=False)
    rows={};rhs=[]
    for x in shared:
        for s in range(p):rows['s',x,s]=len(rhs);rhs.append(int(source[s,x]))
        for d in range(p):rows['d',x,d]=len(rhs);rhs.append(int(dest[x,d]))
    n=len(variables);eq=lil_matrix((len(rhs),n+1));ub=lil_matrix((p*(p-1),n+1))
    peer={(s,d):i for i,(s,d) in enumerate((s,d) for s in range(p) for d in range(p) if s!=d)}
    for i,(s,x,d) in enumerate(variables):
        eq[rows['s',x,s],i]=1;eq[rows['d',x,d],i]=1
        if s!=d:ub[peer[s,d],i]=1
    for row in peer.values():ub[row,n]=-1
    b=np.array([-fixed[s,d] for s,d in peer],float);objective=np.zeros(n+1);objective[n]=1
    result=linprog(objective,A_ub=ub.tocsr(),b_ub=b,A_eq=eq.tocsr(),b_eq=rhs,method='highs')
    if not result.success:raise RuntimeError(result.message)
    optimum=float(result.fun)
    # Among relaxation optima prefer less remote volume. Separate solve
    # avoids relying on an unbounded epsilon-weighted objective.
    objective[:n]=[int(s!=d) for s,x,d in variables];objective[n]=0
    bounds=[(0,None)]*n+[(optimum,optimum+1e-5)]
    second=linprog(objective,A_ub=ub.tocsr(),b_ub=b,A_eq=eq.tocsr(),b_eq=rhs,bounds=bounds,method='highs')
    if not second.success:raise RuntimeError(second.message)
    for value,(s,x,d) in zip(second.x[:n],variables):q[s,x,d]=int(np.floor(max(0,value)+1e-7))
    # Integer transportation repair only within each expert's permitted
    # destinations; it cannot add a replica, change GEMM shape or fan-in.
    aggregate=links(q)
    for x in shared:
        supply=source[:,x].astype(np.int64)-q[:,x,:].sum(1)
        demand=dest[x].astype(np.int64)-q[:,x,:].sum(0)
        if np.any(supply<0) or np.any(demand<0):raise RuntimeError('LP rounding exceeded a marginal')
        while supply.sum():
            choices=[(max(int(aggregate[s,d]),0) if s!=d else -1,s,d)
                     for s in np.flatnonzero(supply) for d in np.flatnonzero(demand)]
            _,s,d=min(choices);take=min(supply[s],demand[d]);q[s,x,d]+=take;supply[s]-=take;demand[d]-=take
            if s!=d:aggregate[s,d]+=take
    if not np.array_equal(q.sum(2),source) or not np.array_equal(q.sum(0),dest):raise RuntimeError('Integer flow conservation failed')
    peak=int(links(q).max(initial=0))
    if peak<analytic:raise RuntimeError('Analytic bound violated')
    return q,dict(lp_solver_optimum=optimum,analytic_row_lower_bound=analytic,integer_peak_rows=peak,
        integer_gap_over_lp=peak-optimum,variables=n,lp_numerical_certificate=False,
        scope='Rows on the largest aggregate off-rank link; no bandwidth, latency, prefix readiness or owner service model')


def main():
    p=argparse.ArgumentParser();p.add_argument('--routes',type=Path,required=True);p.add_argument('--fixtures',type=int,default=50)
    args=p.parse_args();records=[];started=time.monotonic()
    for vio in (1,5,20):
        for regime in ('original','regrouped'):
            path=args.routes/f'maxvio-{vio:g}.npy' if regime=='original' else ROOT/f'artifacts/source-heterogeneity-bank-v1/regrouped/maxvio-{vio:g}.npy'
            ids=np.load(path,allow_pickle=False);initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids])
            for call in range(args.fixtures):
                perm=np.random.default_rng(42+100003*call).permutation(256);source=np.empty_like(initial);source[:,perm]=initial
                for policy in ('native','ultra_ep'):
                    proposal=getattr(contract,policy)(source,32,32);dest=proposal.quota.sum(0)
                    flow,stats=optimize(source,dest)
                    local=contract.local_quota(source,dest)
                    records.append(dict(maxvio=vio,regime=regime,call=call,planner=policy,
                        baseline_peak_rows=int(links(proposal.quota).max()),local_peak_rows=int(links(local).max()),
                        baseline_remote_rows=int(links(proposal.quota).sum()),local_remote_rows=int(links(local).sum()),
                        optimized_remote_rows=int(links(flow).sum()),**stats))
            print(vio,regime,'FIXED_PLACEMENT_LINK_ORACLES_DONE',flush=True)
    result=dict(scope='Offline aggregate network-row headroom only; not native performance or new planner',
        scipy_version=__import__('scipy').__version__,elapsed_seconds=time.monotonic()-started,
        domain='All nonnegative continuous source quotas on each fixed replica set, preserving source and destination expert marginals; integer repair feasible',
        limitations=['Numerical LP optimum, not a certified rational dual bound','Analytic row lower bound is certified; neither bound is a latency bound',
                     'No compute/source-prefix/gradient-owner release model','No NPU execution or online LP overhead measured'],records=records)
    (ROOT/'analysis/source-quota-network-floor.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
