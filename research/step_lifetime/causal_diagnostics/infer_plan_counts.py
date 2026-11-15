"""Semantic proposal: derive global execution/contributor counts from quota.

This is an offline oracle over actual compiled plans, not a timed planner.
It tests whether repeated backward CPU observations are logically redundant.
"""
from pathlib import Path
import argparse,json
import numpy as np


def infer(copies,quota,home):
    copies=np.asarray(copies);quota=np.asarray(quota)
    if copies.ndim!=2 or quota.ndim!=3 or home<=0:
        raise ValueError('Invalid metadata dimensions')
    ep,slots=copies.shape;experts=ep*home
    if quota.shape!=(ep,experts,ep) or np.any(quota<0):raise ValueError('Invalid source quota')
    if np.any(copies< -1) or np.any(copies>=experts):raise ValueError('Invalid replica expert')
    routed=quota.astype(np.int64).sum(axis=0)
    counts=np.zeros((ep,home+slots),dtype=np.int64)
    for rank in range(ep):
        counts[rank,:home]=routed[rank*home:(rank+1)*home,rank]
        seen=set()
        for slot,expert in enumerate(copies[rank].tolist()):
            if expert<0:continue
            if expert in seen:raise ValueError('Duplicate physical copy within destination')
            seen.add(expert)
            if expert//home!=rank:counts[rank,home+slot]=routed[expert,rank]
    if counts.sum()!=quota.sum():raise ValueError('Quota refers to unavailable copies or does not conserve tasks')
    return counts


def audit(folders):
    records=[];failures=[];layouts=set()
    for folder in folders:
        for path in sorted((folder/'timer').glob('*.npz')):
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata']))
                layouts.add(meta['layout']['home'])
                predicted=infer(data['copies'],data['source_quota'],meta['layout']['home'])
                rank=meta['rank'];actual=data['execution_counts'].astype(np.int64)
                passed=np.array_equal(predicted[rank],actual)
                row=dict(file=str(path),rank=rank,phase=meta['phase'],recompute=meta['recompute'],
                    invocation=meta['invocation'],passed=passed,
                    active_remote=int(np.count_nonzero(predicted[rank,meta['layout']['home']:])) )
                records.append(row)
                if not passed:failures.append(dict(**row,expected=predicted[rank].tolist(),actual=actual.tolist()))
    return dict(passed=bool(records) and not failures,records=len(records),failures=failures,
                covered_home_layouts=sorted(layouts),
                positive_replica_records=sum(row['active_remote']>0 for row in records),
                scope='offline integer semantic audit; does not establish online speedup or error recovery')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folders',nargs='+',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=audit(args.folders);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='failures'}));raise SystemExit(0 if result['passed'] else 1)
