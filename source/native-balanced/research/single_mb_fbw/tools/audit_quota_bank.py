"""Independent vectorized assignment compiler for captured quota-bank digests."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def physical_ids(ids,quota,copies,home):
    p,m,k=ids.shape;slots=copies.shape[1];e=p*home
    mapping=np.full((e,p),-1,np.int32)
    for d in range(p):
        mapping[d*home:(d+1)*home,d]=np.arange(home)
        for slot,x in enumerate(copies[d]):
            if x>=0:mapping[x,d]=home+slot
    result=[]
    for source in range(p):
        logical=ids[source].ravel();order=np.argsort(logical,kind='stable')
        counts=np.bincount(logical,minlength=e);starts=np.cumsum(counts)-counts
        ordinal=np.empty(len(logical),np.int64)
        ordinal[order]=np.arange(len(logical))-np.repeat(starts,counts)
        ends=quota[source].cumsum(axis=1,dtype=np.int64)
        dest=(ordinal[:,None]>=ends[logical]).sum(axis=1)
        if np.any(dest>=p):raise ValueError('Quota does not cover every logical assignment')
        slot=mapping[logical,dest]
        if np.any(slot<0):raise ValueError('No legal physical replica for an assignment')
        result.append(np.ascontiguousarray((dest*(home+slots)+slot).reshape(m,k),dtype=np.int32))
    return np.stack(result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--bank',required=True,type=Path);p.add_argument('--route',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();meta=json.loads(args.bank.with_suffix('.json').read_text())
    if hashlib.sha256(args.bank.read_bytes()).hexdigest()!=meta['npz_sha256']:raise ValueError('Frozen bank changed')
    if hashlib.sha256(args.route.read_bytes()).hexdigest()!=meta['route_sha256']:raise ValueError('Frozen source bank changed')
    base=np.load(args.route,allow_pickle=False)
    with np.load(args.bank,allow_pickle=False) as data:copies=data['copies'];quotas=data['quota']
    manifests=[json.loads((args.directory/f'prepared-plan-manifest-rank{r}.json').read_text()) for r in range(8)]
    checked=changed=0
    for record in manifests[0]['records']:
        step,layer=record['key'];call=step*2+layer
        perm=np.random.default_rng(42+100003*call).permutation(256)
        ids=np.stack([np.ascontiguousarray(perm[base[r,np.random.default_rng(
            42+100003*call+997*(r+1)).permutation(base.shape[1])]],dtype=np.int32) for r in range(8)])
        expected=physical_ids(ids,quotas[call],copies[call],32)
        for r,manifest in enumerate(manifests):
            row=next(x for x in manifest['records'] if x['key']==[step,layer])
            if hashlib.sha256(expected[r].tobytes()).hexdigest()!=row['physical_sha256']:
                raise RuntimeError('Captured physical plan differs from independent CPU compiler')
            checked+=1
        changed+=bool(record['changed_from_original'])
    if checked!=48 or changed<2:raise RuntimeError('Insufficient changed physical-plan gate coverage')
    result=dict(passed=True,captured_rank_call_digests=checked,changed_global_invocations=changed,
        bank_sha256=meta['npz_sha256'],route_sha256=meta['route_sha256'],
        scope='Actual physical IDs; numerical and performance qualification are separate')
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
