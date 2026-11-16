"""Exhaust a stated Native-copy deletion subspace on actual route fixtures.

The inherited proxy is deliberately retained as an unqualified ranking model.
Saved banks are forced-plan diagnostics, never online planner performance.
"""
import argparse
import ctypes
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from budget_scan import contract,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--library',type=Path,required=True);p.add_argument('--vios',type=float,nargs='+',default=[.2,1,5,10,15,20])
    args=p.parse_args();payload=(args.baseline/'configs/step-time-full.json').read_bytes()
    model=np.asarray(json.loads(payload)['device_vector'],dtype=np.int32)
    lib=ctypes.CDLL(str(args.library.resolve()));score=lib.score_reference
    arr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    score.argtypes=[arr,arr,arr,arr,ctypes.POINTER(ctypes.c_int64)];score.restype=ctypes.c_int
    all_cases=[];start=time.monotonic()
    for vio in args.vios:
        path=args.baseline/f'routes/maxvio-{vio:g}.npy';ids=np.load(path,allow_pickle=False)
        initial=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        banks={name:dict(copies=[],quota=[]) for name in ('proxy_best','sparse_best')};records=[]
        for step in range(25):
            for layer in range(2):
                permutation=np.random.default_rng(42+100003*(step*2+layer)).permutation(256)
                source=np.empty_like(initial);source[:,permutation]=initial
                native=contract.native(source,32,32);cp0=native.copies;quota0=native.quota
                edges=[(int(d),int(j),int(cp0[d,j])) for d,j in np.argwhere(cp0>=0)]
                if len(edges)>12:raise ValueError('Exhaustive domain exceeds explicit twelve-copy limit')
                destination=quota0.sum(axis=0,dtype=np.int64);load=destination.sum(axis=0)
                proxy_best=sparse_best=None;legal=0;native_score=None
                for mask in range(1<<len(edges)):
                    loads=load.copy()
                    removed=[edge for bit,edge in enumerate(edges) if mask&(1<<bit)]
                    for d,slot,e in removed:
                        n=destination[e,d];loads[d]-=n;loads[e//32]+=n
                    if np.any((loads+127)//128*128>int(model[9])):continue
                    cp=cp0.copy();q=quota0.copy()
                    for d,slot,e in removed:
                        q[:,e,e//32]+=q[:,e,d];q[:,e,d]=0;cp[d,slot]=-1
                    value=ctypes.c_int64()
                    if score(source,model,cp,q,ctypes.byref(value)):raise RuntimeError('Proxy score failed')
                    legal+=1
                    if mask==0:native_score=value.value
                    count=len(edges)-len(removed)
                    row=dict(mask=mask,remaining=count,max_rank_rows=int(loads.max()),score_units=value.value,
                        copies=cp,quota=q)
                    proxy_key=(value.value,count,int(loads.max()),mask)
                    sparse_key=(count,int(loads.max()),value.value,mask)
                    if proxy_best is None or proxy_key<proxy_best[0]:proxy_best=(proxy_key,row)
                    if sparse_best is None or sparse_key<sparse_best[0]:sparse_best=(sparse_key,row)
                if native_score is None:raise ValueError('Original Native plan exceeds common capacity')
                selected={}
                for name,best in [('proxy_best',proxy_best),('sparse_best',sparse_best)]:
                    row=best[1];cp=row['copies'].copy();q=row['quota'].copy()
                    if not np.array_equal(q.sum(axis=2),source):raise AssertionError('Assignment conservation')
                    dest=q.sum(axis=0);remote={(e,d) for e,d in np.argwhere(dest>0) if e//32!=d}
                    declared={(int(cp[d,j]),int(d)) for d,j in np.argwhere(cp>=0)}
                    if remote!=declared:raise AssertionError('Contributor/replica mismatch')
                    banks[name]['copies'].append(cp);banks[name]['quota'].append(q)
                    selected[name]={k:v for k,v in row.items() if k not in ('copies','quota')}
                    selected[name].update(copies_sha256=hashlib.sha256(cp.tobytes()).hexdigest(),
                        quota_sha256=hashlib.sha256(q.tobytes()).hexdigest(),changed_from_native=bool(row['mask']))
                records.append(dict(step=step,layer=layer,native_copies=len(edges),
                    subsets_total=1<<len(edges),capacity_legal_subsets=legal,native_score_units=native_score,
                    token_load_lower_bound=int(source.sum()//8),selected=selected))
        archives={}
        fixture_hashes={str(rank):{} for rank in range(8)}
        for record in records:
            step,layer=record['step'],record['layer'];counter=step*2+layer
            perm=np.random.default_rng(42+100003*counter).permutation(256)
            for rank in range(8):
                tokens=np.random.default_rng(42+100003*counter+997*(rank+1)).permutation(ids.shape[1])
                actual=np.ascontiguousarray(perm[ids[rank,tokens]],dtype=np.int32)
                fixture_hashes[str(rank)][f'{step}:{layer}']=hashlib.sha256(actual.tobytes()).hexdigest()
        for name,bank in banks.items():
            output=ROOT/f'artifacts/native-subsets-vio{vio:g}-{name}.npz'
            buffer=io.BytesIO();np.savez_compressed(buffer,copies=np.stack(bank['copies']),quota=np.stack(bank['quota']))
            payload_bank=buffer.getvalue()
            if output.exists() and output.read_bytes()!=payload_bank:raise ValueError('Refusing to overwrite a different candidate bank')
            output.write_bytes(payload_bank)
            archives[name]=dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())
            metadata=dict(schema=1,diagnostic_only=True,online_score=False,candidate=name,maxvio=vio,
                npz_sha256=archives[name]['sha256'],route_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_sha256=hashlib.sha256(payload).hexdigest(),capacity=163840,
                shape=dict(ep=8,experts=256,hidden=7168,ffn=2048,tokens=16384,topk=8),
                fixture_hashes=fixture_hashes,records=[dict(key=[r['step'],r['layer']],**r['selected'][name]) for r in records])
            output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
        case=dict(maxvio=vio,route_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),archives=archives,records=records)
        all_cases.append(case)
        print(json.dumps(dict(vio=vio,all_subsets=sum(r['subsets_total'] for r in records),
            legal_subsets=sum(r['capacity_legal_subsets'] for r in records),
            changed={name:sum(r['selected'][name]['changed_from_native'] for r in records) for name in banks})),flush=True)
    result=dict(schema=1,diagnostic_only=True,online_score=False,model_sha256=hashlib.sha256(payload).hexdigest(),
        oracle_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),elapsed_seconds=time.monotonic()-start,
        domain='All subsets of each actual Native copy set; removed source quotas return to fixed owner; placement/remaining quota otherwise fixed',
        limitations=['Not exhaustive placement, source transportation or pipeline search',
            'Inherited proxy has unqualified ranking and does not encode redispatch fence correctly',
            'Token-load lower bound is not a latency lower bound',
            'Proxy minima are exact only in this stated finite subspace, not native execution optima'],cases=all_cases)
    (ROOT/'analysis/native-subsets.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
