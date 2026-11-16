"""Compare the actual cached-key header with original source ordering semantics."""
import ctypes
import hashlib
import json
from pathlib import Path
import numpy as np
from budget_scan import ROOT,contract


def main():
    rng=np.random.default_rng(20261116);plans=[]
    for vio in (.2,20):
        ids=np.load(ROOT.parent/f'planner-executor-ep8-lab/routes/maxvio-{vio:g}.npy',allow_pickle=False)
        counts=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        for step in range(25):
            for layer in range(2):
                perm=np.random.default_rng(42+100003*(step*2+layer)).permutation(256)
                source=np.empty_like(counts);source[:,perm]=counts
                for policy in ('native','ultra_ep','eplb_fixed'):
                    plans.append((8,32,32,getattr(contract,policy)(source,32,32).copies.copy(),f'{vio:g}/{policy}'))
    for _ in range(400):
        ranks=int(rng.choice([2,4,8]));home=int(rng.choice([1,4,16,32]));slots=int(rng.choice([0,1,4,32]))
        copies=np.full((ranks,slots),-1,np.int32)
        for rank in range(ranks):
            candidates=[e for e in range(ranks*home) if e//home!=rank]
            used=int(rng.integers(min(slots,len(candidates))+1))
            positions=rng.choice(slots,size=used,replace=False)
            copies[rank,positions]=rng.choice(candidates,size=used,replace=False)
        plans.append((ranks,home,slots,copies,'random'))
    cases=0;counts=[]
    arr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    for priority in (0,1):
        lib=ctypes.CDLL(str(ROOT/f'analysis/gradient_order{priority}_oracle.dll'));fn=lib.build_gradient_order
        fn.argtypes=[arr,*([ctypes.c_int]*5),arr,ctypes.POINTER(ctypes.c_int)];fn.restype=ctypes.c_int
        for ranks,home,slots,cp,label in plans:
            fanin=np.bincount(cp[cp>=0],minlength=ranks*home)
            for rank in range(ranks):
                extent=home+slots;logical=list(range(rank*home,(rank+1)*home))+cp[rank].tolist()
                def key(e):
                    ex=logical[e]
                    if ex<0:return 100000+e
                    return 10000+ex if not fanin[ex] else (ranks-int(fanin[ex]))*256+ex if priority else ex
                expected=sorted(range(extent),key=key)
                order=np.full(64,-77,np.int32);reads=ctypes.c_int()
                assert fn(cp,ranks,rank,home,slots,extent,order,ctypes.byref(reads))==0
                assert order[:extent].tolist()==expected,(priority,label,rank)
                assert np.all(order[extent:]==-77),'Output extent overwritten'
                assert reads.value==ranks*slots,'Cached path rescanned global metadata'
                # Source-level logical Fresh-read count, not machine-code or time.
                old=list(range(extent));old_reads=0
                def old_key(e):
                    nonlocal old_reads
                    old_reads+=int(e>=home)+(ranks*slots if logical[e]>=0 else 0)
                    return key(e)
                for i in range(1,extent):
                    value=old[i];j=i-1
                    while j>=0 and old_key(old[j])>old_key(value):old[j+1]=old[j];j-=1
                    old[j+1]=value
                assert old==expected
                if label!='random':
                    active=bool(np.any(cp>=0))
                    counts.append(dict(priority=priority,label=label,active=active,
                        old_logical_reads=old_reads if active else 0,cached_reads=reads.value if active else 0))
                cases+=1
        bad=np.full((8,32),-1,np.int32);bad[0,0]=256;out=np.zeros(64,np.int32);reads=ctypes.c_int()
        assert fn(bad,8,0,32,32,64,out,ctypes.byref(reads))==1
        assert fn(bad,16,0,16,32,48,out,ctypes.byref(reads))==1
    aggregates=[]
    for priority in (0,1):
        for label in sorted({r['label'] for r in counts}):
            selected=[r for r in counts if r['priority']==priority and r['label']==label]
            aggregates.append(dict(priority=priority,label=label,cases=len(selected),
                active_cases=sum(r['active'] for r in selected),
                old_source_reads_mean=float(np.mean([r['old_logical_reads'] for r in selected])),
                cached_reads_mean=float(np.mean([r['cached_reads'] for r in selected]))))
    result=dict(passed=True,cases=cases,records=aggregates,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'source/native-balanced/include/mega_moe_gradient_order.h',
            ROOT/'source/native-balanced/include/mega_moe_gradient_priority.h',ROOT/'tools/gradient_order_oracle.cpp')},
        oracle_sha256={str(p):hashlib.sha256((ROOT/f'analysis/gradient_order{p}_oracle.dll').read_bytes()).hexdigest() for p in (0,1)},
        scope='Cached header equals original ordering for actual endpoint fixtures and random EP2/4/8 placements; unsupported/malformed cases reject',
        limitation='Read counts describe C++ source evaluation, not optimized instructions or NPU time; inactive-copy kernels skip initialization entirely')
    (ROOT/'evidence/gradient-key-cache-cpu.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
