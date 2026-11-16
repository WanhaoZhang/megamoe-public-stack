"""Cross-check portable online policy math against independent NumPy and compiler."""
import ctypes
import json
from pathlib import Path
import sys
import numpy as np
from budget_scan import contract, ROOT


def main():
    lib=ctypes.CDLL(sys.argv[1]);fn=lib.policy_reference
    arr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*4+[arr]*3;fn.restype=ctypes.c_int
    rng=np.random.default_rng(20261115);cases=0
    for p,h,s in ((2,4,4),(4,8,8),(8,32,32)):
        for scenario in range(16):
            ids=rng.integers(0,p*h,size=(p,64,4),dtype=np.int32)
            if scenario==0:ids[:]=0
            if scenario==1:ids[:]=p*h-1
            if scenario>3:
                prob=rng.lognormal(0,scenario/3,p*h);prob/=prob.sum()
                ids=rng.choice(p*h,size=ids.shape,p=prob).astype(np.int32)
            source=np.stack([np.bincount(x.ravel(),minlength=p*h) for x in ids]).astype(np.int32)
            variants=[(i,getattr(contract,n)(source,h,s)) for i,n in enumerate(('moonep','native','ultra_ep','echo','eplb_fixed'))]
            variants += [(100+b,contract.eplb_fixed(source,h,s,b)) for b in sorted(set((0,1,2,4,min(8,p*s),p*s//2,p*s)))]
            for code,expected in variants:
                cp=np.empty((p,s),np.int32);q=np.empty((p,p*h,p),np.int32)
                assert fn(p,h,s,code,source,cp,q)==0,(p,h,scenario,code)
                np.testing.assert_array_equal(cp,expected.copies)
                np.testing.assert_array_equal(q,expected.quota)
                capacity=(int(q.sum(axis=(0,1)).max())+127)//128*128
                body=contract.compile_plan(ids,expected,contract.PlannerResources(h,s,capacity))
                assert body.execution_counts.sum()==ids.size
                cases+=1
            full=contract.eplb_fixed(source,h,s,p*s)
            np.testing.assert_array_equal(full.quota,contract.eplb_fixed(source,h,s).quota)
            for invalid in (-1,p*s+1,0.5):
                try:contract.eplb_fixed(source,h,s,invalid)
                except ValueError:pass
                else:raise AssertionError('invalid budget accepted')
    receipt=dict(status='pass',cases=cases,scope='Portable C++ vs NumPy plus full assignment materialization; not NPU qualification')
    (ROOT/'evidence/budget-cpu-tests.json').write_text(json.dumps(receipt,indent=2))
    print(receipt)


if __name__=='__main__':main()
