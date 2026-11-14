"""Independent NumPy vs portable/device policy implementation; no NPU imports."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('planner_contract',root/'source/native-balanced/ascend_deepep/planner_contract.py')
ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
policies=['moonep','native','ultra_ep','echo','eplb_fixed']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--library',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    lib=ctypes.CDLL(str(args.library.resolve()));fn=lib.policy_reference
    arr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int]*4+[arr]*3;fn.restype=ctypes.c_int
    rng=np.random.default_rng(9317);records=[];start=time.time()
    for p,h,slots in [(2,4,4),(4,8,8),(8,16,16),(8,32,32)]:
        e=p*h
        fixtures=[]
        # Empty experts, repeated assignments, rotating hotspots and replicas.
        for scenario in range(24):
            prob=rng.lognormal(0,scenario%7*.6,e);prob/=prob.sum()
            ids=rng.choice(e,size=(p,128,8),p=prob).astype(np.int32)
            if scenario==0:ids[:]=0
            if scenario==1:ids[:]=(np.arange(128*8).reshape(128,8)%e).astype(np.int32)
            if scenario==2:ids[:]=e-1
            fixtures.append(ids)
        for scenario,ids in enumerate(fixtures):
            source=np.stack([np.bincount(x.ravel(),minlength=e) for x in ids]).astype(np.int32)
            for index,name in enumerate(policies):
                proposal=getattr(ref,name)(source,h,slots)
                cp=np.empty((p,slots),np.int32);quota=np.empty((p,e,p),np.int32)
                status=fn(p,h,slots,index,source,cp,quota)
                if status:raise AssertionError((p,h,scenario,name,'C++ status',status))
                if not np.array_equal(cp,proposal.copies) or not np.array_equal(quota,proposal.quota):
                    diff=np.argwhere(quota!=proposal.quota)
                    raise AssertionError((p,h,scenario,name,'mismatch',diff[:10].tolist(),cp.tolist(),proposal.copies.tolist()))
                capacity=max(1024,int(quota.sum(axis=(0,1)).max()+127)//128*128)
                body=ref.compile_plan(ids,proposal,ref.PlannerResources(h,slots,capacity))
                assert int(body.execution_counts.sum())==ids.size
                assert not body.physical_ids.flags.writeable
                assert all(body.execution_counts[d, h+s]>0 for d in range(p) for s,x in enumerate(cp[d]) if x>=0)
                records.append(dict(p=p,home=h,scenario=scenario,policy=name,sha=body.digest))
    # A malformed source row, duplicate replica and insufficient receive pool
    # must fail in the common compiler; never replace with another policy.
    ids=np.zeros((2,128,1),np.int32);source=np.array([[128,0],[128,0]],np.int32)
    good=ref.moonep(source,1,1)
    badq=good.quota.copy();badq[0,0,0]+=1
    rejected=0
    for prop,res in [(ref.PlannerProposal(good.copies,badq),ref.PlannerResources(1,1,128)),
                     (good,ref.PlannerResources(1,1,0)),
                     (ref.PlannerProposal(np.array([[0],[0]],np.int32),good.quota),ref.PlannerResources(1,1,128))]:
        try:ref.compile_plan(ids,prop,res)
        except ValueError:rejected+=1
        else:raise AssertionError('invalid plan accepted')
    result=dict(status='pass',cases=len(records),invalid_rejections=rejected,elapsed_s=time.time()-start,records=records)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'}))

if __name__=='__main__':main()
