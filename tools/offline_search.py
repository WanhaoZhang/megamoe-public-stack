"""Broader candidate search using the inherited, imperfect frozen DAG model.

No predicted candidate is promoted to an online policy without native timing.
"""
import argparse
import ctypes
import hashlib
import json
import numpy as np
from budget_scan import contract, ROOT, metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',required=True)
    ap.add_argument('--library',required=True)
    args=ap.parse_args()
    from pathlib import Path
    baseline=Path(args.baseline)
    payload=(baseline/'configs/step-time-full.json').read_bytes()
    model=np.asarray(json.loads(payload)['device_vector'],np.int32)
    lib=ctypes.CDLL(args.library)
    arr=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    score=lib.score_reference
    score.argtypes=[arr,arr,arr,arr,ctypes.POINTER(ctypes.c_int64)]
    score.restype=ctypes.c_int
    result=[]
    for vio in (.2,1,5,10,15,20):
        ids=np.load(baseline/f'routes/maxvio-{vio:g}.npy',allow_pickle=False)
        source=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        bank=[(name,getattr(contract,name)(source,32,32)) for name in ('native','ultra_ep','moonep','echo')]
        bank += [(f'eplb_budget_{b}',contract.eplb_fixed(source,32,32,b)) for b in (0,1,2,4,8,16,32,64,128,256)]
        best=None; evaluations=0;seen=set();scores=[]
        def evaluate(name,proposal):
            nonlocal best,evaluations
            key=hashlib.sha256(proposal.quota.tobytes()).hexdigest()
            if key in seen:return
            seen.add(key)
            if np.any(proposal.quota.sum(axis=(0,1),dtype=np.int64)>model[9]):return
            if not np.array_equal(proposal.quota.sum(axis=2),source):raise AssertionError('conservation')
            value=ctypes.c_int64()
            if score(source,model,proposal.copies.copy(),proposal.quota.copy(),ctypes.byref(value)):raise RuntimeError('scorer')
            evaluations+=1;row=dict(name=name,predicted_ms=value.value*16/1e6,**metrics(source,proposal,32))
            scores.append(row)
            if best is None or value.value<best[0]:best=(value.value,name,proposal)
        for name,proposal in bank:
            evaluate(name,proposal)
            dest=proposal.quota.sum(axis=0,dtype=np.int64).astype(np.int32)
            evaluate(name+'+local_source',contract.PlannerProposal(proposal.copies,contract.local_quota(source,dest)))
        rounds=0
        # Start from every policy, not only the old frozen Native start.
        frontier=list(bank)
        for iteration in range(4):
            previous=best[0];rounds+=1
            for name,proposal in frontier:
                dest=proposal.quota.sum(axis=0).astype(np.int32)
                remote=[(int(dest[e,d]),e,d) for e,d in np.argwhere(dest>0) if d!=e//32]
                for _,e,d in sorted(remote)[:32]:
                    changed=dest.copy();changed[e,e//32]+=changed[e,d];changed[e,d]=0
                    copies=contract._copies(changed,32,32)
                    for local in (False,True):
                        quota=contract.local_quota(source,changed) if local else contract.interval_quota(source,changed)
                        evaluate(f'{name}/return-{e}-{d}/local-{int(local)}',contract.PlannerProposal(copies,quota))
            if best[0]>=previous:break
            frontier=[(best[1],best[2])]
        np.savez_compressed(ROOT/f'analysis/candidate-{vio:g}.npz',copies=best[2].copies,quota=best[2].quota)
        result.append(dict(vio=vio,evaluations=evaluations,rounds=rounds,best=best[1],
            predicted_ms=best[0]*16/1e6,candidates=sorted(scores,key=lambda r:r['predicted_ms'])))
    record=dict(status='prediction_only',model_sha256=hashlib.sha256(payload).hexdigest(),
        limitations=['Inherited 47.40ms calibration uncertainty','Base route bank only',
            'Candidate shape domain and ranking not requalified','No NPU execution or online cost',
            'Bounded search is not an optimality certificate'],records=result)
    (ROOT/'analysis/offline-search.json').write_text(json.dumps(record,indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='candidates'} for r in result],indent=2))


if __name__=='__main__':main()
