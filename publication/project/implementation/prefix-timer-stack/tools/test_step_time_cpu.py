"""Conservation, independent DAG and bounded deterministic search properties."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from step_time_reference import contract,source_for,propose,simulate,compiled_score
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=ROOT/'calibration/step-time-provisional.json');a=p.parse_args()
    model=json.loads(a.model.read_text());rng=np.random.default_rng(42);ref=contract();cases=[]
    for i in range(48):
        v=list(model['device_vector']);v[18]=10000000;v[19]=64;v[60]=0;v[56]=v[31]=0
        v[5]=257;v[6]=2;v[9]=8192;v[7]=v[8]=512;v[36:38]=[16,8]
        v[11]=1+i%4;v[12]=4 if i<24 else 16;v[16]=i%4
        ids=rng.integers(256,size=(8,257,2),dtype=np.int32)
        if i%6==0:ids[:]=0
        if i%6==1:ids[:]=255
        if i%6==2:ids[:]=np.arange(8)[:,None,None]*32
        if i%6==3:ids[:,:,0]=i%256
        source=np.stack([np.bincount(x.ravel(),minlength=256) for x in ids]).astype(np.int32)
        cp,q,stats=propose(source,v);cp2,q2,stats2=propose(source,v)
        assert np.array_equal(cp,cp2) and np.array_equal(q,q2) and np.array_equal(stats,stats2),'nondeterministic search'
        proposal=ref.PlannerProposal(cp,q);body=ref.compile_plan(ids,proposal,ref.PlannerResources(32,32,8192))
        assert stats[0]<=v[11]*v[12] and stats[1]<=v[11]
        predicted=compiled_score(source,v,cp,q);independent=simulate(v,cp,q)
        assert predicted==independent,(i,predicted,independent)
        if int(stats[6])&2:
            old=getattr(ref,model['initial_policy'])(source,32,32)
            assert np.array_equal(cp,old.copies) and np.array_equal(q,old.quota),'guard did not restore the exact initial proposal'
        else:assert stats[5]<stats[4] and stats[2]>0
        cases.append(dict(case=i,variant=v[16],rounds=int(stats[1]),evaluations=int(stats[0]),adopted=int(stats[2]),flags=int(stats[6]),body_sha=body.digest))
    result=dict(passed=True,cases=cases,adopted_cases=sum(x['adopted']>0 for x in cases),
        independent_task_dag=True,independent_public_compiler=True,model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest())
    (ROOT/'evidence/step-time-cpu.json').write_text(json.dumps(result,indent=2));print('STEP_TIME_CPU_PASS',len(cases),result['adopted_cases'])
if __name__=='__main__':main()
