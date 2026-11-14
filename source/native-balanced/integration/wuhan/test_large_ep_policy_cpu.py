"""Differential tests of the shared GM policy against existing NumPy oracles.

Host sanitizers cover bounds/overflow; this is not multi-node NPU validation.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
import types
import numpy as np

CPP=r'''
#include <iostream>
#include <vector>
#define __aicore__
#include "mega_moe_large_ep_policy.h"
#include "mega_moe_large_ep_limits.h"
int main(){
 int mode,p,h,s,b,pen,g,target;std::cin>>mode>>p>>h>>s>>b>>pen>>g>>target;
 int e=p*h;std::vector<int> counts(e),q(e*p,-777),cp(p*s,-777),quota(p*p,-777);
 for(auto&v:counts)std::cin>>v;
 int before=0;
 bool shape=MegaMoeLargeEP::SupportedShape(p,h,s);
 bool ok=shape&&(mode ? MegaMoeLargePolicy::Moon(counts.data(),e,p,h,target,q.data(),cp.data(),quota.data()) :
  MegaMoeLargePolicy::Native(counts.data(),e,p,h,s,b,pen,g,q.data(),cp.data(),before));
 std::cout<<ok<<" "<<before<<"\n";
 if(ok){for(int v:q)std::cout<<v<<" ";std::cout<<"\n";for(int v:cp)std::cout<<v<<" ";}
}
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    pkg=types.ModuleType('large_ep_oracle');pkg.__path__=[str(root/'ascend_deepep')]
    sys.modules[pkg.__name__]=pkg
    native=importlib.import_module('large_ep_oracle.megaflux_plan')
    moon=importlib.import_module('large_ep_oracle.moonep_plan')
    limits=importlib.import_module('large_ep_oracle.runtime_limits')
    (args.out/'policy.cpp').write_text(CPP)
    cmd=['g++','-std=c++17','-O1','-g','-fsanitize=address,undefined,bounds','-fno-sanitize-recover=all',
         '-I',str(root/'include'),str(args.out/'policy.cpp'),'-o',str(args.out/'policy')]
    subprocess.run(cmd,check=True)
    def call(mode,counts,p,h,s,b,pen,g,target):
        text=' '.join(map(str,[mode,p,h,s,b,pen,g,target,*map(int,counts)]))
        r=subprocess.run([str(args.out/'policy')],input=text,text=True,capture_output=True,timeout=30)
        assert r.returncode==0,r.stderr
        assert not r.stderr,r.stderr
        v=np.fromstring(r.stdout,sep=' ',dtype=np.int64)
        if not v[0]:return None
        e=p*h;return v[2:2+e*p].reshape(e,p),v[2+e*p:].reshape(p,s)
    rng=np.random.default_rng(42);rows=[]
    for p in (2,4,8,16,32,64,128,256):
        h=2;e=p*h;s=h;k=2
        for route in ('balanced','hot','rotate','random'):
            m=2049 if route in ('hot','rotate') else 129
            if route=='balanced':ids=(np.arange(p*m*k).reshape(p,m,k)%e).astype(np.int32)
            elif route in ('hot','rotate'):
                ids=np.broadcast_to([0,1] if route=='hot' else [e-2,e-1],(p,m,k)).copy().astype(np.int32)
            else:
                first=rng.integers(e,size=(p,m));second=(first+rng.integers(1,e,size=(p,m)))%e
                ids=np.stack([first,second],axis=-1).astype(np.int32)
            counts=np.bincount(ids.ravel(),minlength=e)
            for penalty,gain in [(0,0),(2,4)]:
                q,cp=call(0,counts,p,h,s,768,penalty,gain,m*k)
                reference=native.make_plan(ids,e,s,768,penalty_blocks=penalty,min_gain_blocks=gain)
                assert np.array_equal(cp,reference.copies),(p,route,'native copies')
                blocks=(counts+767)//768;offset=0;expected=np.zeros((e,p),np.int64)
                for expert,n in enumerate(blocks):
                    expected[expert]=np.bincount(reference.block_exec[offset:offset+n]//(h+s),minlength=p);offset+=n
                assert np.array_equal(q,expected),(p,route,'native assignment')
                assert np.array_equal(q.sum(1),blocks)
                rows.append(dict(ep=p,route=route,planner='native',penalty=penalty,gain=gain,pass_=True))
            result=call(1,counts,p,h,s,1,0,0,m*k)
            try:reference=moon.make_moonep_plan(ids,e,s)
            except ValueError:
                assert result is None,(p,route,'expected replica-budget error')
                rows.append(dict(ep=p,route=route,planner='moonep',collective_error_expected=True,pass_=True));continue
            assert result is not None,(p,route,'unexpected policy error')
            q,cp=result
            assert np.array_equal(cp,reference.copies),(p,route,'moonep copies')
            allocation=moon.moonep_allocation(counts,p,m*k)
            assert np.array_equal(q,allocation.T),(p,route,'moonep assignment')
            assert np.array_equal(q.sum(1),counts) and np.all(q.sum(0)==m*k)
            rows.append(dict(ep=p,route=route,planner='moonep',pass_=True))
    # Physical table and overflow boundaries, including dimensions beyond the
    # old 256-expert/256-replica limits and the 2048 total-slot ceiling.
    for p,h,s,ok in [(256,2,2,True),(256,4,4,True),(256,8,0,True),(256,8,1,False),(16,24,8,True),(16,24,24,False),(12,2,2,False),(512,1,1,False)]:
        try:limits.validate_layout(p,h,s);actual=True
        except ValueError:actual=False
        assert actual==ok,(p,h,s)
    counts=np.full(2048,129,dtype=np.int64)
    q,cp=call(0,counts,256,8,0,768,2,4,1032)
    expected=np.zeros((2048,256),np.int64);expected[np.arange(2048),np.arange(2048)//8]=1
    assert np.array_equal(q,expected) and cp.shape==(256,0)
    # Declared slots (including empty ones) exceed the old 256-entry cache.
    ids=np.broadcast_to([0,1],(256,2049,2)).copy().astype(np.int32)
    counts=np.bincount(ids.ravel(),minlength=256)
    q,cp=call(0,counts,256,1,7,768,2,4,4098)
    reference=native.make_plan(ids,256,7,768,penalty_blocks=2,min_gain_blocks=4)
    assert cp.size==1792 and np.array_equal(cp,reference.copies)
    assert np.array_equal(q.sum(1),(counts+767)//768)
    assert call(0,[2147483000,0],2,1,1,1,1048576,1048576,1) is not None
    assert call(0,[2147483000,1000],2,1,1,768,2,4,1) is None
    assert call(0,[-1,1],2,1,1,768,2,4,1) is None
    result=dict(pass_=True,policy_cases=len(rows),ranks=[2,4,8,16,32,64,128,256],
        sanitizer='address,undefined,bounds',rows=rows,npu_multi_node_validated=False)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LARGE_EP_POLICY_CPU_PASS',len(rows),flush=True)


if __name__=='__main__':main()
