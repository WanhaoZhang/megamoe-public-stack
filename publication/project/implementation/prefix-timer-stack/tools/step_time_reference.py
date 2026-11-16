"""Independent CPU event-DAG reference. No executor, device or model parameters.

The portable C++ oracle supplies proposals; this explicit per-task simulation
checks its cost arithmetic and the public NumPy compiler checks semantics.
Each graph uses a single logical origin, never independent device timestamps.
"""
import ctypes
import importlib.util
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PHASES=('gmm1','activation','gmm2','dh','swiglu_grad','dw2','dx','dw1')

def source_for(base,step,layer,layers=2):
    counts=np.stack([np.bincount(ids.ravel(),minlength=256) for ids in base]).astype(np.int32)
    permutation=np.random.default_rng(42+100003*(step*layers+layer)).permutation(256)
    result=np.empty_like(counts);result[:,permutation]=counts
    return result

def contract():
    name='_independent_planner_contract'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'source/native-balanced/ascend_deepep/planner_contract.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name]

def library():
    target=ROOT/'analysis/libstep_time_reference.so'
    sources=[ROOT/'tools/step_time_oracle.cpp',ROOT/'source/native-balanced/include/mega_moe_step_time_policy.h',ROOT/'source/native-balanced/include/mega_moe_unified_policy.h']
    if not target.exists() or target.stat().st_mtime<max(p.stat().st_mtime for p in sources):
        target.parent.mkdir(exist_ok=True)
        subprocess.run(['g++','-std=c++17','-O2','-shared','-fPIC','-I'+str(sources[1].parent),str(sources[0]),'-o',str(target)],check=True)
    lib=ctypes.CDLL(str(target))
    return lib

def ptr(value):return value.ctypes.data_as(ctypes.c_void_p)

def propose(source,vector):
    source=np.ascontiguousarray(source,dtype=np.int32);v=np.ascontiguousarray(vector,dtype=np.int32)
    cp=np.empty((8,32),np.int32);flow=np.empty((8,256,8),np.int32);stats=np.empty(8,np.float32)
    if library().search_reference(*map(ptr,(source,v,cp,flow,stats))):raise ValueError('portable policy rejected source counts')
    return cp,flow,stats

def compiled_score(source,vector,cp,flow):
    arrays=[np.ascontiguousarray(x,dtype=np.int32) for x in (source,vector,cp,flow)]
    answer=ctypes.c_int64()
    if library().score_reference(*map(ptr,arrays),ctypes.byref(answer)):raise ValueError('score failed')
    return answer.value

def compiled_phases(source,vector,cp,flow):
    arrays=[np.ascontiguousarray(x,dtype=np.int32) for x in (source,vector,cp,flow)]
    answer=np.empty(3,dtype=np.int64)
    if library().phase_reference(*map(ptr,arrays),ptr(answer)):raise ValueError('phase prediction failed')
    return answer*16/1e6

class DAG:
    def __init__(self):self.nodes=[]
    def task(self,label,duration,*ready):
        start=max(ready,default=0);end=start+duration
        self.nodes.append((label,int(start),int(end)));return end

def simulate(vector,cp,flow,trace=False):
    """Explicit task-by-task max-plus simulation, including every owner tile."""
    v=list(map(int,vector));q=np.asarray(flow,dtype=np.int64).sum(axis=0)
    local=np.array([[flow[d,e,d] for d in range(8)] for e in range(256)],dtype=np.int64)
    physical=np.concatenate((np.arange(256).reshape(8,32),cp),axis=1)
    replica=(q>0).sum(axis=1)-(q[np.arange(256),np.arange(256)//32]>0)
    activity=(q>0).sum(axis=0);copy_count=(np.asarray(cp)>=0).sum(axis=1)
    allcopies=int(copy_count.sum());g=DAG()
    def duration(p,rows,d):
        if rows==0:return 0
        b=v[208+p*5:213+p*5];tm=192 if p<2 else 128;cols=16 if p==0 else 28 if p in (2,6) else 8
        if v[16]==2:
            rate=b[1]*16000//tm+b[2]*16000*(1 if p in (5,7) else cols)//(tm*(2 if p in (5,7) else 32))
            return b[0]+rows*rate//16000+b[4]*int(activity[d])//32
        tiles=(rows+tm-1)//tm;waves=(tiles+1)//2 if p in (5,7) else (tiles*cols+31)//32
        return b[0]+b[1]*tiles+b[2]*waves+b[3]*((rows-1)%tm+1)//tm+b[4]*int(activity[d])//32
    def routing(p,e,d):
        b=v[256+3*p:259+3*p];n=int(q[e,d]);l=int(local[e,d])
        return b[0]+(l*b[1]+(n-l)*b[2])//16000 if n else 0
    rows=np.asarray(flow,dtype=np.int64).sum(axis=1);copies=np.zeros((8,8),np.int64)
    for e,d in zip(*np.where(q>0)):
        if d!=e//32:copies[e//32,d]+=1
    full=6*v[7]*v[8]
    def net(tb,wb,gb):
        cost=np.zeros((8,8),np.int64)
        for s in range(8):
            for d in range(8):
                if s==d:continue
                size=int(rows[s,d])*tb+int(copies[s,d])*wb+int(copies[d,s])*gb
                if size:cost[s,d]=v[64+s*8+d]+size*v[128+s*8+d]//16000000
        result=0
        for axis in (0,1):
            for rank in range(8):
                times=cost[:,rank] if axis==0 else cost[rank,:];n=int((times>0).sum())
                if n:
                    factor=1000000+(v[192+rank]-1000000)*(n-1)//6
                    result=max(result,int(times.max()),((int(times.sum())+n-1)//n)*factor//1000000)
        return result
    copy=net(0,full,0)*v[25]//1000000;inputs=net(2*v[7]+8,0,0)*v[24]//1000000
    fw_net=net(4*v[7]+12,full,0)*v[24]//1000000;bw_net=net(6*v[7]+16,full,full)*v[24]//1000000
    fwd=[];p1=[];p2=[];tail=[];release={}
    for d in range(8):
        schedule=[(k,int(e),int(q[e,d])) for k,e in enumerate(physical[d]) if e>=0 and q[e,d]>0]
        data={};dispatch=0
        for k,e,n in schedule:dispatch=g.task(f'd{d}.dispatch.e{e}',routing(0,e,d),dispatch);data[k]=dispatch
        total=max(dispatch,inputs);aic=act=0;activations={};ordinal=0
        for k,e,n in schedule:
            ready=v[20]+data[k]
            if k>=32:
                first=copy*(ordinal*1000000+v[40])//(max(int(copy_count[d]),1)*1000000)
                ready=max(ready,total+first*2//3);ordinal+=1
            aic=g.task(f'd{d}.g1.e{e}',duration(0,n,d),aic,ready)
            act=g.task(f'd{d}.silu.e{e}',duration(1,n,d),act,aic);activations[k]=act
        combine=0;ordinal=0
        for k,e,n in schedule:
            ready=activations[k]
            if k>=32:ready=max(ready,copy*(ordinal*1000000+v[41])//(max(int(copy_count[d]),1)*1000000));ordinal+=1
            aic=g.task(f'd{d}.g2.e{e}',duration(2,n,d),aic,ready)
            combine=g.task(f'd{d}.combine.e{e}',routing(1,e,d),combine,aic)
        fwd.append(max(aic,combine,fw_net)+v[21])
        by_expert={e:(k,e,n) for k,e,n in schedule}
        seq=[by_expert[e] for e in sorted(by_expert) if replica[e]>0]
        seq += [by_expert[e] for e in range(d*32,(d+1)*32) if e in by_expert and replica[e]==0]
        dh=sg=data_time=0
        for j,(k,e,n) in enumerate(seq):
            data_time+=routing(2,e,d);ready=v[22]+data_time
            if k>=32:ready=max(ready,copy*(j*1000000+v[41])//(max(len(seq),1)*1000000)//3)
            dh=g.task(f'd{d}.dH.e{e}',duration(3,n,d),dh,ready);sg=g.task(f'd{d}.dSilu.e{e}',duration(4,n,d),sg,dh)
        p1.append(max(dh,sg));aic=p1[-1];returned=aic
        for k,e,n in seq:
            length=duration(5,n,d);begin=aic;aic=g.task(f'd{d}.dW2.e{e}',length,aic)
            release[d,e,1]=(begin+length*v[39]//1000000,aic)
            if k>=32:aic=max(aic,copy)
            aic=g.task(f'd{d}.dX.e{e}',duration(6,n,d),aic)
            returned=g.task(f'd{d}.returnX.e{e}',routing(3,e,d),returned,aic)
        p2.append(max(aic,returned)+v[27]*((64 if allcopies else 32)-int(activity[d])))
        for k,e,n in seq:
            length=duration(7,n,d);begin=aic;aic=g.task(f'd{d}.dW1.e{e}',length,aic)
            release[d,e,0]=(begin+length*v[38]//1000000,aic)
        tail.append(max(aic,p2[-1]))
    owner_end=[]
    for owner in range(8):
        finish=tail[owner]
        if v[16]!=1:
            for parity in (0,1):
                now=p1[owner] if parity else p2[owner]
                for projection in (1,0):
                    tiles=v[36+projection];residue=min(63 if parity else 62,tiles-1)
                    if residue%2!=parity:residue-=1
                    if residue<0:continue
                    tasks=(tiles-1-residue)//64+1
                    for e in range(owner*32,(owner+1)*32):
                        if not replica[e]:continue
                        first=last=0
                        for d in np.flatnonzero(q[e]):
                            begin,end=release[int(d),e,projection];span=end-begin
                            first=max(first,begin+span*residue//max(tiles-1,1))
                            last=max(last,begin+span*(residue+64*(tasks-1))//max(tiles-1,1))
                        b=v[248+projection*3:251+projection*3]
                        service=(b[0]+b[1]*int(replica[e])+(b[2] if q[e,owner] else 0))*v[26]//1000000
                        stride=(last-first+tasks-2)//(tasks-1) if tasks>1 else 0
                        for t in range(tasks):now=g.task(f'owner{owner}.aiv{parity}.w{projection}.e{e}.tile{t}',service,now,first+t*stride)
                finish=max(finish,now)
        owner_end.append(finish)
    forward=max(fwd);backward=max(max(tail),max(owner_end),bw_net)+v[23]
    if v[16]==1:backward=max(tail)+net(0,0,full)*v[24]//1000000+v[23]
    now=ag=rs=0
    for layer in range(v[55]):
        now=g.task(f'L{layer}.attentionF',v[44]+v[52],now,ag);ag=now+v[48]
        now=g.task(f'L{layer}.moeF',v[42]+v[29]+forward+v[46],now)
    now+=v[53]
    for layer in reversed(range(v[55])):
        now=g.task(f'L{layer}.replay_and_backward',v[44]+v[42]+forward+v[46]+v[47]+v[42]+backward+v[45]+v[54],now,ag)
        rs=max(rs,now)+v[49];ag=now+v[48]
    result=max(now,rs)+v[50]+v[51]+v[28]
    return (result,g.nodes) if trace else result
