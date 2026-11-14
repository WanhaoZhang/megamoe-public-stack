"""Fixed-plan CPU oracle. No torch/NPU imports; online planner is out of scope."""
from dataclasses import dataclass
import hashlib
import numpy as np
from .runtime_limits import LOCAL_EXPERT_CAPACITY

@dataclass(frozen=True)
class ReplicaPlan:
    copies: np.ndarray
    route_blocks: np.ndarray
    block_exec: np.ndarray
    physical_ids: np.ndarray
    logical_counts: np.ndarray
    exec_counts: np.ndarray
    home: int
    slots: int
    beta: int
    max_recv: int
    fingerprint: str

def make_plan(ids, experts, slots=1, beta=768, rebalance=True, penalty_blocks=0, min_gain_blocks=0):
    ids=np.asarray(ids)
    if ids.ndim!=3:raise ValueError('expected [EP,M,TopK]')
    ranks,m,k=ids.shape
    if min(ranks,m,k)<=0 or experts<=0 or experts%ranks or beta<=0 or slots<0 or penalty_blocks<0 or min_gain_blocks<0:raise ValueError('invalid planner dimensions')
    if not np.issubdtype(ids.dtype,np.integer):raise ValueError('expert IDs must be integers')
    home=experts//ranks; es=home+slots
    if es>LOCAL_EXPERT_CAPACITY or es*ranks>2048 or ids.min()<0 or ids.max()>=experts or k>experts:raise ValueError('routing/capacity violation')
    ids=ids.astype(np.int32)
    counts=np.bincount(ids.ravel(),minlength=experts).astype(np.int64)
    blocks=(counts+beta-1)//beta
    q=np.zeros((experts,ranks),np.int64)
    q[np.arange(experts),np.arange(experts)//home]=blocks
    copies=np.full((ranks,slots),-1,np.int32)
    load=q.sum(0); target=int((blocks.sum()+ranks-1)//ranks)
    if rebalance:
        for _ in range(int(blocks.sum())+1):
            moved=False
            for donor in sorted(range(ranks),key=lambda r:(-load[r],r)):
                if load[donor]<=target:continue
                for e in sorted(range(experts),key=lambda e:(-q[e,donor],e)):
                    if not q[e,donor]:continue
                    owner=e//home
                    for receiver in sorted(range(ranks),key=lambda r:(load[r],r)):
                        if receiver==donor or load[receiver]>=target:continue
                        existing=np.flatnonzero(copies[receiver]==e)
                        free=np.flatnonzero(copies[receiver]<0)
                        if receiver!=owner and not len(existing) and not len(free):continue
                        take=min(int(q[e,donor]),target-int(load[receiver]),int(load[donor])-target)
                        if take<=0:continue
                        if receiver!=owner and not len(existing):copies[receiver,free[0]]=e
                        q[e,donor]-=take;q[e,receiver]+=take;load[donor]-=take;load[receiver]+=take;moved=True
                        if load[donor]<=target:break
                    if load[donor]<=target:break
            if not moved:break
    # Owner-side reduction penalty, in measured block-equivalent units. The
    # caller supplies a calibrated value; zero preserves the S3 reference.
    if penalty_blocks or min_gain_blocks:
        original=np.bincount(np.arange(experts)//home,weights=blocks,minlength=ranks).astype(np.int64)
        def augmented():
            remote=np.zeros(ranks,np.int64)
            for r in range(ranks):
                for e in copies[r]:
                    if e>=0 and q[e,r]>0:remote[e//home]+=1
            return load+penalty_blocks*remote
        for _ in range(ranks*slots):
            donor=int(np.argmax(augmented()))
            candidates=sorted((int(q[e,r]),int(e),r,s) for r in range(ranks)
                              for s,e in enumerate(copies[r]) if e>=0 and e//home==donor and q[e,r]>0)
            if not candidates or candidates[0][0]>=penalty_blocks:break
            v,e,r,s=candidates[0];q[e,r]=0;q[e,donor]+=v
            load[r]-=v;load[donor]+=v;copies[r,s]=-1
        if int(augmented().max())+min_gain_blocks>=int(original.max()):
            q[:]=0;q[np.arange(experts),np.arange(experts)//home]=blocks
            copies[:]=-1;load=original
    # Prune replicas left with no assigned blocks.
    for r in range(ranks):
        for s,e in enumerate(copies[r]):
            if e>=0 and q[e,r]==0:copies[r,s]=-1
    table=[]; offsets=np.zeros(experts+1,np.int64)
    for e in range(experts):
        for r in range(ranks):
            n=int(q[e,r])
            if not n:continue
            slot=e%home if r==e//home else home+int(np.flatnonzero(copies[r]==e)[0])
            table.extend([r*es+slot]*n)
        offsets[e+1]=len(table)
    table=np.asarray(table,np.int32)
    route_blocks=np.empty_like(ids); physical=np.empty_like(ids)
    flat=ids.ravel(); routeflat=route_blocks.ravel()
    # Stable ordering is source-rank, token, original TopK slot.
    order=np.argsort(flat,kind='stable'); sortedids=flat[order]
    starts=np.r_[0,np.flatnonzero(sortedids[1:]!=sortedids[:-1])+1,len(order)]
    for begin,end in zip(starts[:-1],starts[1:]):
        e=int(sortedids[begin]); routeflat[order[begin:end]]=offsets[e]+np.arange(end-begin)//beta
    physical[:]=table[route_blocks]
    exec_counts=np.bincount(physical.ravel(),minlength=ranks*es).reshape(ranks,es).astype(np.int32)
    logical_counts=counts.reshape(ranks,home).astype(np.int32)
    original_rank=logical_counts.sum(1)
    max_recv=max(int(original_rank.max()),int(exec_counts.sum(1).max()),1)
    max_recv=(max_recv+127)//128*128
    assert exec_counts.sum()==ids.size and np.array_equal(q.sum(1),blocks)
    slotmap=np.empty((ranks,es),np.int32)
    for r in range(ranks):slotmap[r,:home]=np.arange(r*home,(r+1)*home);slotmap[r,home:]=copies[r]
    assert np.array_equal(slotmap.ravel()[physical],ids)
    digest=hashlib.sha256(ids.tobytes()+copies.tobytes()+table.tobytes()).hexdigest()
    for array in (copies,route_blocks,table,physical,logical_counts,exec_counts):array.flags.writeable=False
    return ReplicaPlan(copies,route_blocks,table,physical,logical_counts,exec_counts,home,slots,beta,max_recv,digest)
