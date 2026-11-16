"""Pure CPU semantic oracle for the fixed-owner planner ABI (no runtime imports).

The online implementation is in mega_moe_unified_planner.cpp. This module is
deliberately independent and is used only by tests/offline diagnostics.
"""
from dataclasses import dataclass
import hashlib
import numpy as np


def _readonly(value):
    result = np.array(value, dtype=np.int32, order='C', copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PlannerResources:
    home: int
    replica_slots: int
    capacity: int
    local_limit: int = 64


@dataclass(frozen=True)
class PlannerProposal:
    copies: np.ndarray
    quota: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, 'copies', _readonly(self.copies))
        object.__setattr__(self, 'quota', _readonly(self.quota))


@dataclass(frozen=True)
class CompiledPlanBody:
    """No epoch, stream, model Parameter or pool storage belongs in this body."""
    logical_ids: np.ndarray
    physical_ids: np.ndarray
    copies: np.ndarray
    quota: np.ndarray
    logical_counts: np.ndarray
    execution_counts: np.ndarray
    contributors: tuple
    digest: str


def compile_plan(logical_ids, proposal, resources):
    ids = np.asarray(logical_ids)
    cp, quota = proposal.copies, proposal.quota
    if ids.ndim != 3 or ids.dtype != np.int32:
        raise ValueError('logical ids must be int32[EP,M,K]')
    p, m, k = ids.shape
    h, slots = resources.home, resources.replica_slots
    e = p * h
    if h <= 0 or slots < 0 or h+slots>resources.local_limit or resources.capacity <= 0 or resources.capacity % 128:
        raise ValueError('invalid resources')
    if cp.shape != (p, slots) or quota.shape != (p, e, p):
        raise ValueError('invalid proposal shape')
    if np.any(ids < 0) or np.any(ids >= e) or np.any(quota < 0):
        raise ValueError('invalid expert or negative quota')
    # TopK can contain repeated expert IDs: each assignment remains distinct.
    source = np.stack([np.bincount(x.ravel(), minlength=e) for x in ids])
    if not np.array_equal(quota.sum(axis=2, dtype=np.int64), source):
        raise ValueError('source quota violates conservation')
    for d in range(p):
        copies = cp[d][cp[d] >= 0]
        if np.any(cp[d] < -1) or np.any(copies >= e) or len(set(copies.tolist())) != len(copies):
            raise ValueError('invalid or duplicate replica')
        if np.any(copies // h == d):
            raise ValueError('replica duplicates fixed home')
        for expert in range(e):
            rows = int(quota[:, expert, d].sum())
            if expert // h != d and ((rows > 0) != (expert in copies)):
                raise ValueError('nonzero quota and replica/contributor set disagree')
    receive = quota.sum(axis=(0, 1), dtype=np.int64)
    if np.any((receive + 127) // 128 * 128 > resources.capacity):
        raise ValueError('receive capacity exceeded')
    physical = np.empty_like(ids)
    execution = np.zeros((p, h + slots), dtype=np.int32)
    for s in range(p):
        ordinal = np.zeros(e, dtype=np.int64)
        ends = quota[s].cumsum(axis=1, dtype=np.int64)
        for pos, expert in enumerate(ids[s].ravel()):
            d = int(np.searchsorted(ends[expert], ordinal[expert], side='right'))
            ordinal[expert] += 1
            slot = int(expert % h) if expert // h == d else h + int(np.flatnonzero(cp[d] == expert)[0])
            physical[s].ravel()[pos] = d * (h + slots) + slot
            execution[d, slot] += 1
    contributors = tuple(tuple(d for d in range(p) if quota[:, expert, d].sum() > 0)
                         for expert in range(e))
    arrays = tuple(_readonly(x) for x in (ids, physical, cp, quota, source.sum(axis=0), execution))
    digest = hashlib.sha256(b''.join(x.tobytes() for x in arrays)).hexdigest()
    return CompiledPlanBody(*arrays, contributors, digest)


def interval_quota(source, destination):
    """Stable source-major/global-expert ordinal used by legacy MoonEP/Native."""
    p, e = source.shape
    result = np.zeros((p, e, p), np.int32)
    for expert in range(e):
        source_ends = source[:, expert].cumsum()
        dest_ends = destination[expert].cumsum()
        result[:, expert, :] = np.maximum(0, np.minimum(source_ends[:, None], dest_ends[None, :]) -
            np.maximum((source_ends-source[:, expert])[:, None], (dest_ends-destination[expert])[None, :]))
    return result


def local_quota(source, destination, order=None):
    """Local fill + integer proportional remainder, preserving both marginals.

    UltraEP's source-local largest remainder is followed by deterministic
    column repair: its independently rounded source rows can exceed a column.
    The repair does not change expert placement or global execution quota.
    """
    p, e = source.shape
    result = np.zeros((p, e, p), np.int32)
    for expert in range(e):
        available = source[:, expert].astype(np.int64).copy()
        need = destination[expert].astype(np.int64).copy()
        for d in range(p):
            v = min(available[d], need[d])
            result[d, expert, d] = v
            available[d] -= v
            need[d] -= v
        denom = int(need.sum())
        if not denom:
            continue
        # A joint floor is below every column and every source row. Complete
        # the transportation residual by stable highest fractional remainder.
        product = available[:, None] * need[None, :]
        floor = product // denom
        result[:, expert, :] += floor.astype(np.int32)
        left = available - floor.sum(axis=1)
        room = need - floor.sum(axis=0)
        fractions = product % denom
        while left.sum():
            choices = [(int(fractions[s, d]), -d, -s, s, d)
                       for s in range(p) for d in range(p) if left[s] and room[d]]
            _, _, _, s, d = max(choices)
            result[s, expert, d] += 1
            left[s] -= 1
            room[d] -= 1
            fractions[s, d] = -1
    return result


def _copies(destination, home, slots, order=None):
    e, p = destination.shape
    cp = np.full((p, slots), -1, np.int32)
    for d in range(p):
        active = [x for x in range(e) if x // home != d and destination[x, d] > 0]
        if order == 'moonep':
            active.sort(key=lambda x: (-int(destination[x, d]), -x))
        if len(active) > slots:
            raise ValueError('policy exhausted replica slots')
        cp[d, :len(active)] = active
    return cp


def moonep(source, home, slots):
    p, e = source.shape
    counts = source.sum(axis=0)
    total = int(counts.sum())
    if slots != home or total % p:
        raise ValueError('MoonEP requires slots=home and integer average')
    target = total // p
    dest = np.zeros((e, p), np.int32)
    dest[np.arange(e), np.arange(e)//home] = counts
    balance = counts.reshape(p, home).sum(axis=1) - target
    transfers = np.zeros((p, p), np.int64)
    for _ in range(p*p):
        donor, recv = int(balance.argmax()), int(balance.argmin())
        if balance[donor] <= 0 or balance[recv] >= 0:
            break
        take = -int(balance[recv])
        transfers[donor, recv] = take
        balance[donor] -= take
        balance[recv] = 0
    for owner in range(p):
        available = counts[owner*home:(owner+1)*home].copy()
        for _ in range(p+home):
            d, j = int(transfers[owner].argmax()), int(available.argmax())
            if transfers[owner, d] <= 0:
                break
            take = min(int(available[j]), int(transfers[owner, d]))
            if not take:
                raise ValueError('infeasible MoonEP transfer')
            expert = owner*home+j
            dest[expert, owner] -= take
            dest[expert, d] += take
            available[j] -= take
            transfers[owner, d] -= take
    return PlannerProposal(_copies(dest, home, slots, 'moonep'), interval_quota(source, dest))


def echo(source, home, slots):
    p, e = source.shape
    counts = source.sum(axis=0)
    average = int(counts.sum())//p
    spill = np.zeros(e, np.int64)
    capacity = np.maximum(average-counts.reshape(p, home).sum(axis=1), 0)
    for d in range(p):
        remaining = average
        for expert in sorted(range(d*home, (d+1)*home), key=lambda x:(int(counts[x]), x)):
            keep = min(remaining, int(counts[expert]))
            remaining -= keep
            spill[expert] = int(counts[expert])-keep
    es = sorted(range(e), key=lambda x:(-int(spill[x]), x))
    ds = sorted(range(p), key=lambda d:(-int(capacity[d]), d))
    ends_e, ends_d = np.cumsum(spill[es]), np.cumsum(capacity[ds])
    overlap = np.maximum(0, np.minimum(ends_e[:, None], ends_d[None, :]) -
        np.maximum((ends_e-spill[es])[:, None], (ends_d-capacity[ds])[None, :]))
    dest = np.zeros((e, p), np.int32)
    dest[np.arange(e), np.arange(e)//home] = counts
    moves = np.zeros((e,p),np.int64)
    for col, d in enumerate(ds):
        chosen = sorted(range(e), key=lambda j:(-int(overlap[j, col]), j))[:slots]
        for j in chosen:
            expert, move = es[j], int(overlap[j, col])
            if not move:
                continue
            moves[expert,d] = move
            dest[expert, expert//home] -= move
            dest[expert, d] += move
    q = np.zeros((p,e,p),np.int32)
    for expert in range(e):
        # All breadth-first floors are reserved before depth-first consumes
        # the residual, exactly as in the frozen ECHO two-pass implementation.
        if not counts[expert]: continue
        first = source[:,expert,None].astype(np.int64)*moves[expert,None,:]//counts[expert]
        available = source[:,expert]-first.sum(axis=1)
        room = moves[expert]-first.sum(axis=0)
        q[:,expert,:] = first
        for d in range(p):
            for s in range(p):
                take = min(int(available[s]),int(room[d]))
                q[s,expert,d] += take
                available[s] -= take
                room[d] -= take
        q[:,expert,expert//home] += available.astype(np.int32)
    return PlannerProposal(_copies(dest, home, slots), q)


def native(source, home, slots, beta=768, penalty=2, gain=4):
    p,e = source.shape
    counts=source.sum(axis=0)
    blocks=(counts+beta-1)//beta
    q=np.zeros((e,p),np.int64)
    q[np.arange(e),np.arange(e)//home]=blocks
    load=blocks.reshape(p,home).sum(axis=1)
    original=load.copy()
    cp=np.full((p,slots),-1,np.int32)
    target=(int(blocks.sum())+p-1)//p
    for _ in range(int(blocks.sum())+1):
        moved=False
        for d in sorted(range(p),key=lambda r:(-int(load[r]),r)):
            if load[d]<=target: continue
            for expert in sorted(range(e),key=lambda x:(-int(q[x,d]),x)):
                if not q[expert,d]: continue
                for r in sorted(range(p),key=lambda r:(int(load[r]),r)):
                    if r==d or load[r]>=target: continue
                    existing=np.flatnonzero(cp[r]==expert)
                    free=np.flatnonzero(cp[r]<0)
                    if r!=expert//home and not len(existing) and not len(free): continue
                    take=min(int(q[expert,d]),target-int(load[r]),int(load[d])-target)
                    if take<=0: continue
                    if r!=expert//home and not len(existing): cp[r,free[0]]=expert
                    q[expert,d]-=take; q[expert,r]+=take
                    load[d]-=take; load[r]+=take; moved=True
                    if load[d]<=target: break
                if load[d]<=target: break
        if not moved: break
    if penalty or gain:
        for _ in range(p*slots):
            aug=load.copy()
            for r in range(p):
                for expert in cp[r]:
                    if expert>=0 and q[expert,r]>0: aug[expert//home]+=penalty
            d=int(aug.argmax())
            choices=[(int(q[x,r]),int(x),r,s) for r in range(p) for s,x in enumerate(cp[r])
                     if x>=0 and x//home==d and q[x,r]>0]
            if not choices: break
            value,expert,r,s=min(choices)
            if value>=penalty: break
            q[expert,r]=0; q[expert,d]+=value
            load[r]-=value; load[d]+=value; cp[r,s]=-1
        aug=load.copy()
        for r in range(p):
            for expert in cp[r]:
                if expert>=0 and q[expert,r]>0: aug[expert//home]+=penalty
        if int(aug.max())+gain>=int(original.max()):
            q.fill(0); q[np.arange(e),np.arange(e)//home]=blocks; cp.fill(-1)
    # Match legacy saturation of the last partial block in rank order.
    dest=np.zeros((e,p),np.int32)
    for expert in range(e):
        left=int(counts[expert])
        for d in range(p):
            take=min(left,int(q[expert,d])*beta); dest[expert,d]=take; left-=take
    for r in range(p):
        for s,x in enumerate(cp[r]):
            if x>=0 and dest[x,r]==0: cp[r,s]=-1
    return PlannerProposal(cp,interval_quota(source,dest))


def eplb_fixed(source, home, slots):
    p, e = source.shape
    counts = source.sum(axis=0).astype(np.float32)
    multiplicity = np.ones(e, np.int32)
    replicas = []
    for _ in range(p*slots):
        expert = int(np.argmax((counts/multiplicity).astype(np.float32)))
        replicas.append(expert)
        multiplicity[expert] += 1
    instance = (counts/multiplicity).astype(np.float32)
    # Match the scalar AscendC reduction order; GPU reduction association can
    # otherwise move a tie by one float32 ULP in this greedy placement.
    loads = np.zeros(p,np.float32)
    for d in range(p):
        for x in range(d*home,(d+1)*home):loads[d]=np.float32(loads[d]+instance[x])
    used = np.zeros(p, np.int32)
    placement = [[] for _ in range(e)]
    for expert in range(e):
        placement[expert].append(expert//home)
    for index in sorted(range(len(replicas)), key=lambda i:(-float(instance[replicas[i]]), i)):
        expert = replicas[index]
        d = min((d for d in range(p) if used[d] < slots), key=lambda d:(float(loads[d]), d))
        placement[expert].append(d)
        used[d] += 1
        loads[d] += instance[expert]
    dest = np.zeros((e, p), np.int32)
    quota=np.zeros((p,e,p),np.int32)
    for expert in range(e):
        # Frozen EPLB uses global ordinal % replica_count, with physical
        # instances ordered by rank then slot. Coalesce only after counting.
        places=sorted(placement[expert]);count=len(places)
        lo=0
        for s in range(p):
            hi=lo+int(source[s,expert])
            for j,d in enumerate(places):
                quota[s,expert,d]+=hi//count+int(hi%count>j)-lo//count-int(lo%count>j)
            lo=hi
        dest[expert]=quota[:,expert,:].sum(axis=0)
    return PlannerProposal(_copies(dest, home, slots),quota)


def ultra_ep(source, home, slots, min_tokens=1024, allow_zero=False, oracle_eps=0.01):
    """Scalar port of UltraEP stage1 quota oracle, including fast retries."""
    p, e = source.shape
    counts = source.sum(axis=0).astype(np.int64)
    loads = counts.reshape(p, home).sum(axis=1)
    source_order = sorted(range(p), key=lambda r:(-int(loads[r]), r))
    expert_order = [sorted(range(r*home,(r+1)*home), key=lambda x:(-int(counts[x]),x)) for r in range(p)]
    def oracle(threshold, dynamic):
        dest = np.zeros((e, p), np.int32)
        dest[np.arange(e), np.arange(e)//home] = counts
        slack = np.maximum(threshold-loads, 0)
        used = np.zeros(p, np.int32)
        for owner in source_order:
            need = max(int(loads[owner])-threshold,0)
            for expert in expert_order[owner]:
                available = int(counts[expert])-int(not allow_zero and counts[expert]>0)
                while need and available:
                    candidates = [d for d in range(p) if d != owner and slack[d]>0 and used[d]<slots and dest[expert,d]==0]
                    if not candidates:
                        break
                    floor = max(1,(need+len(candidates)-1)//len(candidates)) if dynamic else min_tokens
                    candidates = [d for d in candidates if min(need,int(slack[d]),available)>=floor]
                    if not candidates:
                        break
                    d = min(candidates, key=lambda d:(-int(slack[d]),d))
                    take = min(need,int(slack[d]),available)
                    dest[expert,owner] -= take
                    dest[expert,d] += take
                    need -= take
                    available -= take
                    slack[d] -= take
                    used[d] += 1
                if not need:
                    break
            if need:
                return None
        return dest
    low, high = (int(counts.sum())+p-1)//p, max(int(loads.max()), (int(counts.sum())+p-1)//p)
    fast = max(low,int(np.ceil(np.float32(low)*np.float32(1+oracle_eps))))
    dest = None
    for _ in range(3):
        dest = oracle(fast, True)
        if dest is not None:
            break
        fast = min(high,fast+max(1,(high+99)//100))
    if dest is None:
        dest = oracle(low, False)
        if dest is None:
            low = min(low+1, high)
            while low < high:
                span = high-low
                probes = [low+w for w in range(4) if low+w<high] if span<=4 else [low+span*(w+1)//5 for w in range(4)]
                feasible = [(t,oracle(t,False)) for t in probes]
                first = next(((i,t,x) for i,(t,x) in enumerate(feasible) if x is not None),None)
                if first is not None:
                    i,t,x = first
                    if span<=4:
                        low=high=t
                    else:
                        high=t
                        if i: low=probes[i-1]+1
                else:
                    low=high if span<=4 else probes[-1]+1
            dest=oracle(low,False)
    if dest is None:
        raise ValueError('UltraEP quota oracle infeasible')
    return PlannerProposal(_copies(dest,home,slots),local_quota(source,dest))
