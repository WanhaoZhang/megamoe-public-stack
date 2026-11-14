"""CPU MoonEP allocation policy, pinned to Ascend dev_server_dedup@36be681.

Only placement is reused: dispatch, weight transfer and reduction remain MegaFlux.
Allocation is [destination, logical expert]; routes retain source/token/K order.
"""
import numpy as np
from .execution_plan import ExecutionPlanSpec

SOURCE_REVISION = '36be6812f78669e23f6b4f5a9a26af8fc7be67bc'


def moonep_allocation(counts, ep, target):
    """Exact row quotas; ties choose the lowest rank/expert, as upstream."""
    counts = np.asarray(counts)
    if (counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer)
            or isinstance(ep, bool) or not isinstance(ep, (int, np.integer))
            or not (2 <= ep <= 8 or ep in (16,32,64,128,256)) or not len(counts) or len(counts) % ep
            or not isinstance(target, (int, np.integer)) or isinstance(target, bool)
            or target <= 0 or np.any(counts < 0)
            or int(counts.sum(dtype=np.int64)) != ep * target):
        raise ValueError('invalid counts, EP or per-rank M*K target')
    counts = counts.astype(np.int64, copy=True)
    home = len(counts) // ep
    balance = counts.reshape(ep, home).sum(1) - target
    quota = np.zeros((ep, ep), dtype=np.int64)
    while balance.max() > 0:
        owner, dest = int(balance.argmax()), int(balance.argmin())
        take = -int(balance[dest])
        quota[owner, dest] = take
        balance[owner] -= take
        balance[dest] = 0
    allocation = np.zeros((ep, len(counts)), dtype=np.int64)
    allocation[np.arange(len(counts)) // home, np.arange(len(counts))] = counts
    for owner in range(ep):
        remaining = counts[owner*home:(owner+1)*home].copy()
        while quota[owner].max() > 0:
            dest = int(quota[owner].argmax())
            local = int(remaining.argmax())
            take = min(int(remaining[local]), int(quota[owner, dest]))
            if take <= 0:
                raise RuntimeError('MoonEP quota cannot be satisfied')
            expert = owner*home + local
            allocation[owner, expert] -= take
            allocation[dest, expert] += take
            remaining[local] -= take
            quota[owner, dest] -= take
    if not (np.array_equal(allocation.sum(0), counts)
            and np.all(allocation >= 0) and np.all(allocation.sum(1) == target)):
        raise RuntimeError('MoonEP allocation invariant failed')
    return allocation


def make_moonep_plan(logical_ids, experts, slots):
    """Return an immutable canonical plan; native policy requires slots=home."""
    ids = np.asarray(logical_ids)
    if (ids.ndim != 3 or not np.issubdtype(ids.dtype, np.integer)
            or any(v <= 0 for v in ids.shape)
            or isinstance(experts, bool) or not isinstance(experts, (int, np.integer))):
        raise ValueError('expected integer IDs [EP,M,K] and expert count')
    ep, m, k = ids.shape
    if (not (2 <= ep <= 8 or ep in (16,32,64,128,256)) or experts <= 0 or experts % ep or experts > 2048
            or experts//ep > 16 or m > 16384 or k > experts
            or isinstance(slots, bool) or not isinstance(slots, (int, np.integer))
            or slots != experts//ep or ep*slots > 2048 or 2*experts > 2048):
        raise ValueError('MoonEP requires supported dimensions and slots=home')
    if np.any((ids < 0) | (ids >= experts)):
        raise ValueError('logical expert out of range')
    ids = ids.astype(np.int32, copy=True)
    home = experts//ep
    counts = np.bincount(ids.ravel(), minlength=experts)
    allocation = moonep_allocation(counts, ep, m*k)
    copies = np.full((ep, slots), -1, dtype=np.int32)
    mapping = np.full((ep, experts), -1, dtype=np.int32)
    for rank in range(ep):
        mapping[rank, rank*home:(rank+1)*home] = rank*(home+slots)+np.arange(home)
        remote = sorted((int(e) for e in np.flatnonzero(allocation[rank]) if e//home != rank),
                        key=lambda e: (int(allocation[rank, e]), e), reverse=True)
        if len(remote) > slots:
            raise ValueError('MoonEP live replicas exceed reserved slots')
        for slot, expert in enumerate(remote):
            copies[rank, slot] = expert
            mapping[rank, expert] = rank*(home+slots)+home+slot
    order = np.argsort(ids.ravel(), kind='stable')
    boundaries = np.r_[0, np.cumsum(counts)]
    physical = np.empty(ids.size, dtype=np.int32)
    for expert in range(experts):
        rows = order[boundaries[expert]:boundaries[expert+1]]
        offset = 0
        for rank in range(ep):
            take = int(allocation[rank, expert])
            if take:
                physical[rows[offset:offset+take]] = mapping[rank, expert]
            offset += take
    return ExecutionPlanSpec(ids, physical.reshape(ids.shape), copies, home)
