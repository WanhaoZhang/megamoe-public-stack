from .runtime_limits import LOCAL_EXPERT_CAPACITY
"""CPU specification for bounded four-candidate placement; not the online planner.

MegaFlux v1 section 3.1 / Appendix A supplies the search and pruning rules.
Ascending IDs resolve unspecified ties; single-block refinement is bounded by
E*P moves by default. The Ascend beta/penalty remain explicit caller policy.
"""
from dataclasses import dataclass
import hashlib
import numpy as np
from .megaflux_plan import ReplicaPlan


@dataclass(frozen=True)
class SearchAssignment:
    blocks: np.ndarray
    assignment: np.ndarray
    copies: np.ndarray
    original_load: np.ndarray
    load: np.ndarray
    owner_replicas: np.ndarray
    candidates: tuple
    selected: int
    refinement_moves: int
    redundant_removed: int
    reduction_removed: int
    proxy: int
    reverted: bool
    policy: str
    beta: int
    penalty_blocks: int


def _copies(q, home):
    return [(e, r) for e in range(len(q)) for r in range(q.shape[1])
            if r != e // home and q[e, r] > 0]


def _owner_replicas(q, home):
    result = np.zeros(q.shape[1], np.int64)
    for e, _ in _copies(q, home):
        result[e // home] += 1
    return result


def _candidate(blocks, ranks, slots, priority, fanout):
    home = len(blocks) // ranks
    q = np.zeros((len(blocks), ranks), np.int64)
    q[np.arange(len(blocks)), np.arange(len(blocks)) // home] = blocks
    load = q.sum(0)
    target = (int(blocks.sum()) + ranks - 1) // ranks
    rounds = moves = 0
    for _ in range(slots + 2):
        rounds += 1
        moved = False
        if priority == 'overload':
            donors = sorted(range(ranks), key=lambda d: (-int(load[d]), d))
        else:
            donors = sorted(range(ranks), key=lambda d:
                            (-int(q[d*home:(d+1)*home, d].max()), d))
        for donor in donors:
            if load[donor] <= target:
                continue
            # Existing replicas are ordered by capacity at visit entry.
            existing = sorted(((e, r) for e, r in _copies(q, home)
                               if e // home == donor),
                              key=lambda er: (int(load[er[1]]), er[0], er[1]))
            for e, receiver in existing:
                take = min(int(q[e, donor]), max(0, target-int(load[receiver])))
                if take:
                    q[e, donor] -= take; q[e, receiver] += take
                    load[donor] -= take; load[receiver] += take
                    moved = True; moves += 1
                if load[donor] <= target:
                    break
            if load[donor] <= target:
                continue
            experts = sorted(range(donor*home, (donor+1)*home),
                             key=lambda e: (-int(q[e, donor]), e))
            for e in experts:
                opened = 0
                while q[e, donor] > 0 and load[donor] > target:
                    # Owned experts do not consume replica slots, including
                    # owned experts whose allocation is currently zero.
                    used = [sum(q[x, r] > 0 for x in range(len(blocks))
                                if x // home != r) for r in range(ranks)]
                    eligible = [r for r in range(ranks) if r != donor
                                and load[r] < target and q[e, r] == 0 and used[r] < slots]
                    if not eligible or (fanout == 'single' and opened):
                        break
                    receiver = min(eligible, key=lambda r: (int(load[r]), r))
                    take = min(int(q[e, donor]), target-int(load[receiver]))
                    q[e, donor] -= take; q[e, receiver] += take
                    load[donor] -= take; load[receiver] += take
                    moved = True; moves += 1; opened += 1
                if load[donor] <= target:
                    break
        if not moved:
            break
    return q, dict(priority=priority, fanout=fanout, rounds=rounds,
                   transfers=moves, max_load=int(load.max()), replicas=len(_copies(q, home)))


def search_blocks(counts, ranks, slots=1, beta=768, *, policy='training',
                  penalty_blocks=2, min_gain_blocks=0, refinement_limit=None):
    """Return deterministic E x P block counts and diagnostics, without routes.

    CPU-only, not placed in timed execution. penalty is a block proxy, not
    measured communication time. Four candidates are evaluated sequentially.
    """
    counts = np.asarray(counts)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError('counts must be a one-dimensional integer histogram')
    integers = (ranks, slots, beta, penalty_blocks, min_gain_blocks)
    if any(not isinstance(x, (int, np.integer)) or isinstance(x, (bool, np.bool_)) for x in integers):
        raise ValueError('integer policy dimensions required')
    if ranks < 1 or ranks > 8 or not len(counts) or len(counts) > 256 or len(counts) % ranks:
        raise ValueError('invalid expert/rank dimensions')
    home = len(counts) // ranks
    if slots < 0 or home+slots > LOCAL_EXPERT_CAPACITY or beta <= 0 or beta > np.iinfo(np.int32).max or not 0 <= penalty_blocks <= np.iinfo(np.int32).max or not 0 <= min_gain_blocks <= np.iinfo(np.int32).max:
        raise ValueError('invalid capacity or policy')
    if policy not in ('training', 'load_only'):
        raise ValueError('policy must be training or load_only')
    if counts.min() < 0 or counts.max() > np.iinfo(np.int32).max:
        raise ValueError('counts must fit nonnegative int32')
    counts = counts.astype(np.int64)
    if counts.sum() > np.iinfo(np.int32).max:
        raise ValueError('total routes must fit int32')
    if refinement_limit is None:
        refinement_limit = len(counts)*ranks
    if not isinstance(refinement_limit, (int, np.integer)) or isinstance(refinement_limit, (bool, np.bool_)) or refinement_limit < 0:
        raise ValueError('refinement_limit must be a nonnegative integer')
    blocks = (counts + beta - 1) // beta
    baseline = np.zeros((len(counts), ranks), np.int64)
    baseline[np.arange(len(counts)), np.arange(len(counts)) // home] = blocks
    original = baseline.sum(0)
    candidates = [_candidate(blocks, ranks, slots, p, f)
                  for p in ('overload', 'largest_home') for f in ('single', 'multiple')]
    selected = min(range(4), key=lambda i: (candidates[i][1]['max_load'],
                                           candidates[i][1]['replicas'], i))
    q = candidates[selected][0].copy(); load = q.sum(0); ceiling = int(load.max())
    redundant = 0
    for e, receiver in sorted(_copies(q, home), key=lambda er: (int(q[er]), *er)):
        value = int(q[e, receiver]); owner = e // home
        targets = [r for r in range(ranks) if r != receiver and
                   (r == owner or q[e, r] > 0) and load[r]+value <= ceiling]
        if not value or not targets:
            continue
        target = min(targets, key=lambda r: (r != owner, int(load[r]), r))
        q[e, receiver] = 0; q[e, target] += value
        load[receiver] -= value; load[target] += value; redundant += 1
    refined = 0
    for _ in range(refinement_limit):
        donor = int(np.argmax(load)); receiver = int(np.argmin(load))
        if load[donor]-load[receiver] <= 1:
            break
        shared = [e for e in range(len(q)) if q[e, donor] > 0 and
                  (receiver == e // home or q[e, receiver] > 0)]
        if not shared:
            break
        e = shared[0]
        q[e, donor] -= 1; q[e, receiver] += 1
        load[donor] -= 1; load[receiver] += 1; refined += 1
    assert int(load.max()) <= ceiling
    removed = 0
    if policy == 'training':
        for _ in range(ranks*slots):
            augmented = load + penalty_blocks*_owner_replicas(q, home)
            owner = int(np.argmax(augmented))
            remote = sorted((int(q[e, r]), e, r) for e, r in _copies(q, home) if e // home == owner)
            if not remote or remote[0][0] >= penalty_blocks:
                break
            value, e, receiver = remote[0]
            q[e, receiver] = 0; q[e, owner] += value
            load[receiver] -= value; load[owner] += value; removed += 1
            assert int((load+penalty_blocks*_owner_replicas(q, home)).max()) <= int(augmented.max())
    effective_penalty = penalty_blocks if policy == 'training' else 0
    proxy = int((load+effective_penalty*_owner_replicas(q, home)).max())
    reverted = (policy == 'training' or min_gain_blocks > 0) and proxy+min_gain_blocks >= int(original.max())
    if reverted:
        q = baseline.copy(); load = original.copy(); proxy = int(original.max())
    copies = np.full((ranks, slots), -1, np.int32)
    for rank in range(ranks):
        remote = [e for e in range(len(q)) if e // home != rank and q[e, rank] > 0]
        assert len(remote) <= slots
        copies[rank, :len(remote)] = remote
    owners = _owner_replicas(q, home)
    assert np.array_equal(q.sum(1), blocks) and (q >= 0).all()
    for array in (blocks, q, copies, original, load, owners):
        array.flags.writeable = False
    return SearchAssignment(blocks, q, copies, original, load, owners,
                            tuple(c[1] for c in candidates), selected, refined, redundant,
                            removed, proxy, bool(reverted), policy, beta, penalty_blocks)


def make_search_plan(ids, experts, slots=1, beta=768, **policy):
    """Materialize an offline ReplicaPlan plus its search diagnostics.

    Stable source-rank/token/TopK order, ascending destination ranks; compatible
    with DeviceReplicaPlan and the staged executor, not FullTrain online ABI.
    """
    ids = np.asarray(ids)
    if ids.ndim != 3 or not np.issubdtype(ids.dtype, np.integer) or min(ids.shape) <= 0:
        raise ValueError('expected nonempty integer [EP,M,TopK]')
    if not isinstance(experts, (int, np.integer)) or isinstance(experts, (bool, np.bool_)) or not 0 < experts <= 256 or ids.shape[2] > experts:
        raise ValueError('invalid experts/TopK')
    if ids.min() < 0 or ids.max() >= experts:
        raise ValueError('invalid logical expert IDs')
    ids = ids.astype(np.int32)
    ranks = ids.shape[0]
    counts = np.bincount(ids.ravel(), minlength=experts).astype(np.int64)
    result = search_blocks(counts, ranks, slots, beta, **policy)
    home = experts // ranks; es = home+slots
    table = []; offsets = np.zeros(experts+1, np.int64)
    for e in range(experts):
        for rank in range(ranks):
            n = int(result.assignment[e, rank])
            if not n:
                continue
            slot = e % home if rank == e // home else home+int(np.flatnonzero(result.copies[rank] == e)[0])
            table.extend([rank*es+slot]*n)
        offsets[e+1] = len(table)
    table = np.asarray(table, np.int32); route_blocks = np.empty_like(ids)
    flat = ids.ravel(); dest = route_blocks.ravel(); order = np.argsort(flat, kind='stable')
    sorted_ids = flat[order]
    starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1])+1, len(order)]
    for begin, end in zip(starts[:-1], starts[1:]):
        e = int(sorted_ids[begin]); dest[order[begin:end]] = offsets[e]+np.arange(end-begin)//beta
    physical = table[route_blocks]
    execution = np.bincount(physical.ravel(), minlength=ranks*es).reshape(ranks, es).astype(np.int32)
    logical = counts.reshape(ranks, home).astype(np.int32)
    maximum = max(int(logical.sum(1).max()), int(execution.sum(1).max()), 1)
    maximum = (maximum+127)//128*128
    slotmap = np.concatenate((np.arange(experts).reshape(ranks, home), result.copies), axis=1)
    assert np.array_equal(slotmap.ravel()[physical], ids)
    assert execution.sum() == ids.size
    fingerprint = hashlib.sha256(ids.tobytes()+result.copies.tobytes()+table.tobytes()).hexdigest()
    for array in (table, route_blocks, physical, execution, logical):
        array.flags.writeable = False
    plan = ReplicaPlan(result.copies, route_blocks, table, physical, logical, execution,
                       home, slots, beta, maximum, fingerprint)
    return plan, result
