from .runtime_limits import LOCAL_EXPERT_CAPACITY
"""CPU diagnostics for the current 128x256x256 BF16 GM WGrad schedule.

These are work/traffic counts, not measured latency or a proven runtime bound.
Do not use them to enable a planner without calibration and device validation.
"""
import numpy as np


def training_plan_features(logical_counts, exec_counts, copies, hidden, ffn):
    """Validate a physical plan and count per-rank WGrad work and owner traffic.

    Home experts precede replica slots. Each nonempty physical expert writes
    a complete dW1/dW2 plane, even if it executes only a few token rows. An
    empty owner still sums every remote contribution to its logical expert.
    Canonical BF16 shapes are dW1=[2F,H], dW2=[H,F], with token rows as K.
    """
    for value in (hidden, ffn):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError('positive integer hidden/ffn required')
    arrays = [np.asarray(v) for v in (logical_counts, exec_counts, copies)]
    if any(a.ndim != 2 or not np.issubdtype(a.dtype, np.integer) for a in arrays):
        raise ValueError('integer rank-by-expert/slot matrices required')
    if any(a.size and (int(a.max()) > 2147483647 or int(a.min()) < -1) for a in arrays):
        raise ValueError('plan values must fit int32')
    logical, execution, cp = [a.astype(np.int64) for a in arrays]
    ranks, home = logical.shape
    if not 1 <= ranks <= 8 or not home or ranks*home > 256 or home+cp.shape[1] > LOCAL_EXPERT_CAPACITY or cp.shape[0] != ranks or execution.shape != (ranks, home+cp.shape[1]):
        raise ValueError('inconsistent rank/home/slot dimensions')
    if (logical < 0).any() or (execution < 0).any():
        raise ValueError('negative route counts')
    if sum(map(int, logical.ravel())) > 2147483520:
        raise ValueError('total routes exceed device int32 capacity')
    experts = ranks*home
    if (cp < -1).any() or (cp >= experts).any():
        raise ValueError('invalid replica logical expert')
    received = np.zeros(experts, np.int64)
    contributors = [[] for _ in range(experts)]
    for rank in range(ranks):
        received[rank*home:(rank+1)*home] += execution[rank, :home]
        seen = set()
        for slot, expert in enumerate(cp[rank]):
            rows = int(execution[rank, home+slot])
            if expert < 0:
                if rows:
                    raise ValueError('unassigned slot has routes')
                continue
            expert = int(expert)
            if expert//home == rank or expert in seen or not rows:
                raise ValueError('owner-local, duplicate or empty active replica')
            seen.add(expert); received[expert] += rows; contributors[expert].append(rank)
    if not np.array_equal(received, logical.ravel()):
        raise ValueError('physical routes do not conserve logical expert counts')
    ceil = lambda n, d: (int(n)+d-1)//d
    dw1_tiles = ceil(2*ffn, 128)*ceil(hidden, 256)
    dw2_tiles = ceil(hidden, 128)*ceil(ffn, 256)
    plane_tiles = dw1_tiles+dw2_tiles
    payload = 6*int(hidden)*int(ffn)
    rows = []
    for rank in range(ranks):
        active = execution[rank] > 0
        shared = [e for e in range(rank*home, (rank+1)*home) if contributors[e]]
        remote_count = sum(len(contributors[e]) for e in shared)
        replicas = int((cp[rank] >= 0).sum())
        nonempty = int(active.sum())
        rows.append(dict(rank=rank,execution_rows=int(execution[rank].sum()),
            nonempty_home_experts=int(active[:home].sum()),nonempty_replica_experts=replicas,
            wgrad_output_tiles=nonempty*plane_tiles,
            wgrad_l1_k_visits=sum(ceil(n, 256)*plane_tiles for n in execution[rank] if n),
            shared_owned_experts=len(shared),owner_remote_contributors=remote_count,
            empty_shared_owners=[e for e in shared if execution[rank,e%home] == 0],
            owner_sum_output_tiles=len(shared)*plane_tiles,
            weight_receive_bytes=replicas*payload,weight_send_bytes=remote_count*payload,
            replica_partial_payload_bytes=replicas*payload,owner_partial_read_bytes=remote_count*payload))
    assert sum(v['weight_receive_bytes'] for v in rows) == sum(v['weight_send_bytes'] for v in rows)
    return dict(hidden=int(hidden),ffn=int(ffn),ep=ranks,home=home,slots=cp.shape[1],
        l1_tile=[128,256,256],dw1_tiles_per_expert=dw1_tiles,dw2_tiles_per_expert=dw2_tiles,
        bf16_weight_or_partial_bytes_per_replica=payload,rank_features=rows,
        predicts_time=False,scope='canonical BF16 GM WGrad schedule; not a calibrated cost model')
