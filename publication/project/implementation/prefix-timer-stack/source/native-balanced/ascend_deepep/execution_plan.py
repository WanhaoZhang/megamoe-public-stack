"""Planner-independent, validated BF16 expert placement and route contract.

Only preparation imports torch. CPU policies can use ExecutionPlanSpec without
an NPU. The execution representation contains no beta, penalty or policy score.
"""
from dataclasses import dataclass
from .runtime_limits import LOCAL_EXPERT_CAPACITY
import hashlib
import numpy as np


def _immutable_i32(value, name):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_):
        raise ValueError(name + ' must contain integers')
    if a.size and (a.min() < -2147483648 or a.max() > 2147483647):
        raise ValueError(name + ' exceeds int32')
    # A bytes-backed array cannot be made writeable again with setflags().
    return np.frombuffer(np.asarray(a, dtype=np.int32).tobytes(), dtype=np.int32).reshape(a.shape)


@dataclass(frozen=True, init=False)
class ExecutionPlanSpec:
    """Global canonical routes [EP,M,K], with fixed uniform optimizer owners.

    Physical ID = destination_rank * (home + slots) + execution_slot.
    Slot [0,home) is that rank's home expert; remaining slots use copies[r,s].
    Every original (source rank, token, TopK slot) has exactly one destination.
    Identical logical IDs in different TopK slots remain separate routes.
    """
    schema: int
    home: int
    slots: int
    logical_ids: np.ndarray
    physical_ids: np.ndarray
    copies: np.ndarray
    logical_counts: np.ndarray
    exec_counts: np.ndarray
    required_capacity: int
    fingerprint: str

    def __init__(self, logical_ids, physical_ids, copies, home):
        logical = _immutable_i32(logical_ids, 'logical_ids')
        physical = _immutable_i32(physical_ids, 'physical_ids')
        replicas = _immutable_i32(copies, 'copies')
        if logical.ndim != 3 or physical.shape != logical.shape or replicas.ndim != 2:
            raise ValueError('expected routes [EP,M,K] and copies [EP,slots]')
        ep, m, k = logical.shape
        if not isinstance(home, (int, np.integer)) or isinstance(home, (bool, np.bool_)):
            raise ValueError('home must be an integer')
        home = int(home); slots = replicas.shape[1]; width = home + slots
        experts = ep * home
        # These are representational limits, not a transport capability claim.
        # Fixed uniform owners require the global physical-slot budget too.
        # Large online planning uses a separate GM kernel, not larger UB Q.
        if not ((2 <= ep <= 8 or ep in (16,32,64,128,256)) and ep*slots <= 2048 and
                home > 0 and width <= LOCAL_EXPERT_CAPACITY and experts <= 2048 and ep*width<=2048 and
                0 < m <= 16384 and 0 < k <= experts and replicas.shape[0] == ep and
                logical.size < 2147483392):
            raise ValueError('unsupported EP/home/slot/token/TopK dimensions')
        if np.any((logical < 0) | (logical >= experts)):
            raise ValueError('logical expert out of range')
        if np.any((replicas < -1) | (replicas >= experts)):
            raise ValueError('replica expert out of range')
        for rank, row in enumerate(replicas):
            live = row[row >= 0]
            if len(live) != len(set(map(int, live))):
                raise ValueError('duplicate replica of one expert on the same rank')
            if np.any(live // home == rank):
                raise ValueError('replicas must be on ranks other than the fixed owner')
        if np.any((physical < 0) | (physical >= ep * width)):
            raise ValueError('physical execution ID out of range')
        slotmap = np.concatenate((np.arange(experts, dtype=np.int32).reshape(ep, home), replicas), axis=1)
        if not np.array_equal(slotmap.ravel()[physical], logical):
            raise ValueError('physical route changes a logical expert or uses an empty slot')
        ec = np.bincount(physical.ravel(), minlength=ep*width).reshape(ep, width)
        if np.any((replicas >= 0) & (ec[:, home:] == 0)):
            # Reduce waits for every declared replica's produced partials.
            # An unused declared slot would never publish those ready epochs.
            raise ValueError('declared replica must receive at least one route')
        lc = np.bincount(logical.ravel(), minlength=experts).reshape(ep, home)
        # The ABI2 executor uses physical rows; zero-replica placement has the
        # same owner rows. Do not reserve the old pre-replication hotspot size.
        required = max(int(ec.sum(1).max()), 1)
        required = (required + 127) // 128 * 128
        digest = hashlib.sha256()
        digest.update(np.asarray([1, home, slots, ep, m, k], dtype='<i8').tobytes())
        for a in (logical, physical, replicas): digest.update(a.tobytes())
        for key, value in dict(schema=1, home=home, slots=slots, logical_ids=logical,
                physical_ids=physical, copies=replicas,
                logical_counts=_immutable_i32(lc, 'logical_counts'),
                exec_counts=_immutable_i32(ec, 'exec_counts'), required_capacity=required,
                fingerprint=digest.hexdigest()).items():
            object.__setattr__(self, key, value)

    @classmethod
    def from_replica_plan(cls, logical_ids, plan):
        """Adapter for the existing CPU reference; beta is not executor state."""
        return cls(logical_ids, plan.physical_ids, plan.copies, plan.home)

    @classmethod
    def from_blocks(cls, logical_ids, route_blocks, block_exec, copies, home):
        blocks = _immutable_i32(route_blocks, 'route_blocks')
        table = _immutable_i32(block_exec, 'block_exec')
        if blocks.shape != np.asarray(logical_ids).shape or table.ndim != 1 or not table.size:
            raise ValueError('block assignment shape mismatch')
        if np.any((blocks < 0) | (blocks >= table.size)):
            raise ValueError('route block index out of range')
        return cls(logical_ids, table[blocks], copies, home)

    @property
    def contributors(self):
        """Replica (rank,slot) contributors; owner is fixed by expert//home."""
        return tuple(tuple((r, s) for r in range(len(self.copies))
                           for s in range(self.slots) if self.copies[r, s] == e)
                     for e in range(self.home * len(self.copies)))


class PreparedExecutionPlan:
    """Collectively checked, owned device snapshot; reusable across SGD epochs.

    Preparation is outside the kernel hot path and includes validation/H2D.
    Properties expose tensors for interoperability; in-place edits invalidate
    this object. Raw storage/.data writes are unsupported, as for autograd.
    """
    def __init__(self, spec, rank, device, capacity):
        import torch
        self.spec = spec; self.rank = rank; self.device = torch.device(device)
        self.capacity = capacity
        put = lambda a: torch.tensor(np.array(a, copy=True), device=device, dtype=torch.int32)
        self.logical_ids = put(spec.logical_ids[rank])
        self.physical_ids = put(spec.physical_ids[rank])
        self.copies = put(spec.copies)
        self.logical_counts = put(spec.logical_counts.ravel())
        self._tensors = (self.logical_ids, self.physical_ids, self.copies, self.logical_counts)
        self._versions = tuple(t._version for t in self._tensors)
        self._pointers = tuple(t.data_ptr() for t in self._tensors)
        self.stream_id = torch.npu.current_stream(device).npu_stream
        self._frozen = True

    def __setattr__(self, name, value):
        if getattr(self, '_frozen', False):
            raise AttributeError('prepared execution plans are immutable; prepare a new plan')
        object.__setattr__(self, name, value)

    def check(self, module, ids):
        import torch
        if (module.home, module.slots, module.shmem.rank(), module.shmem.world_size()) != (
                self.spec.home, self.spec.slots, self.rank, len(self.spec.copies)):
            raise ValueError('prepared plan does not match executor topology')
        if module.w1.device != self.logical_ids.device:
            raise ValueError('prepared plan device differs from executor weights')
        if tuple(t._version for t in self._tensors) != self._versions or tuple(t.data_ptr() for t in self._tensors) != self._pointers:
            raise RuntimeError('prepared plan was mutated')
        if ids is not self.logical_ids:
            raise ValueError('replay uses the prepared logical_ids snapshot')
        if torch.npu.current_stream(self.device).npu_stream != self.stream_id:
            raise RuntimeError('prepared plan must be acquired on its preparation stream')


def prepare_execution_plan(spec, *, device, group=None, capacity=None):
    """Collective preparation on exactly the SHMEM rank group, in rank order.

    Pass the same global spec on every rank. Local validation failures must be
    represented by the caller before this collective (see import_device_plan).
    No per-call route D2H, all-gather or counts rebuild happens during replay.
    """
    import torch.distributed as dist
    local_error = None
    try:
        if not isinstance(spec, ExecutionPlanSpec): raise TypeError('expected ExecutionPlanSpec')
        cap = spec.required_capacity if capacity is None else int(capacity)
        if cap < spec.required_capacity or cap > 2147483392 or cap % 128:
            raise ValueError('capacity must fit int32, align128 and cover the plan')
        stamp = (spec.schema, spec.fingerprint, cap, len(spec.copies))
    except Exception as exc:
        local_error = str(exc); stamp = None
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, (local_error, stamp), group=group)
    if any(p[0] for p in peers): raise ValueError('collective plan preparation rejected: '+repr(peers))
    if any(p[1] != stamp for p in peers) or stamp[-1] != len(peers):
        raise ValueError('ranks supplied inconsistent plan/topology/capacity')
    return PreparedExecutionPlan(spec, dist.get_rank(group), device, cap)


def import_device_plan(logical_ids, physical_ids, copies, *, home, group=None, capacity=None):
    """Strict generic adapter for arbitrary device planners; explicitly costly.

    Input is each rank's local [M,K] routes and global [EP,slots] placement.
    D2H/gather/validation/H2D are preparation costs, not a fast online planner.
    Algorithms needing fully device-resident dynamic planning should implement
    a checked device adapter rather than hide these costs in execution timing.
    """
    import torch.distributed as dist
    error = None; payload = None
    try:
        payload = tuple(t.detach().cpu().numpy().copy() for t in (logical_ids, physical_ids, copies))
    except Exception as exc: error = str(exc)
    peers = [None] * dist.get_world_size(group)
    dist.all_gather_object(peers, (error, payload), group=group)
    if any(p[0] for p in peers): raise ValueError('device plan import failed: '+repr([p[0] for p in peers]))
    error = None; spec = None
    try:
        replicas = peers[0][1][2]
        if any(not np.array_equal(p[1][2], replicas) for p in peers):
            raise ValueError('placement differs across ranks')
        spec = ExecutionPlanSpec(np.stack([p[1][0] for p in peers]),
                                 np.stack([p[1][1] for p in peers]), replicas, home)
    except Exception as exc: error = str(exc)
    errors = [None] * len(peers); dist.all_gather_object(errors, error, group=group)
    if any(errors): raise ValueError('device plan import rejected: '+repr(errors))
    return prepare_execution_plan(spec, device=logical_ids.device, group=group, capacity=capacity)
