"""Policy adapters. Execution code consumes their identical device contract."""
from dataclasses import dataclass
from types import SimpleNamespace
from .control_timing import control_timing_scope

_CHECKED_PLAN = object()


import torch
_LEGACY_ALGORITHM = object()


class NativeOnlinePlan:
    def __init__(self, module, ids, capacity, beta, penalty, gain, algorithm=_LEGACY_ALGORITHM):
        self.reference = SimpleNamespace(home=module.home, slots=module.slots)
        self.logical_ids = ids
        self.epoch = module._epoch
        self.owner = module._plan_owner
        self.max_recv = capacity
        shape = (tuple(ids.shape), ids.device)
        if module._buffer_shape != shape:
            ep = module.shmem.world_size()
            experts = module.home * ep
            make = lambda size: torch.empty(size, device=ids.device, dtype=torch.int32)
            module._plan_buffers = (make((ep, module.slots)), make(ids.shape),
                                    make((experts,)), make((experts, ep)), make((8,)))
            module._buffer_shape = shape
        self.copies, self.physical_ids, self.logical_counts, self.assignment, self.header = module._plan_buffers
        if getattr(module, '_cache_control_views', False):
            if getattr(module, '_cached_plan_buffers', None) is not module._plan_buffers:
                # Only stable addresses/rank metadata are cached. Tensor
                # values, route pointer, epoch and capacity stay per-call.
                descriptor = tuple(module.workspace.descriptor(module.shmem, self.copies))
                counts = self.logical_counts.narrow(0, descriptor[12]*module.home, module.home)
                module._cached_descriptor_base = descriptor
                module._cached_owner_counts = counts
                module._cached_plan_buffers = module._plan_buffers
            self.owner_counts = module._cached_owner_counts
            self.device_descriptor = module._cached_descriptor_base + (
                self.header.data_ptr(), ids.data_ptr(), self.epoch, capacity, 2)
        args=(module.shmem, ids, int(beta), int(penalty), int(gain),
              self.epoch, *module._plan_buffers, capacity)
        if algorithm is _LEGACY_ALGORITHM: algorithm=getattr(module,'_planner_algorithm',None)
        with control_timing_scope('planner.call'):
            if algorithm is None:module.workspace.online_plan_device(*args)
            else:module.workspace.online_plan_optimized(*args,algorithm)


@dataclass(frozen=True)
class PlanningContext:
    """Executor-owned per-call lease and runtime services, not policy state."""
    executor: object
    logical_ids: object
    capacity: int
    epoch: int


class NativeOnlinePlanner:
    """Unchanged native online algorithm, including its collective error gate."""
    def __init__(self, beta=768, penalty_blocks=0, min_gain_blocks=0, algorithm=None):
        self.beta, self.penalty, self.gain = beta, penalty_blocks, min_gain_blocks
        if algorithm not in (None,0,1,2): raise ValueError('unknown native planner algorithm')
        self.algorithm = algorithm
        self.calls=0

    def plan(self, context):
        self.calls+=1
        plan = NativeOnlinePlan(context.executor, context.logical_ids, context.capacity,
                                self.beta, self.penalty, self.gain, self.algorithm)
        plan._certificate = _CHECKED_PLAN
        return plan


class ReplayPlanner:
    """Adapter for any policy's prepared placement and per-route assignment."""
    def __init__(self, prepared):
        self.prepared = prepared
        self.capacity_hint = prepared.capacity

    def plan(self, context):
        import torch
        m = context.executor; prepared = self.prepared
        prepared.check(m, context.logical_ids)
        if context.capacity != prepared.capacity:
            raise ValueError('replay capacity must match collective preparation')
        active = int((prepared.spec.copies >= 0).sum())
        # Only the per-call 32B control/epoch is produced here. Route buffers
        # remain resident; no planner, histogram, route gather or weight cache.
        header = torch.tensor([active, prepared.spec.required_capacity, context.epoch,
                               0, 0, int(active > 0), active, context.capacity],
                              device=prepared.device, dtype=torch.int32)
        return SimpleNamespace(reference=SimpleNamespace(home=m.home, slots=m.slots),
            logical_ids=prepared.logical_ids, physical_ids=prepared.physical_ids,
            copies=prepared.copies, logical_counts=prepared.logical_counts,
            header=header, epoch=context.epoch, owner=m._plan_owner,
            max_recv=context.capacity, prepared=prepared, _certificate=_CHECKED_PLAN)
