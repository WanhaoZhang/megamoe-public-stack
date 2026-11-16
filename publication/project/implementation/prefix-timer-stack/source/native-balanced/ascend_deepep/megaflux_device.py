"""Experimental async device-plan executor with an explicit checked fence.

Call finish() after backward (or forward-only) before consuming results/SGD or
reusing the instance. It waits for the current stream and reports collective
planner/capacity errors. Include finish() in complete-call benchmarks.
"""
from types import SimpleNamespace
import torch
from torch.autograd.function import once_differentiable
from .megaflux_mixed import MegaFluxBF16MixedTrain
from .saved_gate_up_mode import validate_save_gmm1_output, resolve_save_gmm1_output


def receive_capacity(tokens, topk, capacity=None):
    """Shape-only budget; insufficient budgets fail collectively at finish()."""
    if tokens <= 0 or topk <= 0:
        raise ValueError('positive token/TopK dimensions required')
    value = 2 * tokens * topk if capacity is None else int(capacity)
    if value <= 0 or value > 2147483392:
        raise ValueError('receive capacity must fit aligned int32')
    return (value + 127) // 128 * 128


# Backward-compatible import name; policy implementation lives in the adapter.
from .megaflux_planners import NativeOnlinePlan as AsyncDevicePlan


class _DeviceReplica(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, x, p, w1, w2, plan, limit):
        module._active = True
        try:
            y, logical, counts, state, dispatch = module._forward_device(x, p, plan, limit)
            ctx.module, ctx.plan, ctx.limit = module, plan, limit
            ctx.save_gmm1_output = plan._save_gmm1_output
            ctx.save_for_backward(p, w1, w2, counts, state, dispatch)
            ctx.mark_non_differentiable(logical)
            return y, logical
        except Exception:
            module._active = False
            raise

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, unused=None):
        m = ctx.module
        p, w1, w2, counts, state, dispatch = ctx.saved_tensors
        m._check_device_plan(ctx.plan)
        try:
            dx, dp, dw1, dw2 = m.op.backward_replica(
                m.shmem, dy.contiguous(), p, w1, w2, counts, state, dispatch,
                ctx.plan.max_recv, ctx.limit, ctx.save_gmm1_output, m._descriptor(ctx.plan))
            return None, dx, dp, dw1[:m.home], dw2[:m.home], None, None
        finally:
            m._active = False


class MegaFluxBF16DeviceTrain(MegaFluxBF16MixedTrain):
    """One pending call per instance/stream; explicit checked completion."""
    def __init__(self, *args, save_gmm1_output='bf16', **kwargs):
        self._save_gmm1_output = validate_save_gmm1_output(save_gmm1_output)
        super().__init__(*args, **kwargs)
        self._plan_buffers = None
        self._buffer_shape = None
        self._plan_owner = object()
        self._pending = False
        self._stream_id = None

    @property
    def save_gmm1_output(self):
        return self._save_gmm1_output

    def _record_saved_mode(self, plan, limit):
        # A plan owns one forward.  Backward freezes this resolved mode in its
        # context instead of reading the executor's latest invocation metadata.
        mode = resolve_save_gmm1_output(self.save_gmm1_output, limit)
        plan._save_gmm1_output = mode
        self.last_save_gmm1_output = mode
        return mode

    def _check_device_plan(self, plan):
        if plan.owner is not self._plan_owner or plan.epoch != self._epoch:
            raise RuntimeError('foreign or expired device plan')
        if torch.npu.current_stream().npu_stream != self._stream_id:
            raise RuntimeError('device plan must execute and finish on its original stream')
        tensors = getattr(plan, '_lease_tensors', ())
        if any(getattr(plan,n) is not t or t._version != v or t.data_ptr() != ptr for n,t,v,ptr in tensors):
            raise RuntimeError('execution plan tensor was mutated during its lease')
        if getattr(plan, '_lease_capacity', plan.max_recv) != plan.max_recv:
            raise RuntimeError('execution plan capacity was mutated during its lease')

    def plan_online(self, *args, **kwargs):
        raise RuntimeError('use forward_online and the checked finish fence for device plans')

    def __call__(self, x, p, plan, limit=0.0):
        return self.forward_plan(x, p, plan, limit)

    def forward_only(self, *args, **kwargs):
        raise RuntimeError('use forward_online under torch.no_grad(), followed by finish')

    def forward_cpu_online(self, *args, **kwargs):
        raise RuntimeError('the device entry does not accept host-generated plans')

    def _descriptor(self, plan):
        if getattr(plan,'_sft_host_only_zero',False):
            raise RuntimeError('private CPU-only SFT zero metadata cannot describe a replica kernel')
        cached = getattr(plan, 'device_descriptor', None)
        if cached is not None:
            return list(cached)
        return self.workspace.descriptor(self.shmem, plan.copies) + [
            plan.header.data_ptr(), plan.logical_ids.data_ptr(), plan.epoch, plan.max_recv, 2]

    def _forward_device(self, x, p, plan, limit):
        self._check_device_plan(plan)
        save_mode = self._record_saved_mode(plan, limit)
        y, counts, state, dispatch, _ = self.op.compute_train_replica(
            self.shmem, x, plan.physical_ids, p, self.w1, self.w2,
            plan.max_recv, float(limit), save_mode, self._descriptor(plan))
        self.last_exec_counts = counts
        rank = self.shmem.rank()
        logical = plan.logical_counts[rank*self.home:(rank+1)*self.home]
        return y, logical, counts, state, dispatch

    def forward_online(self, x, logical_ids, p, limit=0.0, beta=768,
                       penalty_blocks=0, min_gain_blocks=0, capacity=None):
        from .megaflux_planners import NativeOnlinePlanner
        return self.forward_with_planner(x, logical_ids, p,
            NativeOnlinePlanner(beta, penalty_blocks, min_gain_blocks,
                algorithm=getattr(self, '_planner_algorithm', None)), limit, capacity)

    def forward_with_planner(self, x, logical_ids, p, planner, limit=0.0, capacity=None):
        """Dynamic policy -> checked plan -> the same fused executor."""
        if tuple(p.shape) != tuple(logical_ids.shape) or x.shape[0] != p.shape[0]:
            raise ValueError('route/probability/input shape mismatch')
        plan = self.create_plan(logical_ids, planner, capacity)
        return self.forward_plan(x, p, plan, limit)

    def create_plan(self, logical_ids, planner, capacity=None):
        """Run a policy adapter; reserve one executor-owned call lease.

        A prepared/replay adapter does not execute a planning algorithm. Dynamic
        adapters must return a checked canonical plan, as the native provider
        does. Do not pass unchecked arbitrary NPU pointers to this interface.
        """
        from .megaflux_planners import PlanningContext, ReplayPlanner, _CHECKED_PLAN
        from .execution_plan import PreparedExecutionPlan
        if self._closed:
            raise RuntimeError('device operator closed')
        if self._pending or self._active:
            raise RuntimeError('finish the previous device call before replanning')
        if self._epoch >= 2147483647:
            raise RuntimeError('executor epoch exhausted; create a new executor after collective retirement')
        self._epoch += 1
        self._stream_id = torch.npu.current_stream().npu_stream
        if logical_ids.ndim != 2 or logical_ids.dtype != torch.int32 or not logical_ids.is_contiguous():
            raise ValueError('logical_ids must be contiguous int32 [M,TopK]')
        if capacity is None: capacity = getattr(planner, 'capacity_hint', None)
        budget = receive_capacity(logical_ids.shape[0], logical_ids.shape[1], capacity)
        plan = planner.plan(PlanningContext(self, logical_ids, budget, self._epoch))
        if isinstance(plan, PreparedExecutionPlan):
            plan = ReplayPlanner(plan).plan(PlanningContext(self, logical_ids, budget, self._epoch))
        if getattr(plan, '_certificate', None) is not _CHECKED_PLAN:
            raise ValueError('planner must return a validated PreparedExecutionPlan or a checked native adapter result')
        self._check_device_plan(plan)
        plan._lease_tensors = tuple((n, getattr(plan,n), getattr(plan,n)._version, getattr(plan,n).data_ptr()) for n in (
            'logical_ids', 'physical_ids', 'copies', 'logical_counts', 'header'))
        plan._lease_capacity = plan.max_recv
        plan._consumed = False
        self.last_plan = plan
        self.last_exec_counts = None
        self._pending = True
        return plan

    def forward_plan(self, x, p, plan, limit=0.0):
        """Execute a bound call or acquire a reusable PreparedExecutionPlan.

        This same entry owns actual weight replication, original compute and
        owner reduction in backward; it never chooses a placement or route.
        """
        from .execution_plan import PreparedExecutionPlan
        from .megaflux_planners import ReplayPlanner
        if isinstance(plan, PreparedExecutionPlan):
            plan = self.create_plan(plan.logical_ids, ReplayPlanner(plan), plan.capacity)
        if not self._pending or plan is not self.last_plan:
            raise RuntimeError('execution needs the current executor-owned plan lease')
        self._check_device_plan(plan)
        if plan._consumed or self._active:
            raise RuntimeError('execution plan lease may be consumed only once')
        if tuple(p.shape) != tuple(plan.logical_ids.shape) or x.shape[0] != p.shape[0]:
            raise ValueError('route/probability/input shape mismatch')
        plan._consumed = True
        if torch.is_grad_enabled():
            return _DeviceReplica.apply(self, x, p, self.w1, self.w2, plan, float(limit))
        y, logical, *_ = self._forward_device(x, p, plan, limit)
        return y, logical

    def release_pending(self):
        """Drop the call lease without draining the stream.

        The next create_plan is enqueued on the same stream, after any saved
        placement/header clones, so planner scratch can be overwritten in
        order. Host finish_device remains for callers that must read the
        control header on the CPU.
        """
        if self._active:
            raise RuntimeError('backward must complete before releasing the plan lease')
        if not self._pending:
            raise RuntimeError('no pending device call')
        self._check_device_plan(self.last_plan)
        self._pending = False

    def finish(self):
        """Checked fence, mandatory before accepting results or doing SGD."""
        if self._active:
            raise RuntimeError('backward must complete before finish')
        if not self._pending:
            raise RuntimeError('no pending device call')
        self._check_device_plan(self.last_plan)
        try:
            status = self.workspace.finish_device(self.last_plan.header, self._epoch, self.last_plan.max_recv)
            if not status[0] and self.last_exec_counts is not None:
                self.last_exec_counts = self.last_exec_counts[:self.home]
            self.last_status = status
            return status
        finally:
            self._pending = False

    def close(self):
        if self._pending:
            raise RuntimeError('finish pending device call before close')
        super().close()
