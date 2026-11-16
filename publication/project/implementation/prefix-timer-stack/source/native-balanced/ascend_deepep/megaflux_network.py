"""External-parameter bridge for the accepted S9 v2 executor.

The model owns optimizer/FSDP parameters. Symmetric home weights are a per-call
snapshot, never replacement Parameters. One live training invocation per bridge
is supported; use a separate bridge for each layer/microbatch lease. Framework
activation checkpoint replay must not reuse a live lease.
"""
import torch
from torch.autograd.function import once_differentiable
from .megaflux_early import MegaFluxBF16EarlyTrain
from .megaflux_planners import NativeOnlinePlanner, ReplayPlanner


class _ExternalWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, x, p, w1, w2, ids, limit, capacity, planner):
        m = bridge.executor
        if m._pending or m._active:
            raise RuntimeError('live MegaFlux network lease: backward/finish before reuse')
        if tuple(w1.shape) != tuple(m.w1.shape) or tuple(w2.shape) != tuple(m.w2.shape):
            raise ValueError('external expert weights do not match the symmetric owner layout')
        if any(t.dtype != torch.bfloat16 for t in (x, p, w1, w2)):
            raise ValueError('network bridge requires BF16 inputs, probabilities and expert weights')
        with torch.profiler.record_function('MegaFlux.publish_home_weights'):
            m.w1.copy_(w1)
            m.w2.copy_(w2)
            # Source copies must be visible on every owner before a peer starts
            # copying replica weights. This conservative bridge fence is timed.
            torch.npu.current_stream().synchronize()
            bridge.publish_fence()
            torch.npu.current_stream().synchronize()
        with torch.profiler.record_function('MegaFlux.prepare_call_plan'):
            plan = m.create_plan(ids, planner, capacity)
        plan._consumed = True
        try:
            with torch.profiler.record_function('MegaFlux.forward_replica_pipeline'):
                y, logical, counts, state, dispatch = m._forward_device(x, p, plan, float(limit))
            ctx.bridge, ctx.plan, ctx.limit = bridge, plan, float(limit)
            ctx.save_for_backward(p, counts, state, dispatch)
            m._active = True
            bridge.last_counts = logical
            return y
        except BaseException:
            # Do not silently accept an incomplete or failed collective call.
            bridge.failed = True
            raise

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        bridge = ctx.bridge
        m = bridge.executor
        p, counts, state, dispatch = ctx.saved_tensors
        m._check_device_plan(ctx.plan)
        try:
            with torch.profiler.record_function('MegaFlux.backward_owner_reduce_pipeline'):
                dx, dp, a, b = m.op.backward_replica(
                    m.shmem, dy.contiguous(), p, m.w1, m.w2, counts, state, dispatch,
                    ctx.plan.max_recv, ctx.limit, 'bf16', m._descriptor(ctx.plan))
            m._active = False
            with torch.profiler.record_function('MegaFlux.finish_before_parameter_gradients'):
                m.finish()
            return None, dx, dp, a[:m.home], b[:m.home], None, None, None, None
        except BaseException:
            bridge.failed = True
            raise


class MegaFluxNetworkBridge:
    """BF16 model tensors -> S9 v2 -> gradients of those same model tensors.

    ``publish_fence`` is a collective on exactly this SHMEM EP group. All ranks
    must construct bridges and call them in the same order. No tensor/weight
    values are cached across optimizer updates. This initial integration keeps
    weight-publication synchronization explicit for profiling and correctness.
    """
    def __init__(self, shmem, home_experts, slots, hidden, ffn, local_mem_size,
                 device, publish_fence, beta=768, penalty=2, gain=4,
                 executor_class=MegaFluxBF16EarlyTrain, planner=None):
        self.executor = executor_class(shmem, home_experts, slots, hidden, ffn,
                                       local_mem_size, device)
        self.publish_fence = publish_fence
        self.beta, self.penalty, self.gain = beta, penalty, gain
        self.planner = NativeOnlinePlanner(beta, penalty, gain) if planner is None else planner
        self.last_counts = None
        self.failed = False

    def __call__(self, x, ids, p, w1, w2, limit=0., capacity=None):
        if self.failed:
            raise RuntimeError('failed network bridge cannot be reused')
        training = torch.is_grad_enabled() and any(t.requires_grad for t in (x, p, w1, w2))
        y = _ExternalWeights.apply(self, x, p, w1, w2, ids, float(limit), capacity, self.planner)
        if not training:
            self.executor._active = False
            self.executor.finish()
        return y

    def forward_plan(self, x, p, w1, w2, prepared, limit=0.):
        """External parameters with any policy's validated, resident plan."""
        if self.failed: raise RuntimeError('failed network bridge cannot be reused')
        training = torch.is_grad_enabled() and any(t.requires_grad for t in (x,p,w1,w2))
        y = _ExternalWeights.apply(self, x, p, w1, w2, prepared.logical_ids,
                                  float(limit), prepared.capacity, ReplayPlanner(prepared))
        if not training:
            self.executor._active = False
            self.executor.finish()
        return y

    def close(self):
        self.executor.close()
