"""Live FSDP parameters and a bounded, serial S9 weight pool.

The autograd tape owns all per-layer tensors (including the checked placement).
No autograd context retains pooled weights or the mutable online planner scratch.
EP owner SUM is native; FSDP owns its own reduce-scatter and gradient scaling.
"""
from types import SimpleNamespace
import hashlib
import os

import torch
from .saved_gate_up_mode import validate_save_gmm1_output, resolve_save_gmm1_output

from .control_timing import control_timing_scope


_PLAN_TENSORS = ('logical_ids', 'physical_ids', 'copies', 'logical_counts')


def _owner_gradient(gradient, home):
    """Native outputs are fresh allocations; the prefix view owns their lifetime.

    Keep the previous extra copy available for a controlled ablation. Never
    apply this rule to mutable pool or planner scratch tensors.
    """
    prefix = gradient[:home]
    if os.environ.get('MEGAFLUX_CLONE_OWNER_GRAD', '0') == '1':
        return prefix.clone()
    return prefix


def _require_panel_ready(mode, module):
    if mode == 'panel' and getattr(module, 'restore_operand_ready', 0) != 1:
        raise RuntimeError('panel restore requires operand-ready=1; rebuild the worker SO '
                           'and select the verified variant, not a first-panel-only library')


class _SavedPlacementReplay:
    """Replay a forward placement on a new epoch.

    ``status`` is either the device header cloned after compute (no host
    fence) or a host length-8 list from an older capture.
    """

    def __init__(self, home, slots, capacity, status, tensors):
        self.home, self.slots, self.capacity_hint = home, slots, capacity
        self.tensors = dict(zip(_PLAN_TENSORS, tensors))
        if len(tensors) != len(_PLAN_TENSORS):
            raise RuntimeError('incomplete autograd placement')
        if torch.is_tensor(status):
            if tuple(status.shape) != (8,) or status.dtype != torch.int32:
                raise RuntimeError('replay header must be int32[8]')
            self.header = status
            self.status = None
        else:
            self.header = None
            self.status = tuple(status)
            if len(self.status) != 8 or self.status[3] or self.status[1] > capacity or self.status[7] != capacity:
                raise RuntimeError('cannot replay an unsuccessful forward placement')

    def plan(self, context):
        from .megaflux_planners import _CHECKED_PLAN
        m = context.executor
        if (m.home, m.slots, context.capacity) != (self.home, self.slots, self.capacity_hint):
            raise RuntimeError('saved placement/executor capacity mismatch')
        if context.logical_ids is not self.tensors['logical_ids']:
            raise RuntimeError('backward must bind the saved forward logical routes')
        if self.header is not None:
            header = self.header.detach().to(context.logical_ids.device).clone()
            header[2] = int(context.epoch)
        else:
            status = list(self.status)
            status[2] = context.epoch
            header = torch.tensor(status, device=context.logical_ids.device, dtype=torch.int32)
        return SimpleNamespace(**self.tensors, header=header, epoch=context.epoch,
            owner=m._plan_owner, max_recv=context.capacity,
            reference=SimpleNamespace(home=m.home, slots=m.slots), _certificate=_CHECKED_PLAN)


class _ExternalExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bridge, x, ids, prob, w1, w2, limit):
        ctx.save_gmm1_output = resolve_save_gmm1_output(
            getattr(bridge.pool, 'save_gmm1_output', 'bf16'), limit)
        y, header, saved = bridge.pool.forward(x, ids, prob, w1, w2, float(limit))
        ctx.bridge, ctx.limit = bridge, float(limit)
        ctx.weight_meta = ((tuple(w1.shape), w1.dtype), (tuple(w2.shape), w2.dtype))
        ctx.prob_dtype = prob.dtype
        ctx.routed_redispatch = getattr(bridge.pool, 'routed_input_mode', 'saved') == 'redispatch'
        if ctx.routed_redispatch:
            counts, state, dispatch, *placement = saved
            sink = getattr(bridge.pool, 'x_redispatch_debug_sink', None)
            if sink is not None:
                sink('forward', dispatch, counts, bridge.pool.hidden, bridge.pool.capacity, int(header[2]))
            saved = (counts, state, torch.empty(0, dtype=torch.uint8, device=x.device), *placement)
        # Header stays on the execution stream. Backward patches the epoch
        # there; a host copy is only for the explicit CPU-tape diagnostic.
        if getattr(bridge.pool, 'control_tape_device', 'npu') == 'cpu':
            header = header.detach().to('cpu')
        capture = getattr(bridge.pool, 'capture', None)
        # The tape schema must stay identical during checkpoint recomputation,
        # even after another layer has already written the one selected capture.
        ctx.capture = capture is not None
        # Only the explicit diagnostic run saves the original local input/Y.
        # Both remain visible to checkpoint/UAM saved-tensor hooks.
        ctx.save_for_backward(prob, *saved, header,
                              *((x,) if ctx.routed_redispatch else ()),
                              *((x, y) if ctx.capture else ()))
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        saved = ctx.saved_tensors
        if ctx.capture:
            saved, (capture_x, capture_y) = saved[:-2], saved[-2:]
        original_x = None
        if getattr(ctx, 'routed_redispatch', False):
            saved, original_x = saved[:-1], saved[-1]
        prob, counts, state, dispatch, *placement_and_control = saved
        placement, header = placement_and_control[:-1], placement_and_control[-1]
        w1, w2 = ctx.bridge.current_weights()
        for w, (shape, _) in zip((w1, w2), ctx.weight_meta):
            if tuple(w.shape) != shape:
                raise RuntimeError('FSDP backward unshard did not restore the full home shape')
        dx, dp, dw1, dw2 = ctx.bridge.pool.backward(
            grad, prob, w1, w2, header, placement, counts, state, dispatch, ctx.limit,
            save_gmm1_output=ctx.save_gmm1_output,
            **({'original_x': original_x} if original_x is not None else {}))
        if ctx.capture:
            ctx.bridge.pool.capture.save(ctx.bridge.pool, capture_x, capture_y, grad, prob,
                w1, w2, header.detach().cpu().tolist(), placement, (dx, dp, dw1, dw2), ctx.limit)
        # BF16 native partial/owner rounding is preserved. Casting here only
        # satisfies the original Parameter/router dtype at the framework edge.
        if dw1 is None:
            return None, dx, None, dp.to(ctx.prob_dtype), None, None, None
        return (None, dx, None, dp.to(ctx.prob_dtype), dw1.to(ctx.weight_meta[0][1]),
                dw2.to(ctx.weight_meta[1][1]), None)


class FSDP2ExpertsBridge:
    def __init__(self, pool, weights, shared_weights=None):
        self.pool, self.current_weights = pool, weights
        self.current_shared_weights = shared_weights

    def __call__(self, x, ids, prob, limit=0.):
        w1, w2 = self.current_weights()
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, prob, w1, w2)):
            return _ExternalExperts.apply(self, x, ids, prob, w1, w2, float(limit))
        return self.pool.forward(x, ids, prob, w1, w2, float(limit))[0]

    def forward_shared(self, x, ids, prob, weight_gate, weight_up, weight_down,
                       limit=0., shared_save_mode='recompute', *, shared_weight_provider=None,
                       routed_input_mode='saved'):
        from .shared_expert import forward_shared
        if shared_weight_provider is not None:
            if not callable(shared_weight_provider):
                raise ValueError('shared_weight_provider must be an owner callback')
            # This callback must read the module at use time, not close over
            # the forward's local parameter tensors. Each autograd ctx retains
            # its callback even if a later forward updates the bridge callback.
            self.current_shared_weights = shared_weight_provider
        return forward_shared(self, x, ids, prob, weight_gate, weight_up,
                              weight_down, limit, shared_save_mode, routed_input_mode)


class NativeFSDP2Pool:
    """One weight/window allocation per EP group and static expert shape."""

    def __init__(self, shmem, group, home, slots, hidden, ffn, tokens, topk,
                 window_bytes, capacity, device, planner='moonep', backward_restore='staged', plan_validation='fused', direct_permille=0, timer=False, *, enable_diagnostics=False, save_gmm1_output='bf16', defer_completion_checks=False, routed_input_mode='saved',cost_model=None):
        self.defer_completion_checks = bool(defer_completion_checks)
        if routed_input_mode not in ('saved', 'redispatch'):
            raise ValueError('unknown routed input save contract')
        if routed_input_mode == 'redispatch' and backward_restore != 'panel':
            raise ValueError('routed redispatch requires panel restore')
        self.routed_input_mode = routed_input_mode
        self._save_gmm1_output = validate_save_gmm1_output(save_gmm1_output)
        import torch.distributed as dist
        from .megaflux_worker_pipeline import MegaFluxBF16WorkerPipelineTrain
        from .megaflux_planners import NativeOnlinePlanner
        from .moonep_planner import MoonEPPlanner
        from .runtime_limits import LARGE_EPS,gm_planner_requested,validate_layout,require_gm_binary
        from .execution_plan_device import normalize_validation
        checked_mode=normalize_validation(plan_validation)
        unified=(os.environ.get('MEGAFLUX_PLANNER_INTERFACE','unified')=='unified' and shmem.world_size() in (2,4,8))
        if checked_mode == 'native' and planner != 'moonep' and not unified:
            raise ValueError('native plan validation requires the built-in MoonEP provider')
        self.plan_validation_implementation='common-fused-v2' if unified else (('native-moonep' if checked_mode=='native' else 'fused-v2') if planner=='moonep' else 'native')
        ep=shmem.world_size()
        if timer:
            if (not unified and planner != 'moonep') or ep not in (2,4,8)+LARGE_EPS:
                raise ValueError('detailed Timer capture requires the matching MoonEP planner and worker')
            from .megaflux_worker_timer import MegaFluxBF16WorkerTimerTrain
            executor_type = MegaFluxBF16WorkerTimerTrain
        else:
            executor_type = MegaFluxBF16WorkerPipelineTrain
        if ep not in (2,4,8)+LARGE_EPS:
            raise ValueError('Wuhan integration requires EP2/4/8/16/32/64/128/256')
        if gm_planner_requested(ep):
            signature=(ep,int(home),int(slots),hidden,ffn,tokens,topk,int(capacity),planner,backward_restore,'fused-v2',self.save_gmm1_output)
            peers=[None]*ep;dist.all_gather_object(peers,signature,group=group)
            if any(p!=signature for p in peers):
                raise ValueError('inconsistent large EP pool shape/policy/capacity across ranks')
            if dist.get_world_size(group)!=ep or dist.get_rank(group)!=shmem.rank():
                raise ValueError('SHMEM/HCCL EP group topology mismatch')
        validate_layout(ep,home,slots,tokens,topk)
        require_gm_binary(ep)
        if planner not in ('native','moonep','ultra_ep','echo','eplb_fixed','step_time') or (not unified and planner not in ('native','moonep')):
            raise ValueError('unknown balanced planner')
        if direct_permille and (planner != 'moonep' or ep != 8 or gm_planner_requested(ep)):
            raise ValueError('direct routing policy is an opt-in small EP8 MoonEP experiment')
        if backward_restore not in ('staged', 'expert', 'panel'):
            raise ValueError('unknown backward restore mode')
        control_tape_device = os.environ.get('WUHAN_CONTROL_TAPE_DEVICE', 'npu')
        if control_tape_device not in ('cpu', 'npu'):
            raise ValueError('WUHAN_CONTROL_TAPE_DEVICE must be cpu or npu')
        if planner == 'moonep' and slots != home:
            raise ValueError('MoonEP requires slots=home; no replica truncation')
        self.shmem, self.group, self.home, self.slots = shmem, group, int(home), int(slots)
        self.hidden, self.ffn, self.tokens, self.topk = hidden, ffn, tokens, topk
        self.window_bytes = int(window_bytes)
        self.capacity, self.dist = int(capacity), dist
        self.planner_name, self.backward_restore = planner, backward_restore
        self.control_tape_device = control_tape_device
        self.busy = self.failed = False
        self.module = executor_type(
            shmem, home, slots, hidden, ffn, window_bytes, device,
            save_gmm1_output=self.save_gmm1_output)
        m = self.module
        if routed_input_mode == 'redispatch':
            cap = getattr(m._worker_native, 'megaflux_routed_redispatch_version', None)
            if ep not in (2, 4, 8, 16, 32, 64) or cap is None or int(cap()) != 2:
                raise RuntimeError('routed redispatch requires worker capability2 with read-lease fence and EP<=64')
            if not callable(getattr(m.op, 'backward_replica_redispatch', None)):
                raise RuntimeError('routed redispatch requires a matching rebuilt Host binding')
        expected_mode = {'staged': 0, 'expert': 1, 'panel': 2}[backward_restore]
        if m.backward_restore_mode != expected_mode:
            raise RuntimeError('backward restore selection does not match the isolated native SO')
        _require_panel_ready(backward_restore, m)
        if (m.worker_flags, m.worker_shared_prefix, m.worker_external_refresh,
                m.worker_owner_rows, m.worker_owner_compact) != (0, 0, 0, 16, 0):
            raise RuntimeError('integration requires S9 control, all worker features off, owner rows16')
        if unified:
            from .unified_planner import CommonOnlinePlanner
            self.policy=CommonOnlinePlanner(name=planner,home=home,slots=slots,tokens=tokens,topk=topk,device=device,group=group,cost_model=cost_model)
        else:
            self.policy = (NativeOnlinePlanner(768, 2, 4) if planner == 'native' else
                MoonEPPlanner(home=home, slots=slots, tokens=tokens, topk=topk, device=device, group=group, validation=plan_validation, direct_permille=direct_permille, timer=timer))
        self.unified_planner=unified
        self._plan_bodies={}
        self._plan_backward={}
        self.replay_audit=dict(original=0,recompute=0,backward=0)
        self.enable_diagnostics = bool(enable_diagnostics)
        self.stats = (dict(forwards=0, backwards=0, home_refresh_bytes=0, backward_restores=0)
                      if self.enable_diagnostics else None)
        self.audit = ([] if self.enable_diagnostics and os.environ.get('WUHAN_POOL_AUDIT', '0') == '1'
                      else None)
        self.capture = None
        self.shared_timer_enabled = bool(timer)
        self.last_shared_timer = None
        self.last_shared_native_timer = None
        self.last_shared_backward_timer = None
        self.last_x_redispatch_backward_timer = None
        # Explicitly opt-in, untimed diagnostic only. The callback must retain
        # hashes/metadata rather than device payloads and is never enabled by default.
        self.x_redispatch_debug_sink = None
        if self.enable_diagnostics and os.environ.get('WUHAN_REPLAY_CAPTURE_DIR'):
            from .wuhan_replay_capture import WuhanReplayCapture
            self.capture = WuhanReplayCapture(os.environ['WUHAN_REPLAY_CAPTURE_DIR'])

    @property
    def save_gmm1_output(self):
        return self._save_gmm1_output

    def _enter(self):
        if self.busy or self.failed:
            raise RuntimeError('failed or concurrently used FSDP2 executor pool')
        self.busy = True

    def _record_native_inputs(self, *values):
        # Raw shared descriptor pointers are not ATen operands. Keep side-stream
        # FSDP/UAM allocations alive through the consuming execution stream.
        tensors=[v for v in values if torch.is_tensor(v) and v.device.type=='npu']
        if tensors:
            stream=torch.npu.current_stream(tensors[0].device)
            for value in tensors:value.record_stream(stream)

    def _defer_completion(self, plan, header=None):
        from .deferred_completion import enqueue
        self.module._check_device_plan(plan)
        if header is None:header=plan.header.detach().clone()
        enqueue(header,plan.epoch,plan.max_recv,self)
        self.module.release_pending()
        return header

    def _publish(self, w1, w2):
        if tuple(w1.shape) != (self.home, 2*self.ffn, self.hidden) or tuple(w2.shape) != (self.home, self.hidden, self.ffn):
            raise ValueError('expected unsharded EP-local expert weights')
        with torch.no_grad():
            self.module.w1.copy_(w1.detach())
            self.module.w2.copy_(w2.detach())
        if self.stats is not None:
            self.stats['home_refresh_bytes'] += (w1.numel() + w2.numel()) * 2

    def forward(self, x, ids, prob, w1, w2, limit):
        self._enter()
        try:
            if x.shape != (self.tokens, self.hidden) or ids.shape != (self.tokens, self.topk):
                raise ValueError('FSDP2 pool requires its configured static input shape')
            self._publish(w1, w2)
            m = self.module
            from .plan_replay import invocation
            request=invocation() if self.unified_planner else None
            if request is not None:
                key,recompute=request
                if recompute:
                    if key not in self._plan_bodies:raise RuntimeError('checkpoint requested a missing/stale plan body')
                    body=self._plan_bodies[key]
                    body.replay()
                    plan=m.create_plan(body.logical_ids,body,self.capacity)
                    self.replay_audit['recompute']+=1
                else:
                    if key in self._plan_bodies:raise RuntimeError('original forward submitted a duplicate plan body')
                    plan=m.create_plan(ids,self.policy,self.capacity)
                    from .unified_planner import ExecutionPlanBody
                    from .plan_replay import PlanTicket
                    body=PlanTicket(key,ExecutionPlanBody.freeze(plan,plan.source_quota,self.home,self.slots,self.capacity))
                    self._plan_bodies[key]=body
                    self._plan_backward[key]=0
                    self.replay_audit['original']+=1
            else:
                plan = m.create_plan(ids, self.policy, self.capacity)
            plan._consumed = True
            y, _, counts, state, dispatch = m._forward_device(
                x, prob.to(torch.bfloat16).contiguous(), plan, limit)
            lifetime=getattr(m,'lifetime',None)
            if lifetime is not None:lifetime.commit(plan)
            # Freeze planner scratch on this stream before the next layer reuses it.
            placement = (tuple(getattr(body,n) for n in _PLAN_TENSORS) if request is not None else
                         tuple(getattr(plan, n).detach().clone() for n in _PLAN_TENSORS))
            header = plan.header.detach().clone()
            if getattr(self,'defer_completion_checks',False):self._defer_completion(plan,header)
            else:m.release_pending()
            if self.audit is not None:
                status = header.detach().cpu().tolist()
                self.audit.append(dict(status=status, placement={name: dict(
                    shape=list(value.shape), sha256=hashlib.sha256(
                        value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
                    for name, value in zip(_PLAN_TENSORS, placement)},
                    copies=placement[2].cpu().tolist(), execution_counts=counts.cpu().tolist()))
            if self.stats is not None:
                self.stats['forwards'] += 1
            return y, header, (counts, state, dispatch, *placement)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def forward_shared(self, x, ids, prob, w1, w2, weight_gate, weight_up,
                       weight_down, limit, shared_save_mode='recompute', routed_input_mode='saved'):
        """One native fused forward, with hook-visible shared saved planes."""
        from .shared_expert import (SHARED_FORWARD_ABI, validate_shared_inputs,
                                    validate_shared_save_mode, shared_saved_state,
                                    prepare_shared_timer_capture, validate_routed_input_mode,
                                    SHARED_REDISPATCH_BACKWARD_ABI)
        from .shared_expert import require_shared_saved_mode
        save_mode = resolve_save_gmm1_output(self.save_gmm1_output, limit)
        require_shared_saved_mode(self.module, save_mode)
        validate_shared_save_mode(shared_save_mode)
        routed_code = validate_routed_input_mode(routed_input_mode)
        validate_shared_inputs(x, weight_gate, weight_up, weight_down, limit)
        ep = self.shmem.world_size()
        cap = getattr(self.module._worker_native, 'megaflux_shared_max_ep', None)
        if ep not in (2, 4, 8, 16, 32, 64) or ep > (int(cap()) if cap else 8):
            raise ValueError('shared EP exceeds selected worker capability')
        abi = getattr(self.module._worker_native, 'megaflux_shared_expert_forward_abi', None)
        if abi is None or int(abi()) != SHARED_FORWARD_ABI:
            raise RuntimeError('shared forward requires a rebuilt shared-enabled S9 ABI1 worker')
        if routed_code:
            capability = getattr(self.module._worker_native, 'megaflux_shared_expert_backward_v3_version', None)
            method = 'backward_replica_shared_v3' + ('_with_timer' if self.shared_timer_enabled else '')
            if (capability is None or int(capability()) != SHARED_REDISPATCH_BACKWARD_ABI
                    or not callable(getattr(self.module.op, method, None))):
                raise RuntimeError('kernel X redispatch requires matching rebuilt ABI3 Host and worker')
        self._enter()
        try:
            if x.shape != (self.tokens, self.hidden) or ids.shape != (self.tokens, self.topk):
                raise ValueError('FSDP2 pool requires its configured static input shape')
            with control_timing_scope('forward.publish'):
                self._publish(w1, w2)
            self._record_native_inputs(x,prob,weight_gate,weight_up,weight_down)
            m = self.module
            with control_timing_scope('forward.plan'):
                plan = m.create_plan(ids, self.policy, self.capacity)
            plan._consumed = True
            with control_timing_scope('forward.prepare'):
                m._check_device_plan(plan)
            self.last_shared_timer = self.last_shared_native_timer = None
            if self.shared_timer_enabled:
                with control_timing_scope('forward.timer_zero'):
                    prepare_shared_timer_capture(m, plan, x)
                with control_timing_scope('forward.native_launch'):
                    outputs = m.op.compute_train_replica_shared_with_timer(
                        self.shmem, x, plan.physical_ids,
                        prob.to(torch.bfloat16).contiguous(), m.w1, m.w2,
                        plan.max_recv, float(limit), save_mode, m._descriptor(plan),
                        weight_gate, weight_up, weight_down)
                outputs, native_timer = outputs[:-1], outputs[-1]
                self.last_shared_native_timer = native_timer
                m.trace_timers['forward'] = native_timer
            else:
                with control_timing_scope('forward.native_launch'):
                    outputs = m.op.compute_train_replica_shared(
                        self.shmem, x, plan.physical_ids,
                        prob.to(torch.bfloat16).contiguous(), m.w1, m.w2,
                        plan.max_recv, float(limit), save_mode, m._descriptor(plan),
                        weight_gate, weight_up, weight_down)
            y, counts, state, dispatch, _scratch, raw_gate, raw_up, activation, _ready, shared_timer = outputs
            m.last_exec_counts = counts
            with control_timing_scope('forward.placement_snapshot'):
                placement = tuple(getattr(plan, name).detach().clone() for name in _PLAN_TENSORS)
            with control_timing_scope('forward.finish'):
                status = self._defer_completion(plan) if getattr(self,'defer_completion_checks',False) else m.finish()
            if self.shared_timer_enabled:
                self.last_shared_timer = shared_timer
                schema = getattr(m._worker_native, 'megaflux_shared_forward_timer_schema', None)
                self.last_shared_timer_geometry = dict(
                    tokens=x.shape[0], hidden=x.shape[1], ffn=weight_gate.shape[0],
                    schema=int(schema()) if schema else 1,
                    tile_enabled=os.environ.get('MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE', '0') == '1')
            if self.stats is not None:
                self.stats['forwards'] += 1
            if routed_code:
                sink = self.x_redispatch_debug_sink
                if sink is not None:
                    sink('forward', dispatch, counts, self.hidden, self.capacity, int(status[2]))
                if self.stats is not None:
                    self.stats['forward_dispatch_bytes_dropped'] = (
                        self.stats.get('forward_dispatch_bytes_dropped', 0) + dispatch.numel())
                dispatch = torch.empty(0, dtype=torch.uint8, device=x.device)
            saved = (counts, state, dispatch, *placement)
            return y, status, saved, shared_saved_state(
                x, raw_gate, raw_up, activation, shared_save_mode, routed_input_mode)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def backward(self, grad, prob, w1, w2, status, placement, counts, state, dispatch, limit,
                 *, save_gmm1_output=None, original_x=None):
        # The autograd context supplies its own resolved forward format.  Auto
        # may resolve differently for other layers sharing this serial pool.
        save_mode = resolve_save_gmm1_output(
            self.save_gmm1_output if save_gmm1_output is None else save_gmm1_output, limit)
        self._enter()
        try:
            self._publish(w1, w2)
            m = self.module
            replay = _SavedPlacementReplay(self.home, self.slots, self.capacity, status, placement)
            plan = m.create_plan(placement[0], replay, self.capacity)
            if self.backward_restore == 'staged':
                m.workspace.restore_weights(self.shmem, plan.copies)
            if self.stats is not None:
                self.stats['backward_restores'] += 1
            if original_x is None:
                dx, dp, dw1, dw2 = m.op.backward_replica(self.shmem, grad.contiguous(),
                    prob.to(torch.bfloat16).contiguous(),
                    m.w1, m.w2, counts, state, dispatch, self.capacity, limit, save_mode, m._descriptor(plan))
            else:
                if dispatch.numel() != 0:
                    raise ValueError('redispatch tape must not retain packed dispatch')
                self._record_native_inputs(grad, prob, counts, state, original_x)
                timer = getattr(self, 'shared_timer_enabled', False)
                native = getattr(m.op, 'backward_replica_redispatch' + ('_with_timer' if timer else ''))
                outputs = native(self.shmem, grad.contiguous(), prob.to(torch.bfloat16).contiguous(),
                    m.w1, m.w2, counts, state, self.capacity, limit, save_mode, m._descriptor(plan), original_x)
                if timer:
                    outputs, native_timer, x_timer = outputs[:-2], outputs[-2], outputs[-1]
                    m.trace_timers['backward'] = native_timer
                    self.last_x_redispatch_backward_timer = x_timer
                dx, dp, dw1, dw2, rebuilt = outputs
                sink = getattr(self, 'x_redispatch_debug_sink', None)
                if sink is not None:
                    sink('backward', rebuilt, counts, self.hidden, self.capacity, int(status[2]))
            m.last_exec_counts = counts
            if self.unified_planner:
                for key,body in self._plan_bodies.items():
                    if placement[0].data_ptr()==body.logical_ids.data_ptr():
                        self._plan_backward[key]+=1
                        if self._plan_backward[key]!=1:raise RuntimeError('plan body received multiple backwards')
                        self.replay_audit['backward']+=1
            if getattr(self,'defer_completion_checks',False):self._defer_completion(plan)
            else:m.release_pending()
            if self.stats is not None:
                self.stats['backwards'] += 1
            # Native allocates fresh torch-owned dW outputs per call. A home view
            # retains that storage until its consumer retires; it never aliases
            # pool gradient scratch. The old clone remains an ablation switch.
            lifetime=getattr(m,'lifetime',None)
            if lifetime is not None:
                lifetime.gradients(plan,dw1,dw2)
                return dx,dp,None,None
            return dx, dp, _owner_gradient(dw1, self.home), _owner_gradient(dw2, self.home)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def backward_shared(self, grad, prob, w1, w2, status, placement, counts, state,
                        dispatch, limit, x, weight_gate, weight_up, weight_down,
                        raw_gate, raw_up, activation, shared_save_mode='recompute',
                        routed_input_mode='saved', *, save_gmm1_output=None):
        """Replay routed placement once, then launch native shared/routed Bwd V2.

        Recompute is internal to this launch and only reconstructs shared GU/A;
        it never regenerates the routed forward or its communication plan.
        """
        from .shared_expert import (SHARED_BACKWARD_ABI, SHARED_REDISPATCH_BACKWARD_ABI,
                                    validate_shared_inputs, validate_shared_save_mode,
                                    validate_routed_input_mode)
        from .shared_expert import require_shared_saved_mode
        save_mode = resolve_save_gmm1_output(
            self.save_gmm1_output if save_gmm1_output is None else save_gmm1_output, limit)
        require_shared_saved_mode(self.module, save_mode, backward=True)
        code = validate_shared_save_mode(shared_save_mode)
        routed_code = validate_routed_input_mode(routed_input_mode)
        tokens, _, ffn = validate_shared_inputs(x, weight_gate, weight_up, weight_down, limit)
        for value in (raw_gate, raw_up, activation):
            if (value.dtype != x.dtype or value.device != x.device or not value.is_contiguous()
                    or (tuple(value.shape) != (tokens, ffn) if code else value.numel() != 0)):
                raise ValueError('shared backward requires kept [M,Fs] planes or three empty recompute tensors')
        if self.backward_restore != 'panel':
            raise RuntimeError('shared backward V2 requires panel restore')
        m = self.module
        abi_name = 'megaflux_shared_expert_backward_v3_version' if routed_code else 'megaflux_shared_expert_backward_v2_version'
        abi = getattr(m._worker_native, abi_name, None)
        timer = getattr(self, 'shared_timer_enabled', False)
        method = ('backward_replica_shared_v3' if routed_code else 'backward_replica_shared_v2') + ('_with_timer' if timer else '')
        native = getattr(m.op, method, None)
        expected_abi = SHARED_REDISPATCH_BACKWARD_ABI if routed_code else SHARED_BACKWARD_ABI
        if abi is None or int(abi()) != expected_abi or not callable(native):
            if routed_code:
                raise RuntimeError('kernel X redispatch requires matching S9 V3 worker and Host')
            raise RuntimeError('shared backward requires a rebuilt shared-enabled S9 V2 worker and host')
        if routed_code and (dispatch.dtype != torch.uint8 or dispatch.numel() != 0):
            raise ValueError('kernel X redispatch must consume an empty dispatch tape slot')
        self._enter()
        try:
            with control_timing_scope('backward.publish'):
                self._publish(w1, w2)
            self._record_native_inputs(grad,prob,counts,state,dispatch,x,weight_gate,weight_up,weight_down,raw_gate,raw_up,activation)
            replay = _SavedPlacementReplay(self.home, self.slots, self.capacity, status, placement)
            with control_timing_scope('backward.replay'):
                plan = m.create_plan(placement[0], replay, self.capacity)
            if self.stats is not None:
                self.stats['backward_restores'] += 1
            self.last_shared_backward_timer = None
            self.last_x_redispatch_backward_timer = None
            with control_timing_scope('backward.native_launch'):
                outputs = native(
                    self.shmem, grad.contiguous(), prob.to(torch.bfloat16).contiguous(),
                    m.w1, m.w2, counts, state, dispatch, self.capacity, float(limit),
                    save_mode, m._descriptor(plan), x, weight_gate, weight_up, weight_down,
                    raw_gate, raw_up, activation, float(limit), not bool(code))
            if timer:
                if routed_code:
                    outputs, native_timer, shared_timer, x_timer = outputs[:-3], outputs[-3], outputs[-2], outputs[-1]
                    self.last_x_redispatch_backward_timer = x_timer
                else:
                    outputs, native_timer, shared_timer = outputs[:-2], outputs[-2], outputs[-1]
                m.trace_timers['backward'] = native_timer
                self.last_shared_backward_timer = shared_timer
            rebuilt_dispatch = None
            if routed_code:
                outputs, rebuilt_dispatch = outputs[:-1], outputs[-1]
            dx, dp, dw1, dw2, dwg, dwu, dwd = outputs
            m.last_exec_counts = counts
            with control_timing_scope('backward.finish'):
                if getattr(self,'defer_completion_checks',False):self._defer_completion(plan)
                else:m.finish()
            if routed_code:
                if self.stats is not None:
                    self.stats['native_x_redispatch_backwards'] = self.stats.get('native_x_redispatch_backwards', 0) + 1
                sink = self.x_redispatch_debug_sink
                if sink is not None:
                    sink('backward', rebuilt_dispatch, counts, self.hidden, self.capacity, int(status[2]))
            if self.stats is not None:
                self.stats['backwards'] += 1
            # Native already performs owner SUM and shared dX addition. Shared
            # dW stays on the original ordinary-FSDP Parameters/mesh.
            lifetime=getattr(m,'lifetime',None)
            if lifetime is not None:
                lifetime.gradients(plan,dw1,dw2)
                return dx,dp,None,None
            return dx, dp, _owner_gradient(dw1, self.home), _owner_gradient(dw2, self.home), dwg, dwu, dwd
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False

    def retire_training_step(self,step):
        keys=[k for k in self._plan_bodies if k[0]==step]
        for key in keys:
            if self._plan_backward[key]!=1:raise RuntimeError('cannot retire a plan before exactly one backward')
            self._plan_bodies[key].retire()
            del self._plan_bodies[key]
            del self._plan_backward[key]
        return len(keys)

    def close(self):
        if self.busy or self.failed:
            raise RuntimeError('cannot close an active/failed native pool as successful')
        if self._plan_bodies:raise RuntimeError('cannot close with live training plan bodies')
        if getattr(self,'defer_completion_checks',False):
            from .deferred_completion import pending_for
            if pending_for(self):
                raise RuntimeError('flush deferred statuses before closing the pool')
        self.module.close()
