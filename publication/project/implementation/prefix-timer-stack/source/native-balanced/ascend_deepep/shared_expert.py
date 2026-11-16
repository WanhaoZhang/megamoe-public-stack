"""Contracts for opt-in fused shared experts and the V2 backward bridge.

This module never substitutes Python GEMMs or Add for the native shared path.
Shared backward is delegated to the matching native V2 worker; Python never
substitutes routed-only gradients or recomputes the routed forward.
"""
import math

import torch
from .saved_gate_up_mode import resolve_save_gmm1_output


SHARED_FORWARD_ABI = 1
SHARED_BACKWARD_ABI = 2
SHARED_REDISPATCH_BACKWARD_ABI = 3
_SAVE_MODES = {'recompute': 0, 'keep': 1}
_SHARED_ROUTED_SAVE_BITS = {'bf16': 1, 'none': 2, 'mxfp8': 4}


def require_shared_saved_mode(module, mode, *, backward=False):
    """Check the actual loaded worker; legacy shared images support BF16 only."""
    bit = _SHARED_ROUTED_SAVE_BITS.get(mode, 0)
    symbol = 'megaflux_shared_expert_' + ('backward' if backward else 'forward') + '_saved_modes'
    getter = getattr(module._worker_native, symbol, None)
    mask = int(getter()) if getter is not None else 1
    if not bit or not mask & bit:
        raise RuntimeError('shared routed saved format requires a matching rebuilt Host/worker: ' + str(mode))


def validate_routed_input_mode(mode):
    if mode not in ('saved', 'redispatch'):
        raise ValueError("routed_input_mode must be 'saved' or 'redispatch'")
    return int(mode == 'redispatch')


def validate_shared_save_mode(mode):
    if mode not in _SAVE_MODES:
        raise ValueError("shared_save_mode must be 'keep' or 'recompute'")
    return _SAVE_MODES[mode]


def validate_shared_inputs(x, weight_gate, weight_up, weight_down, limit=0.):
    """Validate the dense layout; the native boundary also requires NPU tensors."""
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError('shared expert requires contiguous BF16 x [M,H]')
    if not math.isfinite(float(limit)) or float(limit) < 0:
        raise ValueError('shared expert limit must be finite and nonnegative')
    for name, value in (('gate', weight_gate), ('up', weight_up), ('down', weight_down)):
        if (value.ndim != 2 or value.dtype != torch.bfloat16 or
                not value.is_contiguous() or value.device != x.device):
            raise ValueError(f'shared {name} weight must be contiguous rank-2 BF16 on x.device')
    tokens, hidden = x.shape
    ffn = weight_gate.shape[0]
    if (tokens <= 0 or hidden <= 0 or hidden % 256 or ffn <= 0 or ffn % 256 or
            tuple(weight_gate.shape) != (ffn, hidden) or
            tuple(weight_up.shape) != (ffn, hidden) or
            tuple(weight_down.shape) != (hidden, ffn)):
        raise ValueError('shared weights require [Fs,H], [Fs,H], [H,Fs] with H/Fs multiples of 256')
    return int(tokens), int(hidden), int(ffn)


def shared_saved_state(x, raw_gate, raw_up, activation, mode, routed_input_mode='saved'):
    """Expose saved tensors individually to checkpoint/UAM hooks; never clone X."""
    code = validate_shared_save_mode(mode)
    tokens, hidden = x.shape
    ffn = raw_gate.shape[1]
    routed_code = validate_routed_input_mode(routed_input_mode)
    metadata = [SHARED_FORWARD_ABI, code, tokens, hidden, ffn]
    if routed_code:
        metadata.append(routed_code)
    meta = torch.tensor(metadata,
                        dtype=torch.int32, device='cpu')
    if not code:
        raw_gate, raw_up, activation = (x.new_empty((0,)) for _ in range(3))
    return x, raw_gate, raw_up, activation, meta


def prepare_shared_timer_capture(module, plan, x):
    """Initialize the same coordinate Timer buffers used by the routed entry."""
    from .megaflux_stage_layout import MAGIC, PREFIX, HEADER, LANES, WORDS
    hidden, ffn = module.w1.shape[2], module.w1.shape[1] // 2
    ep = module.shmem.world_size()
    if module.stage_capacity != plan.max_recv:
        for phase, attr in (('forward', 'pipeline_trace'), ('backward', 'gradient_trace')):
            spec = module._stage_spec(ep, module.home, module.slots, hidden, ffn,
                                      plan.max_recv, phase)
            module.stage_layouts[phase] = spec
            size = PREFIX[phase] + HEADER + LANES * 8 + spec['total'] * WORDS
            tensor=torch.zeros(size,device=x.device,dtype=torch.int64)
            header=[MAGIC,ep,module.home,module.slots,hidden,ffn,plan.max_recv,0,spec['total'],spec['version']]+[0]*6
            tensor[PREFIX[phase]:PREFIX[phase]+HEADER].copy_(torch.tensor(header,device=x.device,dtype=torch.int64))
            setattr(module,attr,tensor)
        module.stage_capacity = plan.max_recv
    for phase, timer in (('forward', module.pipeline_trace), ('backward', module.gradient_trace)):
        offset=PREFIX[phase]
        timer[:offset].zero_();timer[offset+HEADER:].zero_()
        timer[offset+7].fill_(plan.epoch)


class _ExternalSharedExperts(torch.autograd.Function):
    """Forward argument order and saved tape are explicit, independent ABIs.

    apply(bridge, X, ids, prob, routedW1, routedW2, Wg, Wu, Wd, limit, mode)
    saves (prob, counts, state, dispatch, logical_ids, physical_ids, copies,
           logical_counts, control, X, rawG, rawU, A, shared_meta).

    control belongs to this successful forward's plan and travels through
    checkpoint/UAM saved-tensor hooks. Context attributes contain metadata and
    the bridge and an owner callback, never a cached shared/routed weight
    tensor. The callback reads the original shared module after FSDP unshard.
    """
    @staticmethod
    def forward(ctx, bridge, x, ids, prob, w1, w2, weight_gate, weight_up,
                weight_down, limit, shared_save_mode, routed_input_mode='saved'):
        validate_routed_input_mode(routed_input_mode)
        routed_kwargs = ({'routed_input_mode': routed_input_mode}
                         if routed_input_mode != 'saved' else {})
        routed_save = resolve_save_gmm1_output(
            getattr(bridge.pool, 'save_gmm1_output', 'bf16'), limit)
        y, status, routed_saved, shared_saved = bridge.pool.forward_shared(
            x, ids, prob, w1, w2, weight_gate, weight_up, weight_down,
            float(limit), shared_save_mode=shared_save_mode, **routed_kwargs)
        ctx.bridge = bridge
        ctx.shared_weight_provider = bridge.current_shared_weights
        ctx.limit = float(limit)
        ctx.save_gmm1_output = routed_save
        ctx.shared_save_mode = shared_save_mode
        ctx.routed_input_mode = routed_input_mode
        ctx.input_arity = len(getattr(ctx, 'needs_input_grad', ())) or (12 if routed_kwargs else 11)
        ctx.weight_meta = tuple((tuple(weight.shape), weight.dtype) for weight in
                                (w1, w2, weight_gate, weight_up, weight_down))
        ctx.prob_dtype = prob.dtype
        control_device = ('cpu' if getattr(bridge.pool, 'control_tape_device', 'npu') == 'cpu'
                          else prob.device)
        control = (status.detach().to(control_device) if torch.is_tensor(status)
                   else torch.tensor(status, device=control_device, dtype=torch.int32))
        # Preserve the explicitly selected save contract through standard hooks.
        # The control tensor must be restored with these exact saved routes;
        # an ordinary ctx.status tuple would survive a checkpoint replay with
        # its old epoch. No external weight pointers are pinned on this tape.
        ctx.save_for_backward(prob, *routed_saved, control, *shared_saved)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        from .control_timing import control_timing_scope
        (prob, counts, state, dispatch, logical_ids, physical_ids, copies,
         logical_counts, control, x, raw_gate, raw_up, activation, meta) = ctx.saved_tensors
        with control_timing_scope('backward.live_weights'):
            w1, w2 = ctx.bridge.current_weights()
            shared_weights = tuple(ctx.shared_weight_provider())
        if len(shared_weights) != 3:
            raise RuntimeError('live shared provider must return Wg/Wu/Wd')
        all_weights = (w1, w2, *shared_weights)
        for weight, (shape, dtype) in zip(all_weights, ctx.weight_meta):
            if tuple(weight.shape) != shape or weight.dtype != dtype:
                raise RuntimeError('FSDP backward unshard did not restore the original weight shape/dtype')
        wg, wu, wd = shared_weights
        validate_shared_inputs(x, wg, wu, wd, ctx.limit)
        expected = [SHARED_FORWARD_ABI, validate_shared_save_mode(ctx.shared_save_mode),
                    x.shape[0], x.shape[1], wg.shape[0]]
        routed_mode = getattr(ctx, 'routed_input_mode', 'saved')
        if validate_routed_input_mode(routed_mode):
            expected.append(1)
        if meta.dtype != torch.int32 or meta.numel() != len(expected) or meta.cpu().tolist() != expected:
            raise RuntimeError('shared saved-state metadata does not match this forward')
        placement = (logical_ids, physical_ids, copies, logical_counts)
        gradients = ctx.bridge.pool.backward_shared(
            grad, prob, w1, w2,
            control if getattr(ctx.bridge.pool,'defer_completion_checks',False) else control.cpu().tolist(),
            placement, counts,
            state, dispatch, ctx.limit, x, wg, wu, wd, raw_gate, raw_up,
            activation, shared_save_mode=ctx.shared_save_mode,
            save_gmm1_output=ctx.save_gmm1_output,
            **({'routed_input_mode': routed_mode} if routed_mode != 'saved' else {}))
        if len(gradients) != 7:
            raise RuntimeError('native shared backward must return exactly seven gradients')
        dx, dp, dw1, dw2, dwg, dwu, dwd = gradients
        # Native dx already contains routed-direct + shared dX. The Router
        # consumes dp through its own graph; neither contribution is added here.
        dweights = tuple(value.to(dtype) for value, (_, dtype) in
                         zip((dw1, dw2, dwg, dwu, dwd), ctx.weight_meta))
        result = (None, dx, None, dp.to(ctx.prob_dtype), *dweights, None, None)
        return result + (None,) if getattr(ctx, 'input_arity', 11) == 12 else result


def forward_shared(bridge, x, ids, prob, weight_gate, weight_up, weight_down,
                   limit=0., shared_save_mode='recompute', routed_input_mode='saved'):
    """Run fused forward and use live owner callbacks for the V2 backward."""
    validate_shared_save_mode(shared_save_mode)
    validate_routed_input_mode(routed_input_mode)
    validate_shared_inputs(x, weight_gate, weight_up, weight_down, limit)
    w1, w2 = bridge.current_weights()
    args = (x, prob, w1, w2, weight_gate, weight_up, weight_down)
    if torch.is_grad_enabled() and any(value.requires_grad for value in args):
        if not callable(getattr(bridge, 'current_shared_weights', None)):
            raise RuntimeError('grad-enabled shared fusion requires a live shared-weight provider')
        arguments = (bridge, x, ids, prob, w1, w2, weight_gate, weight_up, weight_down,
                     float(limit), shared_save_mode)
        return _ExternalSharedExperts.apply(
            *(arguments + (routed_input_mode,) if routed_input_mode != 'saved' else arguments))
    return bridge.pool.forward_shared(
        x, ids, prob, w1, w2, weight_gate, weight_up, weight_down,
        float(limit), shared_save_mode=shared_save_mode,
        **({'routed_input_mode': routed_input_mode} if routed_input_mode != 'saved' else {}))[0]
