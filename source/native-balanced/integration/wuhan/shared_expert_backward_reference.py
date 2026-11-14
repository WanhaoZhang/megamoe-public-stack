"""Independent, unfused BF16 shared-expert oracle for the S0 backward gate.

The graph uses PyTorch Linear, SiLU, clamp and autograd. Gate/up get separate
input leaves so each Linear's dX is rounded to BF16 before their BF16 sum.
This is the TP1, no-bias, asymmetric Triton-clamp contract, not routed MoE.
"""
import math


def require_active_replica(case, active, ep_group=None, iteration=None):
    """Validate each required case/epoch, never a flag accumulated by earlier cases."""
    required = case in ('hot', 'rotate', 'boundary')
    if required and not active:
        raise AssertionError(f'{case} did not exercise an active replica '
                             f'(EP group={ep_group}, iteration={iteration})')
    return required


def reference(x, wg, wu, wd, dy, limit):
    import torch
    import torch.nn.functional as functional

    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('shared limit must be positive and finite')
    xg, xu, gate_w, up_w, down_w = [
        value.detach().clone().requires_grad_(True)
        for value in (x, x, wg, wu, wd)]
    raw_g = functional.linear(xg, gate_w)
    raw_u = functional.linear(xu, up_w)
    # clamp derivatives include equality. In particular gate has no low clip.
    g = raw_g.float().clamp_max(limit)
    u = raw_u.float().clamp(min=-limit, max=limit)
    activation = (functional.silu(g) * u).to(torch.bfloat16)
    y = functional.linear(activation, down_w)
    dxg, dxu, dwg, dwu, dwd = torch.autograd.grad(
        y, (xg, xu, gate_w, up_w, down_w), dy)
    if any(t.dtype != torch.bfloat16 for t in (dxg, dxu, dwg, dwu, dwd)):
        raise AssertionError('oracle expects BF16 Linear inputs and gradients')
    dx_shared = (dxg.float() + dxu.float()).to(torch.bfloat16)
    return dict(raw_gu=torch.cat((raw_g, raw_u), dim=1).detach().contiguous(),
                activation=activation.detach().contiguous(), y=y.detach(),
                dx_shared=dx_shared.detach(), dwg=dwg.detach(),
                dwu=dwu.detach(), dwd=dwd.detach())


def boundary_inputs(tokens, hidden, shared_ffn, device, limit=1.):
    """Exact BF16 +/-limit and adjacent BF16 values, with an observable dX.

    Sparse identity projections make raw G/U equal the first two input
    columns. A large exact downstream gradient exposes boundary-mask errors.
    The graph above still generates all saved state and expected gradients.
    """
    import torch

    if limit != 1.:
        raise ValueError('the exact boundary fixture is qualified at limit=1')
    gate_values = [1., 1.0078125, .99609375, -1., -1.0078125, -.99609375, 0.]
    up_values = [1., 1.0078125, .99609375, -1., -1.0078125, -.99609375, .5]
    x = torch.zeros((tokens, hidden), device=device, dtype=torch.bfloat16)
    wg = torch.zeros((shared_ffn, hidden), device=device, dtype=torch.bfloat16)
    wu = torch.zeros_like(wg)
    wd = torch.zeros((hidden, shared_ffn), device=device, dtype=torch.bfloat16)
    x[:, 0] = torch.tensor([gate_values[i % len(gate_values)] for i in range(tokens)],
                           device=device, dtype=torch.bfloat16)
    x[:, 1] = torch.tensor([up_values[(i // len(gate_values)) % len(up_values)]
                           for i in range(tokens)], device=device, dtype=torch.bfloat16)
    wg[0, 0], wu[0, 1], wd[:, 0] = 1., 1., .25
    dy = torch.full_like(x, .125)
    return x, wg, wu, wd, dy


def routed_progress_range(tokens, topk, ep, physical_experts):
    """Only the original BF16 saved-state progress suffix is writable.

    TrainingSavedStateLayout(embedDispatchOutput=false), BF16 keep mode.
    The route metadata and appended raw G/U on both sides remain immutable.
    This excludes exactly 256 int32 counters, never the numerical payload.
    """
    align = lambda value, multiple: (value + multiple - 1) // multiple * multiple
    begin = align(ep * physical_experts * 4, 512)
    begin += 2 * align(tokens, 256) * topk * 4
    stride = align(ep * physical_experts, 128)
    begin += stride * 4 + align(ep * stride * 4, 512)
    return begin, begin + 256 * 4
