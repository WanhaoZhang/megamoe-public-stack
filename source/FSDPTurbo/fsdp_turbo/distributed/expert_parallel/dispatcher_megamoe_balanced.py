"""S9 feature dispatcher. Ordinary FSDP owns Parameters and its reductions.

The author MoonEP runtime is deliberately not initialized by this dispatcher.
Both planner policies use one ascend_deepep runtime and the same executor SO.
"""
import threading

import torch
from torch.distributed.tensor import DTensor

from .dispatcher_mega_moe import _ensure_mega_moe_runtime, _local_mem_bytes
from .megamoe_balanced_diag import megamoe_balanced_diagnostics_enabled
from .utils import fixed_router_for_debug, normalize_expert_args


_POOLS = {}
_STEP_COMPLETION_BOUNDARY = False
_LOCK = threading.Lock()


def _local(weight):
    return weight.to_local() if isinstance(weight, DTensor) else weight


def _read_expert_parameters(module, convert):
    """Read live Parameters once; shape inspection never needs a dtype cast."""
    w1, w2 = _local(module.gate_up_proj), _local(module.down_proj)
    if convert:
        if w1.dtype != torch.bfloat16:
            w1 = w1.to(torch.bfloat16)
        if w2.dtype != torch.bfloat16:
            w2 = w2.to(torch.bfloat16)
    return w1, w2


def get_megamoe_balanced_forward_fn(ep_group, fixed_router=False, config=None):
    from fsdp_turbo.fsdp_turbo_config import MegaMoeBalancedConfig
    config = config or MegaMoeBalancedConfig()

    def balanced_forward(self, hidden_states, top_k_index, top_k_weights, *,
                         shared_expert_weights=None, shared_expert_limit=None,
                         shared_expert_weight_provider=None):
        from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge, NativeFSDP2Pool
        ids, prob = normalize_expert_args(top_k_index, top_k_weights)
        ep = torch.distributed.get_world_size(ep_group)
        if config.shared_expert_mode == 'fused':
            if ep not in (2, 4, 8, 16, 32, 64):
                raise ValueError('fused shared experts require EP2/4/8/16/32/64 and a matching capable worker')
            if shared_expert_weights is None or shared_expert_limit is None:
                raise ValueError('fused shared experts require explicit Wg/Wu/Wd and a positive limit')
            if len(shared_expert_weights) != 3:
                raise ValueError('fused shared experts require exactly Wg/Wu/Wd')
            if not callable(shared_expert_weight_provider):
                raise ValueError('fused shared experts require a live shared owner callback')
            shared_limit = float(shared_expert_limit)
            if not 0. < shared_limit < float('inf'):
                raise ValueError('fused shared experts require a finite positive SwiGLU limit')
        elif (shared_expert_weights is not None or shared_expert_limit is not None
              or shared_expert_weight_provider is not None):
            raise ValueError('shared expert parameters require shared_expert_mode=fused')
        if ep not in (2,4,8,16,32,64,128,256):
            raise ValueError('megamoe_balanced requires EP2/4/8/16/32/64/128/256')
        if self.num_global_experts%ep:
            raise ValueError('uniform expert owners require num_global_experts divisible by EP')
        if ep>8 and config.recv_capacity is None:
            raise ValueError('large EP requires explicit aligned recv_capacity; no unbounded worst-case allocation')
        if ids.ndim == 1:
            ids, prob = ids.unsqueeze(-1), prob.unsqueeze(-1)
        if fixed_router:
            ids, prob = fixed_router_for_debug(ids, prob, self.num_global_experts, ep_size=ep)
        shape = hidden_states.shape
        hidden = int(self.hidden_dim)
        x = hidden_states.reshape(-1, hidden).contiguous()
        if x.dtype != torch.bfloat16:
            raise ValueError('megamoe_balanced expects BF16 expert activations')
        ids = ids.reshape(x.shape[0], -1).to(torch.int32).contiguous()
        # v4lab: the fp32 router path yields fp32 probabilities; the native pool
        # consumes BF16 weights like the raw MegaMoE dispatcher.
        prob = prob.reshape_as(ids).to(torch.bfloat16).contiguous()
        tokens, topk = ids.shape
        # Read attributes every invocation, after FSDP unshard. Never capture
        # a pre-wrap Parameter or a pooled replica pointer in this closure.
        def weights():
            return _read_expert_parameters(self, config.parameter_read_mode != 'fp32_to_home')

        # Preserve the old duplicate cast as an explicit causal control. The
        # bridge calls weights() again for its actual autograd inputs. In the
        # direct mode, the existing native home copy converts FP32 to BF16.
        w1, w2 = (weights() if config.parameter_read_mode == 'legacy'
                  else _read_expert_parameters(self, False))
        home, out, actual_hidden = w1.shape
        ffn = out // 2
        if actual_hidden != hidden or tuple(w2.shape) != (home, hidden, ffn):
            raise ValueError('unsupported expert Parameter layout')
        if config.shared_expert_mode == 'fused':
            # Resolve live FSDP weights on every call; never retain a shared
            # Parameter or data_ptr in a closure, module, or reusable pool.
            wg, wu, wd = (_local(w) for w in shared_expert_weights)
            # v4lab: public parent FSDP2 may promote shared MLP params to fp32.
            wg, wu, wd = tuple(
                w.to(torch.bfloat16) if w.dtype != torch.bfloat16 else w
                for w in (wg, wu, wd)
            )
            if any(w.dtype != torch.bfloat16 for w in (wg, wu, wd)):
                raise ValueError('fused shared experts require BF16 weights')
            if any(w.device != x.device for w in (wg, wu, wd)):
                raise ValueError('fused shared expert weights must be on the activation device')
            if wg.ndim != 2 or tuple(wu.shape) != tuple(wg.shape):
                raise ValueError('fused shared Wg/Wu require matching [F,H] matrices')
            shared_ffn, shared_hidden = wg.shape
            if shared_hidden != hidden or tuple(wd.shape) != (hidden, shared_ffn):
                raise ValueError('unsupported fused shared expert Parameter layout')
            if shared_ffn != ffn:
                raise ValueError('the initial fused shared kernel requires shared F equal to routed F')
            if any(not w.is_contiguous() for w in (wg, wu, wd)):
                raise ValueError('fused shared expert weights must be contiguous after FSDP unshard')
            limit = getattr(self, 'limit', 0.)
            if limit is None or shared_limit != float(limit):
                raise ValueError('the initial fused shared kernel requires the routed SwiGLU limit')
            if not callable(getattr(FSDP2ExpertsBridge, 'forward_shared', None)):
                raise RuntimeError('fused shared experts require the matching ascend_deepep forward_shared bridge')
        slots = home if config.replica_slots is None else int(config.replica_slots)
        from ascend_deepep.runtime_limits import validate_layout
        validate_layout(ep,home,slots,tokens,topk)
        from ascend_deepep.runtime_limits import require_runtime_capacity
        require_runtime_capacity()
        if config.planner == 'moonep' and slots != home:
            raise ValueError('MoonEP balanced planner requires replica_slots=home')
        capacity = tokens * topk if config.planner == 'moonep' else tokens * topk * ep
        capacity = (capacity + 127) // 128 * 128
        if config.recv_capacity is not None:
            # Explicit bounded-workload allocation. Native collective plan
            # validation still rejects routes that exceed this capacity.
            capacity = int(config.recv_capacity)
            if capacity < (tokens * topk + 127) // 128 * 128:
                raise ValueError('recv_capacity is smaller than the EP average route load')
        window = _local_mem_bytes(tokens, hidden, topk, ep, home + slots, capacity)
        # Split S9 storage: homes + replicas + BF16 replica partials; include
        # generous control/ready padding. This is one pool, not per-layer HBM.
        replica_storage = (home + 2*slots) * 3*ffn*hidden*2 + 32*1024*1024
        # One executor window plus its split symmetric weights and gradients.
        # The legacy token-window ring spare must not duplicate the replica bank.
        heap_bytes = window + replica_storage + 256*1024*1024
        key = (id(ep_group), home, slots, hidden, ffn, tokens, topk,
               config.planner, config.backward_restore, capacity, config.plan_validation,
               config.shared_expert_mode, config.shared_expert_save_mode, config.routed_input_mode,
               config.save_gmm1_output, config.shared_backward_schedule,config.timer,config.step_time_model,
               config.checkpoint_home_reuse,config.parameter_read_mode)
        with _LOCK:
            pool = _POOLS.get(key)
            if pool is None:
                if _POOLS:
                    raise RuntimeError('balanced integration requires one static expert shape/policy/save mode per process')
                # A mixed save mode changes both the forward tape and the
                # backward schedule. Check every EP size before SHMEM init;
                # this collective runs once for the static process pool.
                signature=(home,slots,hidden,ffn,tokens,topk,capacity,config.planner,config.timer,
                           config.backward_restore,config.plan_validation,
                           config.shared_expert_mode,config.shared_expert_save_mode,config.routed_input_mode,config.save_gmm1_output,
                           config.shared_backward_schedule,config.checkpoint_home_reuse,config.parameter_read_mode)
                peers=[None]*ep
                torch.distributed.all_gather_object(peers,signature,group=ep_group)
                if any(p!=signature for p in peers):
                    raise ValueError('inconsistent balanced EP shape/policy/capacity/save mode before SHMEM initialization')
                if ep>8:
                    from .balanced_bootstrap import resolve_balanced_bootstrap
                    endpoint=resolve_balanced_bootstrap(ep_group,torch.distributed)
                    shmem = _ensure_mega_moe_runtime(ep_group, window_pool_size=1,
                        window_bytes=window + replica_storage,bootstrap_ip_port=endpoint,total_bytes=heap_bytes)
                else:
                    shmem = _ensure_mega_moe_runtime(ep_group, window_pool_size=1,
                        window_bytes=window + replica_storage,total_bytes=heap_bytes)
                diagnostics = megamoe_balanced_diagnostics_enabled(config)
                pool = NativeFSDP2Pool(shmem, ep_group, home, slots, hidden, ffn,
                    tokens, topk, window, capacity, x.device, config.planner, config.backward_restore,
                    config.plan_validation, timer=config.timer, cost_model=config.step_time_model,enable_diagnostics=diagnostics,
                    save_gmm1_output=config.save_gmm1_output,
                    defer_completion_checks=_STEP_COMPLETION_BOUNDARY,
                    routed_input_mode=config.routed_input_mode,
                    **({'checkpoint_home_reuse': True} if config.checkpoint_home_reuse else {}),
                    **({'shared_backward_schedule': config.shared_backward_schedule}
                       if config.shared_expert_mode == 'fused' else {}))
                _POOLS[key] = pool
                pool.parameter_read_mode = config.parameter_read_mode
                pool.master_parameter_dtypes = tuple(str(_local(w).dtype)
                    for w in (self.gate_up_proj, self.down_proj))
                if diagnostics:
                    print(f'[MegaMoeBalanced] planner={config.planner} slots={slots} '
                          f'restore={config.backward_restore} home={home} shape={tokens,hidden,ffn,topk} '
                          f'max_recv={capacity} validation={config.plan_validation} '
                          f'save={config.save_gmm1_output} worker_flags=0 owner_rows=16', flush=True)
        bridge = getattr(self, '_megamoe_balanced_bridge', None)
        if bridge is None:
            bridge = FSDP2ExpertsBridge(pool, weights)
            self._megamoe_balanced_bridge = bridge
        elif bridge.pool is not pool:
            raise RuntimeError('expert module changed its balanced execution configuration')
        limit = getattr(self, 'limit', 0.)
        if config.shared_expert_mode == 'fused':
            def live_shared_weights():
                # The incoming callback weakly refers to the original owner;
                # do not close over this forward's local wg/wu/wd tensors.
                out = []
                for weight in shared_expert_weight_provider():
                    tensor = _local(weight)
                    if tensor.dtype != torch.bfloat16:
                        tensor = tensor.to(torch.bfloat16)
                    out.append(tensor)
                return tuple(out)
            return bridge.forward_shared(x, ids, prob, wg, wu, wd,
                limit=shared_limit, shared_save_mode=config.shared_expert_save_mode,
                shared_weight_provider=live_shared_weights,
                **({'routed_input_mode': config.routed_input_mode} if config.routed_input_mode != 'saved' else {})).view(shape)
        return bridge(x, ids, prob, 0. if limit is None else float(limit)).view(shape)

    # Bound methods preserve function attributes. Model integration can opt in
    # before its first call without storing shared Parameters on experts.
    balanced_forward._megamoe_balanced_shared_expert_mode = config.shared_expert_mode
    balanced_forward._megamoe_balanced_shared_expert_save_mode = config.shared_expert_save_mode
    return balanced_forward


def check_step_completions():
    """Mandatory before clipping/optimizer when this dispatcher owns pools."""
    if any(getattr(pool,'defer_completion_checks',False) for pool in _POOLS.values()):
        from ascend_deepep.deferred_completion import flush
        return flush()
    return 0


def enable_step_completion_boundary():
    """Only a trainer that flushes before optimizer may enable async status."""
    global _STEP_COMPLETION_BOUNDARY
    _STEP_COMPLETION_BOUNDARY=True
    for pool in _POOLS.values():pool.defer_completion_checks=True


def close_megamoe_balanced():
    """Release task pools and their SHMEM world before process-group teardown."""
    if not _POOLS:
        return
    check_step_completions()
    torch.accelerator.synchronize()
    for pool in _POOLS.values():
        pool.close()
    _POOLS.clear()
    from .dispatcher_mega_moe import _MEGA_MOE_STATE
    shmem = _MEGA_MOE_STATE['shmem']
    if shmem is not None:
        torch.distributed.barrier(group=_MEGA_MOE_STATE.get('ep_group'))
        shmem.destroy()
        _MEGA_MOE_STATE['shmem'] = None
        _MEGA_MOE_STATE.pop('ep_group', None)
