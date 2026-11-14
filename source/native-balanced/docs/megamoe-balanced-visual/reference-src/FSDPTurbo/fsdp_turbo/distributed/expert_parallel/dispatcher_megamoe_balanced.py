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
_LOCK = threading.Lock()


def _local(weight):
    return weight.to_local() if isinstance(weight, DTensor) else weight


def get_megamoe_balanced_forward_fn(ep_group, fixed_router=False, config=None):
    from fsdp_turbo.fsdp_turbo_config import MegaMoeBalancedConfig
    config = config or MegaMoeBalancedConfig()

    def balanced_forward(self, hidden_states, top_k_index, top_k_weights):
        from ascend_deepep.fsdp2_bridge import FSDP2ExpertsBridge, NativeFSDP2Pool
        ids, prob = normalize_expert_args(top_k_index, top_k_weights)
        ep = torch.distributed.get_world_size(ep_group)
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
        prob = prob.reshape_as(ids).contiguous()
        tokens, topk = ids.shape
        # Read attributes every invocation, after FSDP unshard. Never capture
        # a pre-wrap Parameter or a pooled replica pointer in this closure.
        weights = lambda: (_local(self.gate_up_proj), _local(self.down_proj))
        w1, w2 = weights()
        home, out, actual_hidden = w1.shape
        ffn = out // 2
        if actual_hidden != hidden or tuple(w2.shape) != (home, hidden, ffn):
            raise ValueError('unsupported expert Parameter layout')
        slots = home if config.replica_slots is None else int(config.replica_slots)
        from ascend_deepep.runtime_limits import validate_layout
        validate_layout(ep,home,slots,tokens,topk)
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
        key = (id(ep_group), home, slots, hidden, ffn, tokens, topk,
               config.planner, config.backward_restore, capacity, config.plan_validation,
               config.save_gmm1_output)
        with _LOCK:
            pool = _POOLS.get(key)
            if pool is None:
                if _POOLS:
                    raise RuntimeError('balanced integration requires one static expert shape/policy/save mode per process')
                # A mixed save mode changes both the forward tape and the
                # backward schedule. Check every EP size before SHMEM init;
                # this collective runs once for the static process pool.
                signature=(home,slots,hidden,ffn,tokens,topk,capacity,config.planner,
                           config.backward_restore,config.plan_validation,config.save_gmm1_output)
                peers=[None]*ep
                torch.distributed.all_gather_object(peers,signature,group=ep_group)
                if any(p!=signature for p in peers):
                    raise ValueError('inconsistent balanced EP shape/policy/capacity/save mode before SHMEM initialization')
                if ep>8:
                    from .balanced_bootstrap import resolve_balanced_bootstrap
                    endpoint=resolve_balanced_bootstrap(ep_group,torch.distributed)
                    shmem = _ensure_mega_moe_runtime(ep_group, window_pool_size=1,
                        window_bytes=window + replica_storage,bootstrap_ip_port=endpoint)
                else:
                    shmem = _ensure_mega_moe_runtime(ep_group, window_pool_size=1,
                        window_bytes=window + replica_storage)
                diagnostics = megamoe_balanced_diagnostics_enabled(config)
                pool = NativeFSDP2Pool(shmem, ep_group, home, slots, hidden, ffn,
                    tokens, topk, window, capacity, x.device, config.planner, config.backward_restore,
                    config.plan_validation, enable_diagnostics=diagnostics,
                    save_gmm1_output=config.save_gmm1_output)
                _POOLS[key] = pool
                if diagnostics:
                    print(f'[MegaMoeBalanced] planner={config.planner} slots={slots} '
                          f'restore={config.backward_restore} home={home} shape={tokens,hidden,ffn,topk} '
                          f'max_recv={capacity} validation={config.plan_validation} '
                          f'save={config.save_gmm1_output} worker_flags=0 owner_rows=16',
                          flush=True)
        bridge = getattr(self, '_megamoe_balanced_bridge', None)
        if bridge is None:
            bridge = FSDP2ExpertsBridge(pool, weights)
            self._megamoe_balanced_bridge = bridge
        elif bridge.pool is not pool:
            raise RuntimeError('expert module changed its balanced execution configuration')
        limit = getattr(self, 'limit', 0.)
        return bridge(x, ids, prob, 0. if limit is None else float(limit)).view(shape)

    return balanced_forward
