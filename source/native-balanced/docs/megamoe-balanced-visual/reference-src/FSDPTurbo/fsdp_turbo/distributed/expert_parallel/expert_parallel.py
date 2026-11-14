# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import types
from functools import partial
from typing import Callable, Optional

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, DTensor, Replicate, distribute_module
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from fsdp_turbo.distributed.expert_parallel.dispatcher import get_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.dispatcher_mc2 import get_experts_forward_mc2_fn
from fsdp_turbo.distributed.expert_parallel.domino_dispatcher import get_domino_experts_forward_fn
from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import get_mega_moe_forward_fn
from fsdp_turbo.fsdp_turbo_config import EPPlanConfig
from fsdp_turbo.utils.log import print_rank
from fsdp_turbo.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def _annotate_ep_module(module: torch.nn.Module, ep_rank: int, ep_size: int) -> None:
    module.num_global_experts = len(module) if not hasattr(module, 'num_experts') else module.num_experts
    if module.num_global_experts % ep_size != 0:
        raise AssertionError(
            f'Number of experts({module.num_global_experts}) is not divisible by ep size({ep_size}).'
        )
    module.num_local_experts = module.num_global_experts // ep_size
    local_expert_indices_offset = ep_rank * module.num_local_experts
    module.local_expert_indices = [local_expert_indices_offset + i for i in range(module.num_local_experts)]
    if module.num_local_experts > 1:
        module.expert_ids_per_ep_rank = torch.tensor(
            [i % module.num_local_experts for i in range(module.num_global_experts)],
            dtype=torch.int32,
            device=torch.accelerator.current_device_index(),
        )


def expert_parallelize_modules(
    modules: torch.nn.Module,
    ep_mesh: DeviceMesh,
    plan: EPPlanConfig,
    edp_mesh: Optional[DeviceMesh] = None,
    efsdp_mesh: Optional[DeviceMesh] = None,
    weight_mesh: Optional[DeviceMesh] = None,
    fsdp_owned_weights: Optional[bool] = None,
    attach_moonep: bool = True,
):
    """Apply expert parallel plans.

    For MoonEP/Hybrid with ``fsdp_owned_weights=True`` (efsdp>1), pass
    ``attach_moonep=False`` so this only EP-shards experts; call
    :func:`attach_moonep_after_expert_fsdp` **after** ``expert_fully_shard_modules``
    so mirror bind sees the live FSDP2 Parameters.
    """
    ep_modules = get_ep_modules(modules, plan)

    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    moonep_runtime = None
    # efsdp>1: FSDP2 owns shards + one pooled VMM full-home mirror.
    # efsdp=1: per-layer VMM homes (no pool mirror / refill copy).
    if fsdp_owned_weights is None:
        fsdp_owned_weights = True

    for module in ep_modules:
        _annotate_ep_module(module, ep_rank, ep_size)

        if plan.dispatcher in ("moonep", "moonep_megamoe"):
            if fsdp_owned_weights and not attach_moonep:
                # Phase 1 (efsdp>1): EP Shard only; MoonEP mirror attaches after FSDP2.
                #只执行EP切权重（专家）
                distribute_experts_module(module, ep_mesh)
                continue
            moonep_runtime = parallelize_moonep(
                module,
                moonep_runtime,
                modules,
                ep_mesh,
                plan,
                edp_mesh=edp_mesh,
                efsdp_mesh=efsdp_mesh,
                weight_mesh=weight_mesh,
                fsdp_owned_weights=fsdp_owned_weights,
                skip_ep_distribute=False,
            )
            continue

        # distribute experts weights
        distribute_experts_module(module, ep_mesh)

        # Select the complete EP forward explicitly through the dispatcher plan.
        experts_forward_fn = get_dispatcher_fn(plan.dispatcher, ep_group, fixed_router=plan.fixed_router,
                                              megamoe_precision=getattr(plan, "megamoe_precision", "bf16"),
                                              balanced_config=getattr(plan, "megamoe_balanced_config", None))
        module.forward = types.MethodType(experts_forward_fn, module)

    if moonep_runtime is not None:
        # Every symmetric allocation is now registered, so the backend can bring
        # up its communication buffer and (on Ascend) its symmetric heap. This
        # must happen before the first forward.
        moonep_runtime.finalize()
    return modules


def attach_moonep_after_expert_fsdp(
    modules: torch.nn.Module,
    ep_mesh: DeviceMesh,
    plan: EPPlanConfig,
):
    """Attach MoonEP/Hybrid mirrors after expert FSDP2 wrap (efsdp>1 only).

    EP Shard must already have been applied via
    ``expert_parallelize_modules(..., fsdp_owned_weights=True, attach_moonep=False)``.
    Binding after FSDP2 means ``owned_parameter`` is the live sharded Parameter
    (no stale pre-wrap handle).
    """
    if plan.dispatcher not in ("moonep", "moonep_megamoe"):
        raise RuntimeError(
            f"attach_moonep_after_expert_fsdp requires moonep/moonep_megamoe, got {plan.dispatcher!r}."
        )
    moonep_runtime = None
    for module in get_ep_modules(modules, plan):
        moonep_runtime = parallelize_moonep(
            module,
            moonep_runtime,
            modules,
            ep_mesh,
            plan,
            edp_mesh=None,
            efsdp_mesh=None,
            weight_mesh=None,
            fsdp_owned_weights=True,
            skip_ep_distribute=True,
        )
    if moonep_runtime is None:
        raise RuntimeError("attach_moonep_after_expert_fsdp found no expert modules to attach.")
    moonep_runtime.finalize()
    return modules


def parallelize_moonep(
    module,
    moonep_runtime,
    model,
    ep_mesh: DeviceMesh,
    plan: EPPlanConfig,
    edp_mesh: Optional[DeviceMesh] = None,
    efsdp_mesh: Optional[DeviceMesh] = None,
    weight_mesh: Optional[DeviceMesh] = None,
    fsdp_owned_weights: bool = True,
    skip_ep_distribute: bool = False,
):
    """Attach one MoonEP runtime to the model and parallelize one expert module.

    The runtime is shared by every MoE layer: MoonEP's symmetric weight mappings,
    prefetch slots and FP32 reduction buffers are pooled by matrix shape, which
    is what keeps its memory footprint independent of the layer count.

    ``fsdp_owned_weights=True`` (efsdp>1): FSDP2 owns Parameters; MoonEP only
    mirrors into one pooled VMM full-home. Prefer calling this **after** expert
    FSDP2 wrap (``skip_ep_distribute=True``) so bind sees live Parameters.
    ``fsdp_owned_weights=False`` (efsdp=1): per-layer VMM homes as Parameter
    payload — no pool full-home and no refill copy.
    """
    from fsdp_turbo.distributed.expert_parallel.moonep import (
        MoonEPRuntime,
        parallelize_moonep_module,
    )

    del efsdp_mesh, weight_mesh  # out-dim AG/RS is FSDP2 (efsdp>1) or unused (efsdp=1)
    if moonep_runtime is None:
        if fsdp_owned_weights:
            # Expert FSDP2 owns RS / divide / EDP HSDP.
            divide = 1.0
            edp_group = None
        else:
            # efsdp=1: MoonEP owns divide + EDP SUM; per-layer VMM homes, no pool mirror.
            divide = float(plan.gradient_divide_factor or 1.0)
            edp_group = edp_mesh.get_group() if edp_mesh is not None else None
        moonep_runtime = MoonEPRuntime(
            ep_mesh.get_group(),
            ep_mesh,
            module.num_global_experts,
            plan.moonep_config,
            gradient_divide_factor=divide,
            edp_group=edp_group,
            efsdp_group=None,
            weight_mesh=None,
            fsdp_owned_weights=fsdp_owned_weights,
        )
        object.__setattr__(model, "_moonep_runtime", moonep_runtime)
    elif moonep_runtime.num_experts != module.num_global_experts:
        raise RuntimeError(
            'All MoonEP modules sharing a runtime must have the same number of experts; '
            f'expected {moonep_runtime.num_experts}, got {module.num_global_experts}.'
        )
    elif bool(moonep_runtime.fsdp_owned_weights) != bool(fsdp_owned_weights):
        raise RuntimeError(
            "MoonEPRuntime fsdp_owned_weights mismatch across expert modules: "
            f"runtime={moonep_runtime.fsdp_owned_weights}, requested={fsdp_owned_weights}."
        )
    if fsdp_owned_weights and not skip_ep_distribute:
        # EP Shard(0) so Parameters match mega_moe before FSDP2 wrap.
        distribute_experts_module(module, ep_mesh)
    experts_forward_fn = None
    if plan.dispatcher == "moonep_megamoe":
        from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import (
            _resolve_window_pool_size,
        )

        moonep_runtime.megamoe_windows = True
        pool_size = _resolve_window_pool_size(module)
        requested = moonep_runtime._megamoe_requested_pool
        if requested is None:
            moonep_runtime._megamoe_requested_pool = pool_size
        elif requested != pool_size:
            raise ValueError(
                f"MegaMoe window pool resolved to {pool_size}, but an earlier expert "
                f"module on this MoonEP runtime requested {requested}."
            )
        from fsdp_turbo.distributed.expert_parallel.dispatcher_moonep_megamoe import (
            get_moonep_megamoe_forward_fn,
        )

        experts_forward_fn = get_moonep_megamoe_forward_fn(
            moonep_runtime, ep_mesh.get_group(), fixed_router=plan.fixed_router
        )
    parallelize_moonep_module(
        module,
        moonep_runtime,
        ep_mesh,
        fixed_router=plan.fixed_router,
        experts_forward_fn=experts_forward_fn,
        fsdp_mirror=fsdp_owned_weights,
    )
    return moonep_runtime


def get_ep_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    ep_modules = []
    for plan_name in plan.apply_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Parallel]: Apply ep to module <{name}>')
                ep_modules.append(module)
    if len(ep_modules) == 0:
        raise RuntimeError(f'[Expert Parallel] No module named {plan} or not be ModuleList')
    return ep_modules


def prepare_distribute_input_fn(module, inputs, device_mesh):
    inputs = list(inputs)
    for idx, input_tensor in enumerate(inputs):
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (Replicate(),), run_check=False)
            inputs[idx] = input_tensor
    return (*inputs,)


def prepare_distribute_output_fn(module, outputs, device_mesh):
    return outputs.to_local()


def distribute_expert_weight(module_name, module, ep_mesh):
    placements = [Shard(0)]
    for name, param in module.named_parameters(recurse=False):
        local_shape, global_offset = compute_local_shape_and_global_offset(param.shape, ep_mesh, placements)
        slices = [slice(offset, offset + size) for offset, size in zip(global_offset, local_shape)]
        dist_param = torch.nn.Parameter(DTensor.from_local(param[slices].detach().clone(), ep_mesh, placements))
        module.register_parameter(name, dist_param)

    for name, children_module in module.named_children():
        distribute_expert_weight(name, children_module, ep_mesh)


def distribute_experts_module(module: torch.nn.Module, ep_mesh: DeviceMesh):
    return distribute_module(
        module=module,
        device_mesh=ep_mesh,
        partition_fn=distribute_expert_weight,
    )


def get_dispatcher_fn(dispatcher, ep_group, fixed_router=False, megamoe_precision="bf16", balanced_config=None):
    if megamoe_precision not in ("bf16", "mxfp8"):
        raise ValueError(f"Unsupported MegaMoE precision: {megamoe_precision}")
    if megamoe_precision != "bf16" and dispatcher != "mega_moe":
        raise ValueError("megamoe_precision=mxfp8 requires ep_dispatcher=mega_moe")
    forward_fn = None
    if isinstance(dispatcher, Callable):
        forward_fn = partial(dispatcher, ep_group)
    elif isinstance(dispatcher, str):
        if dispatcher == 'eager':
            forward_fn = get_experts_forward_fn(ep_group, fused=False, fixed_router=fixed_router)
        elif dispatcher == 'fused':
            forward_fn = get_experts_forward_fn(ep_group, fused=True, fixed_router=fixed_router)
        elif dispatcher == 'mc2':
            forward_fn = get_experts_forward_mc2_fn(ep_group, fixed_router=fixed_router)
        elif dispatcher == 'domino':
            forward_fn = get_domino_experts_forward_fn(ep_group, fixed_router=fixed_router)
        elif dispatcher == 'mega_moe' and megamoe_precision == 'mxfp8':
            from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe_mxfp8 import get_mega_moe_mxfp8_forward_fn
            forward_fn = get_mega_moe_mxfp8_forward_fn(ep_group, fixed_router=fixed_router)
        elif dispatcher == 'megamoe_balanced':
            from .dispatcher_megamoe_balanced import get_megamoe_balanced_forward_fn
            forward_fn = get_megamoe_balanced_forward_fn(ep_group, fixed_router, balanced_config)
        elif dispatcher == 'mega_moe':
            forward_fn = get_mega_moe_forward_fn(ep_group, fixed_router=fixed_router)
        elif dispatcher.startswith('custom_'):
            method_name = dispatcher.removeprefix('custom_')

            def custom_ep_forward(self, hidden_states, top_k_index, top_k_weights):
                method = getattr(self, method_name, None)
                if not callable(method):
                    raise RuntimeError(
                        f'EP dispatcher requires {type(self).__name__}.{method_name} to be a callable method.'
                    )
                return method(
                    hidden_states,
                    top_k_index,
                    top_k_weights,
                    ep_group=ep_group,
                    fixed_router=fixed_router,
                )

            forward_fn = custom_ep_forward

    if forward_fn is None:
        raise RuntimeError(f'Unsupported dispatcher {dispatcher}.')

    return forward_fn
