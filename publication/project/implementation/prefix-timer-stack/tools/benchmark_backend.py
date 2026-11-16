"""Attach the original MoonEP communication backend to framework-owned weights."""
from __future__ import annotations

from types import MethodType
import torch


def unattached_moonep(module, hidden_states, top_k_index, top_k_weights):
    """Reject execution until all layer mirrors have been registered together."""
    raise RuntimeError('MoonEP mirrors must be attached after parameter initialization')


def attach_moonep(engine, tokens: int, topk: int):
    """Create one shared mirror bank and retain the original trainable Parameters."""
    from fsdp_turbo.distributed.expert_parallel.moonep.runtime import MoonEPRuntime
    from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings
    from fsdp_turbo.distributed.expert_parallel.moonep.dispatcher import parallelize_moonep_module

    mesh = engine.parallel_state.get_ep_device_mesh()
    config = MoonEPSettings(tokens_per_rank=tokens, top_k=topk, recompute_dispatch=True,
                            async_finish=True, npu_async_comm=True, shmem_stream_barrier=True)
    experts = [layer.mlp.experts for layer in engine.model.model.layers]
    runtime = MoonEPRuntime(mesh.get_group(), mesh, experts[0].num_global_experts,
                            config, gradient_divide_factor=1.0, fsdp_owned_weights=True)
    for module in experts:
        parallelize_moonep_module(module, runtime, mesh, fsdp_mirror=True)
    runtime.finalize()
    return runtime
