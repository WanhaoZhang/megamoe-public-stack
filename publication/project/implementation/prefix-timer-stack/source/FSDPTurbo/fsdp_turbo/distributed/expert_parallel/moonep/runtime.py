# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MoonEP resource ownership: symmetric expert weights and per-step state.

The device-independent half of the integration (Kimi K3 report S5.2.1): every
rank receives exactly ``S*K`` tokens so all activation shapes are constants, the
redundant experts are planned per micro-batch and prefetched on the comm stream,
their gradients are staged in FP32 and reduced back to their home ranks, and no
host reads a device scalar on the critical path. All device-specific work is
delegated to a :class:`~moonep.backend.MoonEPBackend`.

Notation: ``S`` tokens/rank, ``K`` top-k, ``E`` routed experts, ``R`` EP ranks,
``epn = E/R``, ``B = epn`` prefetch slots/rank, ``NvS`` dispatched rows/rank.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from fsdp_turbo.distributed.expert_parallel.moonep.backend import get_moonep_backend
from fsdp_turbo.distributed.expert_parallel.moonep.plan import (
    compact_group_ends,
    full_group_ends,
    verify_plan_locality,
    zero_static_tail_,
)
from fsdp_turbo.distributed.expert_parallel.moonep.settings import MoonEPSettings
from fsdp_turbo.distributed.expert_parallel.moonep.streams import (
    _record_stream,
    device_stream_module,
)
from fsdp_turbo.distributed.expert_parallel.moonep.trace import trace
from fsdp_turbo.utils.device import get_accelerator_name


def _loaded_libshmem_paths() -> list[str]:
    """Distinct absolute paths of ``libshmem.so`` mapped into this process."""
    paths: list[str] = []
    try:
        handle = open("/proc/self/maps", encoding="utf-8", errors="replace")
    except OSError:
        return paths
    with handle:
        for line in handle:
            if "libshmem.so" not in line:
                continue
            path = line.split()[-1]
            if path.startswith("/") and path not in paths:
                paths.append(path)
    return paths

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh

    from fsdp_turbo.fsdp_turbo_config import MoonEPConfig


logger = logging.getLogger(__name__)


class MoonEPSymmetricProjection:
    """One layer's expert projection backed by MoonEP symmetric memory.

    Three views over the same memory: ``home_weight`` ``[epn, out, in]`` is the
    parameter payload, ``full_weight`` ``[E+B, out, in]`` is what prefetch reads,
    ``gemm_weight`` is the GEMM operand, global or compact per the backend.
    """

    def __init__(
        self,
        runtime: "MoonEPRuntime",
        role: str,
        source: torch.Tensor,
        ep_mesh: "DeviceMesh",
        transposed: bool = False,
        owned_parameter: torch.nn.Parameter | None = None,
    ):
        self.runtime = runtime
        self.role = role
        self.layer_index = -1
        self.closed = False
        #: Whether the stored matrix is ``[in, out]`` instead of ``[out, in]``.
        #: Only the grouped GEMMs care; every buffer below mirrors what the model
        #: brought, which is what keeps the parameter's shape and the model's
        #: state-dict layout unchanged.
        self.transposed = transposed
        # The two matrix dimensions as stored, not as named by the linear-layer
        # convention: under ``transposed`` these are (in_features, out_features).
        self.out_features = int(source.shape[-2])
        self.in_features = int(source.shape[-1])
        self.pool = runtime.get_projection_pool(role, self.out_features, self.in_features)
        self.resources = runtime.backend.create_projection(runtime, self.pool, self.out_features, self.in_features)

        # For fsdp_owned: refill must getattr(module, name) each step. FSDP2
        # unshard may swap the module attribute to a full tensor while the
        # attach-time Parameter handle still to_local()'s as the shard.
        self._param_module: torch.nn.Module | None = None
        self._param_name: str | None = None

        if runtime.fsdp_owned_weights:
            # FSDP2 owns Parameter shards (attach after expert FSDP2 wrap).
            # VMM has only the pooled full-home (home_weight aliases pool.gathered_home).
            # Refill is begin_expert_layer → refill_home_from_parameter.
            if owned_parameter is None:
                raise RuntimeError(
                    "fsdp_owned_weights requires owned_parameter=module.*.proj "
                    "(do not copy weights into a per-layer VMM home)."
                )
            self.parameter = owned_parameter
            return

        # efsdp=1: Parameter storage aliases VMM home.
        if owned_parameter is not None:
            raise RuntimeError("owned_parameter requires MoonEPRuntime.fsdp_owned_weights=True.")
        epn = runtime.experts_per_rank
        start = runtime.rank * epn
        local_source = source.to_local() if isinstance(source, DTensor) else source[start : start + epn]
        self.resources.home_weight.copy_(local_source.detach().contiguous())
        dist.barrier(group=runtime.group)
        local_dtensor = DTensor.from_local(self.resources.home_weight, ep_mesh, (Shard(0),), run_check=False)
        self.parameter = torch.nn.Parameter(local_dtensor, requires_grad=source.requires_grad)

    def bind_module_parameter(self, module: torch.nn.Module, name: str) -> None:
        """Record module attribute for refill after FSDP2 unshard swaps it."""
        self._param_module = module
        self._param_name = name
        self.parameter = getattr(module, name)

    def module_parameter(self) -> torch.nn.Parameter:
        """Module attribute as of now (post-unshard full tensor when FSDP2 is active)."""
        if self._param_module is not None and self._param_name is not None:
            return getattr(self._param_module, self._param_name)
        return self.parameter

    @property
    def full_weight(self) -> torch.Tensor:
        return self.resources.full_weight

    @property
    def gemm_weight(self) -> torch.Tensor:
        """The weight operand for the forward and data-gradient GEMMs."""
        return self.resources.gemm_weight

    @torch.no_grad()
    def refill_home_from_parameter(self) -> None:
        """Copy FSDP2 unsharded weight into the single pooled VMM full-home.

        Only the current layer is mirrored. Must use :meth:`module_parameter`
        (getattr), not the attach-time handle: after unshard the module attribute
        is the full tensor, while ``Parameter.to_local()`` on the sharded DTensor
        stays half-sized.
        """
        param = self.module_parameter()
        local = param.to_local() if isinstance(param, DTensor) else param
        home = self.resources.pool.gathered_home.tensor
        if tuple(local.shape) != tuple(home.shape):
            raise RuntimeError(
                f"MoonEP {self.role} FSDP local shape {tuple(local.shape)} does not match "
                f"pooled VMM home {tuple(home.shape)}."
            )
        home.copy_(local.detach())

    def split_gradient(self, gemm_grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice a weight gradient into its home and prefetch-slot halves."""
        runtime = self.runtime
        epn = runtime.experts_per_rank
        slots = runtime.prefetch_slots
        return gemm_grad[:epn], gemm_grad[epn : epn + slots]

    def prefetch(self, buffer, plan) -> None:
        """Copy this step's planned redundant expert weights into the slots."""
        self.runtime.backend.prefetch(self.runtime, self.resources, buffer, plan)

    def stage_gradient(self, gemm_grad: torch.Tensor) -> None:
        """Publish the gradient halves into the FP32 reduction buffers.

        ``gemm_grad`` is ``[epn+B, out, in]``: the weight gradient always uses
        the compact group set, whichever layout the forward's weight operand
        had. There is no tail group to drop, because ``compact_group_ends``
        folds MoonEP's reserved rows into the final slot group.
        """
        home_slice, slot_slice = self.split_gradient(gemm_grad)
        self.runtime.backend.stage_gradient(self.runtime, self.resources, home_slice, slot_slice)
        self._staged_dtype = gemm_grad.dtype

    def finish_gradient(self, buffer, plan) -> torch.Tensor:
        """Rendezvous, then reduce the staged slot gradients home.

        Must run after :meth:`stage_gradient` on every rank. The rendezvous is per
        projection, not shared: every MoonEP barrier uses the same slot and
        generation counter, so the sequence of arrivals is part of the protocol.

        With ``fsdp_owned_weights`` (efsdp>1): return private FP32 full-home grad;
        FSDP2 owns out-dim RS / divide / EDP HSDP.
        With efsdp=1 legacy VMM homes: apply ``gradient_divide_factor`` and cast to
        the staged dtype (no pool mirror, no FSDP expert wrap).
        """
        backend = self.runtime.backend
        backend.rendezvous_before_reduce(self.runtime, buffer)
        home_grad = backend.reduce_staged_gradient(self.runtime, self.resources, buffer, plan)
        # Pooled reduce buffer is reused by the next layer; autograd must not
        # retain a view into it.
        private = torch.empty(home_grad.shape, dtype=home_grad.dtype, device=home_grad.device)
        private.copy_(home_grad)
        if self.runtime.fsdp_owned_weights:
            return private
        factor = self.runtime.gradient_divide_factor
        if factor != 1.0:
            private = private.div(factor)
        staged = getattr(self, "_staged_dtype", torch.bfloat16)
        return private.to(dtype=staged)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.resources.close()


class MoonEPCallState:
    """State of a single expert forward: plan, group layout, pending events."""

    def __init__(self, runtime: "MoonEPRuntime", buffer, tokens_per_rank: int, hidden: int, top_k: int):
        self.runtime = runtime
        self.buffer = buffer
        self.tokens_per_rank = tokens_per_rank
        self.hidden = hidden
        self.top_k = top_k
        self.validate = runtime.should_check_plan_invariants()
        self.plan = None
        self.cu_seqlens = None
        self.group_ends = None
        self.wgrad_group_ends = None
        self.dispatch_rows = None
        self._wgrad_group_ends_host: list[int] | None = None
        self._prefetch_events: dict[int, Any] = {}

    def wgrad_group_ends_host(self) -> list[int] | None:
        """The weight-gradient group offsets on the host, read at most once.

        Both projections of a layer share this cache, so a layer costs one
        device-to-host read rather than two.
        """
        if self.runtime.backend.device_type == "npu":
            return None
        if self._wgrad_group_ends_host is None:
            self._wgrad_group_ends_host = self.wgrad_group_ends.detach().to(device="cpu", dtype=torch.int64).tolist()
        return self._wgrad_group_ends_host

    @property
    def _comm_stream(self):
        if not self.runtime.config.async_finish or not self.runtime.backend.supports_async_comm:
            return None
        stream = self.buffer._comm_stream
        if stream is None:
            raise RuntimeError("MoonEP communication stream is unavailable.")
        return stream

    def wait(self, event) -> None:
        if event is not None:
            event.wait(self.runtime.device_module.current_stream())

    def adopt(self, *tensors: torch.Tensor | None) -> None:
        """Record tensors produced on the comm stream on the compute stream."""
        if self._comm_stream is None:
            return
        stream = self.runtime.device_module.current_stream()
        for tensor in tensors:
            _record_stream(tensor, stream)

    def set_plan(self, plan, cu_seqlens: torch.Tensor, dispatch_rows: int) -> None:
        """Record the plan and derive this step's two group layouts.

        ``group_ends`` must match the backend's weight layout; the weight gradient
        reads no weights and can always use the compact set.
        """
        runtime = self.runtime
        self.plan = plan
        self.cu_seqlens = cu_seqlens
        self.dispatch_rows = dispatch_rows

        if self.validate:
            verify_plan_locality(cu_seqlens, runtime.rank, runtime.experts_per_rank, runtime.num_experts)
        compact = compact_group_ends(
            cu_seqlens,
            runtime.rank,
            runtime.experts_per_rank,
            runtime.num_experts,
            runtime.prefetch_slots,
            dispatch_rows,
        )
        self.wgrad_group_ends = compact
        self.group_ends = (
            compact if runtime.backend.expert_weight_layout == "compact" else full_group_ends(cu_seqlens, dispatch_rows)
        )

    def neutralize_static_tail(self, rows: torch.Tensor) -> torch.Tensor:
        """Zero the rows past the last live group, where the backend has them.

        Both layouts extend their final offset to the full buffer, so the
        undefined tail would contaminate the last group's weight gradient.
        """
        if not self.runtime.backend.has_static_tail:
            return rows
        return zero_static_tail_(rows, self.cu_seqlens[-1])

    def start_prefetch(self, projection: MoonEPSymmetricProjection) -> None:
        """Queue a projection's redundant-weight copy on the comm stream.

        Slots are pooled across layers, so the copy overwrites memory an adjacent
        layer's GEMM reads: the comm stream must wait for the compute stream's
        queued work first. After that the copy proceeds concurrently, which is
        what lets the ``down`` prefetch overlap the first GEMM.
        """
        if self.plan is None:
            raise RuntimeError("MoonEP dispatch plan is not initialized.")
        trace(self.runtime, "prefetch.start", projection.role)
        with torch.profiler.record_function("moonep.prefetch"):
            comm_stream = self._comm_stream
            if comm_stream is None:
                projection.prefetch(self.buffer, self.plan)
                return
            device_module = self.runtime.device_module
            comm_stream.wait_event(device_module.current_stream().record_event())
            with device_module.stream(comm_stream):
                projection.prefetch(self.buffer, self.plan)
                self._prefetch_events[id(projection)] = comm_stream.record_event()

    def await_prefetch(self, projection: MoonEPSymmetricProjection) -> None:
        """Block the compute stream until this projection's slots are filled."""
        self.wait(self._prefetch_events.pop(id(projection), None))
        trace(self.runtime, "prefetch.await", projection.role)


class MoonEPRuntime:
    """Owns every MoonEP resource associated with one FSDPTurbo model."""

    def __init__(
        self,
        group,
        ep_mesh: "DeviceMesh",
        num_experts: int,
        config: "MoonEPConfig",
        gradient_divide_factor: float | None = None,
        edp_group=None,
        efsdp_group=None,
        weight_mesh=None,
        fsdp_owned_weights: bool = False,
    ):
        # ``get_accelerator_name`` is FSDPTurbo's canonical detection and reports
        # "npu" for torch_npu builds, which ``torch.accelerator`` may surface as
        # a PrivateUse1 device name instead.
        device_type = get_accelerator_name()
        self.backend = get_moonep_backend(device_type)
        # Resolved once: every stream and event this runtime touches has to come
        # from this module so it stays in MoonEP's own API family.
        self.device_module = device_stream_module(device_type)
        self.group = group
        self.ep_mesh = ep_mesh
        # Optional process group along the expert-data-parallel (replica) axis.
        # ``None`` or size 1 leaves the already-verified ``edp==1`` path unchanged.
        self.edp_group = edp_group
        self.fsdp_owned_weights = bool(fsdp_owned_weights)
        # Out-dim sharding / AG/RS belong to expert FSDP2. Reject leftover meshes.
        if efsdp_group is not None or weight_mesh is not None:
            raise ValueError(
                "MoonEP no longer owns out-dim AG/RS; pass efsdp_mesh=None and "
                "weight_mesh=None, and wrap experts with expert_fully_shard_modules "
                "(fsdp_owned_weights=True)."
            )
        self.efsdp_group = None
        self.weight_mesh = None
        self.rank = dist.get_rank(group=group)
        self.size = dist.get_world_size(group=group)
        self.num_experts = int(num_experts)
        # An expert's gradient collects tokens from every EP rank, and those
        # ranks hold different micro-batches, so the sum the home-rank reduction
        # produces has to be averaged the same way expert FSDP's reduce-scatter
        # averages it for the other dispatchers. Without this the experts train
        # at ``ep_size`` times the effective learning rate of the dense
        # parameters -- largely hidden by Adam's second moment, but not by weight
        # decay and not by ``Trainer``'s clip-by-norm. With fsdp_owned_weights,
        # expert FSDP2 applies this via set_gradient_divide_factor instead.
        self.gradient_divide_factor = float(gradient_divide_factor or 1.0)
        self.config = MoonEPSettings.from_config(config)
        if device_type == "npu":
            self.backend.supports_async_comm = self.config.npu_async_comm
        # efsdp>1: one pooled VMM full-home per matrix shape (mirror of FSDP local).
        # efsdp=1: per-layer VMM homes as Parameter payload — no pool mirror.
        self._layers: list[tuple[MoonEPSymmetricProjection, MoonEPSymmetricProjection]] = []
        if self.fsdp_owned_weights:
            logger.info(
                "MoonEP FSDP2-mirror (efsdp>1): one pooled VMM full-home per matrix shape; "
                "no per-layer VMM home (FSDP2 owns Parameter shards)."
            )
        else:
            logger.info(
                "MoonEP efsdp=1: per-layer VMM homes as Parameter payload; "
                "no pooled full-home mirror / refill copy."
            )
        self.config.warn_about_inert_knobs(device_type, self.backend.supports_async_comm)
        self.config.report_effective_async(device_type, self.backend.supports_async_comm)
        self.closed = False
        self.finalized = False
        self.hidden_size: int | None = None
        self._static_tokens_per_rank: int | None = self.config.tokens_per_rank
        self._static_top_k: int | None = self.config.top_k
        self._plan_invariants_checked = False
        #: Monotonic expert-forward counter, used only to label trace output.
        self.call_index = -1
        self._buffers: dict[tuple[int, int, int], Any] = {}
        self._projection_pools: dict[tuple[str, int, int], Any] = {}
        self._projections: list[MoonEPSymmetricProjection] = []

        if self.num_experts % self.size:
            raise RuntimeError(f"MoonEP experts ({self.num_experts}) must be divisible by EP size ({self.size}).")
        self.experts_per_rank = self.num_experts // self.size
        # Appendix E of the K3 report proves a balanced plan always exists with
        # at most E/R redundant experts per rank and that the bound is tight, so
        # one prefetch slot per local expert makes planning unconditionally
        # feasible and training uninterruptible.
        self.prefetch_slots = self.experts_per_rank
        # Hybrid dispatcher: MegaMoe token windows are extra VMM chunks in this
        # heap. They are allocated before ``initialize_symmetric_heap`` and bound
        # to symmetric addresses afterwards. Pure MoonEP leaves this false.
        self.megamoe_windows = False
        self._megamoe_requested_pool: int | None = None
        self._megamoe_window_tensors: list[torch.Tensor] = []
        self._megamoe_ops: list[Any] = []
        self._megamoe_shmem = None
        self._megamoe_next = 0
        self._megamoe_pool_size = 0
        self._megamoe_window_bytes = 0
        self.backend.validate_environment(self)

    def register_expert_layer(self, gate_up: MoonEPSymmetricProjection, down: MoonEPSymmetricProjection) -> None:
        """Remember expert modules in forward order so later layers can be prefetched."""
        index = len(self._layers)
        gate_up.layer_index = index
        down.layer_index = index
        self._layers.append((gate_up, down))

    def begin_expert_layer(self, projection: MoonEPSymmetricProjection, *, backward: bool) -> None:
        """Make this layer's full homes visible for GEMM / SHMEM prefetch.

        With ``fsdp_owned_weights``, refill the single pooled VMM home from
        FSDP2 ``to_local()`` (no EP barrier; plan/prefetch order the copy).
        Legacy non-FSDP homes already live in VMM.
        """
        del backward
        if not self.fsdp_owned_weights:
            return
        gate_up, down = self._layers[projection.layer_index]
        for item in sorted((gate_up, down), key=lambda p: p.role):
            item.refill_home_from_parameter()

    def sync_device_before_shmem_comm(self) -> None:
        """Optionally drain framework side-stream traffic before SHMEM entry."""
        if self.config.device_sync_before_shmem:
            self.device_module.synchronize()

    def reduce_replica_gradients(self) -> None:
        """SUM accumulated expert grads across EDP replicas, once per optimizer step.

        Each backward only returns this rank's EP-local gradient. Autograd adds
        those into ``param.grad`` across gradient-accumulation micro-batches. One
        all-reduce of that sum matches summing every micro-batch and then adding,
        without re-reading a gradient that already includes earlier steps.
        ``edp==1`` is a no-op.

        With ``fsdp_owned_weights``, replica reduction is left to expert FSDP2
        HSDP; calling MoonEP here would double-count.
        """
        if self.fsdp_owned_weights:
            return
        group = self.edp_group
        if group is None or dist.get_world_size(group=group) <= 1:
            return
        seen = set()
        for projection in self._projections:
            grad = projection.module_parameter().grad
            if grad is None:
                continue
            local = grad.to_local() if isinstance(grad, DTensor) else grad
            ptr = local.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            if local.is_contiguous():
                dist.all_reduce(local, op=dist.ReduceOp.SUM, group=group)
                continue
            synced = local.contiguous()
            dist.all_reduce(synced, op=dist.ReduceOp.SUM, group=group)
            local.copy_(synced)

    def register_hidden_size(self, hidden_size: int) -> None:
        hidden_size = int(hidden_size)
        if self.hidden_size is None:
            self.hidden_size = hidden_size
        elif self.hidden_size != hidden_size:
            raise RuntimeError(
                "All MoonEP expert modules sharing a runtime must have the same hidden "
                f"size; expected {self.hidden_size}, got {hidden_size}."
            )

    def projection_pools(self) -> list:
        """Every pooled resource set, in the deterministic order they were created."""
        return list(self._projection_pools.values())

    def projections(self) -> list:
        """Every registered projection, in the order the modules were visited."""
        return list(self._projections)

    @property
    def configured_tokens_per_rank(self) -> int | None:
        """Reserved token capacity, or ``None`` before the first lock.

        On NPU this is filled from the data config before the heap is created.
        On CUDA it stays ``None`` until the first expert forward.
        """
        return self._static_tokens_per_rank

    @property
    def static_shape(self) -> tuple[int, int, int]:
        """The locked ``(tokens_per_rank, hidden, top_k)`` triple."""
        if self._static_tokens_per_rank is None or self._static_top_k is None or self.hidden_size is None:
            raise RuntimeError("MoonEP static token shape is not known yet.")
        return self._static_tokens_per_rank, self.hidden_size, self._static_top_k

    def get_projection_pool(self, role: str, out_features: int, in_features: int):
        key = (role, out_features, in_features)
        pool = self._projection_pools.get(key)
        if pool is None:
            pool = self.backend.create_projection_pool(self, role, out_features, in_features)
            self._projection_pools[key] = pool
        return pool

    def distribute_projection(
        self,
        role: str,
        source: torch.Tensor,
        ep_mesh: "DeviceMesh",
        transposed: bool = False,
        owned_parameter: torch.nn.Parameter | None = None,
    ) -> MoonEPSymmetricProjection:
        if self.finalized:
            raise RuntimeError("MoonEP symmetric weights must be allocated before the runtime is finalized.")
        _validate_projection_source(
            role,
            source,
            self.size,
            self.backend.device_type,
            allow_fp32=self.fsdp_owned_weights,
        )
        if owned_parameter is not None and not self.fsdp_owned_weights:
            raise RuntimeError(
                "distribute_projection(owned_parameter=...) requires "
                "MoonEPRuntime.fsdp_owned_weights=True."
            )
        projection = MoonEPSymmetricProjection(
            self, role, source, ep_mesh, transposed, owned_parameter=owned_parameter
        )
        self._projections.append(projection)
        return projection

    def finalize(self) -> None:
        """Complete initialization once every expert module is parallelized.

        Ascend requires this: its symmetric heap is initialized once from the
        descriptors of every registered allocation, so all communication buffers
        and expert weights must exist first. On CUDA this only pre-creates the
        communication buffer when the static token shape is configured.
        """
        if self.finalized:
            return
        if not self._projections:
            raise RuntimeError("MoonEP runtime was finalized without any expert projections.")
        if self._static_tokens_per_rank is not None and self._static_top_k is not None:
            self._create_buffer(self._static_tokens_per_rank, self.hidden_size, self._static_top_k)
        elif self.backend.requires_preconfigured_static_shape:
            raise RuntimeError(
                f"The MoonEP {self.backend.device_type} backend needs a static "
                "tokens_per_rank and top_k before the symmetric heap is initialized. "
                "Both are derived automatically: S from data.batch_size * "
                "data.max_seq_length / ulysses_size, K from the model's routed top-k. "
                "Set data.max_seq_length (and a positive batch_size) or override "
                "moonep_config if derivation could not run."
            )
        if self.megamoe_windows:
            self._register_megamoe_windows()
        self.backend.initialize_symmetric_heap(self)
        if self.megamoe_windows:
            self._open_megamoe_windows()
        self._prewarm_collectives()
        self.finalized = True

    def _prewarm_collectives(self) -> None:
        """Compile the collective kernels now rather than in the first backward.

        A kernel first compiled inside the backward pass deadlocks; see
        :meth:`CudaMoonEPBackend.prewarm_collectives`. Not fatal if a backend
        cannot do it, since the kernels still compile lazily, so a failure only
        warns.
        """
        buffer = next(iter(self._buffers.values()), None)
        if buffer is None:
            return
        trace(self, "prewarm.collectives enter")
        try:
            self.backend.prewarm_collectives(self, buffer)
        except Exception as error:
            logger.warning(
                "MoonEP could not pre-compile its collective kernels (%s). They will "
                "be compiled on first use instead, which has been seen to hang in the "
                "backward pass; report this if the run stalls at the first step.",
                error,
            )
        trace(self, "prewarm.collectives exit")

    def should_check_plan_invariants(self) -> bool:
        """Run the device-side plan checks on the first call, then opt-in only."""
        if self.config.validate_plan_invariants:
            return True
        if self._plan_invariants_checked:
            return False
        self._plan_invariants_checked = True
        return True

    def new_call(self, tokens_per_rank: int, hidden: int, top_k: int) -> MoonEPCallState:
        if self._static_tokens_per_rank is None:
            self._static_tokens_per_rank = tokens_per_rank
        elif self._static_tokens_per_rank != tokens_per_rank:
            raise RuntimeError(
                f"MoonEP uses a static token shape: expected S={self._static_tokens_per_rank}, "
                f"got S={tokens_per_rank}. Pad/drop the batch or use another dispatcher."
            )
        if self._static_top_k is None:
            self._static_top_k = top_k
        elif self._static_top_k != top_k:
            raise RuntimeError(f"MoonEP uses a static top-k: expected K={self._static_top_k}, got K={top_k}.")

        key = (tokens_per_rank, hidden, top_k)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = self._create_buffer(tokens_per_rank, hidden, top_k)
        self.call_index += 1
        return MoonEPCallState(self, buffer, tokens_per_rank, hidden, top_k)

    def _create_buffer(self, tokens_per_rank: int, hidden: int, top_k: int):
        if hidden is None:
            raise RuntimeError("MoonEP cannot create a buffer before the hidden size is known.")
        key = (tokens_per_rank, hidden, top_k)
        buffer = self._buffers.get(key)
        if buffer is not None:
            return buffer
        if self.finalized and self.backend.requires_preconfigured_static_shape:
            raise RuntimeError(
                f"The MoonEP {self.backend.device_type} backend cannot create a "
                "communication buffer after the symmetric heap is initialized; the "
                f"static shape (S={tokens_per_rank}, K={top_k}) differs from the "
                "configured one."
            )
        buffer = self.backend.create_buffer(self, tokens_per_rank, hidden, top_k)
        self._buffers[key] = buffer
        self.backend.prepare_buffer(self, buffer)
        return buffer

    def megamoe_call(self, layer_index: int | None = None):
        """Return the borrowed SHMEM handle and the next fused-op window.

        Windows were registered into the MoonEP heap at finalize. A positive
        pool rotates across layers; pool size 0 keeps one window per layer.
        Backward reuses the operator stored on the autograd context, so only
        the forward call advances the rotation.
        """
        if not self._megamoe_ops or self._megamoe_shmem is None:
            raise RuntimeError(
                "MegaMoe windows are not bound. They must be registered before "
                "MoonEP initializes its symmetric heap."
            )
        if self._megamoe_pool_size <= 0:
            if layer_index is None or layer_index < 0 or layer_index >= len(self._megamoe_ops):
                raise RuntimeError(
                    f"MegaMoe per-layer window index {layer_index} is outside "
                    f"0..{len(self._megamoe_ops) - 1}."
                )
            return self._megamoe_shmem, self._megamoe_ops[layer_index]
        op = self._megamoe_ops[self._megamoe_next % len(self._megamoe_ops)]
        self._megamoe_next += 1
        return self._megamoe_shmem, op

    def _register_megamoe_windows(self) -> None:
        """Allocate MegaMoe token windows and register them on the descriptor list.

        ``create_vmm_physical`` appends each chunk to the process registry.
        ``initialize_symmetric_heap`` then publishes that list in one
        ``init_with_buffer``. Calling this after the heap exists is rejected
        by the registry.
        """
        if self.backend.device_type != "npu":
            raise RuntimeError("MegaMoe windows can only be registered on the Ascend MoonEP heap.")
        if self.hidden_size is None or self._static_tokens_per_rank is None or self._static_top_k is None:
            raise RuntimeError("MegaMoe window registration needs the locked MoonEP static shape.")
        from fsdp_turbo.distributed.expert_parallel.dispatcher_mega_moe import (
            _local_mem_bytes,
            _resolve_window_pool_size,
        )

        tokens = int(self._static_tokens_per_rank)
        top_k = int(self._static_top_k)
        local_e = self.experts_per_rank + self.prefetch_slots
        # Hybrid keeps MoonEP's balanced NvS=S*K recv (align 128 like MegaMoe).
        hybrid_max_recv = (tokens * top_k + 127) // 128 * 128
        window_bytes = _local_mem_bytes(
            tokens,
            int(self.hidden_size),
            top_k,
            self.size,
            local_e,
            max_recv_tokens=hybrid_max_recv,
        )
        pool_size = (
            self._megamoe_requested_pool
            if self._megamoe_requested_pool is not None
            else _resolve_window_pool_size(None)
        )
        count = pool_size if pool_size > 0 else len(self._layers)
        if count <= 0:
            raise RuntimeError("MegaMoe window registration found no expert layers and no window pool.")
        device = torch.accelerator.current_device()
        tensors = []
        for _ in range(count):
            tensor, _, _ = self.backend.imports.buffer_c.create_vmm_physical(
                [window_bytes], torch.uint8, device
            )
            tensors.append(tensor)
        self._megamoe_window_tensors = tensors
        self._megamoe_window_bytes = window_bytes
        self._megamoe_pool_size = pool_size
        logger.info(
            "Registered %d MegaMoe window(s) of %d bytes into the MoonEP symmetric-heap descriptor list "
            "(local_experts=%d, tokens=%d).",
            count,
            window_bytes,
            local_e,
            tokens,
        )

    def _open_megamoe_windows(self) -> None:
        """Map registered windows to symmetric addresses and build the fused ops.

        The fused kernel's ``aclshmem_ptr`` must see this heap, so ascend_deepep
        is imported only after MoonEP has loaded ``libshmem``. ``ShmemInstance.attach``
        borrows that runtime instead of calling ``aclshmemx_init_attr``.
        """
        loaded = _loaded_libshmem_paths()
        import ascend_deepep
        from ascend_deepep import MegaMoeBF16Train, ShmemInstance

        loaded_after = _loaded_libshmem_paths()
        if len(loaded_after) != 1:
            raise RuntimeError(
                "MegaMoe windows must live in MoonEP's libshmem, but this process "
                f"mapped {loaded_after or loaded}."
            )
        syms = []
        for tensor in self._megamoe_window_tensors:
            syms.append(
                self.backend.imports.buffer_c.map_to_sym_ptr(int(tensor.data_ptr()), "megamoe_window")
            )
        shmem = ShmemInstance(self.rank, self.size, "tcp://127.0.0.1:1")
        attach = getattr(shmem, "attach", None)
        if attach is None:
            raise RuntimeError(
                "Hybrid requires ShmemInstance.attach() (ascend_deepep Hybrid patch)."
            )
        attach()
        ops = [
            MegaMoeBF16Train(shmem, self._megamoe_window_bytes, external_ptr=sym)
            for sym in syms
        ]
        self._megamoe_shmem = shmem
        self._megamoe_ops = ops
        logger.info(
            "Bound %d MegaMoe window(s) into the MoonEP heap via %s.",
            len(ops),
            loaded_after[0],
        )
        del ascend_deepep

    def _close_megamoe_windows(self) -> None:
        for op in self._megamoe_ops:
            close = getattr(op, "close", None)
            if close is not None:
                close()
        self._megamoe_ops = []
        if self._megamoe_shmem is not None:
            destroy = getattr(self._megamoe_shmem, "destroy", None)
            if destroy is not None:
                destroy()
        self._megamoe_shmem = None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._close_megamoe_windows()
        self.backend.release(self)
        for buffer in self._buffers.values():
            buffer.destroy()
        self._buffers.clear()
        if dist.is_available() and dist.is_initialized():
            dist.barrier(group=self.group)
        for projection in self._projections:
            projection.close()
        self._projections.clear()
        for pool in self._projection_pools.values():
            pool.close()
        self._projection_pools.clear()


def reduce_moonep_replica_gradients(root: torch.nn.Module) -> None:
    """SUM MoonEP expert grads across EDP once, after gradient accumulation.

    No-op when the model has no MoonEP runtime or ``edp==1``. Safe to call on
    every optimizer step, including runs that do not use MoonEP.
    """
    seen: set[int] = set()
    for module in root.modules():
        runtime = getattr(module, "_moonep_runtime", None)
        if runtime is None or id(runtime) in seen:
            continue
        seen.add(id(runtime))
        reduce = getattr(runtime, "reduce_replica_gradients", None)
        if reduce is not None:
            reduce()


def _validate_projection_source(
    name: str,
    tensor: torch.Tensor,
    ep_size: int,
    device_type: str,
    allow_fp32: bool = False,
) -> None:
    """Reject expert weights MoonEP's kernels cannot consume.

    ``allow_fp32`` is the efsdp>1 path: FSDP keeps the dev_gp FP32 master weight
    and the BF16 VMM mirror is filled by ``refill_home_from_parameter``.
    """
    if tensor.device.type != device_type:
        raise RuntimeError(
            f"MoonEP {name} must live on {device_type}, got {tensor.device}. "
            f"MoonEP maps expert weights into symmetric memory during FSDPTurbo "
            f"construction, so move the model to the accelerator before calling "
            f"FSDPTurbo(config, model)."
        )
    if tensor.dtype != torch.bfloat16 and not (allow_fp32 and tensor.dtype == torch.float32):
        raise RuntimeError(f"MoonEP {name} must be BF16, got {tensor.dtype}.")
    if not tensor.is_contiguous():
        raise RuntimeError(f"MoonEP {name} must be contiguous.")
    if tensor.ndim != 3:
        raise RuntimeError(
            f"MoonEP {name} must have [experts, out_features, in_features] shape, got {tuple(tensor.shape)}."
        )
    if tensor.shape[0] % ep_size:
        raise RuntimeError(f"MoonEP {name} expert dimension {tensor.shape[0]} is not divisible by EP size {ep_size}.")
    if tensor.shape[-2] % 128 or tensor.shape[-1] % 128:
        raise RuntimeError(
            f"MoonEP {name} matrix dimensions must be multiples of 128 for the prefetch and "
            f"gradient kernels, got {tuple(tensor.shape[-2:])}."
        )
