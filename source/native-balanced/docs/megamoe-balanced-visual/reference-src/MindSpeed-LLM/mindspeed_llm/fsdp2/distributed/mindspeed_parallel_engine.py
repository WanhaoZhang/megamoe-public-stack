# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional, Any
import os

import torch

from fsdp_turbo.distributed.fully_shard_parallel.fully_shard_parallel import fully_shard_parallel_modules
from fsdp_turbo.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from fsdp_turbo.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from fsdp_turbo.memory.recompute.recompute import recompute_modules
from fsdp_turbo.memory.chunk_batch.chunk_batch import chunk_batch_modules
from mindspeed_llm.fsdp2.distributed.parallel_state import init_parallel_state
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import ParallelEngineConfig
from mindspeed_llm.fsdp2.distributed.context_parallel.context_parallel_manager import apply_context_parallelize_modules
from mindspeed_llm.fsdp2.distributed.expert_parallel.expert_parallel import (
    expert_parallelize_modules,
)
from mindspeed_llm.fsdp2.models.model_loader import WeightLoader
from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)


def _materialize_fp32_init(model: torch.nn.Module, device: torch.device) -> None:
    """Draw FP32 initializers on the full (unsharded) model, then store BF16.

    MoonEP must do this before it registers symmetric views: a later ``to_empty``
    would drop them. MegaMoe's default path shards first and initializes each
    local shard, which is a different model. ``SCRATCH_INIT_BEFORE_SHARD=1``
    makes that path use this same full-then-shard sequence so the two
    dispatchers start from the same weights.
    """
    from mindspeed_llm.fsdp2.models.model_loader import reset_hf_initialized_flag

    if not any(param.is_meta for param in model.parameters()):
        return
    logger.info_rank0(f"> Materializing full model on {device} for FP32 init")
    model.to_empty(device=device)
    reset_hf_initialized_flag(model)
    if hasattr(model, "init_weights"):
        model.float()
        model.init_weights()
    model.to(dtype=torch.bfloat16)


class MindSpeedParallelEngine(torch.nn.Module):
    def __init__(
        self,
        config: ParallelEngineConfig,
        model: torch.nn.Module,
        init_device: str = "cpu",
        weights_path: Optional[str] = None,
        hf_config: Optional[Any] = None,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.init_device = init_device
        self.weights_path = weights_path
        self.hf_config = hf_config

        self.parallel_state = init_parallel_state(self.config)
        self.apply_quantization_modules()
        self.apply_tp_modules()
        self.apply_ep_modules()
        self.apply_cp_modules()
        self.apply_recompute_modules()
        self.apply_chunk_batch_modules()
        self.apply_fsdp_modules()

        # dev_gp initializes after FSDP: to_empty on the sharded model, then
        # init_weights(). A full-model FP32 init before EP consumes a different
        # RNG stream, so MoonEP/Hybrid would not match fused/MegaMoe. efsdp>1
        # only mirrors those parameters, so attach the mirror after this init.
        # efsdp=1 replaces parameter storage with a VMM home during EP setup and
        # still initializes before that replacement.
        dispatcher = getattr(self.config.ep_plan, "dispatcher", None)
        moonep = dispatcher in ("moonep", "moonep_megamoe")
        scratch_initialized = (
            os.environ.get("SCRATCH_INIT_BEFORE_SHARD", "0") == "1"
            and self.weights_path is None
            and self.config.expert_parallel_size > 1
            and not moonep
        )
        if self.init_device == "meta" and not scratch_initialized:
            logger.info_rank0("> Loading weights after FSDP wrapping...")
            WeightLoader.load(model=self.model, weights_path=self.weights_path, device=None, hf_config=self.hf_config)
        elif scratch_initialized:
            logger.info_rank0("> Keeping pre-shard scratch init; skip WeightLoader re-init")

        attach_plan = getattr(self, "_moonep_attach_plan", None)
        if attach_plan is not None:
            from fsdp_turbo.distributed.expert_parallel.expert_parallel import (
                attach_moonep_after_expert_fsdp,
            )

            logger.info_rank0("> Attaching MoonEP mirrors to baseline-initialized parameters")
            self.model = attach_moonep_after_expert_fsdp(
                self.model,
                self.parallel_state.get_ep_device_mesh(),
                attach_plan,
            )

        # After FSDP so policy globs match named_modules(); after recompute so
        # an empty recompute_plan cannot be confused with UAM still being off.
        self.apply_uam_modules()

    def apply_fsdp_modules(self):
        # Mirror FSDPTurbo.apply_fsdp_modules: 2D (dp, fsdp) HSDP is only
        # required when the replicate (dp) dim is actually > 1. With dp=1,
        # FSDP2 still treats a (1, N) mesh as HSDP (empty dp all-reduce, 2D
        # DTensor dispatch), which slows the whole train_step; keep the 1D
        # fsdp mesh there (a14-proven behavior).
        if self.parallel_state.data_parallel_size > 1:
            fsdp_mesh = self.parallel_state.get_hsdp_device_mesh('dp_fsdp')
        else:
            fsdp_mesh = self.parallel_state.get_fsdp_device_mesh()
        self.model = fully_shard_parallel_modules(
            self.model, fsdp_mesh, self.config.fsdp_plan
        )

    def apply_tp_modules(self):
        if self.config.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(self.model, self.parallel_state.get_tp_device_mesh(), self.config.tp_plan)

    def apply_ep_modules(self):
        if self.config.expert_parallel_size > 1:
            plan = self.config.ep_plan
            if (
                os.environ.get("SCRATCH_INIT_BEFORE_SHARD", "0") == "1"
                and self.weights_path is None
                and plan.dispatcher not in ("moonep", "moonep_megamoe")
            ):
                device = torch.device("npu", torch.npu.current_device())
                _materialize_fp32_init(self.model, device)
            if plan.dispatcher in ("moonep", "moonep_megamoe"):
                self._apply_moonep_modules(plan)
                return
            self.model = expert_parallelize_modules(
                self.model, self.parallel_state.get_ep_device_mesh(), plan
            )
            # Mirror FSDPTurbo.apply_ep_modules: with edp=1 a (1, E) mesh is
            # still ndim=2 HSDP (extra dispatch overhead), so keep the 1D
            # efsdp mesh there; with edp>1 pass the 2D (edp, efsdp) mesh for
            # expert HSDP.
            if self.parallel_state.expert_data_parallel_size > 1:
                efsdp_mesh = self.parallel_state.get_hsdp_device_mesh('edp_efsdp')
            else:
                efsdp_mesh = self.parallel_state.get_efsdp_device_mesh()
            self.model = expert_fully_shard_modules(
                self.model, efsdp_mesh, self.config.ep_plan, self.config.fsdp_plan
            )

    def _apply_moonep_modules(self, plan):
        """MoonEP/Hybrid experts: efsdp=1 uses VMM homes; efsdp>1 uses FSDP2 then mirror."""
        from mindspeed_llm.fsdp2.utils.global_vars import get_args
        from fsdp_turbo.distributed.expert_parallel.expert_parallel import (
            expert_parallelize_modules as turbo_expert_parallelize_modules,
        )
        from fsdp_turbo.distributed.expert_parallel.moonep.plan import fill_moonep_static_shape

        args = get_args()
        fill_moonep_static_shape(
            plan,
            self.model,
            batch_size=args.per_device_train_batch_size,
            max_seq_length=args.cutoff_len,
            ulysses_size=1,
        )
        efsdp_size = self.config.expert_fully_shard_parallel_size
        # efsdp=1: per-layer VMM homes, no pool mirror / expert FSDP wrap.
        # efsdp>1: EP shard → expert FSDP2 → same WeightLoader init as dev_gp →
        # MoonEP pooled VMM mirror. The mirror must be attached after init;
        # to_empty inside WeightLoader would drop an earlier symmetric view.
        fsdp_owned = efsdp_size > 1
        if not fsdp_owned:
            device = torch.device("npu", torch.npu.current_device())
            _materialize_fp32_init(self.model, device)
        edp_mesh = None
        if not fsdp_owned and self.parallel_state.get_edp_group_size() > 1:
            edp_mesh = self.parallel_state.get_device_mesh("edp")
        self.model = turbo_expert_parallelize_modules(
            self.model,
            self.parallel_state.get_ep_device_mesh(),
            plan,
            edp_mesh=edp_mesh,
            efsdp_mesh=None,
            weight_mesh=None,
            fsdp_owned_weights=fsdp_owned,
            attach_moonep=not fsdp_owned,
        )
        if fsdp_owned:
            if self.parallel_state.get_edp_group_size() > 1:
                efsdp_mesh = self.parallel_state.device_mesh_map["efsdp"][("edp", "efsdp")]
            else:
                efsdp_mesh = self.parallel_state.get_efsdp_device_mesh()
            self.model = expert_fully_shard_modules(
                self.model, efsdp_mesh, plan, self.config.fsdp_plan
            )
            self._moonep_attach_plan = plan

    def apply_cp_modules(self):
        VALID_CP_TYPES = ("ulysses", "ring", "kvallgather")
        cp_size = self.config.context_parallel_size
        cp_type = self.config.context_parallel_type

        if cp_size > 1:
            if cp_type not in VALID_CP_TYPES:
                raise ValueError(f"context_parallel_type must be one of {VALID_CP_TYPES}")
            if cp_type == "ulysses" and self.model.config.num_attention_heads % cp_size != 0:
                raise ValueError(f"num_attention_heads must be divisible by context_parallel_size (current: {cp_size})")
            apply_context_parallelize_modules(self.model, self.config.cp_plan)

    def apply_recompute_modules(self):
        if not self.config.recompute:
            return
        self.model = recompute_modules(self.model, self.config.recompute_plan)

    def apply_uam_modules(self):
        from mindspeed_llm.fsdp2.features.unified_activation_memory import apply_uam_modules

        apply_uam_modules(self)

    def apply_chunk_batch_modules(self):
        if not self.config.enable_chunk_batch:
            return
        self.model = chunk_batch_modules(self.model, self.config.chunkbatch_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization_plan.quant_recipe:
            return
        try:
            if self.config.recompute:
                self.config.quantization_plan.fsdp_low_precision_all_gather_mode = "all"

            from fsdp_turbo.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.config.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError("Failed to convert quantization plan") from e

    def apply_optimizer_hook(self, optimizer: torch.optim.Optimizer):
        if not self.config.quantization_plan.quant_recipe:
            return
        from mindspeed.fsdp.quantization.core.cache import hook_optimizer_step

        hook_optimizer_step(self.model, optimizer)

    def apply_ngram_hook(self):
        model_type = str(getattr(self.hf_config, "model_type", "") or "").lower()
        if model_type != "longcat_flash_ngram":
            return

        if hasattr(self.model, "_install_ngram_embedding_early_post_backward"):
            self.model._install_ngram_embedding_early_post_backward()
            logger.info_rank0("> Applied LongCat-Flash-Lite N-gram embedding FSDP2 optimization.")

    def apply_model_hooks(self, optimizer: torch.optim.Optimizer):
        self.apply_optimizer_hook(optimizer)
        self.apply_ngram_hook()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
