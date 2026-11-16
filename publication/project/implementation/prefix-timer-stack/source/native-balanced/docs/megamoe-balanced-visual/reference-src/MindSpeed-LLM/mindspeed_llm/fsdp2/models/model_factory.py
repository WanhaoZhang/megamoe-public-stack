import os
import importlib
import torch
import torch.distributed as dist
from transformers import AutoConfig

from mindspeed_llm.fsdp2.models.model_registry import ModelRegistry
from mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine import MindSpeedParallelEngine
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import (
    ParallelEngineConfig,
    FSDPPlanConfig,
    TPPlanConfig,
    EPPlanConfig,
    CPPlanConfig,
    QuantizeConfig,
    ChunkBatchPlanConfig,
)

from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.models.model_loader import ModelLoader
from mindspeed_llm.fsdp2.utils.global_vars import get_args

# FSDPPlanConfig / TPPlanConfig / ChunkBatchPlanConfig inherit their constructor fields from
# the external `fsdp_turbo` package, which pylint cannot resolve in the lint environment, so
# every keyword passed to those constructors is a false E1123 (unexpected-keyword-arg).
# pylint: disable=unexpected-keyword-arg

logger = get_logger(__name__)

# 未通过 CLI 指定架构参数时的回退值（= 当前 256p 部署基线 192/61）。
DEFAULT_N_ROUTED_EXPERTS = 192
DEFAULT_NUM_HIDDEN_LAYERS = 61



def _get_config_class(model_id: str):
    """Dynamically import the config class by model_id."""
    try:
        module_path = f"transformers.models.{model_id}.configuration_{model_id}"
        config_module = importlib.import_module(module_path)
        # Convert model_id like "qwen3_next" to "Qwen3NextConfig"
        parts = model_id.split("_")
        class_name = "".join(p.capitalize() for p in parts) + "Config"
        return getattr(config_module, class_name)
    except (ImportError, AttributeError):
        return None


# ==============================================================================
# ModelFactory
# ==============================================================================
class ModelFactory:
    """
    Responsible for building HuggingFace native models and wrapping them
    as MindSpeed FSDP instances based on parallelization arguments.

    Supports two initialization modes controlled by model_args.init_model_with_meta_device:
    - False: Load model fully on CPU (original behavior)
    - True: Create empty model on meta device, load weights after FSDP wrapping
    """

    @staticmethod
    def create(model_args, parallel_args):
        """
        Creates a MindSpeed FSDP wrapped model.

        Args:
            model_args: Contains model_name_or_path, trust_remote_code, train_from_scratch,
                        init_model_with_meta_device, etc.
            parallel_args: Contains tp_size, fsdp_size, recompute, ep_size, etc.
        """
        # 1. Setup Device
        # Ensure NPU is being used
        if torch.accelerator.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            target_device = torch.device(torch.accelerator.current_accelerator().type, local_rank)
            torch.accelerator.set_device(target_device)
        else:
            target_device = torch.device("cpu")

        # 2. Determine initialization device based on init_model_with_meta_device flag
        use_meta_device = getattr(model_args, 'init_model_with_meta_device', False)
        init_device = "meta" if use_meta_device else "cpu"
        logger.info_rank0(
            f"> Model initialization device: {init_device} (init_model_with_meta_device={use_meta_device})"
        )

        # 3. Load HF Config
        logger.info_rank0(f"> Loading AutoConfig from {model_args.model_name_or_path}...")
        trust_remote_code = model_args.trust_remote_code
        attn_implementation = "eager" if parallel_args.cp_size > 1 else None

        hf_config = None
        model_id = getattr(model_args, 'model_id', None)
        if model_id:
            config_cls = _get_config_class(model_id)
            if config_cls is not None:
                logger.info_rank0(f"> Using direct config class: {config_cls.__name__}")
                hf_config = config_cls.from_pretrained(
                    model_args.model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    attn_implementation=attn_implementation,
                )
        if hf_config is None:
            hf_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )

        # 4. Load HF Model
        # Decide loading method based on init_device and whether training from scratch or fine-tuning.
        # if model_id is configured, load model according to model_id.
        model_cls = None
        if getattr(model_args, 'model_id', None):
            logger.info_rank0(f"> Using factory mode with model_id: {model_args.model_id}")
            model_cls = ModelRegistry.get_model_class(model_args.model_id)
            if hasattr(model_cls, 'register_patches'):
                model_cls.register_patches(model_args)

        # 模型架构覆盖：CLI 优先（--model.n_routed_experts 等），未指定则回退
        # 到用例基线值。见 patch_model_factory_args.py 顶部说明。
        hf_config.n_routed_experts = model_args.n_routed_experts if model_args.n_routed_experts else DEFAULT_N_ROUTED_EXPERTS
        hf_config.num_hidden_layers = model_args.num_hidden_layers if model_args.num_hidden_layers else DEFAULT_NUM_HIDDEN_LAYERS
        if model_args.moe_intermediate_size:
            hf_config.moe_intermediate_size = model_args.moe_intermediate_size
        if model_args.num_experts_per_tok:
            hf_config.num_experts_per_tok = model_args.num_experts_per_tok

        # num_hidden_layers 被覆盖后必须重算派生的逐层 schedule，否则
        # DeepseekV4Config.validate_layer_type() 会因长度不匹配直接抛错。
        n_layers = hf_config.num_hidden_layers
        n_hash = getattr(hf_config, "default_num_hash_layers", 3) or 3
        base_layer_types = getattr(hf_config, "layer_types", None)
        if base_layer_types:
            if len(base_layer_types) >= n_layers:
                layer_types = list(base_layer_types[:n_layers])
            else:
                layer_types = list(base_layer_types) + [
                    base_layer_types[-1] if base_layer_types else "compressed_sparse_attention"
                ] * (n_layers - len(base_layer_types))
        else:
            interleave = [
                "compressed_sparse_attention" if i % 2 else "heavily_compressed_attention"
                for i in range(max(n_layers - 2, 0))
            ]
            layer_types = ["heavily_compressed_attention"] * min(n_layers, 2) + interleave
        hf_config.layer_types = layer_types
        hf_config.mlp_layer_types = ["hash_moe"] * min(n_layers, n_hash) + ["moe"] * max(0, n_layers - n_hash)

        logger.info_rank0(
            f"> MoE arch override -> n_routed_experts={hf_config.n_routed_experts}, "
            f"num_hidden_layers={hf_config.num_hidden_layers}, "
            f"moe_intermediate_size={hf_config.moe_intermediate_size}, "
            f"num_experts_per_tok={hf_config.num_experts_per_tok}, "
            f"hash_moe_layers={hf_config.mlp_layer_types.count('hash_moe')}"
        )

        # Use ModelLoader to create model based on init_device
        loader = ModelLoader(model_args, init_device=init_device, hf_config=hf_config)
        model, weights_path = loader.create_model(model_cls=model_cls)
        hf_config = loader.hf_config

        # 5. Build MindSpeed FSDP Configuration
        # Dynamically calculate Data Parallel (DP) Size
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Guard against division by zero if args are not set correctly
        tp_size = parallel_args.tp_size
        fsdp_size = parallel_args.fsdp_size
        cp_size = parallel_args.cp_size
        dp_size = world_size // (tp_size * fsdp_size * cp_size)

        parallel_config = ModelFactory._build_parallel_config(model_args, parallel_args, dp_size)

        # 6. Wrap & Move
        logger.info_rank0(f"> Wrapping model with MindSpeed FSDP (TP={tp_size},CP={cp_size}, FSDP={fsdp_size})...")

        # MindSpeed FSDP will shard and wrap the CPU model based on the config.
        # The wrapped model automatically handles forward/backward communication.
        # Pass init_device, weights_path, and hf_config for meta device support.
        model = MindSpeedParallelEngine(
            config=parallel_config, model=model, init_device=init_device, weights_path=weights_path, hf_config=hf_config
        )

        # 7. Move to target device
        # For cpu mode: move to NPU after wrapping
        # For meta mode: weights are already loaded to device in MindSpeedParallelEngine
        if init_device == "cpu":
            model = model.to(target_device)

        return model

    @staticmethod
    def _build_parallel_config(model_args, parallel_args, dp_size) -> 'ParallelEngineConfig':
        """
        Builds the Config based on parallel_args and hardcoded layer name rules.
        Note: The wildcards here (e.g., 'model.layers.{*}') are suitable for standard structures like Llama/Qwen.
        If using other non-standard models, these strings might need adjustment.
        """
        # --- 1. FSDP Plan ---
        # Requirement: Apply FSDP to transformer layers
        apply_modules = {
            parallel_args.fsdp_modules[0]: {
                'reshard_after_forward': parallel_args.reshard_after_forward,
                'shard_placement_fn': parallel_args.shard_placement_fn,
            },
        }
        for modules in parallel_args.fsdp_modules[1:]:
            apply_modules[modules] = {
                'reshard_after_forward': parallel_args.reshard_after_forward,
            }
        fsdp_plan = FSDPPlanConfig(
            ignored_modules=parallel_args.ignored_modules if parallel_args.ignored_modules else [],
            apply_modules=apply_modules,
            param_dtype=parallel_args.param_dtype,
            reduce_dtype=parallel_args.reduce_dtype,
            num_to_forward_prefetch=parallel_args.num_to_forward_prefetch,
            num_to_backward_prefetch=parallel_args.num_to_backward_prefetch,
            hook_modules=parallel_args.hook_modules,
            fsdp_implementation=parallel_args.fsdp_implementation,
        )

        # --- 2. Tensor Parallel Plan ---
        # Requirement: Column Parallel for Q/K/V/Gate/Up, Row Parallel for O/Down
        tp_plan = TPPlanConfig(colwise_parallel=parallel_args.tp_colwise, rowwise_parallel=parallel_args.tp_rowwise)

        # --- 3. Expert Parallel Plan ---
        # For Mixture-of-Experts (MoE) models
        ep_size = parallel_args.ep_size
        ep_fsdp_size = parallel_args.ep_fsdp_size
        args = get_args()
        # edp = world // (ep * ep_fsdp)
        ep_world_size = dist.get_world_size() if dist.is_initialized() else 1
        edp_size = ep_world_size // (ep_size * ep_fsdp_size)

        moon_cfg = None
        if parallel_args.ep_dispatcher in ("moonep", "moonep_megamoe"):
            from fsdp_turbo.fsdp_turbo_config import MoonEPConfig

            # Same depths expert FSDP would have used for set_modules_to_*_prefetch.
            # Each step of depth adds one registered full-home buffer per matrix shape.
            moon_cfg = MoonEPConfig(
                num_to_forward_prefetch=parallel_args.num_to_forward_prefetch,
                num_to_backward_prefetch=parallel_args.num_to_backward_prefetch,
                # Multi-node symmetric-heap bootstrap endpoint. Each EP group's
                # rank0 hosts the config store, so the URL must point at *this
                # node's EP-group root* -- injected per node as
                # MOONEP_SHMEM_IP_PORT="tcp6://[<EP-group-root-IPv6>]:<port>"
                # (same root set as deepep's SHMEM_ADDR). Single-node runs leave
                # it unset and use the UID-broadcast path (no TCP bootstrap).
                shmem_ip_port=os.environ.get("MOONEP_SHMEM_IP_PORT"),
            )
        balanced_cfg = None
        if parallel_args.ep_dispatcher == "megamoe_balanced":
            from fsdp_turbo.fsdp_turbo_config import MegaMoeBalancedConfig
            balanced_cfg = MegaMoeBalancedConfig(
                planner=args.megamoe_balanced_planner,
                replica_slots=args.megamoe_balanced_replica_slots,
                backward_restore=args.megamoe_balanced_backward_restore,
                recv_capacity=args.megamoe_balanced_recv_capacity,
                plan_validation=args.megamoe_balanced_plan_validation,
                save_gmm1_output=args.megamoe_balanced_save_gmm1_output,
                enable_diagnostics=args.megamoe_balanced_enable_diagnostics,
            )
        ep_plan = EPPlanConfig(
            apply_modules=parallel_args.ep_modules,
            apply_efsdp_modules=parallel_args.ep_fsdp_modules,
            dispatcher=parallel_args.ep_dispatcher,
            fixed_router=args.fix_router,
            moonep_config=moon_cfg,
            megamoe_balanced_config=balanced_cfg,
        )

        cp_plan = CPPlanConfig(context_parallel_type=parallel_args.cp_type, is_pack=getattr(model_args, "pack", False))

        # --- 4. Recompute Plan ---
        # Activation Checkpointing
        recompute_plan = parallel_args.recompute_modules if parallel_args.recompute else []

        # --- 5. Quantization Config ---
        quantization_plan = QuantizeConfig(
            quant_recipe=model_args.quant_recipe_name,
            quant_format=model_args.quant_format,
            block_size=model_args.quant_block_size,
            quant_apply_modules=model_args.quant_apply_modules,
            quant_ignored_modules=model_args.quant_ignored_modules,
            converters=model_args.quant_converters,
            enable_fsdp_low_precision_all_gather=model_args.enable_fsdp_low_precision_all_gather,
            fsdp_low_precision_all_gather_mode=model_args.fsdp_low_precision_all_gather_mode,
        )
        # --- 6. ChunkMBS Plan ---
        chunkbatch_plan = ChunkBatchPlanConfig(
            apply_modules=parallel_args.chunk_mbs_modules,
            chunk_mbs=parallel_args.chunk_mbs,
            batch_dim=parallel_args.chunk_mbs_batch_dim,
            chunk_arg_indexes=parallel_args.chunk_mbs_arg_indexs,
            chunk_kwarg_names=parallel_args.chunk_mbs_kwarg_names,
        )

        # --- 7. Assemble Config ---
        # Get parallel sizes safely
        tp_size = parallel_args.tp_size
        fsdp_size = parallel_args.fsdp_size

        config = ParallelEngineConfig(
            # Parallelism parameters
            data_parallel_size=dp_size,
            fully_shard_parallel_size=fsdp_size,
            fsdp_plan=fsdp_plan,
            tensor_parallel_size=tp_size,
            tp_plan=tp_plan,
            # Expert Parallelism
            expert_parallel_size=ep_size,
            expert_fully_shard_parallel_size=ep_fsdp_size,
            expert_data_parallel_size=edp_size,
            ep_plan=ep_plan,
            # Context Parallelism
            context_parallel_size=parallel_args.cp_size,
            context_parallel_type=parallel_args.cp_type,
            cp_plan=cp_plan,
            # Recomputation
            recompute=parallel_args.recompute,
            recompute_plan=recompute_plan,
            # Quantization
            quantization_plan=quantization_plan,
            # ChunkBatch
            enable_chunk_batch=parallel_args.enable_chunk_batch,
            chunkbatch_plan=chunkbatch_plan,
        )

        return config


# ==============================================================================
# [Facade] AutoModelFactory
# Unified entry point used by AutoTrainer
# ==============================================================================
class AutoModelFactory:
    """
    Unified Factory for creating models.
    Dispatches to ModelFactory or McoreModelFactory based on the environment.
    """

    @staticmethod
    def create(*args, **kwargs):
        """
        Factory method that forwards arguments to the specific implementation.
        """
        return ModelFactory.create(*args, **kwargs)
