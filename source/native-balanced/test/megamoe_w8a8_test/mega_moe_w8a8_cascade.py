"""W8A8 MegaMoe reference assembled from CANN small operators."""

from typing import NamedTuple


QUANT_MODE_MXFP8 = 4
COMM_QUANT_MODE_NONE = 0
MXFP8_E4M3FN_CODE = 24
HCCL_EP_BUFFER_SIZE = 6000


class CascadeResult(NamedTuple):
    y: object
    expert_token_nums: object


def _require_callable(owner, name: str):
    function = getattr(owner, name, None)
    if function is None or not callable(function):
        raise RuntimeError(
            f"required cascade operator {name} is unavailable; install a "
            "torch_npu/CANN ops-transformer build that provides it"
        )
    return function


def _reinterpret_storage(tensor, dtype, name: str):
    import torch

    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(
            f"{name} storage dtype must be a torch.dtype, got "
            f"{dtype!r} ({type(dtype).__name__})"
        )
    target_item_size = torch.empty((), dtype=dtype).element_size()
    if tensor.element_size() != target_item_size:
        raise RuntimeError(
            f"{name} uses {tensor.element_size()}-byte storage but {dtype} "
            f"uses {target_item_size}-byte storage"
        )
    return tensor.contiguous().view(dtype)


def _normalize_mx_scale(scale):
    if scale.dim() == 2:
        if scale.shape[-1] % 2 != 0:
            raise RuntimeError(
                "MX scale count must be even for the packed [..., groups, 2] "
                "layout"
            )
        return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)
    if scale.dim() == 3 and scale.shape[-1] == 2:
        return scale
    raise RuntimeError(
        f"unexpected dispatch MX scale shape {tuple(scale.shape)}"
    )


def init_hccl_group(rank: int, world_size: int, init_method: str, device):
    """Create the HCCL group required by MoE distribute operators."""
    import torch
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialized before MegaMoe cascade"
        )
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    options.hccl_config = {
        "hccl_buffer_size": HCCL_EP_BUFFER_SIZE,
    }
    group = dist.new_group(
        ranks=list(range(world_size)),
        backend="hccl",
        pg_options=options,
    )
    backend = group._get_backend(torch.device(device.type))
    comm_name = backend.get_hccl_comm_name(rank)
    return group, comm_name


def destroy_hccl_group() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _run_gmm1_fused(
    torch_npu,
    expand_x,
    dynamic_scales,
    expert_token_nums,
    weight1,
    weight_scales1,
    output_dtype,
    e8m0_dtype,
):
    grouped_matmul_swiglu_quant = _require_callable(
        torch_npu,
        "npu_grouped_matmul_swiglu_quant_v2",
    )
    return grouped_matmul_swiglu_quant(
        x=expand_x,
        weight=[weight1.transpose(1, 2)],
        weight_scale=[weight_scales1.transpose(1, 2)],
        x_scale=dynamic_scales,
        group_list=expert_token_nums,
        dequant_mode=2,
        dequant_dtype=output_dtype,
        quant_mode=2,
        quant_dtype=MXFP8_E4M3FN_CODE,
        group_list_type=1,
        weight_scale_dtype=e8m0_dtype,
        x_scale_dtype=e8m0_dtype,
    )


def _run_gmm1_split(
    torch_npu,
    expand_x,
    dynamic_scales,
    expert_token_nums,
    weight1,
    weight_scales1,
    output_dtype,
    fp8_dtype,
    e8m0_dtype,
):
    import torch

    try:
        import cann_ops_nn.ops  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "cascade GMM1 split mode requires the cann_ops_nn Python package"
        ) from error

    grouped_matmul = _require_callable(torch_npu, "npu_grouped_matmul")
    y1 = grouped_matmul(
        x=[expand_x],
        weight=[weight1.transpose(1, 2)],
        group_list=expert_token_nums,
        group_type=0,
        group_list_type=1,
        per_token_scale=[dynamic_scales],
        scale=[weight_scales1.transpose(1, 2)],
        split_item=2,
        output_dtype=output_dtype,
        per_token_scale_dtype=e8m0_dtype,
        scale_dtype=e8m0_dtype,
    )[0]

    swiglu_group_quant = getattr(
        torch.ops.cann_ops_nn,
        "swiglu_group_quant",
        None,
    )
    if swiglu_group_quant is None:
        raise RuntimeError(
            "cann_ops_nn.swiglu_group_quant is not registered"
        )
    y, y_scale, _ = swiglu_group_quant(
        y1,
        weight=None,
        group_index=None,
        scale=None,
        dst_type=fp8_dtype,
        quant_mode=1,
        block_size=32,
        round_scale=True,
        clamp_limit=-1.0,
        output_origin=False,
    )
    return y, y_scale


def run_cascade(
    inputs,
    case: dict,
    rank: int,
    group_ep: str,
    gmm1_mode: str = "fused",
) -> CascadeResult:
    """Run dispatch, expert FFN and combine with CANN small operators."""
    import torch
    import torch_npu

    (
        x,
        expert_ids,
        expert_scales,
        weight1_raw,
        weight2_raw,
        weight_scales1_raw,
        weight_scales2_raw,
    ) = inputs

    fp8_storage_dtype = getattr(torch, "float8_e4m3fn", None)
    e8m0_storage_dtype = getattr(torch, "float8_e8m0fnu", None)
    e8m0_api_dtype = getattr(
        torch_npu,
        "float8_e8m0fnu",
        e8m0_storage_dtype,
    )
    if fp8_storage_dtype is None or e8m0_storage_dtype is None:
        raise RuntimeError(
            "the active torch build does not expose the FP8 E4M3 and E8M0 "
            "storage dtypes required by the W8A8 cascade"
        )
    if e8m0_api_dtype is None:
        raise RuntimeError(
            "the active torch_npu/torch build does not expose the E8M0 "
            "operator dtype required by the W8A8 cascade"
        )

    weight1 = _reinterpret_storage(
        weight1_raw,
        fp8_storage_dtype,
        "weight1",
    )
    weight2 = _reinterpret_storage(
        weight2_raw,
        fp8_storage_dtype,
        "weight2",
    )
    weight_scales1 = _reinterpret_storage(
        weight_scales1_raw,
        e8m0_storage_dtype,
        "weight_scales1",
    )
    weight_scales2 = _reinterpret_storage(
        weight_scales2_raw,
        e8m0_storage_dtype,
        "weight_scales2",
    )

    print(f"[CASCADE] rank={rank} dispatch begin", flush=True)
    dispatch = _require_callable(
        torch_npu,
        "npu_moe_distribute_dispatch_v2",
    )
    dispatch_output = dispatch(
        x=x,
        expert_ids=expert_ids,
        x_active_mask=None,
        group_ep=group_ep,
        ep_world_size=case["EP"],
        ep_rank_id=rank,
        tp_world_size=1,
        moe_expert_num=case["EP"] * case["expertPerRank"],
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        quant_mode=QUANT_MODE_MXFP8,
        global_bs=case["M"] * case["EP"],
        y_dtype=MXFP8_E4M3FN_CODE,
    )
    (
        expand_x,
        dynamic_scales,
        assist_info_for_combine,
        expert_token_nums,
        ep_recv_counts,
        tp_send_counts,
        _,
    ) = dispatch_output
    dynamic_scales = _normalize_mx_scale(dynamic_scales)
    print(
        f"[CASCADE] rank={rank} dispatch finished "
        f"expert_token_nums={tuple(expert_token_nums.shape)}",
        flush=True,
    )

    print(
        f"[CASCADE] rank={rank} gmm1_swiglu begin "
        f"mode={gmm1_mode}",
        flush=True,
    )
    if gmm1_mode == "fused":
        y_swiglu_quant, y_swiglu_scale = _run_gmm1_fused(
            torch_npu,
            expand_x,
            dynamic_scales,
            expert_token_nums,
            weight1,
            weight_scales1,
            torch.float32,
            e8m0_api_dtype,
        )
    elif gmm1_mode == "split":
        y_swiglu_quant, y_swiglu_scale = _run_gmm1_split(
            torch_npu,
            expand_x,
            dynamic_scales,
            expert_token_nums,
            weight1,
            weight_scales1,
            torch.bfloat16,
            fp8_storage_dtype,
            e8m0_api_dtype,
        )
    else:
        raise ValueError(f"unsupported cascade GMM1 mode: {gmm1_mode}")
    print(f"[CASCADE] rank={rank} gmm1_swiglu finished", flush=True)

    print(f"[CASCADE] rank={rank} gmm2 begin", flush=True)
    grouped_matmul = _require_callable(torch_npu, "npu_grouped_matmul")
    y2 = grouped_matmul(
        x=[y_swiglu_quant],
        weight=[weight2.transpose(1, 2)],
        scale=[weight_scales2.transpose(1, 2)],
        per_token_scale=[y_swiglu_scale],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_token_nums,
        output_dtype=torch.bfloat16,
        scale_dtype=e8m0_api_dtype,
        per_token_scale_dtype=e8m0_api_dtype,
    )[0]
    print(f"[CASCADE] rank={rank} gmm2 finished", flush=True)

    print(f"[CASCADE] rank={rank} combine begin", flush=True)
    combine = _require_callable(
        torch_npu,
        "npu_moe_distribute_combine_v2",
    )
    y = combine(
        expand_x=y2,
        expert_ids=expert_ids,
        assist_info_for_combine=assist_info_for_combine,
        ep_send_counts=ep_recv_counts,
        expert_scales=expert_scales.to(torch.float32),
        x_active_mask=None,
        group_ep=group_ep,
        ep_world_size=case["EP"],
        ep_rank_id=rank,
        moe_expert_num=case["EP"] * case["expertPerRank"],
        tp_send_counts=tp_send_counts,
        tp_world_size=1,
        expert_shard_type=0,
        shared_expert_num=1,
        shared_expert_rank_num=0,
        global_bs=case["M"] * case["EP"],
        comm_quant_mode=COMM_QUANT_MODE_NONE,
    )
    print(f"[CASCADE] rank={rank} combine finished", flush=True)
    return CascadeResult(y, expert_token_nums)
