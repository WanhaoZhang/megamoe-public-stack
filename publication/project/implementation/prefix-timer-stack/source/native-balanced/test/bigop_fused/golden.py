"""bigop: 朴素 MoE 的 autograd.Function（op 级 10 步前向 + 10 步手写反向）。

前向: router -> permute1 -> alltoall -> permute2 -> gmm1 -> act -> gmm2 -> unpermute1 -> alltoall -> unpermute2
反向: 逐 op 手写，全部用前向接口/ops（不依赖 autograd 自动建图，便于逐 op 调试 + 未来替换融合算子）
  - gmm 反向: npu_grouped_matmul（torch_npu 前向接口，group_list_type=1 不 cumsum）+ grouped wgrad
  - alltoall 反向 = alltoall (交换 splits)
  - permute/unpermute 互为对偶
  - act 反向: 重放 F.silu(gate) * up 小算子图并由 autograd 求导
  - sort 反向 = sort (反序)
  - router 反向: 手写对偶（topk_softmax bwd + aux_loss bwd + gate linear bwd）
  - 反向重计算（对齐 MegaMoe bf16 backward，默认开，BIGOP_BWD_RECOMPUTE=0 关闭）:
    forward 不保存 gate_up_output(Z)/intermediate(Hp)（省 R*N + R*F 激活，与融合算子
    保存契约一致），backward 从保存的 Xpacked 重算 Z = X@W1^T（kernel P0-C GMM1
    recompute）与 Hp = probs*SwiGLU(Z)（kernel P1 epilogue）；bwd GEMM 计算量从
    保存中间量口径的 12 RFK -> 16 RFK，与 v2 融合算子同口径公平对比（见
    docs/backward_perf_analysis.md 第五节）

moe_router_dtype: 对齐 megatron core Router.gating 的 config.moe_router_dtype 开关
  （None 默认 / 'fp32'）。fp32 时前向 gating = linear(x.to(fp32), W.to(fp32))
  （权重存储仍 bf16，运行时上转），logits/probs 全程 fp32；
  act = swiglu(bf16) * probs(fp32) 升 fp32 后显式 to(bf16) 进 GMM2
  （megatron moe_utils unpermute 注释 + mindspeed gmm experts activation_func_with_probs 同构）；
  反向按同一条 dtype 链手写对偶（见 backward 内 router_dtype 分支）。

依赖: 无。本文档前身依赖 megatron 的 aux_loss 相关（switch_load_balancing_loss_func /
      save_to_aux_losses_tracker / parallel_state），现已全部用纯 torch + torch.distributed
      复制为本地实现（见下方 _aux_loss / _parallel shim）。all_to_all / permute /
      unpermute / sort_chunks_by_idxs / grouped_matmul 均在 bigop 内实现。
"""
import math
import os

import torch
import torch_npu

import timing


# ---------- 并行状态 shim（替代 megatron.core.parallel_state，仅 aux_loss 序列归约需要）----------
# bigop 不 import megatron。aux_loss 的 sequence_partition_group（CP / TP+CP 组）由外部注入：
#   - megatron 环境里，训练启动后可调用 set_parallel_groups() 传入 initialize_model_parallel 建的组；
#   - standalone 单卡/朴素 alltoall dispatcher 场景可不注入，视为无序列切分（world_size=1, group=None）。
_CP_GROUP = None             # context parallel group
_TP_GROUP = None             # tensor parallel group
_TP_AND_CP_GROUP = None      # tensor + context parallel group（序列被 TP+CP 联合切分时的归约组）


def set_parallel_groups(cp_group=None, tp_group=None, tp_and_cp_group=None):
    """注入并行组（可选，standalone 可不调）。tp_group 用于 alltoall_seq 时调整 aux_loss 系数。"""
    global _CP_GROUP, _TP_GROUP, _TP_AND_CP_GROUP
    _CP_GROUP = cp_group
    _TP_GROUP = tp_group
    _TP_AND_CP_GROUP = tp_and_cp_group


def _get_context_parallel_group():
    return _CP_GROUP


def _get_tensor_and_context_parallel_group():
    return _TP_AND_CP_GROUP


def _get_tensor_model_parallel_world_size():
    if _TP_GROUP is not None:
        return torch.distributed.get_world_size(_TP_GROUP)
    return 1


def _get_tensor_and_context_parallel_world_size():
    if _TP_AND_CP_GROUP is not None:
        return torch.distributed.get_world_size(_TP_AND_CP_GROUP)
    return 1


# aux_loss 反向缩放（模拟 megatron MoEAuxLossAutoScaler.main_loss_backward_scale）
# bf16 训练里主 loss 梯度是 scaled 的（乘 loss_scale），aux_loss 经 MoEAuxLossAutoScaler 也让其梯度
# 绑定 main_loss_backward_scale，保证 unscale 后 aux_loss 贡献是真实梯度。
# bigop 手写 aux_loss 反向直接算 d(aux)/d(logits)，须额外乘该 scale 才能与 baseline 一致。
# standalone/单层测试默认 1.0（此时 main_loss_backward_scale 也为 1.0）；ST 由 st_bigop_run.py 注入真实值。
_AUX_LOSS_BACKWARD_SCALE = 1.0


def set_aux_loss_backward_scale(scale):
    """注入 aux_loss 反向的后向缩放（megatron MoEAuxLossAutoScaler.main_loss_backward_scale）。"""
    global _AUX_LOSS_BACKWARD_SCALE
    _AUX_LOSS_BACKWARD_SCALE = scale


def _get_aux_loss_backward_scale():
    return _AUX_LOSS_BACKWARD_SCALE


# ---------- aux_loss shim（替代 megatron.core.transformer.moe.moe_utils）----------

_MOE_LAYER_WISE_LOGGING_TRACKER = {}


def _switch_load_balancing_loss_func(probs, tokens_per_expert, topk, moe_aux_loss_coeff,
                                     sequence_partition_group=None):
    """Switch Transformer 负载均衡 aux loss（等价 megatron switch_load_balancing_loss_func）。
    probs: (num_tokens, num_experts) router softmax 概率；tokens_per_expert: (num_experts,)。
    序列被切分时对 tokens_per_expert 做 all_reduce（不需 probs 的梯度，故只归约违规计数）。
    """
    num_sub_sequence = 1
    if sequence_partition_group is not None:
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        torch.distributed.all_reduce(tokens_per_expert, group=sequence_partition_group)

    num_tokens = probs.shape[0] * num_sub_sequence
    num_experts = probs.shape[1]
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (num_tokens * num_tokens * topk)
    )
    return aux_loss


def _save_to_aux_losses_tracker(name, loss, layer_number, num_layers,
                                reduce_group=None, avg_group=None):
    """写本地 layer-wise aux loss tracker（等价 megatron save_to_aux_losses_tracker，仅日志用）。"""
    if layer_number is None:
        return
    tracker = _MOE_LAYER_WISE_LOGGING_TRACKER
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["values"] = torch.zeros(num_layers, device=loss.device)
    tracker[name]["values"][layer_number - 1] += loss.detach()
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group


# ---------- 通信/重排原语（从 megatron 挪入，仅朴素分支；fused 分支需 TE，NPU 不支持）----------

def all_to_all(group, input_, output_split_sizes=None, input_split_sizes=None):
    """all_to_all_single 封装（等价 megatron _AllToAll.forward，无 autograd 建图）。"""
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_
    input_ = input_.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(input_)
    else:
        output = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype, device=input_.device,
        )
    torch.distributed.all_to_all_single(
        output, input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def all_to_all_async(group, input_, output_split_sizes=None, input_split_sizes=None):
    """all_to_all_single 非阻塞封装（async_op=True，返回 handle）。

    单流 async（后续计算仍在主流，最后 handle.wait()）能让 HCCL 的 DMA 通信与主流
    AICore 计算并行（910B 实测单流 async 3.5x，勿用双流+wait_stream 会强制串行）。
    """
    world_size = torch.distributed.get_world_size(group=group)
    if world_size == 1:
        return input_, None
    input_ = input_.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(input_)
    else:
        output = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype, device=input_.device,
        )
    handle = torch.distributed.all_to_all_single(
        output, input_,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    return output, handle


class SharedExperts(torch.nn.Module):
    """Ns 个全宽共享专家（K3 风格，SwiGLU，无 gate）。

    W1[num, hidden, 2*ffn], W2[num, ffn, hidden]。前向在 BigOp.forward 内手动调用
    （不 autograd 建图），反向在 BigOp.backward 手写并累加 .grad。
      fc1_act(x): x(N,hidden) -> npu_swiglu(x@W1^T) -> (N,num,ffn)
      fc2(act):   act(N,num,ffn) -> act@W2^T -> (N,num,hidden)
    """
    def __init__(self, hidden, ffn, num=2, dtype=torch.bfloat16):
        super().__init__()
        self.num = num
        self.W1 = torch.nn.Parameter(torch.empty(num, hidden, 2 * ffn, dtype=dtype))
        self.W2 = torch.nn.Parameter(torch.empty(num, ffn, hidden, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        for W in (self.W1, self.W2):
            torch.nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    @staticmethod
    def _fc1(x, W1):
        # x: (N, hidden), W1: (num, hidden, 2*ffn) -> gate_up (N, num, 2*ffn)，逐专家 matmul + stack（autograd-friendly）
        return torch.stack([torch.matmul(x, W1[e]) for e in range(W1.shape[0])], dim=1)  # (N,hidden)@(hidden,G)->(N,G)

    @staticmethod
    def _fc2(act, W2):
        # act: (N, num, ffn), W2: (num, ffn, hidden) -> (N, num, hidden)
        return torch.stack([torch.matmul(act[:, e], W2[e]) for e in range(W2.shape[0])], dim=1)

    def fc1_act(self, x):
        # x: (N, hidden) -> gate_up (N, num, 2*ffn) -> swiglu -> (N, num, ffn)
        gate_up = self._fc1(x, self.W1)
        N, num, G = gate_up.shape
        act = torch_npu.npu_swiglu(gate_up.reshape(N * num, G), dim=-1).reshape(N, num, G // 2)
        return act

    def fc2(self, act):
        # act: (N, num, ffn) -> (N, num, hidden)
        return self._fc2(act, self.W2)


def swiglu_forward(x):
    """(N, G) 的 SwiGLU（共享专家用，2D 安全）。"""
    return torch_npu.npu_swiglu(x, dim=-1)


def swiglu_backward(grad, x):
    """(N, G) 的 SwiGLU 反向。"""
    return torch_npu.npu_swiglu_backward(grad, x, dim=-1)


def permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=False, drop_and_pad=False):
    """按 routing_map 重排 token（同 expert 聚集）。返回 (permuted_input, permuted_probs, sorted_indices)。"""
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[1]
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[
            :, :capacity
        ].contiguous().view(-1)
        if probs is not None:
            routing_map = routing_map.bool()
    else:
        routing_map = routing_map.bool().T.contiguous()
        token_indices = (
            torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        )
        sorted_indices = token_indices.masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)
    permuted_probs = probs.T.contiguous().masked_select(routing_map) if probs is not None else None
    return permuted_input, permuted_probs, sorted_indices


def unpermute(permuted_tokens, sorted_indices, restore_shape, probs=None, routing_map=None,
              fused=False, drop_and_pad=False):
    """还原 permute 顺序。probs 非 None 时先加权再 scatter_add；probs=None 仅 scatter_add（topk 合并靠同址累加）。"""
    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype
    if probs is not None:
        assert routing_map is not None
        if drop_and_pad:
            num_experts = routing_map.size(1)
            capacity = sorted_indices.size(0) // num_experts
            num_unpermuted_tokens = probs.size(0)
            probs_T_1D = probs.T.contiguous().view(-1)
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


def sort_chunks_by_idxs(input_, split_sizes, sorted_idxs, probs=None, fused=False):
    """split + 按 sorted_idxs 重排 chunks。返回 (output, permuted_probs)。"""
    chunks = torch.split(input_, split_sizes.tolist(), dim=0)
    output = torch.cat([chunks[i] for i in sorted_idxs.tolist()], dim=0)
    if probs is not None:
        prob_chunks = torch.split(probs, split_sizes.tolist(), dim=0)
        permuted_probs = torch.cat([prob_chunks[i] for i in sorted_idxs.tolist()], dim=0)
    else:
        permuted_probs = None
    return output, permuted_probs


# ---------- grouped matmul（torch_npu.npu_grouped_matmul，group_list_type=1 不 cumsum）----------

def _grouped_matmul(input_tokens, weight, tokens_per_expert):
    """y = input_tokens @ weight（grouped 前向）。
    input_tokens: (total_tokens, K), weight: (E, K, N) [K 须等于 input_tokens 末维，否则 npu_grouped_matmul 自动转置],
    tokens_per_expert: (E,) 原样每组大小（group_list_type=1，无需 cumsum）。返回 (total_tokens, N)。
    """
    group_list = tokens_per_expert.to(torch.int64).to(input_tokens.device)
    # With one x, one weight and one y, split-M requires a 3-D weight.
    # Keep the expert axis for E=1; multi-weight calls use 2-D matrices.
    weight_list = ([weight.contiguous()] if weight.shape[0] == 1 else
                   [weight[e].contiguous() for e in range(weight.shape[0])])
    return torch_npu.npu_grouped_matmul(
        [input_tokens.contiguous()], weight_list, group_list=group_list,
        split_item=2, group_type=0, group_list_type=1,
    )[0]


def _grouped_wgrad(input_tokens, grad_output, tokens_per_expert):
    """dw[e] = input_tokens[e].T @ grad_output[e]（grouped 权重梯度）。
    input_tokens: (total_tokens, K), grad_output: (total_tokens, N), tokens_per_expert: (E,)。返回 (E, K, N)。
    split_item=3 + group_type=2: input_tokens.T 按 group_list 分组作左矩阵，grad_output 作右矩阵，输出 per-expert 权重梯度。
    注意: group_type=2 要求 input_tokens 为转置视图（非连续），故用 input_tokens.T 且不可 .contiguous()（否则丢失转置标记）。
    """
    group_list = tokens_per_expert.to(torch.int64).to(input_tokens.device)
    return torch_npu.npu_grouped_matmul(
        [input_tokens.T], [grad_output], group_list=group_list,
        split_item=3, group_type=2, group_list_type=1,
    )[0]


def _accum_grad(param, g):
    """累加梯度到 param.grad，并同步到 param.main_grad（若存在）。

    Megatron Float16Optimizer 用 fp32 param.main_grad 驱动 optimizer（param.grad 是 bf16 且
    不被 optimizer 读取）；单层测试/朴素 autograd 用 param.grad。故两者都累加：
      - param.grad: 单层测试/直接 backward 场景
      - param.main_grad: ST（Float16Optimizer）场景，否则 MoE 梯度不更新

    累积精度：baseline 的 MoE 梯度经 DDP 在 main_grad（fp32）逐 micro-batch 累加
    （accumulate_allreduce_grads_in_fp32），故 bigop 也须 fp32 累加（g.to(fp32) 逐次 add_）。
    num_mb=1 时单次 backward 的 MoE 梯度二者 bit 一致（UT/单层验证）；num_mb>=2 时
    fp32 累积与 baseline 一致（iter1 逐层 rel=0）。"""
    if param.grad is None:
        param.grad = g.detach().clone()
    else:
        param.grad.add_(g)
    mg = getattr(param, "main_grad", None)
    if mg is not None:
        mg.add_(g.to(mg.dtype))


def _bwd_recompute_enabled() -> bool:
    """BIGOP_BWD_RECOMPUTE=0 关闭反向重计算，回到"forward 保存 Z/Hp、backward 直接用"的旧口径。

    默认开启（对齐 MegaMoe bf16 backward）：forward 不保存 gate_up_output(Z)/
    intermediate(Hp)，backward 重算 Z（GMM1 recompute，kernel P0-C）与 Hp
    （kernel P1 epilogue）。两种口径数值等价（重算与 forward 同 op 同输入，
    逐位一致），区别仅在显存（是否跨 forward/backward 存活 R*N + R*F 激活）与
    backward GEMM 计算量（16 vs 12 RFK）。旧口径仅供基准对比
    （见 docs/backward_perf_analysis.md 第五节）。
    """
    return os.environ.get("BIGOP_BWD_RECOMPUTE", "1").lower() not in ("0", "false", "off", "no")


def _round_to_e4m3fn(value: torch.Tensor) -> torch.Tensor:
    """Round finite FP32 values to saturated E4M3FN, returned as FP32.

    This arithmetic form avoids requiring an NPU float8 tensor dtype.  E4M3FN
    has three mantissa bits, a 2^-9 subnormal quantum and finite range 448.
    ``torch.round`` supplies the CAST_RINT ties-to-even rule used by the
    forward Ascend C saver.
    """
    magnitude = value.abs().clamp(max=448.0)
    tiny = torch.finfo(torch.float32).tiny
    exponent = torch.floor(torch.log2(magnitude.clamp_min(tiny))).clamp(-6, 8)
    normal_quantum = torch.exp2(exponent - 3.0)
    quantum = torch.where(
        magnitude < (2.0 ** -6),
        2.0 ** -9,
        normal_quantum,
    )
    rounded = (torch.round(magnitude / quantum) * quantum).clamp(max=448.0)
    return torch.where(value < 0, -rounded, rounded)


def _mxfp8_saved_roundtrip(value: torch.Tensor) -> torch.Tensor:
    """Model the forward saved-state MXFP8 encode/decode boundary.

    The production saver chooses one E8M0 power-of-two scale for every 32
    consecutive values, multiplies BF16 input by the reciprocal scale in
    BF16, casts to saturated E4M3FN with RINT, and the backward decoder casts
    the product back to BF16.
    """
    if value.shape[-1] % 32 != 0:
        raise ValueError(
            f"MXFP8 saved-state golden requires last dimension % 32 == 0, "
            f"got {value.shape[-1]}"
        )
    original_shape = value.shape
    blocks = value.reshape(-1, value.shape[-1] // 32, 32)
    fp32 = blocks.to(torch.float32)
    max_abs = fp32.abs().amax(dim=-1, keepdim=True)
    tiny = torch.finfo(torch.float32).tiny
    scale_exp = torch.floor(torch.log2(max_abs.clamp_min(tiny))) - 8.0
    scale_exp = scale_exp.clamp(-127.0, 127.0)
    # E8M0 raw zero represents 2^-127 for an all-zero group.
    scale_exp = torch.where(max_abs == 0, -127.0, scale_exp)
    scale = torch.exp2(scale_exp)
    reciprocal = torch.exp2(-scale_exp).to(torch.bfloat16)
    normalized = (blocks * reciprocal).to(torch.bfloat16).to(torch.float32)
    encoded_value = _round_to_e4m3fn(normalized)
    decoded = (encoded_value * scale).to(torch.bfloat16)
    return decoded.reshape(original_shape)


class BigOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, moe_layer, shared_experts=None):
        _t = timing.get_step_timer("bigop_fwd")  # BIGOP_TIMING=1 时逐步打点，默认 no-op
        _t.tick("start")
        token_dispatcher = moe_layer.token_dispatcher
        config = moe_layer.config
        experts = moe_layer.experts
        fused = config.moe_permute_fusion
        drop_and_pad = token_dispatcher.drop_and_pad

        token_dispatcher.hidden_shape = hidden_states.shape
        hidden_flat = hidden_states.view(-1, token_dispatcher.hidden_shape[-1])  # (N, hidden)

        # 0. router（手写，保存中间，不用 forward_func/autograd，便于正向换融合算子）
        router = moe_layer.router
        aux_loss_coeff = config.moe_aux_loss_coeff
        topk = router.topk
        # moe_router_dtype 对齐 megatron core Router.gating：fp32 时输入与权重运行时上转 fp32
        # （权重存储仍 params_dtype），logits 为 fp32；None 时按输入 dtype（现状，全 bf16）
        router_dtype = getattr(config, "moe_router_dtype", None)
        if router_dtype == "fp32":
            gate_weight = getattr(router, "weight", None)
            if gate_weight is None:
                gate_weight = router.gating.weight  # standalone FakeRouter 包成 nn.Linear
            logits = torch.nn.functional.linear(hidden_flat.to(torch.float32), gate_weight.to(torch.float32))
        else:
            logits = router.gating(hidden_flat)  # (N, E)
        # topk_softmax 朴素分支（softmax, use_pre_softmax=False, 无 capacity/group/expert_bias）
        # type_as(logits): bf16 logits 时 softmax(fp32) 结果转回 bf16（现状）；
        # fp32 logits 时保持 fp32 —— probs 全程 fp32，与 megatron topk_softmax_with_capacity 一致
        topk_values, topk_indices = torch.topk(logits, topk, dim=1)
        probs_topk_fp32 = torch.softmax(topk_values, dim=-1, dtype=torch.float32)
        probs_topk = probs_topk_fp32.type_as(logits)
        probs = torch.zeros_like(logits).scatter(1, topk_indices, probs_topk)
        routing_map = torch.zeros_like(logits).int().scatter(1, topk_indices, 1).bool()
        aux_tokens_per_expert = routing_map.sum(dim=0)  # aux_loss 用
        aux_scores = router.compute_routing_scores_for_aux_loss(logits)  # softmax(logits)
        if aux_loss_coeff > 0:
            sequence_partition_group = None
            adjusted_aux_coeff = aux_loss_coeff
            if config.moe_token_dispatcher_type == "alltoall_seq":
                sequence_partition_group = _get_context_parallel_group()
                adjusted_aux_coeff = aux_loss_coeff / _get_tensor_model_parallel_world_size()
            elif _get_tensor_and_context_parallel_world_size() > 1:
                sequence_partition_group = _get_tensor_and_context_parallel_group()
            aux_loss = _switch_load_balancing_loss_func(
                probs=aux_scores, tokens_per_expert=aux_tokens_per_expert, topk=topk,
                moe_aux_loss_coeff=adjusted_aux_coeff, sequence_partition_group=sequence_partition_group)
            _save_to_aux_losses_tracker("load_balancing_loss", aux_loss / adjusted_aux_coeff,
                router.layer_number, config.num_layers, reduce_group=sequence_partition_group)
        else:
            sequence_partition_group = None
            adjusted_aux_coeff = 0

        token_dispatcher.probs = probs
        token_dispatcher.routing_map = routing_map
        tokens_per_expert = token_dispatcher.preprocess(routing_map)
        _t.tick("router")

        # 1. permute1
        tokens_per_expert = token_dispatcher._maybe_dtoh_and_synchronize("before_permutation_1", tokens_per_expert)
        token_dispatcher.hidden_shape_before_permute = hidden_flat.shape
        (permuted_local_input_tokens, permuted_probs, token_dispatcher.reversed_local_input_permutation_mapping) = permute(
            hidden_flat, routing_map, probs=probs,
            num_out_tokens=token_dispatcher.num_out_tokens, fused=fused, drop_and_pad=drop_and_pad,
        )
        _t.tick("permute1")

        # 2. alltoall (dispatch, 单流 async，窗口插共享专家 fc1+act 与 HCCL 重叠)
        tokens_per_expert = token_dispatcher._maybe_dtoh_and_synchronize("before_ep_alltoall", tokens_per_expert)
        global_input_tokens, input_handle = all_to_all_async(
            token_dispatcher.ep_group, permuted_local_input_tokens,
            token_dispatcher.output_splits, token_dispatcher.input_splits)
        global_probs, probs_handle = all_to_all_async(
            token_dispatcher.ep_group, permuted_probs,
            token_dispatcher.output_splits, token_dispatcher.input_splits)
        shared_gate_up = None
        shared_act = None
        if shared_experts is not None:
            shared_gate_up = SharedExperts._fc1(hidden_flat, shared_experts.W1)  # (N,num,2ffn)
            _N, _num, _G = shared_gate_up.shape
            shared_act = torch_npu.npu_swiglu(shared_gate_up.reshape(_N * _num, _G), dim=-1).reshape(_N, _num, _G // 2)
        if input_handle is not None:
            input_handle.wait()
        if probs_handle is not None:
            probs_handle.wait()
        _t.tick("alltoall_dispatch")

        # 3. permute2 (sort)
        tokens_per_expert = token_dispatcher._maybe_dtoh_and_synchronize("before_permutation_2", tokens_per_expert)
        if token_dispatcher.num_local_experts > 1:
            global_input_tokens, global_probs = sort_chunks_by_idxs(
                global_input_tokens, token_dispatcher.num_global_tokens_per_local_expert.ravel(),
                token_dispatcher.sort_input_by_local_experts, probs=global_probs, fused=fused,
            )
        tokens_per_expert = token_dispatcher._maybe_dtoh_and_synchronize("before_finish", tokens_per_expert)
        _t.tick("permute2")

        # 4. gmm1
        weight1 = experts.weight1.view(token_dispatcher.num_local_experts, config.hidden_size, -1)
        gate_up_output = _grouped_matmul(global_input_tokens, weight1, tokens_per_expert)
        _t.tick("gmm1")

        # 5. act: SwiGLU(gate_up_output) * probs
        intermediate = experts.activation_func_with_probs(gate_up_output, global_probs.unsqueeze(-1))
        # fp32 路由: global_probs 是 fp32，乘法把 intermediate 提升到 fp32；对齐 mindspeed
        # activation_func_with_probs 的 res.to(x.dtype)，显式压回 bf16 再进 GMM2
        if router_dtype == "fp32" and intermediate.dtype != torch.bfloat16:
            intermediate = intermediate.to(torch.bfloat16)
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            # Test-only: exact expert-packed operand consumed by golden dW2.
            moe_layer._precision_golden_hp = intermediate.detach()
        _t.tick("swiglu")

        # 6. gmm2
        weight2 = experts.weight2.view(token_dispatcher.num_local_experts, -1, config.hidden_size)
        expert_output = _grouped_matmul(intermediate, weight2, tokens_per_expert)
        _t.tick("gmm2")

        # 7. unpermute1 (unsort)
        if token_dispatcher.num_local_experts > 1:
            expert_output, _ = sort_chunks_by_idxs(
                expert_output, token_dispatcher.num_global_tokens_per_local_expert.T.ravel(),
                token_dispatcher.restore_output_by_local_experts, fused=fused,
            )
        _t.tick("unpermute1")

        # 8. alltoall (combine, 单流 async，窗口插共享专家 fc2 与 HCCL 重叠)
        permuted_local_output, output_handle = all_to_all_async(
            token_dispatcher.ep_group, expert_output,
            token_dispatcher.input_splits, token_dispatcher.output_splits)
        shared_out = None
        if shared_experts is not None:
            shared_out = SharedExperts._fc2(shared_act, shared_experts.W2)  # (N,num,hidden)
        if output_handle is not None:
            output_handle.wait()
        _t.tick("alltoall_combine")

        # 9. unpermute2 (加权 scatter_add)
        output = unpermute(permuted_local_output, token_dispatcher.reversed_local_input_permutation_mapping,
                           restore_shape=token_dispatcher.hidden_shape_before_permute, routing_map=routing_map,
                           fused=fused, drop_and_pad=drop_and_pad)
        output = output.view(token_dispatcher.hidden_shape)
        if shared_experts is not None:
            output = output + shared_out.sum(dim=1).view(token_dispatcher.hidden_shape)
        _t.tick("unpermute2")
        _t.finish()  # sync + 相邻 Event 做差 -> 每步 ms，累计进 timing 全局 registry

        # 保存中间供 backward。重计算口径（默认）对齐 MegaMoe bf16 的保存契约：
        # 只保存 packed X / 路由元数据 / probs，不保存 gate_up_output(Z)/
        # intermediate(Hp)——融合算子反向从 Xpacked 重算（kernel P0-C/P1），
        # 省 R*N + R*F 的激活存储；BIGOP_BWD_RECOMPUTE=0 时保存中间量（旧口径，
        # backward 免重算，仅基准对比用）。
        bwd_recompute = _bwd_recompute_enabled()
        if bwd_recompute:
            ctx.save_for_backward(
                hidden_states, probs, routing_map,
                global_input_tokens, global_probs, tokens_per_expert,
                logits, topk_indices, probs_topk_fp32, aux_scores, aux_tokens_per_expert,
            )
        else:
            ctx.save_for_backward(
                hidden_states, probs, routing_map,
                global_input_tokens, global_probs, tokens_per_expert,
                gate_up_output, intermediate,
                logits, topk_indices, probs_topk_fp32, aux_scores, aux_tokens_per_expert,
            )
        ctx.bwd_recompute = bwd_recompute
        ctx.moe_layer = moe_layer
        ctx.fused = fused
        ctx.drop_and_pad = drop_and_pad
        ctx.adjusted_aux_coeff = adjusted_aux_coeff
        ctx.sequence_partition_group = sequence_partition_group
        ctx.topk = topk
        ctx.router_dtype = router_dtype
        # Test-only backward contract.  It deliberately does not affect this
        # forward: MXFP8 is compression of the saved GMM1 output, not forward
        # numerical semantics.
        ctx.backward_saved_gate_up_mode = getattr(
            config, "backward_saved_gate_up_mode", "bf16")
        ctx.shared_experts = shared_experts
        ctx.shared_gate_up = shared_gate_up
        ctx.shared_act = shared_act
        # 保存 splits 到 ctx（pipeline 下 token_dispatcher 状态会被其他 micro-batch 覆盖，backward 须用本层 forward 的 splits）
        ctx.output_splits = token_dispatcher.output_splits if hasattr(token_dispatcher, "output_splits") else None
        ctx.input_splits = token_dispatcher.input_splits if hasattr(token_dispatcher, "input_splits") else None
        ctx.num_local_experts = token_dispatcher.num_local_experts
        ctx.num_global_tokens_per_local_expert = token_dispatcher.num_global_tokens_per_local_expert
        ctx.sort_input_by_local_experts = token_dispatcher.sort_input_by_local_experts
        ctx.restore_output_by_local_experts = token_dispatcher.restore_output_by_local_experts
        ctx.reversed_local_input_permutation_mapping = token_dispatcher.reversed_local_input_permutation_mapping
        ctx.hidden_shape_before_permute = token_dispatcher.hidden_shape_before_permute
        ctx.num_out_tokens = token_dispatcher.num_out_tokens
        ctx.hidden_shape = token_dispatcher.hidden_shape
        ctx.ep_group = token_dispatcher.ep_group
        return output

    @staticmethod
    def backward(ctx, grad_output):
        _t = timing.get_step_timer("bigop_bwd")  # BIGOP_TIMING=1 时逐步打点，默认 no-op
        # 打点名 ↔ kernel 反向阶段（docs/backward_timer_stages.md / backward_perf_analysis.md）：
        #   unpermute2/alltoall_combine/unpermute1 ≈ P0-A/P0-B（grad 收集 + dO dispatch）
        #   gmm1_recompute ≈ P0-C（Z/Hp 重算）、dhp ≈ P1 GEMM、swiglu_bwd ≈ P1 epilogue 对偶、
        #   dx ≈ P2、dw1 ≈ P3、dw2 ≈ P4、permute2/alltoall_dispatch/permute1 ≈ P5/P6 对偶、
        #   router 为小算子特有（融合算子的 router 在 v2 侧手写）
        _t.tick("start")
        moe_layer = ctx.moe_layer
        token_dispatcher = moe_layer.token_dispatcher
        config = moe_layer.config
        experts = moe_layer.experts
        fused = ctx.fused
        drop_and_pad = ctx.drop_and_pad
        bwd_recompute = getattr(ctx, "bwd_recompute", True)
        saved_gate_up_mode = getattr(ctx, "backward_saved_gate_up_mode", "bf16")

        if bwd_recompute:
            # 重计算口径：Z/Hp 不在 saved_tensors 里（对齐 MegaMoe bf16 保存契约），
            # 由下方"gmm1 反向重计算"块从 Xpacked 重算
            (hidden_states, probs, routing_map,
             global_input_tokens, global_probs, tokens_per_expert,
             logits, topk_indices, probs_topk_fp32, aux_scores, aux_tokens_per_expert) = ctx.saved_tensors
            gate_up_output = None
            intermediate = None
        else:
            (hidden_states, probs, routing_map,
             global_input_tokens, global_probs, tokens_per_expert,
             gate_up_output, intermediate,
             logits, topk_indices, probs_topk_fp32, aux_scores, aux_tokens_per_expert) = ctx.saved_tensors
        # megatron router exposes .weight; the standalone FakeRouter wraps it
        # in an nn.Linear (.gating.weight) — same tensor, different handle.
        gate_weight = getattr(moe_layer.router, "weight", None)
        if gate_weight is None:
            gate_weight = moe_layer.router.gating.weight
        router_dtype = ctx.router_dtype
        shared_experts = ctx.shared_experts
        shared_gate_up = ctx.shared_gate_up
        shared_act = ctx.shared_act
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])  # (N, hidden)

        # 9. unpermute2 反向 = permute（unpermute 用 scatter_add，反向用 permute 取对应位置）
        grad_permuted_local_output = permute(
            grad_output.view(-1, grad_output.shape[-1]), routing_map,
            num_out_tokens=ctx.num_out_tokens, fused=fused, drop_and_pad=drop_and_pad,
        )[0]
        _t.tick("unpermute2")

        # 8. alltoall (combine) 反向: 交换 splits（单流 async，窗口插共享专家 fc2 反向）
        grad_expert_output, output_handle = all_to_all_async(
            ctx.ep_group, grad_permuted_local_output, ctx.output_splits, ctx.input_splits)
        grad_shared_act = None
        if shared_experts is not None:
            grad_shared_out = grad_output.view(-1, hidden_flat.shape[-1]).unsqueeze(1).expand(-1, shared_experts.num, -1)  # (N,num,hidden)
            num_e = shared_experts.num
            grad_shared_act = torch.stack([
                torch.matmul(grad_shared_out[:, e], shared_experts.W2[e].transpose(0, 1)) for e in range(num_e)
            ], dim=1)  # (N,num,ffn)
            grad_W2 = torch.stack([
                torch.matmul(shared_act[:, e].transpose(0, 1), grad_shared_out[:, e]) for e in range(num_e)
            ], dim=0)  # (num,ffn,hidden)
            _accum_grad(shared_experts.W2, grad_W2)
        if output_handle is not None:
            output_handle.wait()
        _t.tick("alltoall_combine")

        # 7. unpermute1 (unsort) 反向: sort 反向（用 sort_input 的逆索引）
        if ctx.num_local_experts > 1:
            grad_expert_output, _ = sort_chunks_by_idxs(
                grad_expert_output, ctx.num_global_tokens_per_local_expert.ravel(),
                ctx.sort_input_by_local_experts, fused=fused,
            )
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            # Test-only: exact expert-packed dO operand consumed by golden dW2.
            experts._precision_golden_do = grad_expert_output.detach()
        _t.tick("unpermute1")

        # ---- gmm1 反向重计算（对齐 MegaMoe bf16 backward 的 P0-C/P1 重算）----
        # 融合算子为省 R*N + R*F 激活存储，不保存 Z(gate_up_output)/Hp(intermediate)，
        # backward 从保存的 Xpacked 重算：Z = X@W1^T（kernel P0-C GMM1 recompute）、
        # Hp = probs*SwiGLU(Z)（kernel P1 epilogue）。对齐后 bwd GEMM 计算量与融合
        # 算子同为 16 RFK（重算税 25%）。kernel 侧 P0-C 与 dO 的 dispatch（P0-A/P0-B）
        # 在 AIC/AIV 上重叠；小算子链单流串行，按数据流放在 unsort 之后、gmm2 反向
        # 之前（与 kernel 阶段序一致）。BIGOP_BWD_RECOMPUTE=0 时用 forward 保存的
        # Z/Hp 跳过重算（旧口径，数值等价）。
        weight1 = experts.weight1.view(ctx.num_local_experts, config.hidden_size, -1)
        if bwd_recompute:
            gate_up_output = _grouped_matmul(global_input_tokens, weight1, tokens_per_expert)
        raw_gate_up_output = gate_up_output
        if saved_gate_up_mode == "mxfp8":
            gate_up_output = _mxfp8_saved_roundtrip(raw_gate_up_output)
        if bwd_recompute or saved_gate_up_mode == "mxfp8":
            raw_gate, raw_up = torch.chunk(raw_gate_up_output, 2, dim=-1)
            gate, up = torch.chunk(gate_up_output, 2, dim=-1)
            swiglu_limit = getattr(config, "swiglu_limit", None)
            if swiglu_limit is not None and swiglu_limit > 0:
                if saved_gate_up_mode == "mxfp8":
                    # Explicit MXFP8+clipping stores masks from raw forward
                    # BF16. Use those decisions even when decoding crosses a
                    # boundary.
                    gate = torch.where(raw_gate > swiglu_limit, swiglu_limit, gate)
                    up = torch.where(raw_up < -swiglu_limit, -swiglu_limit, up)
                    up = torch.where(raw_up > swiglu_limit, swiglu_limit, up)
                else:
                    gate = torch.clamp_max(gate, swiglu_limit)
                    up = torch.clamp(up, -swiglu_limit, swiglu_limit)
            intermediate = (torch.nn.functional.silu(gate) * up) * global_probs.unsqueeze(-1)
            # fp32 路由：与 forward 同一截断点（fp32 乘积压回 bf16 再进 dW2）
            if router_dtype == "fp32" and intermediate.dtype != torch.bfloat16:
                intermediate = intermediate.to(torch.bfloat16)
        _t.tick("gmm1_recompute")

        # 6. gmm2 反向: dx = grad_output @ weight2.T, dw2 = intermediate.T @ grad_output
        weight2 = experts.weight2.view(ctx.num_local_experts, -1, config.hidden_size)
        weight2_transposed = weight2.transpose(-1, -2).contiguous()  # (E, hidden, ffn)
        _t.tick("w2_transpose")
        grad_intermediate = _grouped_matmul(grad_expert_output, weight2_transposed, tokens_per_expert)
        _t.tick("dhp")  # P1 GEMM: dHp = dO@W2
        # experts.weight2 梯度：_grouped_wgrad 算 bf16 grad_weight，_accum_grad bf16->fp32 累积到 weight.grad/main_grad，
        # 与 baseline(mindspeed op_dw + DDP main_grad.add_) 的 bf16 累积链一致（bit 对齐）。不用 npu_groupmatmul_add_fp32（fp32 直累积，不等价）。
        grad_weight2 = _grouped_wgrad(intermediate, grad_expert_output, tokens_per_expert).view_as(experts.weight2)
        _accum_grad(experts.weight2, grad_weight2)
        _t.tick("dw2")  # P4 wgrad: dW2 = dO^T@Hp

        # 5. act 反向: differentiate the same PyTorch small-op graph used by
        # forward: F.silu(gate) * up, followed by probs multiplication.
        #    fp32 路由时 global_probs 为 fp32，前向乘积升 fp32 后 to(bf16)（截断点），
        #    反向: grad_intermediate(bf16) 先无损上转 fp32 过截断点，*probs 的 bwd 全程 fp32，
        #    probs 的 grad 用 fp32 sum（对偶 megatron autograd 的 cast 节点行为），
        #    swiglu 反向输入须与 swiglu 输出同 dtype（bf16），故 grad_swiglu 过截断点后收口回 bf16。
        probs_expanded = global_probs.unsqueeze(-1)
        raw_gate, raw_up = torch.chunk(raw_gate_up_output, 2, dim=-1)
        gate, up = torch.chunk(gate_up_output, 2, dim=-1)
        swiglu_limit = getattr(config, "swiglu_limit", None)
        if swiglu_limit is not None and swiglu_limit > 0:
            if saved_gate_up_mode == "mxfp8":
                gate = torch.where(raw_gate > swiglu_limit, swiglu_limit, gate)
                up = torch.where(raw_up < -swiglu_limit, -swiglu_limit, up)
                up = torch.where(raw_up > swiglu_limit, swiglu_limit, up)
            else:
                gate = torch.clamp_max(gate, swiglu_limit)
                up = torch.clamp(up, -swiglu_limit, swiglu_limit)
        swiglu_output = torch.nn.functional.silu(gate) * up
        if router_dtype == "fp32":
            grad_intermediate_fp32 = grad_intermediate.to(torch.float32)
            grad_probs_global = (grad_intermediate_fp32 * swiglu_output.to(torch.float32)).sum(dim=-1)
            grad_swiglu = (grad_intermediate_fp32 * probs_expanded).to(gate_up_output.dtype)
        else:
            grad_swiglu = grad_intermediate * probs_expanded                               # *probs 的 bwd
            grad_probs_global = (grad_intermediate * swiglu_output).sum(dim=-1)              # probs 的 grad
        # BigOp is a custom autograd.Function, so explicitly re-enable autograd
        # for this golden subgraph instead of substituting a fused backward
        # with potentially different BF16 rounding points.
        with torch.enable_grad():
            gate_up_replay = gate_up_output.detach().requires_grad_(True)
            replay_gate, replay_up = torch.chunk(gate_up_replay, 2, dim=-1)
            if swiglu_limit is not None and swiglu_limit > 0:
                if saved_gate_up_mode == "mxfp8":
                    replay_gate = torch.where(
                        raw_gate > swiglu_limit, swiglu_limit, replay_gate)
                    replay_up = torch.where(
                        raw_up < -swiglu_limit, -swiglu_limit, replay_up)
                    replay_up = torch.where(
                        raw_up > swiglu_limit, swiglu_limit, replay_up)
                else:
                    replay_gate = torch.clamp_max(replay_gate, swiglu_limit)
                    replay_up = torch.clamp(
                        replay_up, -swiglu_limit, swiglu_limit)
            replay_output = torch.nn.functional.silu(replay_gate) * replay_up
            grad_gate_up_output, = torch.autograd.grad(
                replay_output, gate_up_replay, grad_swiglu)
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            experts._precision_golden_grad_z = grad_gate_up_output.detach()
        _t.tick("swiglu_bwd")  # P1 epilogue 对偶: swiglu 重算 + probs 乘法 bwd + swiglu_backward

        # 4. gmm1 反向: dx = grad_output @ weight1.T, dw1 = global_input.T @ grad_output
        # （weight1 已在上方重计算块 view 好，此处直接转置）
        weight1_transposed = weight1.transpose(-1, -2).contiguous()  # (E, ffn*2, hidden)
        _t.tick("w1_transpose")
        grad_global_input_tokens = _grouped_matmul(grad_gate_up_output, weight1_transposed, tokens_per_expert)
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            experts._precision_golden_packed_dx = grad_global_input_tokens.detach()
        _t.tick("dx")  # P2 GEMM: dX = dZ@W1
        # experts.weight1 梯度：同 weight2 走 bf16 累积链（见上）
        grad_weight1 = _grouped_wgrad(global_input_tokens, grad_gate_up_output, tokens_per_expert).view_as(experts.weight1)
        _accum_grad(experts.weight1, grad_weight1)
        _t.tick("dw1")  # P3 wgrad: dW1 = dZ^T@Xpacked

        # 3. permute2 (sort) 反向
        if ctx.num_local_experts > 1:
            grad_global_input_tokens, grad_probs_global = sort_chunks_by_idxs(
                grad_global_input_tokens, ctx.num_global_tokens_per_local_expert.T.ravel(),
                ctx.restore_output_by_local_experts, probs=grad_probs_global, fused=fused,
            )
        _t.tick("permute2")

        # 2. alltoall (dispatch) 反向: 交换 splits（单流 async，窗口插共享专家 fc1 反向）
        grad_permuted_local_input_tokens, input_handle = all_to_all_async(
            ctx.ep_group, grad_global_input_tokens, ctx.input_splits, ctx.output_splits)
        grad_permuted_probs, probs_handle = all_to_all_async(
            ctx.ep_group, grad_probs_global, ctx.input_splits, ctx.output_splits)
        grad_hidden_shared = None
        if shared_experts is not None:
            _N, _num, _G2 = shared_gate_up.shape
            grad_shared_gate_up = torch_npu.npu_swiglu_backward(
                grad_shared_act.reshape(_N * _num, -1), shared_gate_up.reshape(_N * _num, _G2), dim=-1,
            ).reshape(_N, _num, _G2)
            num_e = shared_experts.num
            grad_hidden_shared = sum(
                torch.matmul(grad_shared_gate_up[:, e], shared_experts.W1[e].transpose(0, 1)) for e in range(num_e)
            )  # (N,hidden)
            grad_W1 = torch.stack([
                torch.matmul(hidden_flat.transpose(0, 1), grad_shared_gate_up[:, e]) for e in range(num_e)
            ], dim=0)  # (num,hidden,2ffn)
            _accum_grad(shared_experts.W1, grad_W1)
        if input_handle is not None:
            input_handle.wait()
        if probs_handle is not None:
            probs_handle.wait()
        _t.tick("alltoall_dispatch")

        # 1. permute1 反向 = unpermute（permute1 没加权，unpermute probs=None 仅 scatter_add 还原）
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            # Test-only: preserve the exact BF16 route rows and destination
            # indices consumed by the golden unpermute.  test_precision uses
            # them to distinguish GMM/communication errors from the fused
            # kernel's FP32-accumulate-then-cast unpermute semantics.
            experts._precision_golden_pre_unpermute_dx = (
                grad_permuted_local_input_tokens.detach())
            experts._precision_golden_unpermute_indices = (
                ctx.reversed_local_input_permutation_mapping.detach())
        grad_hidden_flat = unpermute(
            grad_permuted_local_input_tokens, ctx.reversed_local_input_permutation_mapping,
            restore_shape=ctx.hidden_shape_before_permute, routing_map=routing_map,
            fused=fused, drop_and_pad=drop_and_pad,
        )
        grad_hidden = grad_hidden_flat.view(ctx.hidden_shape)
        grad_hidden_fused_unpermute = None
        if getattr(config, "_precision_match_fused_unpermute", False):
            # Precision-test reference for MegaMoeBF16Backward::UnpermuteBackward:
            # the kernel converts every BF16 route row to FP32, accumulates all
            # TopK routes in FP32, then casts each token row to BF16 once.  Keep
            # BigOp's normal BF16 scatter_add result above unchanged so speed
            # tests and the small-op baseline retain their original semantics.
            route_dx_fp32 = grad_permuted_local_input_tokens.to(torch.float32)
            fp32_accum = torch.zeros(
                ctx.hidden_shape_before_permute,
                dtype=torch.float32,
                device=grad_permuted_local_input_tokens.device,
            )
            expanded_indices = (
                ctx.reversed_local_input_permutation_mapping.view(-1, 1)
                .expand(-1, route_dx_fp32.shape[-1])
            )
            fp32_accum.scatter_add_(0, expanded_indices, route_dx_fp32)
            grad_hidden_fused_unpermute = fp32_accum.to(
                grad_permuted_local_input_tokens.dtype).view(ctx.hidden_shape)
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            experts._precision_golden_expert_dx = grad_hidden.detach()
        _t.tick("permute1")

        # 0. router 反向（手写对偶，不依赖 autograd）
        # grad_permuted_probs -> grad_probs (scatter 到 N×E，与 forward permute 的 (E,N) 空间一致)
        mask_transposed = routing_map.bool().T.contiguous()
        grad_probs_transposed = torch.zeros_like(probs.T)
        grad_probs_transposed.masked_scatter_(mask_transposed, grad_permuted_probs)
        grad_probs = grad_probs_transposed.T.contiguous()  # (N, E)

        # topk_softmax 反向: grad_probs -> grad_logits_from_probs（fp32 算 softmax_bwd，与 forward 一致）
        grad_probs_topk = torch.gather(grad_probs, 1, topk_indices).float()  # (N, topk)
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            experts._precision_golden_dtopk = grad_probs_topk.to(
                global_probs.dtype).detach()
        grad_sum_topk = (grad_probs_topk * probs_topk_fp32).sum(dim=-1, keepdim=True)
        grad_topk_values = probs_topk_fp32 * (grad_probs_topk - grad_sum_topk)  # softmax_bwd
        grad_logits_from_probs = torch.zeros_like(logits).scatter_add(1, topk_indices, grad_topk_values.to(logits.dtype))

        # aux_loss 反向（g=1 经 MoEAuxLossAutoScaler 会乘 main_loss_backward_scale，bigop 手动复现该缩放）
        if ctx.adjusted_aux_coeff > 0:
            num_tokens = logits.shape[0]
            if ctx.sequence_partition_group is not None:
                num_tokens = num_tokens * torch.distributed.get_world_size(ctx.sequence_partition_group)
            num_experts = logits.shape[1]
            aux_loss_factor = num_experts * ctx.adjusted_aux_coeff / (num_tokens * num_tokens * ctx.topk)
            grad_aux_scores = aux_loss_factor * aux_tokens_per_expert.unsqueeze(0).expand_as(aux_scores)  # (N, E)
            grad_sum_aux = (grad_aux_scores * aux_scores).sum(dim=-1, keepdim=True)
            # softmax_bwd, bf16；乘 main_loss_backward_scale 使 aux_loss 梯度与主 loss 同为 scaled（否则 ST 下差 loss_scale 倍）
            # scale 是 fp32 tensor，须转回 logits.dtype，否则 grad_logits 变 fp32 赋给 bf16 gate_weight.grad 会崩
            grad_logits_from_aux = (aux_scores * (grad_aux_scores - grad_sum_aux)).to(logits.dtype)
            grad_logits_from_aux = (grad_logits_from_aux * _get_aux_loss_backward_scale()).to(logits.dtype)
            grad_logits = grad_logits_from_probs + grad_logits_from_aux
        else:
            grad_logits = grad_logits_from_probs

        # gate linear 反向: 对偶 forward 的 linear(x.to(fp32), W.to(fp32)) —— fp32 路由时
        # 两支 matmul 均 fp32（grad_logits 本就是 fp32），grad 收口回 bf16 与 autograd 的
        # cast-节点反向一致（输入梯度无损 fp32，权重梯度 fp32 算完 to(bf16) 再累积）
        if router_dtype == "fp32":
            grad_route_input = (grad_logits @ gate_weight.to(torch.float32)).to(hidden_flat.dtype)  # (N, hidden)
            grad_gate_weight = (grad_logits.t() @ hidden_flat.to(torch.float32)).to(gate_weight.dtype)  # (E, hidden)
        else:
            grad_route_input = grad_logits @ gate_weight  # (N, hidden)
            grad_gate_weight = grad_logits.t() @ hidden_flat  # (E, hidden)
        if os.environ.get("PRECISION_DIAG_DW2", "0") == "1":
            experts._precision_golden_router_dx = grad_route_input.detach()
        _accum_grad(gate_weight, grad_gate_weight)

        grad_hidden_states = grad_hidden + grad_route_input.view(ctx.hidden_shape)
        if shared_experts is not None:
            grad_hidden_states = grad_hidden_states + grad_hidden_shared.view(ctx.hidden_shape)
        if grad_hidden_fused_unpermute is not None:
            fused_unpermute_grad_x = (
                grad_hidden_fused_unpermute
                + grad_route_input.view(ctx.hidden_shape)
            )
            if shared_experts is not None:
                fused_unpermute_grad_x = (
                    fused_unpermute_grad_x
                    + grad_hidden_shared.view(ctx.hidden_shape)
                )
            experts._precision_fused_unpermute_grad_x = (
                fused_unpermute_grad_x.detach())
        _t.tick("router")
        _t.finish()  # sync + 相邻 Event 做差 -> 每步 ms，累计进 timing 全局 registry
        return grad_hidden_states, None, None
