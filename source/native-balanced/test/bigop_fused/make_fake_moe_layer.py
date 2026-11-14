"""纯 torch 构造 fake MoELayer，供 MegaMoeBF16 接入的 bigop baseline 使用（无 megatron/mindspeed 依赖）。

bigop 本体是纯 torch（原语全内置，`golden.py` 顶部仅 import torch/torch_npu）。它只通过
`BigOp.apply(x, moe_layer)` 从 `moe_layer` 对象取：config / router / experts / token_dispatcher。
本文件用纯 torch 复刻 megatron MoELayer 的这些接口（EP=2/TP=1 dropless 场景），
并精确对齐 megatron alltoall dispatcher 的 preprocess 语义（input/output splits、sort 索引），
使 bigop baseline 与 megatron 版行为一致。

依据：megatron `token_dispatcher.py` preprocess（dropless 分支）与
`experts.py` GroupedMLP（gated_linear_unit=True 时 activation_func=glu=swiglu）。
"""
import torch
import torch.distributed as dist
import torch.nn as nn


def glu(x):
    """swiglu：silu(gate)*up。对齐 megatron GroupedMLP activation_func（experts.py:110-113）。"""
    gate, up = torch.chunk(x, 2, dim=-1)
    return torch.nn.functional.silu(gate) * up


class FakeRouter:
    """对齐 megatron TopKRouter 的 forward 所需接口：gating / topk / compute_routing_scores_for_aux_loss。"""

    def __init__(self, hidden_size, num_experts, topk, dtype=torch.bfloat16):
        self.topk = topk
        self.layer_number = 0  # aux_loss tracker 用
        # bf16 训练下 gating 权重跟随 params_dtype（对齐 megatron）；否则 bf16 x fp32 matmul 报 dtype 不匹配
        self.gating = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype).npu()

    def compute_routing_scores_for_aux_loss(self, logits):
        return torch.softmax(logits, dim=-1)


class FakeExperts(nn.Module):
    """对齐 megatron GroupedMLP：weight1 [H, 2FFN*E]（gate+up 拼接 dim1）、weight2 [FFN*E, H]，
    activation_func_with_probs = glu(x)*probs（experts.py:124-127, 110-113）。"""

    def __init__(self, hidden_size, ffn_hidden_size, num_local_experts,
                 init_std=0.01, dtype=torch.bfloat16):
        super().__init__()
        self.num_local_experts = num_local_experts
        # bf16：与输入 token（bf16）匹配，对齐 megatron GroupedMLP（params_dtype=bf16）
        self.weight1 = nn.Parameter(
            torch.empty(hidden_size, 2 * ffn_hidden_size * num_local_experts, dtype=dtype))
        self.weight2 = nn.Parameter(
            torch.empty(ffn_hidden_size * num_local_experts, hidden_size, dtype=dtype))
        with torch.no_grad():
            for p in (self.weight1, self.weight2):
                p.normal_(mean=0.0, std=init_std)

    @staticmethod
    def activation_func_with_probs(x, probs):
        return glu(x) * probs


class FakeTokenDispatcher:
    """纯 torch 复刻 megatron MoEAlltoAllTokenDispatcher（dropless, EP=2/TP=1 场景）。

    对齐点（megatron token_dispatcher.py）：
    - input_splits  = num_local_tokens_per_expert.reshape(ep, localE).sum(dim=1)   （preprocess）
    - output_splits = 跨 rank gather 后各 rank 发给本 rank 的 token 数               （preprocess）
    - num_out_tokens = N * topk（dropless）
    - sort_input_by_local_experts / restore_output_by_local_experts = arange(E).reshape(...).T.ravel()（构造期常量）
    """

    def __init__(self, num_local_experts, num_experts, topk, ep_group):
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.ep_size = ep_group.size()
        assert self.num_experts == self.ep_size * num_local_experts
        self.local_expert_indices = list(range(num_local_experts))  # 本 rank 的 local expert 相对段
        self.drop_and_pad = False
        self.ep_group = ep_group
        self.topk = topk
        self.hidden_shape = None
        self.hidden_shape_before_permute = None
        self.probs = None
        self.routing_map = None
        self.reversed_local_input_permutation_mapping = None
        self.num_out_tokens = None
        self.input_splits = None
        self.output_splits = None
        self.num_global_tokens_per_local_expert = None

        # 构造期固定索引（megatron token_dispatcher.py:293-303，TP=1）
        idx = torch.arange(self.num_experts)
        self.sort_input_by_local_experts = idx.reshape(-1, self.num_local_experts).T.ravel()
        self.restore_output_by_local_experts = idx.reshape(self.num_local_experts, -1).T.ravel()

    @property
    def local_start(self):
        return dist.get_rank(self.ep_group) * self.num_local_experts

    def _maybe_dtoh_and_synchronize(self, name, tensor):
        return tensor  # bigop 需要 CPU 值；此处保持 device 张量即可（bigop 内部处理）

    def preprocess(self, routing_map):
        """routing_map [N, E] bool → num_tokens_per_local_expert [localE]，并设置 splits/索引。"""
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()  # [E]
        self.num_out_tokens = routing_map.size(0) * self.topk  # dropless

        if self.ep_size > 1:
            self.input_splits = num_local_tokens_per_expert.reshape(
                self.ep_size, self.num_local_experts).sum(dim=1).tolist()
            # 跨 rank gather 各 rank 的 token 分布（megatron 用 tp_ep_group，TP=1 即 ep_group）
            gathered = [torch.empty_like(num_local_tokens_per_expert) for _ in range(self.ep_size)]
            dist.all_gather(gathered, num_local_tokens_per_expert, group=self.ep_group)
            num_global = torch.stack(gathered)  # [ep, E]
            local = num_global[:, self.local_start:self.local_start + self.num_local_experts]
            self.output_splits = local.sum(dim=1).tolist()  # [ep]：各 rank 发给本 rank 的 token 数
            num_tokens_per_local_expert = local.sum(dim=0)  # [localE]
            self.num_global_tokens_per_local_expert = local  # [ep, localE]
        else:
            self.output_splits = num_local_tokens_per_expert.tolist()
            num_tokens_per_local_expert = num_local_tokens_per_expert
            self.num_global_tokens_per_local_expert = num_local_tokens_per_expert.reshape(
                1, -1)
        return num_tokens_per_local_expert


class FakeConfig:
    """bigop forward 用到的 config 字段（golden.py BigOp.forward 逐步核对）。

    moe_router_dtype: 对齐 megatron TransformerConfig.moe_router_dtype（None/'fp32'），
    fp32 时 BigOp 走 fp32 路由链（gating 运行时上转、probs 全程 fp32、act 截断回 bf16）。
    """

    def __init__(self, hidden_size=1024, ffn_hidden_size=2048, num_moe_experts=16,
                 router_topk=2, aux_loss_coeff=0.01, moe_router_dtype=None):
        self.hidden_size = hidden_size
        self.moe_ffn_hidden_size = ffn_hidden_size
        self.num_moe_experts = num_moe_experts
        self.moe_router_topk = router_topk
        self.moe_aux_loss_coeff = aux_loss_coeff
        self.moe_router_dtype = moe_router_dtype
        self.moe_permute_fusion = False
        self.moe_token_dispatcher_type = "alltoall"
        self.num_layers = 1


def make_fake_moe_layer(hidden_size=1024, ffn_hidden_size=2048, num_moe_experts=16,
                        router_topk=2, aux_loss_coeff=0.01, init_std=0.01,
                        moe_router_dtype=None):
    """构造纯 torch fake MoELayer（EP 组 = 当前 torch.distributed 全组）。

    返回 (moe_layer, config)，moe_layer 提供 BigOp.apply 需要的 config/router/experts/token_dispatcher。
    """
    ep_group = dist.group.WORLD
    ep_size = dist.get_world_size()
    num_local_experts = num_moe_experts // ep_size
    assert num_moe_experts % ep_size == 0, "num_moe_experts 须能被 EP 整除"

    config = FakeConfig(hidden_size, ffn_hidden_size, num_moe_experts,
                        router_topk, aux_loss_coeff, moe_router_dtype)
    router = FakeRouter(hidden_size, num_moe_experts, router_topk)
    experts = FakeExperts(hidden_size, ffn_hidden_size, num_local_experts, init_std).npu()
    token_dispatcher = FakeTokenDispatcher(num_local_experts, num_moe_experts,
                                           router_topk, ep_group)

    moe_layer = SimpleNamespaceShim(config, router, experts, token_dispatcher)
    return moe_layer, config


class SimpleNamespaceShim:
    """把 config/router/experts/token_dispatcher 组装成 BigOp.apply 期望的 moe_layer 对象。"""

    def __init__(self, config, router, experts, token_dispatcher):
        self.config = config
        self.router = router
        self.experts = experts
        self.token_dispatcher = token_dispatcher
