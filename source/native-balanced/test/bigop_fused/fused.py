"""bigop v2：router 手写（bigop 步骤0）+ 可插拔融合算子（步骤1-9）。

设计原则：
  - bigop v1（golden.py）是纯小算子 10 步前向 + 10 步手写反向的 golden 参考；
    其反向默认为重计算口径（对齐 MegaMoe bf16 backward 的 P0-C/P1 重算：
    不保存 Z/Hp，从 Xpacked 重算 GMM1 与 SwiGLU*probs，bwd 计算量 16 RFK
    与融合算子同口径；BIGOP_BWD_RECOMPUTE=0 可退回保存中间量的旧口径）；
  - bigop v2 保留 bigop 的 router（步骤0：gating+topk_softmax+aux_loss 手写），
    把 dispatch+FFN+combine（步骤1-9）**整段替换**为任意满足契约的融合算子；
  - 融合算子**可插拔**：不写死类名。任何满足 fused_op 契约的算子都能接入
    （ascend_deepep 的 MegaMoeBF16Train / MegaMoeBF16 / 你自己的实现均可）。

fused_op 契约（__call__ 或 forward，位置参数）：
    fused_op(x, topk_ids, topk_weights, w1, w2, max_recv) -> y 或 (y, counts)
      - x            : (M,K) bf16 输入
      - topk_ids     : (M,topk) INT32 全局专家 id
      - topk_weights : (M,topk) bf16 router softmax 概率
      - w1           : Megatron legacy [E,K,N] bf16（N=2FFN）
      - w2           : Megatron legacy [E,N/2,K] bf16
      - max_recv     : int dropless 容量 = M*TopK*EP
      - y            : (M,K) bf16（可带 counts 第二返回值，BigOpV2 只取 y）

  是否可训练取决于 fused_op 是否在 autograd 图中：
    - autograd.Function 包装（如 MegaMoeBF16Train）→ backward 自动回流，可训练
    - 纯前向 op（如 MegaMoeBF16.compute 封装）→ 只支持前向验证

用法（单层验证 / 训练 patch）：
    # 可训练（fwd+bwd 一体）
    fused = build_megamoe_bf16_train_op(shmem, local_mem_bytes)
    v2 = BigOpV2(fused)
    y, aux_loss = v2(x, moe_layer)          # 接口与 BigOp 对齐

    # 只前向（无 autograd 图，仅验证前向精度）
    fused = build_megamoe_bf16_forward_op(shmem, local_mem_bytes)
    v2 = BigOpV2(fused)
    y, _ = v2(x, moe_layer)                 # 不触发 backward
"""
import os

import torch

import timing
from utils import (
    router_forward,
    make_weight_tensors,
)

_V2_LAYOUT_TIMING = timing.env_enabled("V2_LAYOUT_TIMING")


class BigOpV2(torch.nn.Module):
    """bigop v2：router 手写 + 可插拔融合算子。

    forward(hidden_states, moe_layer) -> (y, aux_loss)
      - hidden_states: (seq, batch, K) bf16
      - moe_layer    : bigop 兼容的 MoELayer（router / token_dispatcher / experts / config）
      - 返回 y (与 hidden_states 同 shape)；aux_loss 走 _save_to_aux_losses_tracker
        （与 bigop 一致），此处返回 None。

    构造注入 fused_op（见模块 docstring 的契约），不硬编码算子类名。
    """

    def __init__(self, fused_op, diagnostic_holder=None):
        super().__init__()
        if fused_op is None:
            raise ValueError("fused_op must not be None")
        self._fused = fused_op
        self.diagnostic_holder = diagnostic_holder

    def forward(self, hidden_states, moe_layer, diagnostic_holder=None):
        holder = diagnostic_holder if diagnostic_holder is not None else self.diagnostic_holder
        timing_tag = (
            holder.get("timing_tag", "v2_fwd")
            if holder is not None else "v2_fwd"
        )
        _t = timing.get_step_timer(
            timing_tag, enabled=_V2_LAYOUT_TIMING)
        _t.tick("start")
        token_dispatcher = moe_layer.token_dispatcher
        config = moe_layer.config
        experts = moe_layer.experts
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])  # (N, H)

        # 0. router（bigop 手写步骤0，probs/topk_weights 对齐融合算子契约）
        topk_ids, topk_weights, _holder = router_forward(moe_layer, hidden_flat)
        _t.tick("router")

        # 1-9. 融合算子（dispatch + FFN + combine，EP 通信算子内部完成）
        num_local = token_dispatcher.num_local_experts
        ep = token_dispatcher.ep_group.size()
        topk = topk_ids.size(1)
        w1, w2 = make_weight_tensors(experts, num_local,
                                     config.hidden_size, config.moe_ffn_hidden_size)
        _t.tick("weight_view")
        max_recv = hidden_flat.size(0) * topk * ep  # dropless M*TopK*EP

        if holder is None:
            out = self._fused(hidden_flat, topk_ids, topk_weights, w1, w2, max_recv)
        elif holder.get("_skip_grad_edge_capture", False):
            out = self._fused(
                hidden_flat, topk_ids, topk_weights, w1, w2, max_recv,
                diagnostic_holder=holder,
            )
        else:
            # Split the two x-gradient edges without changing their values:
            # fused_input.grad is the expert path, while topk_weights.grad is
            # the route-weight edge feeding the router path.
            fused_input = hidden_flat.clone()
            fused_input.retain_grad()
            topk_weights.retain_grad()
            holder["fused_input"] = fused_input
            holder["topk_weights"] = topk_weights
            try:
                out = self._fused(
                    fused_input, topk_ids, topk_weights, w1, w2, max_recv,
                    diagnostic_holder=holder,
                )
            except TypeError:
                # Keep compatibility with forward-only/custom wrappers that
                # still expose the original call contract.
                out = self._fused(
                    fused_input, topk_ids, topk_weights, w1, w2, max_recv)
            if isinstance(out, (tuple, list)) and len(out) > 1:
                holder["counts"] = out[1].detach()
        _t.tick("fused_forward")
        _t.finish()
        y = out[0] if isinstance(out, (tuple, list)) else out
        return y.view(hidden_states.shape), None

    def destroy(self):
        """释放算子资源（若 fused_op 有 destroy；不销毁 shmem，由调用方管理）。"""
        d = getattr(self._fused, "destroy", None)
        if callable(d):
            d()


# ===================== 便捷工厂（按需选用） =====================

class _ConfiguredMegaMoeBF16Train(torch.nn.Module):
    """Bind the saved-state mode used by BigOp precision/perf runners."""

    def __init__(self, shmem, local_mem_size: int,
                 save_gmm1_output: str, swiglu_limit: float):
        super().__init__()
        from ascend_deepep import MegaMoeBF16Train
        self.op = MegaMoeBF16Train(shmem, local_mem_size)
        self.save_gmm1_output = save_gmm1_output
        self.swiglu_limit = swiglu_limit

    def forward(self, x, topk_ids, topk_weights, w1, w2, max_recv,
                diagnostic_holder=None):
        return self.op(
            x, topk_ids, topk_weights, w1, w2, max_recv,
            swiglu_limit=self.swiglu_limit,
            save_gmm1_output=self.save_gmm1_output,
            diagnostic_holder=diagnostic_holder,
        )

    def destroy(self):
        self.op.destroy()


def build_megamoe_bf16_train_op(shmem, local_mem_size: int):
    """Build the train op using the mode selected by the runner environment."""
    save_mode = os.environ.get("MEGAMOE_SAVE_GMM1_OUTPUT", "bf16").strip().lower()
    if save_mode not in ("auto", "none", "bf16", "mxfp8"):
        raise ValueError(
            "MEGAMOE_SAVE_GMM1_OUTPUT must be auto, none, bf16 or mxfp8; "
            f"got {save_mode!r}"
        )
    try:
        swiglu_limit = float(os.environ.get("MEGAMOE_SWIGLU_LIMIT", "0"))
    except ValueError as exc:
        raise ValueError("MEGAMOE_SWIGLU_LIMIT must be a floating-point value") from exc
    return _ConfiguredMegaMoeBF16Train(
        shmem, local_mem_size, save_mode, swiglu_limit)


def build_megamoe_bf16_forward_op(shmem, local_mem_size: int):
    """只前向融合算子：ascend_deepep.MegaMoeBF16（compute），无 autograd 图。

    返回满足 BigOpV2 fused_op 契约的可调用对象（只前向，不能 backward）。
    若你的算子接口不同，照此封装即可——BigOpV2 不关心算子内部。
    """
    from ascend_deepep import MegaMoeBF16
    op = MegaMoeBF16(local_mem_size)

    def fused(x, topk_ids, topk_weights, w1, w2, max_recv):
        y, *_rest = op.compute(shmem, x, topk_ids, topk_weights, w1, w2, max_recv)
        return y

    fused._op = op  # destroy 用
    return fused


def build_bigop_v2(shmem, local_mem_size: int, mode: str = "train"):
    """工厂：构造 BigOpV2。

    mode: 'train'（默认，MegaMoeBF16Train，fwd+bwd 可训练）
          'forward'（MegaMoeBF16.compute，只前向）
    其他算子：直接 build_bigop_v2(your_fused_op) 或手动构造 fused 后 BigOpV2(fused)。
    """
    if mode == "train":
        fused = build_megamoe_bf16_train_op(shmem, local_mem_size)
    elif mode == "forward":
        fused = build_megamoe_bf16_forward_op(shmem, local_mem_size)
    else:
        raise ValueError(f"unknown mode={mode}; use 'train' or 'forward'")
    diagnostic_holder = None
    if (mode == "train" and
            os.environ.get("PRECISION_DIAG_DW2", "0") == "1"):
        diagnostic_holder = {"capture_backward_workspace": True}
    return BigOpV2(fused, diagnostic_holder=diagnostic_holder)
