"""bigop_fused 测试公共工具（BigOpV2 可插拔融合算子的通用函数）。

集中 BigOpV2 测试所需的公共函数/常量（router_forward / make_weight_tensors /
group_report / build_gtopk_ref），使测试只依赖本文件（+ bigop /
make_fake_moe_layer / torch），不绑定任何具体融合算子实现。

依赖：torch / torch.distributed / bigop（无 megatron/mindspeed）。
"""
from types import SimpleNamespace

import torch
import torch.distributed as dist

from golden import (
    _switch_load_balancing_loss_func,
    _save_to_aux_losses_tracker,
)

# ---- 引擎常量 ----
ACLSHMEM_DATA_OP_UDMA = 0x08
DEFAULT_IP_PORT = "tcp://127.0.0.1:8899"   # 单机双卡默认；跨机改 ASCEND_DEEPEP_SHMEM_IP_PORT

# ---- BF16 反向精度门禁 ----
# 日常开发硬门禁使用 cosine + 双百；全小值 reference 跳过 cosine，
# BF16 小值域统一改用绝对误差。
# 双千比双百更严格，作为质量等级与回归趋势保留在日志中，但不单独导致
# case 失败。旧 bring-up 容差也只保留为诊断数字，不决定测试是否通过。
BF16_COSINE_MIN = 0.99
BF16_DOUBLE_HUNDRED_REL = 1e-2
BF16_DOUBLE_HUNDRED_BAD_FRAC = 1e-2
BF16_DOUBLE_THOUSAND_REL = 1e-3
BF16_DOUBLE_THOUSAND_BAD_FRAC = 1e-3
BF16_SMALL_VALUE = 2.0 ** -8
BF16_SMALL_ATOL = 2.0 ** -16

# Legacy engineering diagnostic (not a gate).
BWD_RTOL = 0.15
BWD_ATOL = 0.05


def router_forward(moe_layer, hidden_flat):
    """步骤0 router：hidden_flat(N,H) -> (topk_ids, topk_weights, holder)。

    topk_ids = 全局专家 id INT32，topk_weights = probs_topk BF16（对齐 MegaMoeBF16 契约）。
    无 megatron parallel_state 依赖（TP=1 下 sequence_partition_group 恒 None）。
    """
    router = moe_layer.router
    config = moe_layer.config
    topk = router.topk
    logits = router.gating(hidden_flat)  # (N, E)
    topk_values, topk_indices = torch.topk(logits, topk, dim=1)
    probs_topk_fp32 = torch.softmax(topk_values, dim=-1, dtype=torch.float32)
    probs_topk = probs_topk_fp32.type_as(logits)
    probs = torch.zeros_like(logits).scatter(1, topk_indices, probs_topk)
    routing_map = torch.zeros_like(logits).int().scatter(1, topk_indices, 1).bool()
    aux_tokens_per_expert = routing_map.sum(dim=0)
    aux_scores = router.compute_routing_scores_for_aux_loss(logits)

    aux_loss_coeff = config.moe_aux_loss_coeff
    sequence_partition_group = None  # TP=1 无 CP/TP+CP 归约
    if aux_loss_coeff > 0:
        aux_loss = _switch_load_balancing_loss_func(
            probs=aux_scores, tokens_per_expert=aux_tokens_per_expert, topk=topk,
            moe_aux_loss_coeff=aux_loss_coeff, sequence_partition_group=sequence_partition_group)
        _save_to_aux_losses_tracker(
            "load_balancing_loss", aux_loss / aux_loss_coeff,
            router.layer_number, config.num_layers, reduce_group=sequence_partition_group)

    topk_ids = topk_indices.to(torch.int32).contiguous()
    topk_weights = probs_topk.contiguous()
    holder = SimpleNamespace(
        logits=logits, topk_indices=topk_indices, probs_topk_fp32=probs_topk_fp32,
        probs=probs, routing_map=routing_map,
        aux_scores=aux_scores, aux_tokens_per_expert=aux_tokens_per_expert, topk=topk,
    )
    return topk_ids, topk_weights, holder


def make_weight_tensors(experts, num_local_experts, hidden, ffn):
    """把 fake experts 2D 拼接权重零拷贝 view 成 Megatron legacy 布局。

    fake experts：weight1 [H, 2FFN*E]、weight2 [FFN*E, H]。
    bigop view：w1 [E, H, 2FFN] = [E, K, N]，w2 [E, FFN, H] = [E, N/2, K]。
    融合算子会从 shape 自动识别该布局，无需 transpose + contiguous。
    """
    w1 = experts.weight1.view(num_local_experts, hidden, 2 * ffn)
    w2 = experts.weight2.view(num_local_experts, ffn, hidden)
    return w1, w2


def group_report(name, got, ref, rtol=BWD_RTOL, atol=BWD_ATOL):
    """按 BF16 日常开发硬门禁比较梯度，返回 ``(ok, status, detail)``。

    ``finite + cosine + 双百`` 决定 ``ok``。当 reference 全部落在 BF16
    小值域时，cosine 对接近零的非零向量不稳定，因此只保留为诊断数字，
    不参与硬门禁。正常值同时计算双百和双千；
    双千只把状态从 ``SUCCESS`` 降为 ``WARNING``，不单独令开发门禁失败。以 reference 为准的小值
    ``abs(ref) < 2**-8`` 使用 ``2**-16`` 绝对误差，避免接近零时相对
    误差失真。小值不合格点计入双百/双千各自的坏点比例。传入的
    ``rtol/atol`` 仅用于输出旧工程口径的诊断数字，不影响 ``ok``。
    """
    got_f = got.reshape(-1).to(torch.float32)
    ref_f = ref.reshape(-1).to(torch.float32)
    if got_f.numel() != ref_f.numel():
        return (False, "Failed",
                f"shape/numel {got_f.numel()} != {ref_f.numel()}")
    if got_f.numel() == 0:
        return True, "SUCCESS", "empty"

    got_finite = torch.isfinite(got_f)
    ref_finite = torch.isfinite(ref_f)
    finite_ok = bool((got_finite & ref_finite).all().item())
    abs_err = (got_f - ref_f).abs()
    finite_abs_err = torch.where(torch.isfinite(abs_err), abs_err,
                                 torch.full_like(abs_err, float("inf")))
    ref_abs = ref_f.abs()
    small = ref_abs < BF16_SMALL_VALUE
    normal = ~small
    rel_err = finite_abs_err / ref_abs.clamp_min(BF16_SMALL_VALUE)
    small_bad = small & (finite_abs_err > BF16_SMALL_ATOL)
    bad_hundred = small_bad | (normal & (rel_err > BF16_DOUBLE_HUNDRED_REL))
    bad_thousand = small_bad | (normal & (rel_err > BF16_DOUBLE_THOUSAND_REL))

    n = got_f.numel()
    bad_hundred_count = int(bad_hundred.sum().item())
    bad_thousand_count = int(bad_thousand.sum().item())
    bad_hundred_frac = bad_hundred_count / n
    bad_thousand_frac = bad_thousand_count / n
    small_count = int(small.sum().item())
    small_bad_count = int(small_bad.sum().item())

    got_norm = got_f.norm().item()
    ref_norm = ref_f.norm().item()
    if got_norm == 0.0 or ref_norm == 0.0:
        # Cosine is undefined for an all-zero side. In an all-small tensor,
        # let the official small-value absolute-error rule decide instead.
        cosine = 1.0 if small_count == n and small_bad_count == 0 else 0.0
    elif not finite_ok:
        cosine = 0.0
    else:
        cosine = float(torch.dot(got_f, ref_f).item() / (got_norm * ref_norm))

    all_small = small_count == n
    cosine_gated = not all_small
    cosine_ok = not cosine_gated or cosine > BF16_COSINE_MIN
    hundred_ok = bad_hundred_frac < BF16_DOUBLE_HUNDRED_BAD_FRAC
    thousand_ok = bad_thousand_frac < BF16_DOUBLE_THOUSAND_BAD_FRAC
    # Hard development gate: large numerical regressions must fail, while a
    # double-thousand-only miss remains visible as a quality warning.
    ok = finite_ok and cosine_ok and hundred_ok
    status = "Failed" if not ok else ("SUCCESS" if thousand_ok else "WARNING")

    # Preserve the previous loose comparison as a diagnostic only.
    legacy_rel_err = finite_abs_err / (ref_abs + 1e-6)
    legacy_bad = (finite_abs_err > atol) & (legacy_rel_err > rtol)
    legacy_bad_count = int(legacy_bad.sum().item())
    legacy_bad_frac = legacy_bad_count / n

    max_abs = finite_abs_err.max().item()
    n_eq = (got == ref).float().mean().item()
    rel_l2 = finite_abs_err.norm().item() / (ref_norm + 1e-12)
    detail = (
        f"cos={cosine:.8f}("
        f"{'not-gated:all-small' if not cosine_gated else ('>0.99' if cosine_ok else '<=0.99')}) "
        f"double100_bad={bad_hundred_count}/{n}({bad_hundred_frac * 100:.4f}%) "
        f"double1000_bad={bad_thousand_count}/{n}({bad_thousand_frac * 100:.4f}%) "
        f"small_bad={small_bad_count}/{small_count} "
        f"max_abs={max_abs:.3e} rel_l2={rel_l2:.3e} bit_eq={n_eq:.6f} "
        f"legacy_bad={legacy_bad_count}/{n}({legacy_bad_frac * 100:.3f}%)"
    )
    if not ok:
        hard_bad = bad_hundred | ~got_finite | ~ref_finite
        flat_bad = hard_bad.reshape(-1)
        if not bool(flat_bad.any().item()):
            # A pure cosine failure may not have any double-hundred bad point.
            flat_bad = (finite_abs_err > 0).reshape(-1)
    else:
        flat_bad = torch.zeros_like(got_f, dtype=torch.bool)
    if not ok and bool(flat_bad.any().item()):
        idx = int(flat_bad.nonzero()[0].item())
        detail += (f" first_bad[{idx}] got={got_f[idx].item():.6g}"
                   f" ref={ref_f[idx].item():.6g}")
    return ok, status, detail


def build_gtopk_ref(moe_layer, x, grad_y):
    """grad_topk_weights 的解析 golden（纯 torch fp32 重放前向 FFN）。

    dL/dprobs_topk[t, e] = dot(grad_y[t], expert_out_e(x[t]))，
    expert_out_e 为未加权（不含 probs）的 FFN 输出。EP 下 topk 落到远端 expert
    时输出由远端 rank 算，用 all_gather 聚合本 rank 需要的行。
    """
    hidden_flat = x.view(-1, x.shape[-1])
    grad_flat = grad_y.view(-1, grad_y.shape[-1])
    K = hidden_flat.shape[-1]
    td = moe_layer.token_dispatcher
    config = moe_layer.config
    experts = moe_layer.experts

    with torch.no_grad():
        router = moe_layer.router
        logits = router.gating(hidden_flat)
        topk_v, topk_i = torch.topk(logits, router.topk, dim=1)
        topk_ids = topk_i.to(torch.int32)  # (N, topk) 全局 expert id
        n_tok, topk = topk_ids.shape
        ep_rank = dist.get_rank()
        e_local = td.num_local_experts

        # 本 rank 负责的全局 expert 区间 [ep_rank*e_local, (ep_rank+1)*e_local)
        base = ep_rank * e_local
        # 本 rank 能算的 (t, j) 对：topk_ids[t, j] 落在本区间
        mask_local = (topk_ids >= base) & (topk_ids < base + e_local)  # (N, topk)
        t_idx, j_idx = mask_local.nonzero(as_tuple=True)
        e_global = topk_ids[t_idx, j_idx] - base  # 本 rank local expert 下标
        # FFN 前向（fp32）逐对计算：只取各对所属 expert 的权重，不做全 expert einsum。
        # weight1 物理 [H, 2FFN*E]（fake experts 拼接布局）-> [E, H, 2FFN]（同 bigop 前向第 4 步）
        FFN = config.moe_ffn_hidden_size
        w1 = experts.weight1.view(e_local, K, 2 * FFN).to(torch.float32)  # [E,H,2FFN]
        w2 = experts.weight2.view(e_local, FFN, K).to(torch.float32)      # [E,FFN,H]
        rows = hidden_flat[t_idx].to(torch.float32)                       # (R, K)
        w1_sel = w1[e_global]                                             # (R, H, 2FFN)
        w2_sel = w2[e_global]                                             # (R, FFN, H)
        gate_up = torch.einsum("rh,rhn->rn", rows, w1_sel)                # (R, 2FFN)
        gate, up = gate_up.chunk(2, dim=-1)
        inter = torch.nn.functional.silu(gate) * up                       # (R, FFN)
        expert_out = torch.einsum("rn,rnh->rh", inter, w2_sel)            # (R, K)
        g = (expert_out * grad_flat[t_idx].to(torch.float32)).sum(-1)     # (R,)

        # EP 下 topk 可能落到远端 expert：远端贡献由 owner rank 算，
        # all_reduce 聚合（每个 (t,j) 只被一个 rank 算，sum 即拼接）。
        gtopk = torch.zeros(n_tok, topk, device=rows.device, dtype=torch.float32)
        gtopk[t_idx, j_idx] += g
        dist.all_reduce(gtopk, op=dist.ReduceOp.SUM)
        return gtopk
