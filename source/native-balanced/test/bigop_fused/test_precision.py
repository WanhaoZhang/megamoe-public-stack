"""BigOpV2（router 手写 + 可插拔融合算子）验证测试。

验证 bigop v2 与 bigop v1（纯小算子 golden）对齐：
  - 前向 y：BigOpV2 vs BigOp（assert_close + bit 相等率）
  - 反向四组：dx（完整，含 router 路径）/ dw1 / dw2 / gate weight grad
    注：v2 含 router（步骤0 手写 + autograd 回流），故 dx 直接比完整 dx_ref，
        无需像"无 router 融合算子"测试那样扣 router 路径（那是 MoE 主体
        的语义；v2 是"完整 MoE 层"的语义）。
  - 权重梯度：v2 中 w1/w2 由 make_weight_tensors 从 experts.weight 派生且未 detach，
    autograd 会把梯度流回真实 expert 参数（这正是"可训练"的体现）。

模式开关：
  - TEST_MODE=forward|backward|both（默认 both）——只前向/只反向/都测
  - FUSED_MODE=train|forward（默认 train）——融合算子选型：
      train    = MegaMoeBF16Train（fwd+bwd 一体，autograd 图，可训练）
      forward  = MegaMoeBF16.compute（只前向，无 autograd 图，反向自动 SKIP）

为对比干净：共用一份 moe_layer，v1 跑完读梯度 → zero_layer_grads 清零 → 再跑 v2。

用法（EP2，2 卡）：
  bash bigop_fused/run_precision.sh [forward|backward|both] [train|forward]

报告语义：严格通过显示 SUCCESS，双千单独未通过显示 WARNING，硬门禁未通过
显示 Failed。外层脚本对 Failed 返回 2、对运行 ERROR 返回 1，WARNING 返回 0。
grad_x 硬门禁按融合 kernel 的 FP32 unpermute 累加边界生成参考值；BigOp 原始
BF16 scatter_add 结果另作 INFO 输出，不参与成败判定。
MXFP8 保存态下，反向硬门禁使用相同 MXFP8 编解码语义的 golden；相对原始
BF16 backward 的差异另以 ``MXFP8-DRIFT INFO`` 输出，不影响 kernel 成败。

shape 全部可由环境变量覆盖（run_precision.sh 的矩阵逐组注入）：
  BENCH_SEQ / BENCH_HIDDEN / BENCH_FFN / BENCH_EXPERTS / BENCH_TOPK
  约束：HIDDEN%256==0、2*FFN%512==0、EXPERTS%EP==0、1<=TOPK<=EXPERTS、SEQ<=16384
"""
import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist

from golden import BigOp, _grouped_matmul, _grouped_wgrad
from fused import build_bigop_v2
from make_fake_moe_layer import make_fake_moe_layer
from utils import (
    DEFAULT_IP_PORT,
    group_report,
)

ACLSHMEM_DATA_OP_UDMA = 0x08
LOCAL_MEM_BYTES = 1024 * 1024 * 1024

# ---- shape（融合算子约束：K%256==0、N%512==0、SEQ<=16384）----
# 全部可由环境变量覆盖：run_precision.sh 的 shape 矩阵逐组注入（BENCH_*）。
HIDDEN = int(os.environ.get("BENCH_HIDDEN", "1024"))
FFN = int(os.environ.get("BENCH_FFN", "2048"))
NUM_EXPERTS = int(os.environ.get("BENCH_EXPERTS", "16"))
TOPK = int(os.environ.get("BENCH_TOPK", "2"))
SEQ = int(os.environ.get("BENCH_SEQ", "128"))   # 默认与精度一致；调大排查 shape 相关性

FWD_RTOL = 1e-2
FWD_ATOL = 1e-3


@dataclass
class TestConfig:
    """运行开关（环境变量读取）。"""

    ip_port: str
    test_mode: str      # forward | backward | both
    fused_mode: str     # train | forward


@dataclass
class V2Context:
    """一次验证的全部输入/环境对象（setup 阶段一次性构造）。"""

    rank: int
    world: int
    moe_layer: object       # 共用同一份 moe_layer（v1/v2 先后跑，中间清零梯度）
    x: torch.Tensor         # (seq,1,K) bf16
    grad_y: torch.Tensor
    K: int


# ===================== 0. 配置 + 输入 =====================

def read_config() -> TestConfig:
    """读取并校验运行开关。"""
    ip_port = os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", DEFAULT_IP_PORT)
    os.environ.setdefault("SHMEM_UID_SESSION_ID", ip_port.removeprefix("tcp://"))
    test_mode = os.environ.get("TEST_MODE", "both")       # forward | backward | both
    fused_mode = os.environ.get("FUSED_MODE", "train")    # train | forward
    assert test_mode in ("forward", "backward", "both"), \
        f"TEST_MODE must be forward/backward/both, got {test_mode}"
    return TestConfig(ip_port, test_mode, fused_mode)


def configure_shape(seq: int, hidden: int, ffn: int, experts: int, topk: int):
    """Update the module-level shape used by the existing test helpers.

    The normal runner imports one process per case and therefore obtains these
    values from BENCH_* once.  The fast runner keeps one distributed process
    group alive and calls this before constructing each case.
    """
    global SEQ, HIDDEN, FFN, NUM_EXPERTS, TOPK
    SEQ = seq
    HIDDEN = hidden
    FFN = ffn
    NUM_EXPERTS = experts
    TOPK = topk


def setup(ctx: V2Context, *, init_process_group: bool = True):
    """0. 环境初始化 + 输入构造。dist 启动、seed、fake MoELayer、x/grad_y。"""
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(rank % torch.npu.device_count())
    if init_process_group:
        dist.init_process_group(backend="hccl", world_size=world, rank=rank)
    torch.manual_seed(123)

    moe_layer, config = make_fake_moe_layer(
        hidden_size=HIDDEN, ffn_hidden_size=FFN, num_moe_experts=NUM_EXPERTS,
        router_topk=TOPK, aux_loss_coeff=0.0)
    # Keep the BigOp golden clamp contract identical to the fused operator.
    # Saved-state format affects only the fused backward; swiglu_limit affects
    # both forward computations and therefore must also be visible to golden.
    config.swiglu_limit = float(os.environ.get("MEGAMOE_SWIGLU_LIMIT", "0"))
    x = torch.randn(SEQ, 1, HIDDEN, device="npu", dtype=torch.bfloat16)
    grad_y = torch.randn_like(x)
    print(f"[rank{rank}] shape H={HIDDEN} FFN={FFN} N={2 * FFN} "
          f"E={NUM_EXPERTS} EP={world} topk={TOPK} seq={SEQ} "
          f"save_gmm1={os.environ.get('MEGAMOE_SAVE_GMM1_OUTPUT', 'bf16')} "
          f"swiglu_limit={config.swiglu_limit}", flush=True)

    ctx.rank, ctx.world = rank, world
    ctx.moe_layer = moe_layer
    ctx.x, ctx.grad_y = x, grad_y
    ctx.K = HIDDEN


def zero_layer_grads(moe_layer):
    """清空 moe_layer 参数的 .grad（避免 v1/v2 梯度串扰）。

    SimpleNamespaceShim 非 nn.Module 无 .parameters()，显式列三个可导参数。
    """
    for p in (moe_layer.experts.weight1, moe_layer.experts.weight2,
              moe_layer.router.gating.weight):
        p.grad = None


# ===================== ① bigop v1（golden） =====================

def bigop_v1_forward(ctx: V2Context, backward_saved_mode: str = "bf16"):
    """bigop v1 前向参考 y（前向对比基准）。"""
    ctx.moe_layer.config.backward_saved_gate_up_mode = backward_saved_mode
    x_g = ctx.x.detach().clone().requires_grad_(True)
    ref = BigOp.apply(x_g, ctx.moe_layer)
    return x_g, ref


def bigop_v1_backward_golden(ctx: V2Context, x_g, ref):
    """Build backward goldens, then clear parameter grads for v2.

    Weight/router gradients retain the normal BigOp semantics.  For grad_x,
    the hard-gate reference matches the fused kernel's unpermute boundary:
    BF16 route rows are accumulated in FP32 and cast to BF16 once per token.
    The original BigOp BF16 scatter_add result is retained as INFO-only drift.
    """
    config = ctx.moe_layer.config
    experts = ctx.moe_layer.experts
    flag_name = "_precision_match_fused_unpermute"
    result_name = "_precision_fused_unpermute_grad_x"
    missing = object()
    previous_flag = getattr(config, flag_name, missing)
    if hasattr(experts, result_name):
        delattr(experts, result_name)
    setattr(config, flag_name, True)
    try:
        ref.backward(ctx.grad_y)
    finally:
        if previous_flag is missing:
            delattr(config, flag_name)
        else:
            setattr(config, flag_name, previous_flag)

    fused_unpermute_grad_x = getattr(experts, result_name, None)
    if fused_unpermute_grad_x is None:
        raise RuntimeError("BigOp did not produce fused-unpermute grad_x reference")
    delattr(experts, result_name)
    golden = {
        "grad_x": fused_unpermute_grad_x.view(-1, ctx.K).detach(),
        "grad_x_bigop": x_g.grad.view(-1, ctx.K).detach(),
        "grad_w1": ctx.moe_layer.experts.weight1.grad.detach(),
        "grad_w2": ctx.moe_layer.experts.weight2.grad.detach(),
        "grad_gate": ctx.moe_layer.router.gating.weight.grad.detach(),
    }
    zero_layer_grads(ctx.moe_layer)
    return golden


def resolved_saved_gate_up_mode(ctx: V2Context) -> str:
    """Resolve the public auto mode exactly like the fused wrapper."""
    requested = os.environ.get("MEGAMOE_SAVE_GMM1_OUTPUT", "bf16").lower()
    if requested == "auto":
        return "mxfp8" if ctx.moe_layer.config.swiglu_limit <= 0 else "bf16"
    return requested


def build_precision_goldens(ctx: V2Context, need_backward: bool):
    """Build exact-forward plus mode-aware backward reference(s).

    MXFP8 is a lossy saved-state format, so implementation correctness is
    gated against a golden that applies the same encode/decode boundary.  The
    original exact-BF16 backward remains available as a model-drift report.
    """
    x_exact, ref = bigop_v1_forward(ctx, "bf16")
    if not need_backward:
        return ref, None, None, resolved_saved_gate_up_mode(ctx)

    exact = bigop_v1_backward_golden(ctx, x_exact, ref)
    mode = resolved_saved_gate_up_mode(ctx)
    if mode != "mxfp8":
        return ref, exact, None, mode

    x_aware, aware_ref = bigop_v1_forward(ctx, "mxfp8")
    aware = bigop_v1_backward_golden(ctx, x_aware, aware_ref)
    return ref, aware, exact, mode


# ===================== ② bigop v2（被测对象） =====================

def build_v2(ctx: V2Context, cfg: TestConfig):
    """构造 SHMEM + BigOpV2（融合算子按 cfg.fused_mode 选型）。"""
    from ascend_deepep import ShmemInstance

    shmem = ShmemInstance(ctx.rank, ctx.world, cfg.ip_port)
    shmem.set_data_op_engine_type(ACLSHMEM_DATA_OP_UDMA)
    shmem.init(LOCAL_MEM_BYTES)
    v2 = build_bigop_v2(shmem, LOCAL_MEM_BYTES, mode=cfg.fused_mode)
    return shmem, v2


def run_v2_forward(v2, ctx: V2Context):
    """v2 前向。返回 (x_v, y_v2)。"""
    x_v = ctx.x.detach().clone().requires_grad_(True)
    y_v2, _aux = v2(x_v, ctx.moe_layer)
    return x_v, y_v2


def run_v2_backward(y_v2, x_v, ctx: V2Context):
    """v2 反向，返回四组梯度 dict。forward-only 算子无 autograd 图会抛错（调用方处理）。"""
    y_v2.backward(ctx.grad_y)
    torch.npu.synchronize()
    return {
        "grad_x": x_v.grad.view(-1, ctx.K).detach(),
        "grad_w1": ctx.moe_layer.experts.weight1.grad.detach(),
        "grad_w2": ctx.moe_layer.experts.weight2.grad.detach(),
        "grad_gate": ctx.moe_layer.router.gating.weight.grad.detach(),
    }


# ===================== ③ 对比与报告 =====================

def fwd_report(name, got, ref):
    """前向 y 对比：got/ref 同 shape bf16。返回 (ok, detail)。"""
    got_flat = got.reshape(-1)
    ref_flat = ref.reshape(-1)
    if got_flat.numel() != ref_flat.numel():
        raise ValueError(f"{name}: element counts differ: {got.shape} vs {ref.shape}")
    got_f = got_flat.to(torch.float32)
    ref_f = ref_flat.to(torch.float32)
    diff = (got_f - ref_f).abs()
    n = diff.numel()
    # BigOpV2 returns [S,1,K], while the golden is [S,K]. Comparing the
    # originals broadcasts to [S,S,K] (80 GiB of temporaries at S4096/K1024).
    n_eq = (got_flat == ref_flat).float().mean().item()
    sd, _ = torch.sort(diff)
    rel = diff.norm().item() / (ref_f.norm().item() + 1e-12)
    try:
        torch.testing.assert_close(got_f, ref_f, rtol=FWD_RTOL, atol=FWD_ATOL)
        ok = True
    except AssertionError as e:
        ok = False
        first = str(e).strip().splitlines()
        diag = " | ".join(l.strip() for l in first[:3] if l.strip())
        return ok, f"rel={rel:.3e} bit_eq={n_eq:.6f} :: {diag}"
    return ok, (f"rel={rel:.3e} bit_eq={n_eq:.6f} "
                f"mean={diff.mean().item():.3e} P50={sd[n // 2].item():.3e} "
                f"P99={sd[int(n * 0.99) - 1].item():.3e} max={sd[-1].item():.3e}")


def report_forward(ctx: V2Context, y_v2, ref):
    """前向精度报告：y_v2 vs ref。"""
    ok, d = fwd_report("fwd_y", y_v2, ref.view(-1, ctx.K))
    tag = "SUCCESS" if ok else "Failed"
    print(f"[rank{ctx.rank}] [{tag}] fwd_y (BigOpV2 vs BigOp): {d}", flush=True)
    return ok


def report_backward(ctx: V2Context, got, golden):
    """反向精度报告；返回 ``(hard_ok, has_warning)``。"""
    reports = [
        ("grad_x", *group_report("grad_x", got["grad_x"], golden["grad_x"])),
        ("grad_w1", *group_report("grad_w1", got["grad_w1"], golden["grad_w1"])),
        ("grad_w2", *group_report("grad_w2", got["grad_w2"], golden["grad_w2"])),
        ("grad_gate", *group_report("grad_gate", got["grad_gate"], golden["grad_gate"])),
    ]
    for name, _ok, status, detail in reports:
        print(f"[rank{ctx.rank}] [{status}] {name}: {detail}", flush=True)
    _ok, _status, accumulation_drift = group_report(
        "grad_x_accum_drift", golden["grad_x"], golden["grad_x_bigop"])
    print(
        f"[rank{ctx.rank}] [GRAD_X_ACCUM_DRIFT INFO] "
        "kernel FP32-unpermute reference vs BigOp BF16 scatter_add: "
        f"{accumulation_drift}",
        flush=True,
    )
    n_success = sum(1 for _, _, status, _ in reports if status == "SUCCESS")
    n_warning = sum(1 for _, _, status, _ in reports if status == "WARNING")
    n_failed = sum(1 for _, _, status, _ in reports if status == "Failed")
    print(f"[rank{ctx.rank}] == BigOpV2 vs BigOp 对齐 "
          f"{n_success} SUCCESS / {n_warning} WARNING / {n_failed} Failed ==",
          flush=True)
    return n_failed == 0, n_warning != 0


def report_mxfp8_drift(ctx: V2Context, aware_golden, exact_golden):
    """Report pure MXFP8-vs-BF16 golden drift without changing pass/fail."""
    if exact_golden is None:
        return
    print(
        f"[rank{ctx.rank}] == MXFP8 compression drift vs exact BF16 "
        "(INFO only; not a kernel gate) ==",
        flush=True,
    )
    for name in ("grad_x", "grad_w1", "grad_w2", "grad_gate"):
        _ok, _status, detail = group_report(
            f"mxfp8_drift_{name}", aware_golden[name], exact_golden[name])
        print(f"[rank{ctx.rank}] [MXFP8-DRIFT INFO] {name}: {detail}", flush=True)


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def diagnose_grad_w2(v2, ctx: V2Context, got, golden):
    """Disambiguate dW2 input error from the fused CATLASS GMM error.

    Re-run the golden grouped-wgrad with the fused kernel's actual ptrDO/Hp.
    This is opt-in because copying backward workspace is unsuitable for the
    persistent multi-case fast runner.
    """
    holder = getattr(v2, "diagnostic_holder", None)
    if holder is None:
        return
    workspace = holder.get("backward_workspace")
    counts = holder.get("counts")
    if workspace is None or counts is None:
        print(f"[rank{ctx.rank}] [ERROR] DW2 diagnostic capture incomplete", flush=True)
        return

    max_recv = SEQ * TOPK * ctx.world
    ptrdo_bytes = 2 * max_recv * HIDDEN
    gradz_offset = (
        _align_up(ptrdo_bytes, 512)
        + _align_up(2 * max_recv * FFN, 512)       # gate
        + _align_up(2 * max_recv * FFN, 512)       # up
    )
    gradz_bytes = 2 * max_recv * 2 * FFN
    hp_offset = (
        gradz_offset + _align_up(gradz_bytes, 512)
    )
    hp_bytes = 2 * max_recv * FFN
    raw = workspace.detach().cpu().contiguous()
    valid_rows = int(counts.sum().item())
    ptrdo = raw.narrow(0, 0, ptrdo_bytes).view(torch.bfloat16)
    ptrdo = ptrdo.view(max_recv, HIDDEN)[:valid_rows].to(ctx.x.device)
    gradz = raw.narrow(0, gradz_offset, gradz_bytes).view(torch.bfloat16)
    gradz = gradz.view(max_recv, 2 * FFN)[:valid_rows].to(ctx.x.device)
    hp = raw.narrow(0, hp_offset, hp_bytes).view(torch.bfloat16)
    hp = hp.view(max_recv, FFN)[:valid_rows].to(ctx.x.device)
    regrouped = _grouped_wgrad(hp, ptrdo, counts.to(ctx.x.device))

    kernel_dw2 = got["grad_w2"].view_as(regrouped)
    golden_dw2 = golden["grad_w2"].view_as(regrouped)
    checks = (
        ("kernel_dW2 vs regrouped(fused Hp,ptrDO)", kernel_dw2, regrouped),
        ("regrouped(fused Hp,ptrDO) vs BigOp dW2", regrouped, golden_dw2),
    )
    for name, lhs, rhs in checks:
        _ok, status, detail = group_report(name, lhs, rhs)
        print(f"[rank{ctx.rank}] [DW2-DIAG {status}] {name}: {detail}", flush=True)

    golden_hp = getattr(ctx.moe_layer, "_precision_golden_hp", None)
    golden_do = getattr(ctx.moe_layer.experts, "_precision_golden_do", None)
    if golden_hp is not None and golden_hp.numel() == hp.numel():
        golden_hp = golden_hp.view_as(hp)
        _ok, status, detail = group_report(
            "fused Hp vs BigOp Hp", hp, golden_hp)
        print(f"[rank{ctx.rank}] [DW2-DIAG {status}] fused Hp vs BigOp Hp: "
              f"{detail}", flush=True)
    else:
        golden_hp = None
    if golden_do is not None and golden_do.numel() == ptrdo.numel():
        golden_do = golden_do.view_as(ptrdo)
        _ok, status, detail = group_report(
            "fused ptrDO vs BigOp dO", ptrdo, golden_do)
        print(f"[rank{ctx.rank}] [DW2-DIAG {status}] fused ptrDO vs BigOp dO: "
              f"{detail}", flush=True)
    else:
        golden_do = None

    # Cross-swap one operand at a time.  These two lines say whether the
    # observed dW2 miss follows Hp or dO, including their packed row order.
    if golden_hp is not None:
        hp_isolated = _grouped_wgrad(golden_hp, ptrdo, counts.to(ctx.x.device))
        _ok, status, detail = group_report(
            "golden Hp + fused ptrDO vs BigOp dW2", hp_isolated, golden_dw2)
        print(f"[rank{ctx.rank}] [DW2-DIAG {status}] golden Hp + fused ptrDO "
              f"vs BigOp dW2: {detail}", flush=True)
    if golden_do is not None:
        do_isolated = _grouped_wgrad(hp, golden_do, counts.to(ctx.x.device))
        _ok, status, detail = group_report(
            "fused Hp + golden dO vs BigOp dW2", do_isolated, golden_dw2)
        print(f"[rank{ctx.rank}] [DW2-DIAG {status}] fused Hp + golden dO "
              f"vs BigOp dW2: {detail}", flush=True)

    golden_gradz = getattr(ctx.moe_layer.experts, "_precision_golden_grad_z", None)
    if golden_gradz is not None and golden_gradz.numel() == gradz.numel():
        golden_gradz = golden_gradz.view_as(gradz)
        _ok, status, detail = group_report(
            "fused gradZ vs BigOp gradZ", gradz, golden_gradz)
        print(f"[rank{ctx.rank}] [DX-DIAG {status}] fused gradZ vs BigOp gradZ: "
              f"{detail}", flush=True)

        epr = NUM_EXPERTS // ctx.world
        weight1 = ctx.moe_layer.experts.weight1.view(epr, HIDDEN, 2 * FFN)
        weight1_t = weight1.transpose(-1, -2).contiguous()
        regrouped_dx = _grouped_matmul(
            gradz, weight1_t, counts.to(ctx.x.device))
        golden_packed_dx = getattr(
            ctx.moe_layer.experts, "_precision_golden_packed_dx", None)
        if (golden_packed_dx is not None and
                golden_packed_dx.numel() == regrouped_dx.numel()):
            _ok, status, detail = group_report(
                "regrouped(fused gradZ) vs BigOp packed dX",
                regrouped_dx, golden_packed_dx.view_as(regrouped_dx))
            print(f"[rank{ctx.rank}] [DX-DIAG {status}] regrouped(fused gradZ) "
                  f"vs BigOp packed dX: {detail}", flush=True)

    fused_input = holder.get("fused_input")
    fused_expert_dx = None if fused_input is None else fused_input.grad
    golden_expert_dx = getattr(
        ctx.moe_layer.experts, "_precision_golden_expert_dx", None)
    if fused_expert_dx is not None and golden_expert_dx is not None:
        _ok, status, detail = group_report(
            "fused expert dx vs BigOp expert dx",
            fused_expert_dx, golden_expert_dx.view_as(fused_expert_dx))
        print(f"[rank{ctx.rank}] [DX-DIAG {status}] fused expert dx vs "
              f"BigOp expert dx: {detail}", flush=True)

        golden_route_dx = getattr(
            ctx.moe_layer.experts,
            "_precision_golden_pre_unpermute_dx", None)
        golden_route_indices = getattr(
            ctx.moe_layer.experts,
            "_precision_golden_unpermute_indices", None)
        if golden_route_dx is not None and golden_route_indices is not None:
            # MegaMoeBF16Backward::UnpermuteBackward currently converts every
            # BF16 route row to FP32, accumulates all TopK rows in FP32, then
            # casts the token row once.  Reproduce exactly that numerical
            # boundary with golden route rows.  If this matches fused expert
            # dx while BF16 BigOp scatter_add does not, P2 GMM/communication is
            # exonerated and the miss is isolated to unpermute accumulation.
            route_dx_fp32 = golden_route_dx.to(torch.float32)
            fp32_accum = torch.zeros(
                fused_expert_dx.shape,
                dtype=torch.float32,
                device=fused_expert_dx.device,
            )
            expanded_indices = golden_route_indices.view(-1, 1).expand(
                -1, route_dx_fp32.shape[-1])
            fp32_accum.scatter_add_(0, expanded_indices, route_dx_fp32)
            fp32_accum = fp32_accum.to(fused_expert_dx.dtype)
            old_fp32_match, _status, detail = group_report(
                "fused expert dx vs FP32-accum(golden route dX)",
                fused_expert_dx, fp32_accum)
            if old_fp32_match:
                print(f"[rank{ctx.rank}] [DX-DIAG WARNING] fused expert dx "
                      f"still matches old FP32-accum route dX: {detail}",
                      flush=True)
            else:
                print(f"[rank{ctx.rank}] [DX-DIAG SUCCESS] fused expert dx "
                      f"no longer matches old FP32-accum route dX: {detail}",
                      flush=True)

    topk_weights = holder.get("topk_weights")
    fused_dtopk = None if topk_weights is None else topk_weights.grad
    golden_dtopk = getattr(
        ctx.moe_layer.experts, "_precision_golden_dtopk", None)
    if fused_dtopk is not None and golden_dtopk is not None:
        _ok, status, detail = group_report(
            "fused dtopk vs BigOp dtopk", fused_dtopk, golden_dtopk)
        print(f"[rank{ctx.rank}] [DX-DIAG {status}] fused dtopk vs "
              f"BigOp dtopk: {detail}", flush=True)

    golden_router_dx = getattr(
        ctx.moe_layer.experts, "_precision_golden_router_dx", None)
    if fused_expert_dx is not None and golden_router_dx is not None:
        # x.grad is the BF16 autograd accumulation of expert and router paths.
        # Replacing only the router side with golden reveals whether the final
        # miss can already be explained by the fused expert path.
        mixed_total = (
            fused_expert_dx + golden_router_dx.view_as(fused_expert_dx))
        _ok, status, detail = group_report(
            "fused expert dx + golden router dx vs BigOp grad_x",
            mixed_total.view_as(got["grad_x"]), golden["grad_x"])
        print(f"[rank{ctx.rank}] [DX-DIAG {status}] fused expert dx + "
              f"golden router dx vs BigOp grad_x: {detail}", flush=True)


# ===================== main =====================

def main():
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    cfg = read_config()

    # 0a. shape 合法性（矩阵注入错配时给明确报错，而不是到 kernel 里挂死）
    world = int(os.environ.get("WORLD_SIZE", "2"))
    if HIDDEN % 256 != 0 or (2 * FFN) % 512 != 0:
        print("[ERROR] requires HIDDEN % 256 == 0 and 2*FFN % 512 == 0",
              file=sys.stderr)
        sys.exit(2)
    if NUM_EXPERTS % world != 0:
        print(f"[ERROR] NUM_EXPERTS {NUM_EXPERTS} not divisible by EP {world}",
              file=sys.stderr)
        sys.exit(2)
    if not 1 <= TOPK <= NUM_EXPERTS:
        print(f"[ERROR] TOPK {TOPK} out of range [1, {NUM_EXPERTS}]",
              file=sys.stderr)
        sys.exit(2)
    if not 1 <= SEQ <= 16384:
        print(f"[ERROR] SEQ {SEQ} out of range [1, 16384]", file=sys.stderr)
        sys.exit(2)

    # 0. 环境 + 输入
    ctx = V2Context(0, 0, None, None, None, 0)
    setup(ctx)

    # ① bigop v1 golden：前向 ref 恒需要；反向 golden 仅 backward/both 需要
    need_backward = cfg.test_mode in ("backward", "both")
    ref, golden, exact_golden, golden_mode = build_precision_goldens(
        ctx, need_backward)
    if need_backward:
        print(
            f"[rank{ctx.rank}] backward hard-gate golden={golden_mode}; "
            f"exact_bf16_drift_report={'on' if exact_golden is not None else 'off'}",
            flush=True,
        )

    # ② bigop v2（router 手写 + 可插拔融合算子）
    n_fail = 0
    has_warning = False
    runtime_error = False
    shmem, v2 = build_v2(ctx, cfg)
    try:
        x_v, y_v2 = run_v2_forward(v2, ctx)

        # ③a 前向精度（TEST_MODE=forward/both）
        if cfg.test_mode in ("forward", "both"):
            if not report_forward(ctx, y_v2, ref):
                n_fail += 1

        # ③b 反向精度（TEST_MODE=backward/both；forward-only 算子自动 SKIP）
        if cfg.test_mode in ("backward", "both"):
            try:
                got = run_v2_backward(y_v2, x_v, ctx)
            except Exception as e:
                if cfg.fused_mode == "train":
                    print(f"[rank{ctx.rank}] [ERROR] backward execution failed: {e}",
                          flush=True)
                    runtime_error = True
                else:
                    print(f"[rank{ctx.rank}] [SKIP] backward 不可用（{e}）——"
                          f"forward-only 算子无 autograd 图", flush=True)
            else:
                backward_ok, backward_warning = report_backward(ctx, got, golden)
                diagnose_grad_w2(v2, ctx, got, golden)
                report_mxfp8_drift(ctx, golden, exact_golden)
                has_warning |= backward_warning
                if not backward_ok:
                    n_fail += 1
    finally:
        v2.destroy()
        shmem.destroy()
    dist.destroy_process_group()

    # Keep per-rank workers at zero for numerical Failed/WARNING so the outer
    # matrix runner can aggregate every rank/case.  run_precision*.sh converts
    # the final Failed summary to exit 2; only runtime errors exit here.
    tag = ("ERROR" if runtime_error else
           ("Failed" if n_fail else ("WARNING" if has_warning else "SUCCESS")))
    print(f"[rank{ctx.rank}] == FINAL [{tag}] seq={SEQ} hidden={HIDDEN} ffn={FFN} "
          f"E={NUM_EXPERTS} topk={TOPK}: {n_fail} check group(s) failed ==",
          flush=True)
    sys.exit(1 if runtime_error else 0)


if __name__ == "__main__":
    main()
