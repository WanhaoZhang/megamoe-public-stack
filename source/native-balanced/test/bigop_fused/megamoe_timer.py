"""MegaMoe BF16 前反向 kernel timer：run_speed 内联解析（MEGAMOE_TRACE=1 时打印）。

数据源：MegaMoeBF16Train 的 diagnostic_holder["forward_timer"] 和
diagnostic_holder["backward_timer"]，分别由 compute_train_with_timer 和
backward_with_timer 写出 AscendTimerV2 原始 buffer（int64 1-D tensor）。

布局常量与以下两处三方同步，改任何一处须同步其余：
  - test/megamoe_bf16_test/parse_backward_timer.py（离线全量解析/画时间线）
  - kernels/common/timer/AscendTimerV2.hpp（设备侧布局）

前提：算子 .so 须以 --enable-timer 构建：
  bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer
timer 构建每个打点前后带 PipeBarrier，有额外开销——**归因用 timer 构建，
报总数（speedup）用正式构建**，两套数字不要混。

用法（test_speed.py 调用；一般不直接用）：
  summarize_backward(holder["backward_timer"], r=seq * topk, k=HIDDEN, f=FFN,
                     rank_print=timing.rank0_print)
"""

CYCLE_TO_US = 1000.0  # GetSystemCycle 1 GHz：1000 cycle = 1us（与 parse_backward_timer 一致）

SHARED_FIXED_COUNT = 10
SHARED_DYNAMIC_COUNT = 20
BACKWARD_DYNAMIC_BASE = 11  # 反向 dynamic 类型在共享 dynamic 区的起始下标
MAX_DYNAMIC_ITER = 64
CORE_COUNT = 96             # 32 组 × (1 AIC + 2 AIV)
COUNTERS_PER_CORE = (
    (SHARED_FIXED_COUNT + SHARED_DYNAMIC_COUNT * MAX_DYNAMIC_ITER) * 2 + 15
) // 16 * 16
DYNAMIC_ITER_PER_CORE = (SHARED_DYNAMIC_COUNT + 15) // 16 * 16
BASE_COUNTERS = COUNTERS_PER_CORE * CORE_COUNT
TOTAL_SIZE = BASE_COUNTERS + DYNAMIC_ITER_PER_CORE * CORE_COUNT

FIXED_NAMES = [
    "KERNEL", "AIV_GATHER", "AIV_XSYNC_GATHER", "AIV_RESTORE_TABLE",
    "AIC_SPIN_AIV1", "AIC_SPIN_DZ", "AIC_SPIN_AIV0", "AIC_SPIN_HP",
    "AIV_XSYNC_UNPERMUTE", "AIV_UNPERMUTE", "AIV_XSYNC_DP", "AIV_DP_REDUCE",
    "AIV_XSYNC_TOPK", "AIV_TOPK_UNPERMUTE", "AIV_TABLE_RESET", "AIV_P0_TAIL",
    "AIV_RECOMPUTE_PASS", "AIV_P1_WAIT", "AIC_P2_TAIL", "AIC_INIT", "AIV_INIT",
] + [f"AIC_DZ_WAIT_{expert}" for expert in range(64)] + [
    f"AIV_DX_WAIT_{expert}" for expert in range(64)
] + [f"AIC_DO_REUSE_WAIT_{expert}" for expert in range(64)]
DYNAMIC_NAMES = [
    "AIV_DISPATCH", "AIC_RECOMPUTE", "AIC_DO_WAIT", "AIC_GMM2GRAD",
    "AIV_SWIGLU_GRAD", "AIC_GMM1GRAD", "AIV_COMBINE", "AIC_DW1", "AIC_DW2",
]
# 五个 GEMM 阶段的反向 FLOPs 系数（单位 R*F*K，docs/backward_perf_analysis.md）
GEMM_STAGE_COEF = {
    "AIC_RECOMPUTE": 4.0,   # P0-C 重算 Z = X@W1^T（2RKN = 4RFK）
    "AIC_GMM2GRAD": 2.0,    # P1 dHp = dO@W2
    "AIC_GMM1GRAD": 4.0,    # P2 dX = dZ@W1（2RNK = 4RFK）
    "AIC_DW1": 4.0,         # P3 dW1 = dZ^T@X（2NRK = 4RFK）
    "AIC_DW2": 2.0,         # P4 dW2 = dO^T@Hp
}
ROOF_TFLOPS = 410.4  # 432 峰值 × 0.95（docs/backward_perf_analysis.md）

# ---- 前向 timer ----
FWD_DYNAMIC_BASE = 0   # 前向 dynamic 起始下标（AscendTimer::AIV_DISPATCH）
FWD_DYNAMIC_NAMES = [
    "AIV_DISPATCH", "AIV_SWIGLU_WAIT", "AIV_SWIGLU",
    "AIV_COMBINE_WAIT", "AIV_COMBINE",
    "AIC_GMM1_WAIT", "AIC_GMM1", "AIC_GMM2_WAIT", "AIC_GMM2",
    "AIV_GMM1_PROLOGUE", "AIV_GMM2_PROLOGUE",
]
# 前向两个 GEMM 的 FLOPs 系数（单位 R*F*K）
FWD_GEMM_STAGE_COEF = {
    "AIC_GMM1": 4.0,   # gate+up fused: X @ [W1;W2]^T → R×K × K×2F = 2R·K·2F = 4RFK
    "AIC_GMM2": 2.0,   # down: Hp @ W3^T → R×F × F×K = 2R·F·K = 2RFK（SwiGLU 输出为 F）
}

_FIXED_INDEX = {name: i for i, name in enumerate(FIXED_NAMES)}


def _core_base(core_id):
    group, sub = divmod(core_id, 3)
    return group * COUNTERS_PER_CORE * 3 + sub * COUNTERS_PER_CORE


def _event(values, core_id, logical_index):
    idx = _core_base(core_id) + logical_index * 2
    start, end = values[idx], values[idx + 1]
    return (end - start) / CYCLE_TO_US if end > start else 0.0


def _timer_values(timer, rank_print):
    values = [int(v) for v in timer.detach().cpu().contiguous().tolist()]
    if len(values) < TOTAL_SIZE:
        values += [0] * (TOTAL_SIZE - len(values))
    values = values[:TOTAL_SIZE]
    if not any(values):
        rank_print(
            "[megamoe] timer buffer 全零：kernel 未以 --enable-timer 构建"
            "（bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer）。"
            "跳过 kernel 分阶段表")
        return None
    return values


def summarize_forward(timer, rank_print=print, dump_csv=None):
    """解析并报告整个 forward kernel 的跨核最大跨度。"""
    values = _timer_values(timer, rank_print)
    if values is None:
        return None
    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(v) for v in values))
            fh.write("\n")
        rank_print(f"[megamoe] forward 原始 timer 已落盘 {dump_csv}")
    kernel_us = max(
        (_event(values, core_id, 0) for core_id in range(CORE_COUNT)),
        default=0.0)
    rank_print(
        "==== [megamoe] v2 fwd kernel 总时间（us；max over cores；"
        "timer 构建含打点开销，仅作归因） ====")
    rank_print(f"  KERNEL {kernel_us:.1f}")
    return {"kernel_us": kernel_us}


def summarize_backward(timer, r=0, k=0, f=0, roof_tfps=ROOF_TFLOPS,
                       rank_print=print, dump_csv=None):
    """解析 timer buffer 并打印 kernel 分阶段表。

    timer   : int64 1-D tensor（AscendTimerV2 原始 buffer，backward_with_timer 返回）
    r/k/f   : 本 rank 实际行带 / hidden / FFN（>0 时给五个 GEMM 阶段算 TFLOPS）
    dump_csv: 可选路径，一值一行落盘原始 buffer（parse_backward_timer.load_raw 兼容）
    返回 {kernel_us, stages:{name:{kind,n_e,max_us,tflops?}}}（BENCH_JSON_OUT 的
    v2_kernel_stages 字段）；buffer 全零（kernel 未带 timer 构建）返回 None。
    """
    values = _timer_values(timer, rank_print)
    if values is None:
        return None
    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(v) for v in values))
            fh.write("\n")
        rank_print(f"[megamoe] 原始 timer 已落盘 {dump_csv}"
                   f"（parse_backward_timer.py --dir 可全量解析/画时间线）")

    def fixed_us(name):
        li = _FIXED_INDEX[name]
        aic = max((_event(values, c, li) for c in range(0, CORE_COUNT, 3)),
                  default=0.0)
        aiv = max((_event(values, c, li)
                   for c in range(CORE_COUNT) if c % 3 != 0), default=0.0)
        return aic, aiv

    kernel_us = max(fixed_us("KERNEL"))
    result = {"kernel_us": kernel_us, "stages": {}}
    rank_print("==== [megamoe] v2 bwd kernel 分阶段（us；max over cores；"
               "timer 构建含打点开销，仅作归因，speedup 报数用正式构建） ====")
    rank_print(f"  {'stage':<24}{'AIC_max':>10}{'AIV_max':>10}{'all_max':>10}"
               f"{'%KERNEL':>9}  note")

    # 整段 bar（fixed 槽：跨核全阶段一整条）
    for name in FIXED_NAMES:
        aic, aiv = fixed_us(name)
        all_max = max(aic, aiv)
        if all_max <= 0.0:
            continue
        pct = 100.0 * all_max / kernel_us if kernel_us > 0 else 0.0
        result["stages"][name] = {"kind": "fixed", "n_e": 1, "max_us": all_max}
        rank_print(f"  {name:<24}{aic:>10.1f}{aiv:>10.1f}{all_max:>10.1f}"
                   f"{pct:>8.1f}%")

    # 逐 expert 槽（dynamic）：per-core 各 expert 求和 -> worst core = 该阶段
    # makespan 贡献（AIC 各核 tile 均分，最慢核即阶段时长）
    counts = values[BASE_COUNTERS:]
    for t, name in enumerate(DYNAMIC_NAMES):
        type_idx = BACKWARD_DYNAMIC_BASE + t
        iters = max((counts[c * DYNAMIC_ITER_PER_CORE + type_idx]
                     for c in range(CORE_COUNT)), default=0)
        iters = min(iters, MAX_DYNAMIC_ITER)
        if iters <= 0:
            continue
        base_li = SHARED_FIXED_COUNT + type_idx * MAX_DYNAMIC_ITER
        per_core = [sum(_event(values, c, base_li + it) for it in range(iters))
                    for c in range(CORE_COUNT)]
        all_max = max(per_core) if per_core else 0.0
        if all_max <= 0.0:
            continue
        pct = 100.0 * all_max / kernel_us if kernel_us > 0 else 0.0
        note = ""
        tf = None
        coef = GEMM_STAGE_COEF.get(name)
        if coef and r > 0 and k > 0 and f > 0:
            tf = coef * r * f * k / (all_max * 1e-6) / 1e12
            note = (f"≈{tf:.0f} TFLOPS（{100.0 * tf / roof_tfps:.0f}% of "
                    f"roofline）")
        result["stages"][name] = {"kind": "dynamic", "n_e": iters,
                                  "max_us": all_max, "tflops": tf}
        label = f"{name}*{iters}"
        rank_print(f"  {label:<24}{'—':>10}{'—':>10}{all_max:>10.1f}"
                   f"{pct:>8.1f}%  {note}")
    # P1 Vector 等待占比：AIV_P1_WAIT（每 lane WaitForCube 累计，Accumulate）
    # 对照 AIV_SWIGLU_GRAD（每 lane 各 expert 求和后的最慢核）。首个 tile 的
    # 等待含 P0-C recompute 尾巴 + AIC 入口 SPIN 的 ramp。
    wait_us = result["stages"].get("AIV_P1_WAIT", {}).get("max_us", 0.0)
    swiglu_us = result["stages"].get("AIV_SWIGLU_GRAD", {}).get("max_us", 0.0)
    if wait_us > 0.0 and swiglu_us > 0.0:
        rank_print(f"  AIV_P1_WAIT/AIV_SWIGLU_GRAD = {wait_us:.1f}/"
                   f"{swiglu_us:.1f} us（{100.0 * wait_us / swiglu_us:.0f}%，"
                   "P1 Vector 等待占比）")
    rank_print("  （AIC_* 与 AIV_* 阶段跨核型流水重叠，各阶段 max 之和 > KERNEL 属正常；"
               "五个 AIC GEMM 阶段的 TFLOPS 对照 roofline 即 GMM 是否到顶）")
    return result


def summarize_forward(timer, r=0, k=0, f=0, roof_tfps=ROOF_TFLOPS,
                      rank_print=print, dump_csv=None):
    """解析前向 timer buffer 并打印 kernel 分阶段表。

    timer   : int64 1-D tensor（AscendTimerV2 原始 buffer，compute_train_with_timer 返回）
    r/k/f   : 本 rank 实际行带 / hidden / FFN（>0 时给 GEMM 阶段算 TFLOPS）
    dump_csv: 可选路径，一值一行落盘原始 buffer
    返回 {kernel_us, stages:{name:{kind,n_e,max_us,tflops?}}}；buffer 全零返回 None。
    """
    values = [int(v) for v in timer.detach().cpu().contiguous().tolist()]
    if len(values) < TOTAL_SIZE:
        values += [0] * (TOTAL_SIZE - len(values))
    values = values[:TOTAL_SIZE]
    if not any(values):
        rank_print(
            "[megamoe] forward timer buffer 全零：kernel 未以 --enable-timer 构建。"
            "跳过 forward kernel 分阶段表")
        return None
    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8") as fh:
            fh.write("\n".join(str(v) for v in values))
            fh.write("\n")
        rank_print(f"[megamoe] forward timer 已落盘 {dump_csv}")

    def fixed_us(name):
        li = _FIXED_INDEX[name]
        aic = max((_event(values, c, li) for c in range(0, CORE_COUNT, 3)),
                  default=0.0)
        aiv = max((_event(values, c, li)
                   for c in range(CORE_COUNT) if c % 3 != 0), default=0.0)
        return aic, aiv

    kernel_us = max(fixed_us("KERNEL"))
    result = {"kernel_us": kernel_us, "stages": {}}
    rank_print("==== [megamoe] fwd kernel 分阶段（us；max over cores；"
               "timer 构建含打点开销，仅作归因） ====")
    rank_print(f"  {'stage':<24}{'AIC_max':>10}{'AIV_max':>10}{'all_max':>10}"
               f"{'%KERNEL':>9}  note")

    # fixed 槽
    for name in FIXED_NAMES:
        aic, aiv = fixed_us(name)
        all_max = max(aic, aiv)
        if all_max <= 0.0:
            continue
        pct = 100.0 * all_max / kernel_us if kernel_us > 0 else 0.0
        result["stages"][name] = {"kind": "fixed", "n_e": 1, "max_us": all_max}
        rank_print(f"  {name:<24}{aic:>10.1f}{aiv:>10.1f}{all_max:>10.1f}"
                   f"{pct:>8.1f}%")

    # 前向 dynamic 槽（per-expert）
    counts = values[BASE_COUNTERS:]
    for t, name in enumerate(FWD_DYNAMIC_NAMES):
        type_idx = FWD_DYNAMIC_BASE + t
        iters = max((counts[c * DYNAMIC_ITER_PER_CORE + type_idx]
                     for c in range(CORE_COUNT)), default=0)
        iters = min(iters, MAX_DYNAMIC_ITER)
        if iters <= 0:
            continue
        base_li = SHARED_FIXED_COUNT + type_idx * MAX_DYNAMIC_ITER
        per_core = [sum(_event(values, c, base_li + it) for it in range(iters))
                    for c in range(CORE_COUNT)]
        all_max = max(per_core) if per_core else 0.0
        if all_max <= 0.0:
            continue
        pct = 100.0 * all_max / kernel_us if kernel_us > 0 else 0.0
        note = ""
        tf = None
        coef = FWD_GEMM_STAGE_COEF.get(name)
        if coef and r > 0 and k > 0 and f > 0:
            tf = coef * r * f * k / (all_max * 1e-6) / 1e12
            note = (f"≈{tf:.0f} TFLOPS（{100.0 * tf / roof_tfps:.0f}% of "
                    f"roofline）")
        result["stages"][name] = {"kind": "dynamic", "n_e": iters,
                                  "max_us": all_max, "tflops": tf}
        label = f"{name}*{iters}"
        rank_print(f"  {label:<24}{'—':>10}{'—':>10}{all_max:>10.1f}"
                   f"{pct:>8.1f}%  {note}")
    rank_print("  （AIC_* 与 AIV_* 阶段跨核型流水重叠，各阶段 max 之和 > KERNEL 属正常；"
               "AIC_GMM1/GMM2 的 TFLOPS 对照 roofline 即 GMM 是否到顶）")
    return result
