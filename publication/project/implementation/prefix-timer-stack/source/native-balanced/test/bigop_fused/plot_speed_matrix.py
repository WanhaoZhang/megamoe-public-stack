#!/usr/bin/env python3
"""聚合 run_speed 的 JSON 结果：summary.csv + 6 联对比图（shape/配置敏感性分析）。

数据源：test_speed.py 以 BENCH_JSON_OUT=<path> 产出的 per-run JSON
（run_speed_matrix.sh 按矩阵自动设置；手动单跑同样可用）。
一个 JSON = 一个配置点（K/F/E/topK/EP 固定，可含多个 seq）；图上一组 bar =
一个 (配置, seq) 组合。多 JSON 一起读即可对比不同 shape/配置。

用法：
    python plot_speed_matrix.py --dir speed_matrix/20260908_153000
    python plot_speed_matrix.py --jsons a.json,b.json --out my_report
输出（写到 --dir 或首个 json 所在目录；--out 为前缀）：
    summary.csv       全部配置 x 指标扁平表（Excel 可直接开）
    speed_matrix.png  7 面板（matplotlib 缺失时降级 ASCII 表 + 仅 CSV）
上排：fwd speedup | bwd speedup | GEMM efficiency megamoe vs bigop
中排：fwd time    | bwd time    | bigop decomposition
底部（全宽）：megamoe overhead（host wait + kernel wait 堆叠，需 MEGAMOE_TRACE=1）
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

CONFIG_COLS = ["case", "seq", "hidden", "ffn", "experts", "topk", "ep",
               "bwd_mode", "bwd_rfk", "bwd_recompute"]
METRIC_COLS = [
    "megamoe_bwd_event_ms", "bigop_bwd_event_ms", "speedup_bwd",
    "megamoe_fwd_event_ms", "bigop_fwd_event_ms", "speedup_fwd",
    "megamoe_fwd_bwd_event_ms", "bigop_fwd_bwd_event_ms", "speedup_fwd_bwd",
    "megamoe_roof_event_ms", "megamoe_roof_floor_ms", "megamoe_roof_tflops",
    "megamoe_roof_pct_roofline", "megamoe_roof_overhead_ms",
    "megamoe_roof_kernel_ms", "megamoe_roof_kernel_pct_roofline",
    "dec_total_ms", "dec_gemm_ms", "dec_alltoall_ms", "dec_other_ms",
    "dec_gemm_pct", "dec_alltoall_pct", "dec_other_pct", "dec_non_gemm_pct",
    "dec_gemm_tflops", "dec_gemm_pct_roofline", "dec_speedup_ceiling",
]
# ASCII 表列（PNG 之外随时可看的速览）
TABLE_COLS = ["case", "megamoe_bwd_event_ms", "bigop_bwd_event_ms", "speedup_bwd",
              "megamoe_roof_pct_roofline", "dec_gemm_pct", "dec_non_gemm_pct",
              "megamoe_roof_overhead_ms"]


def _num(v):
    """None 安全的 float；NaN/inf 归 None（JSON 里可能是 NaN 字面量）。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def load_rows(paths):
    """JSON 列表 -> 扁平行列表（保持文件顺序 = 矩阵扫描顺序）。"""
    rows = []
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        if isinstance(data, list):
            # 兼容旧格式：JSON 直接是 list，无 meta 包装
            meta = {}
            results = data
        else:
            meta = data.get("meta", {})
            results = data.get("results", [])
        for res in results:
            row = {
                "case": (f"s{res.get('seq')}_K{meta.get('hidden')}"
                         f"_F{meta.get('ffn')}_E{meta.get('experts')}"
                         f"_t{meta.get('topk')}_EP{meta.get('ep')}"),
                "seq": res.get("seq"), "hidden": meta.get("hidden"),
                "ffn": meta.get("ffn"), "experts": meta.get("experts"),
                "topk": meta.get("topk"), "ep": meta.get("ep"),
                "bwd_mode": meta.get("bwd_mode"),
                "bwd_rfk": meta.get("bwd_rfk"),
                "bwd_recompute": meta.get("bwd_recompute"),
            }
            for side in ("megamoe", "bigop"):
                for phase in ("fwd", "bwd", "fwd_bwd"):
                    st = (res.get(side) or {}).get(phase) or {}
                    row[f"{side}_{phase}_event_ms"] = _num(st.get("event_ms"))
            for key in ("fwd", "bwd", "fwd_bwd"):
                row[f"speedup_{key}"] = _num((res.get("speedup") or {}).get(key))
            rl = res.get("megamoe_roofline") or {}
            for key in ("event_ms", "floor_ms", "tflops", "pct_roofline",
                        "overhead_ms", "kernel_ms", "kernel_pct_roofline",
                        "speedup_no_host", "speedup_ideal"):
                row[f"megamoe_roof_{key}"] = _num(rl.get(key))
            fwd_rl = res.get("megamoe_fwd_roofline") or {}
            for key in ("event_ms", "floor_ms", "tflops", "pct_roofline",
                        "overhead_ms", "kernel_ms", "kernel_pct_roofline",
                        "gemm_ms", "gemm_pct_roofline",
                        "speedup_no_host", "speedup_ideal"):
                row[f"megamoe_fwd_roof_{key}"] = _num(fwd_rl.get(key))
            bp_fwd_rl = res.get("bigop_fwd_roofline") or {}
            for key in ("event_ms", "tflops", "pct_roofline"):
                row[f"bigop_fwd_roof_{key}"] = _num(bp_fwd_rl.get(key))
            dec = res.get("bigop_decomposition") or {}
            for key in ("gemm_ms", "alltoall_ms", "other_ms", "total_ms",
                        "non_gemm_ms", "gemm_pct", "alltoall_pct", "other_pct",
                        "non_gemm_pct", "gemm_tflops", "gemm_pct_roofline",
                        "speedup_ceiling"):
                row[f"dec_{key}"] = _num(dec.get(key))
            fwd_dec = res.get("bigop_fwd_decomposition") or {}
            for key in ("gemm_ms", "alltoall_ms", "other_ms", "total_ms",
                        "gemm_pct", "alltoall_pct", "other_pct",
                        "gemm_tflops", "gemm_pct_roofline"):
                row[f"fwd_dec_{key}"] = _num(fwd_dec.get(key))
            # 掩盖率 = (bigop e2e - megamoe e2e) / bigop alltoall 通信时间
            # fwd/bwd 各自用自己的 alltoall
            bp_a2a = row.get("dec_alltoall_ms")
            bp_bwd = row.get("bigop_bwd_event_ms")
            mm_bwd = row.get("megamoe_bwd_event_ms")
            if bp_a2a and bp_a2a > 0 and bp_bwd and mm_bwd:
                row["bwd_comm_coverage"] = 100.0 * (bp_bwd - mm_bwd) / bp_a2a
            fwd_a2a = row.get("fwd_dec_alltoall_ms")
            bp_fwd = row.get("bigop_fwd_event_ms")
            mm_fwd = row.get("megamoe_fwd_event_ms")
            if fwd_a2a and fwd_a2a > 0 and bp_fwd and mm_fwd:
                row["fwd_comm_coverage"] = 100.0 * (bp_fwd - mm_fwd) / fwd_a2a
            rows.append(row)
    return rows


def write_csv(rows, path):
    cols = CONFIG_COLS + METRIC_COLS + [
        c for c in (rows[0] if rows else {}) if c not in CONFIG_COLS + METRIC_COLS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k))
                             for k in cols})
    print(f"[out] csv: {path}  ({len(rows)} rows)")


def print_ascii_table(rows):
    """零依赖速览表。

    列说明：
      meg/bp        = "megamoe时间/bigop时间"（ms）
      speedup       = bigop/megamoe（>1 = megamoe 赢）
      meg roof%     = megamoe 反向整体效率 = floor/event（含 host，偏低）
      kernel gemm%  = megamoe kernel 内 GEMM 效率 = floor/kernel（不含 host）
                      ← 与 bp gemm% 同口径，直接对比
      bp gemm%      = bigop 的纯 matmul 效率（分步打点口径，含 W 转置 copy）
      host wait     = event - kernel = NPU 流空闲等 host 准备的时间
                      （autograd / workspace 分配 / launch）
      kernel wait   = kernel - floor = kernel 内部的等待
                      （P1 握手、barrier、tile 尾块）
      no-host su    = bigop_event / kernel_time = 消掉 host 后的 speedup
                      （"修 host 值不值"的直接答案）
    """
    if not rows:
        return

    def _s(row, key, suffix="", digits=1):
        v = row.get(key)
        return f"{v:.{digits}f}{suffix}" if isinstance(v, (int, float)) else "-"

    def _pair(r, phase):
        mm = _s(r, f"megamoe_{phase}_event_ms")
        bp = _s(r, f"bigop_{phase}_event_ms")
        return f"{mm}/{bp}" if mm != "-" and bp != "-" else "-"

    has_kernel = any(r.get("megamoe_roof_kernel_ms") is not None for r in rows)
    has_fwd_kernel = any(r.get("megamoe_fwd_roof_kernel_ms") is not None
                         for r in rows)

    # ---- 表头 ----
    print(f"{'case':<34} "
          f"{'fwd mmoe/bop':>13} {'speedup':>8} ", end="")
    if has_fwd_kernel:
        print(f"{'mmoe_roof%':>11} {'mmoe_gemm%':>11} {'bop_gemm%':>10} "
              f"{'host':>8} {'no_host_spd':>12} {'ideal_spd':>10} "
              f"{'coverage':>9} ", end="")
    else:
        print(f"{'mmoe_roof%':>11} ", end="")
    print(f"| {'bwd mmoe/bop':>13} {'speedup':>8} ", end="")
    if has_kernel:
        print(f"{'mmoe_roof%':>11} {'mmoe_gemm%':>11} {'bop_gemm%':>10} "
              f"{'host':>8} {'no_host_spd':>12} {'ideal_spd':>10} "
              f"{'coverage':>9} | ", end="")
    else:
        print(f"{'mmoe_roof%':>11} {'event-floor':>12} "
              f"{'bop_gemm%':>10} {'coverage':>9} | ", end="")
    print(f"{'f+b mmoe/bop':>13} {'speedup':>8}  {'mode':<12}")

    # 分隔线
    _fwd_cols = 73 if has_fwd_kernel else 12
    _bwd_cols = 73 if has_kernel else 54
    print("-" * (34 + 22 + _fwd_cols + 2 + 22 + _bwd_cols + 2 + 28))

    # ---- 数据行 ----
    for r in rows:
        line = (f"{r['case']:<34} "
                f"{_pair(r, 'fwd'):>13} "
                f"{_s(r, 'speedup_fwd', 'x', 2):>8} ")
        # fwd 详细
        if has_fwd_kernel:
            fk_ms = r.get("megamoe_fwd_roof_kernel_ms")
            fe_ms = r.get("megamoe_fwd_roof_event_ms")
            if fk_ms and fe_ms:
                host_fwd = fe_ms - fk_ms
                line += (f"{_s(r, 'megamoe_fwd_roof_pct_roofline', '%', 0):>11} "
                         f"{_s(r, 'megamoe_fwd_roof_gemm_pct_roofline', '%', 0):>11} "
                         f"{_s(r, 'fwd_dec_gemm_pct_roofline', '%', 0):>10} "
                         f"{host_fwd:6.1f}ms "
                         f"{_s(r, 'megamoe_fwd_roof_speedup_no_host', 'x', 2):>12} "
                         f"{_s(r, 'megamoe_fwd_roof_speedup_ideal', 'x', 2):>10} "
                         f"{_s(r, 'fwd_comm_coverage', '%', 0):>9} ")
            else:
                line += (f"{_s(r, 'megamoe_fwd_roof_pct_roofline', '%', 0):>11} "
                         f"{'-':>11} {'-':>10} {'-':>8} "
                         f"{'-':>12} {'-':>10} {'-':>9} ")
        else:
            line += f"{_s(r, 'megamoe_fwd_roof_pct_roofline', '%', 0):>11} "
        # bwd 详细
        line += (f"| {_pair(r, 'bwd'):>13} "
                 f"{_s(r, 'speedup_bwd', 'x', 2):>8} ")
        if has_kernel:
            k_ms = r.get("megamoe_roof_kernel_ms")
            e_ms = r.get("megamoe_bwd_event_ms")
            f_ms = r.get("megamoe_roof_floor_ms")
            if k_ms and e_ms and f_ms:
                host = e_ms - k_ms
                line += (f"{_s(r, 'megamoe_roof_pct_roofline', '%', 0):>11} "
                         f"{_s(r, 'megamoe_roof_kernel_pct_roofline', '%', 0):>11} "
                         f"{_s(r, 'dec_gemm_pct_roofline', '%', 0):>10} "
                         f"{host:6.1f}ms "
                         f"{_s(r, 'megamoe_roof_speedup_no_host', 'x', 2):>12} "
                         f"{_s(r, 'megamoe_roof_speedup_ideal', 'x', 2):>10} "
                         f"{_s(r, 'bwd_comm_coverage', '%', 0):>9} | ")
            else:
                line += (f"{'-':>11} {'-':>11} {'-':>10} {'-':>8} "
                         f"{'-':>12} {'-':>10} {'-':>9} | ")
        else:
            line += (f"{_s(r, 'megamoe_roof_pct_roofline', '%', 0):>11} "
                     f"{_s(r, 'megamoe_roof_overhead_ms', 'ms', 1):>12} "
                     f"{_s(r, 'dec_gemm_pct_roofline', '%', 0):>10} "
                     f"{_s(r, 'bwd_comm_coverage', '%', 0):>9} | ")
        line += (f"{_pair(r, 'fwd_bwd'):>13} "
                 f"{_s(r, 'speedup_fwd_bwd', 'x', 2):>8}  "
                 f"{r.get('bwd_mode') or '-':<12}")
        print(line)


def _grouped_bars(ax, rows, title, series, ylabel):
    n = len(series)
    width = 0.8 / n
    xs = range(len(rows))
    for i, (name, key, color) in enumerate(series):
        vals = [r.get(key) if r.get(key) is not None else 0.0 for r in rows]
        ax.bar([x + i * width for x in xs], vals, width=width, label=name,
               color=color)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_xticks([x + 0.4 - width / 2 for x in xs])
    ax.set_xticklabels([r["case"] for r in rows], rotation=40, ha="right",
                       fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot_png(rows, path):
    """渲染 2x3 六面板图；matplotlib 不可用返回 False。

    上排：fwd speedup | bwd speedup | GEMM efficiency (megamoe vs bigop)
    下排：fwd time    | bwd time    | bigop decomposition
    bwd time 有 kernel timer 时，megamoe 柱子堆叠显示 kernel + host wait。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    cases = [r["case"] for r in rows]
    n = len(rows)
    xs = range(n)

    def _annotate(ax, bar, fmt="{:.2f}"):
        h = bar.get_height()
        if h == 0:
            return
        ax.annotate(fmt.format(h),
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    def _speedup_ax(ax, key, title, potential_key=None, ideal_key=None):
        w = 0.35
        actual = [r.get(key) or 0.0 for r in rows]
        if potential_key:
            potential = [r.get(potential_key) or 0.0 for r in rows]
            b1 = ax.bar([x - w / 2 for x in xs], actual, w,
                        color="tab:green", label="current")
            b2 = ax.bar([x + w / 2 for x in xs], potential, w,
                        color="lightgreen", alpha=0.7, label="no-host")
            for b in b1:
                _annotate(ax, b, "{:.2f}x")
            for b in b2:
                _annotate(ax, b, "{:.2f}x")
            ymax = max(actual + potential)
            # 理论上限：gold 柱叠在 current/no-host 后面
            if ideal_key:
                ideals = [r.get(ideal_key) or 0.0 for r in rows]
                ax.bar(xs, ideals, width=2 * w,
                       color="gold", alpha=0.35, zorder=0,
                       label="ideal")
                for x, iv in zip(xs, ideals):
                    if iv > 0:
                        ax.annotate(f"{iv:.2f}x", xy=(x, iv),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha="center", va="bottom",
                                    fontsize=6.5, color="goldenrod",
                                    fontweight="bold")
                if any(iv > 0 for iv in ideals):
                    ymax = max(ymax, max(iv for iv in ideals if iv > 0))
            ax.legend(fontsize=8, loc="lower right")
        else:
            bars = ax.bar(xs, actual, color="tab:green", width=0.55)
            for b in bars:
                _annotate(ax, b, "{:.2f}x")
            ymax = max(actual)
        ax.axhline(1.0, color="red", ls="--", lw=1.2)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(ymax * 1.18, 1.3))

    def _time_ax(ax, phase, title, has_kernel_timer=False):
        mm = [r.get(f"megamoe_{phase}_event_ms") or 0.0 for r in rows]
        bp = [r.get(f"bigop_{phase}_event_ms") or 0.0 for r in rows]
        w = 0.35
        xs_mm = [x - w / 2 for x in xs]

        # 有 kernel timer 时，megamoe 柱子堆叠：kernel(底) + host wait(顶)
        kernel_key = (f"megamoe_{phase}_roof_kernel_ms" if phase == "fwd"
                      else "megamoe_roof_kernel_ms")
        if has_kernel_timer:
            kernel = [r.get(kernel_key) or 0.0 for r in rows]
            host = [m - k for m, k in zip(mm, kernel)]
            b1 = ax.bar(xs_mm, kernel, w,
                        label="megamoe (kernel)", color="tab:blue")
            b1h = ax.bar(xs_mm, host, w, bottom=kernel,
                         label="megamoe (host wait)", color="lightblue")
            for i, (xm, k, h, total) in enumerate(zip(xs_mm, kernel, host, mm)):
                cx = xm  # ax.bar 以 x 为中心，柱子中心就是 xm 本身
                if k > 0:
                    ax.text(cx, k / 2, f"{k:.1f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
                if h > 0:
                    ax.text(cx, k + h / 2, f"{h:.1f}", ha="center", va="center",
                            fontsize=7, color="black", fontweight="bold")
                ax.annotate(f"{total:.1f}", xy=(cx, total),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.5,
                            fontweight="bold")
        else:
            b1 = ax.bar(xs_mm, mm, w,
                        label="megamoe", color="tab:blue")
            for b in b1:
                _annotate(ax, b, "{:.1f}")

        b2 = ax.bar([x + w / 2 for x in xs], bp, w,
                    label="bigop", color="tab:orange")
        for b in b2:
            _annotate(ax, b, "{:.1f}")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("ms")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ymax = max(mm + bp) if mm + bp else 1
        ax.set_ylim(0, ymax * 1.15)

    modes = {r.get("bwd_mode") for r in rows if r.get("bwd_mode")}
    mode_str = " + ".join(sorted(modes)) if modes else "unknown"
    has_kernel = any(r.get("megamoe_roof_kernel_ms") is not None for r in rows)
    has_kernel_pct = any(r.get("megamoe_roof_kernel_pct_roofline") is not None
                         for r in rows)

    # 2x3 网格
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 3)
    fig.suptitle(f"megamoe vs bigop  [{mode_str}]",
                 fontsize=14, fontweight="bold")

    # Row 1 Col 1-2: fwd / bwd speedup
    has_fwd_no_host = any(r.get("megamoe_fwd_roof_speedup_no_host") for r in rows)
    if has_fwd_no_host:
        _speedup_ax(fig.add_subplot(gs[0, 0]), "speedup_fwd",
                    "fwd speedup",
                    potential_key="megamoe_fwd_roof_speedup_no_host",
                    ideal_key="megamoe_fwd_roof_speedup_ideal")
    else:
        _speedup_ax(fig.add_subplot(gs[0, 0]), "speedup_fwd", "fwd speedup")
    # bwd speedup：有 kernel timer 时画 current + no-host + ideal 三层
    has_bwd_no_host = any(r.get("megamoe_roof_speedup_no_host") for r in rows)
    if has_bwd_no_host:
        _speedup_ax(fig.add_subplot(gs[0, 1]), "speedup_bwd",
                    "bwd speedup",
                    potential_key="megamoe_roof_speedup_no_host",
                    ideal_key="megamoe_roof_speedup_ideal")
    else:
        _speedup_ax(fig.add_subplot(gs[0, 1]), "speedup_bwd", "bwd speedup")

    # Row 1 Col 3: GEMM 效率对比
    ax = fig.add_subplot(gs[0, 2])
    has_fwd_kernel = any(r.get("megamoe_fwd_roof_kernel_pct_roofline") is not None
                         for r in rows)
    has_fwd_kernel = any(r.get("megamoe_fwd_roof_kernel_pct_roofline") is not None
                         for r in rows)
    if has_kernel_pct:
        meg_bwd_key = "megamoe_roof_kernel_pct_roofline"
    else:
        meg_bwd_key = "megamoe_roof_pct_roofline"
    if has_fwd_kernel:
        meg_fwd_key = "megamoe_fwd_roof_gemm_pct_roofline"
    else:
        meg_fwd_key = "megamoe_fwd_roof_pct_roofline"
    mm_fwd = [r.get(meg_fwd_key) or 0.0 for r in rows]
    bp_fwd = [r.get("fwd_dec_gemm_pct_roofline") or 0.0 for r in rows]
    mm_bwd = [r.get(meg_bwd_key) or 0.0 for r in rows]
    bp_bwd = [r.get("dec_gemm_pct_roofline") or 0.0 for r in rows]
    w = 0.2
    b1 = ax.bar([x - 1.5 * w for x in xs], mm_fwd, w,
                label="mmoe fwd", color="deepskyblue")
    b2 = ax.bar([x - 0.5 * w for x in xs], bp_fwd, w,
                label="bop fwd", color="lightsalmon")
    b3 = ax.bar([x + 0.5 * w for x in xs], mm_bwd, w,
                label="mmoe bwd", color="tab:blue")
    b4 = ax.bar([x + 1.5 * w for x in xs], bp_bwd, w,
                label="bop bwd", color="tab:orange")
    for b in b1 + b2 + b3 + b4:
        _annotate(ax, b, "{:.0f}%")
    ax.axhline(95.0, color="red", ls="--", lw=1)
    _eff_note = ""
    if has_fwd_kernel and has_kernel_pct:
        _eff_note = " (kernel timer)"
    elif has_fwd_kernel or has_kernel_pct:
        _eff_note = " (mixed)"
    else:
        _eff_note = " (event)"
    ax.set_title(f"GEMM efficiency{_eff_note}", fontsize=10, fontweight="bold")
    ax.set_ylabel("% of roofline")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=6.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 115)

    # Row 2 Col 1-2: fwd / bwd time
    _time_ax(fig.add_subplot(gs[1, 0]), "fwd", "fwd time (ms)",
             has_kernel_timer=has_fwd_kernel)
    _time_ax(fig.add_subplot(gs[1, 1]), "bwd", "bwd time (ms)",
             has_kernel_timer=has_kernel)

    # Row 2 Col 3: 掩盖率（fwd + bwd）
    ax = fig.add_subplot(gs[1, 2])
    fwd_cov = [r.get("fwd_comm_coverage") or 0.0 for r in rows]
    bwd_cov = [r.get("bwd_comm_coverage") or 0.0 for r in rows]
    w = 0.35
    b1 = ax.bar([x - w / 2 for x in xs], fwd_cov, w,
                label="fwd", color="tab:cyan")
    b2 = ax.bar([x + w / 2 for x in xs], bwd_cov, w,
                label="bwd", color="tab:purple")
    for b in b1 + b2:
        _annotate(ax, b, "{:.0f}%")
    ax.axhline(100.0, color="red", ls="--", lw=1)
    ax.set_title("comm coverage (bigger = more hidden)", fontsize=10,
                 fontweight="bold")
    ax.set_ylabel("%")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(cases, rotation=45, ha="right", fontsize=6.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dir", type=Path,
                       help="读该目录下全部 *.json（run_speed_matrix 输出目录）")
    group.add_argument("--jsons", help="逗号分隔的 JSON 路径列表")
    parser.add_argument("--out", type=Path,
                        help="输出前缀（默认 <dir>/summary.csv + speed_matrix.png）")
    args = parser.parse_args()

    if args.dir is not None:
        paths = sorted(args.dir.glob("case_*.json"))
        outdir = args.dir
    else:
        paths = [Path(s.strip()) for s in args.jsons.split(",") if s.strip()]
        outdir = paths[0].parent if paths else Path(".")
    if not paths:
        sys.exit("[error] no input json files found")
    rows = load_rows(paths)
    if not rows:
        sys.exit("[error] loaded jsons contain no results")

    csv_path = args.out.with_suffix(".csv") if args.out else outdir / "summary.csv"
    png_path = args.out.with_suffix(".png") if args.out else outdir / "speed_matrix.png"
    write_csv(rows, csv_path)
    print_ascii_table(rows)
    if plot_png(rows, png_path):
        print(f"[out] png: {png_path}")
    else:
        print("[warn] matplotlib 不可用：仅生成 CSV + ASCII 表"
              "（pip install matplotlib 后重跑本脚本可出 PNG）")


if __name__ == "__main__":
    main()
