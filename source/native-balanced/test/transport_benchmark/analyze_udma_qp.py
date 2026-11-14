#!/usr/bin/env python3

import argparse
import csv
import shlex
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def parse_results(input_dir: Path):
    rows = []
    for path in sorted(input_dir.glob("rank*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("[RESULT] "):
                continue
            row = {"source_log": path.name}
            for token in shlex.split(line[len("[RESULT] "):]):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                row[key] = value
            if row.get("transport") == "remote_udma_qp":
                rows.append(row)
    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "rank", "transport", "op", "mode", "direction", "row_bytes", "rows",
        "total_bytes", "qp_count", "active_qp_count", "blocks", "warmup_operations",
        "measured_groups", "pipe", "total_cycles", "max_core_cycles",
        "global_start_cycle", "global_end_cycle",
        "submitted_operations", "gbps", "verify", "source_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows):
    grouped = {}
    for row in rows:
        key = (
            row["op"], int(row["row_bytes"]), int(row["rows"]),
            int(row["qp_count"]), int(row["active_qp_count"]),
            int(row["measured_groups"]),
        )
        grouped.setdefault(key, {})[int(row["rank"])] = row

    lines = [
        "# DeepEP UDMA QP单向与双向逐行搬运Benchmark",
        "",
        "`read/write`保持rank0发起的单向口径；`read_bi/write_bi`表示两张卡同时工作。",
        "每个AIV独占一个QP；每次`aclshmemx_udma_qp_put_nbi/get_nbi`搬运一行。",
        "每个AIV先预热100次单行搬运，然后quiet一次；预热不计入带宽。",
        "每个QP提交完本组中自己负责的行后执行一次`aclshmemx_udma_qp_quiet`。",
        "每个AIV独立记录所属QP的起止时间；Host以所有QP的最早开始到最晚结束作为本rank makespan。"
        "这样既覆盖AIV启动错位，也避免QP数小于物理AIV数时调用全AIV `SyncAll`。",
        "",
        "## 参数含义",
        "",
        "1. `单行字节`：单次QP PUT/GET搬运的数据量，即一行Token数据的字节数。",
        "2. `行数`：一个完整测量组内，单个rank累计搬运的总行数。",
        "3. `单rank单组字节`：`单行字节 × 行数`。",
        "4. `配置QP数`：SHMEM初始化时通过`aclshmemx_set_qp_num()`配置的QP数量。",
        "5. `活跃QP/AIV数`：该测点实际启动的AIV数量；每个AIV独占一个QP，等于`min(配置QP数, 行数)`。",
        "6. `计时组数`：完整行组在正式计时区间内重复执行的次数，默认100组。",
        "7. `rank0/rank1 GB/s`：各rank按自身makespan计算的单rank有效带宽。",
        "8. `双向合计GB/s`：双向模式下rank0与rank1带宽之和，不表示单条链路带宽。",
        "9. `校验`：目标数据与预期数据一致时为`PASS`。",
        "",
        "每个AIV先执行`warmup_operations=100`次单行搬运，随后quiet一次；预热不进入计时。正式计时中，每组结束时各QP都会执行`aclshmemx_udma_qp_quiet()`，完成后才进入下一组。",
        "",
        "例如64行、16个QP/AIV时，每个AIV每组负责4行；计时100组时，每个AIV共提交400次单行搬运，单rank共搬运6400行。",
        "",
        "```text",
        "单rank GB/s = row_bytes × rows × measured_groups / rank_makespan_seconds / 10^9",
        "rank_makespan = max(all end_cycle) - min(all start_cycle)",
        "双向合计 GB/s = rank0 GB/s + rank1 GB/s",
        "```",
        "",
    ]

    table_titles = {
        "read": "## READ 单向",
        "read_bi": "## READ 双向",
        "write": "## WRITE 单向",
        "write_bi": "## WRITE 双向",
    }
    format_gbps = lambda value: "N/A" if value is None else f"{value:.3f}"
    for operation in ("read", "read_bi", "write", "write_bi"):
        bidirectional = operation.endswith("_bi")
        lines.extend([table_titles[operation], ""])
        if bidirectional:
            lines.extend([
                "| 单行字节 | 行数 | 单rank单组字节 | 配置QP数 | 活跃QP/AIV数 | 计时组数 | rank0 GB/s | rank1 GB/s | 双向合计GB/s | 校验 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ])
        else:
            lines.extend([
                "| 单行字节 | 行数 | 单rank单组字节 | 配置QP数 | 活跃QP/AIV数 | 计时组数 | rank0 GB/s | 校验 |",
                "|---:|---:|---:|---:|---:|---:|---:|---|",
            ])

        operation_keys = sorted(
            key for key in grouped if key[0] == operation
        )
        for key in operation_keys:
            _, row_bytes, row_count, qp_count, active_qp_count, measured_groups = key
            rank_rows = grouped[key]
            rank0 = rank_rows.get(0)
            rank1 = rank_rows.get(1)
            rank0_gbps = float(rank0["gbps"]) if rank0 else None
            rank1_gbps = float(rank1["gbps"]) if rank1 else None
            valid = rank0 is not None and rank0.get("verify") == "PASS"
            if bidirectional:
                valid = (
                    valid and rank1 is not None and
                    rank1.get("verify") == "PASS"
                )
                aggregate_gbps = (
                    rank0_gbps + rank1_gbps
                    if rank0_gbps is not None and rank1_gbps is not None
                    else None
                )
                lines.append(
                    f"| {row_bytes} | {row_count} | "
                    f"{row_bytes * row_count} | {qp_count} | {active_qp_count} | "
                    f"{measured_groups} | {format_gbps(rank0_gbps)} | "
                    f"{format_gbps(rank1_gbps)} | "
                    f"{format_gbps(aggregate_gbps)} | "
                    f"{'PASS' if valid else 'FAIL'} |"
                )
            else:
                lines.append(
                    f"| {row_bytes} | {row_count} | "
                    f"{row_bytes * row_count} | {qp_count} | {active_qp_count} | "
                    f"{measured_groups} | {format_gbps(rank0_gbps)} | "
                    f"{'PASS' if valid else 'FAIL'} |"
                )
        lines.append("")

    lines.extend([
        "## 95%扫描峰值对应的首个饱和行数",
        "",
        "在每个`操作 × 单行字节 × QP/AIV数`组合内，以本次行数扫描的峰值为基准。",
        "单向使用rank0带宽，双向使用rank0与rank1带宽之和；只统计校验通过的测点。",
        "",
        "| 操作 | 单行字节 | 配置QP数 | 扫描峰值GB/s | 95%门槛GB/s | 首个达到门槛的总行数 | 该测点活跃QP/AIV数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    saturation = {}
    for key, rank_rows in grouped.items():
        operation, row_bytes, row_count, qp_count, active_qp_count, _ = key
        rank0 = rank_rows.get(0)
        rank1 = rank_rows.get(1)
        if rank0 is None or rank0.get("verify") != "PASS":
            continue
        value = float(rank0["gbps"])
        if operation.endswith("_bi"):
            if rank1 is None or rank1.get("verify") != "PASS":
                continue
            value += float(rank1["gbps"])
        saturation.setdefault(
            (operation, row_bytes, qp_count), []).append(
                (row_count, active_qp_count, value))
    for key in sorted(saturation):
        operation, row_bytes, qp_count = key
        points = sorted(saturation[key])
        peak = max(value for _, _, value in points)
        threshold = peak * 0.95
        first_rows, first_active_qps, _ = next(
            point for point in points if point[2] >= threshold)
        lines.append(
            f"| {operation} | {row_bytes} | {qp_count} | {peak:.3f} | "
            f"{threshold:.3f} | {first_rows} | {first_active_qps} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = parse_results(args.input_dir)
    if not rows:
        raise SystemExit("No remote_udma_qp result records found")
    write_csv(args.output_csv, rows)
    write_markdown(args.output_markdown, rows)


if __name__ == "__main__":
    main()
