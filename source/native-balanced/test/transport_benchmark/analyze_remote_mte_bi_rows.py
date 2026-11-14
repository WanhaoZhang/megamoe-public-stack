#!/usr/bin/env python3

import argparse
import csv
import shlex
import statistics
from collections import defaultdict
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
            if row.get("transport") == "remote_mte_bi_rows":
                rows.append(row)
    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "rank", "transport", "op", "mode", "row_bytes", "total_rows",
        "rows_per_block", "total_bytes", "blocks", "warmup_operations",
        "measured_groups", "loops", "batch", "total_cycles",
        "max_core_cycles", "global_start_cycle", "global_end_cycle",
        "submitted_operations", "gbps", "verify", "source_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def append_table(lines, rows):
    lines.extend([
        "## Remote MTE BI Token-row Sweep",
        "",
        "每次MTE NBI只搬一行，并访问连续且互不重叠的行地址。每个AIV先预热100次单行搬运，然后quiet一次；预热不计入带宽。",
        "1个AIV搬全部行；2个AIV平分固定总行数，因此两种配置的单rank总字节数相同。正式计时默认执行100个完整行组，每组结束时各AIV都执行一次`aclshmemx_mte_quiet()`。",
        "Host使用所有参与AIV的最早开始到最晚结束作为本rank makespan，与UDMA QP口径一致。",
        "",
        "## 参数含义",
        "",
        "1. `单行字节`：单次MTE GET/PUT搬运的数据量，即一行Token数据的字节数。",
        "2. `总行数`：一个完整测量组内，单个rank累计搬运的总行数。",
        "3. `每AIV行数`：`总行数 ÷ AIV Blocks`，即每个AIV在一组内负责的行数。",
        "4. `AIV Blocks`：该测点实际参与搬运的AIV数量，当前测试1核和2核。",
        "5. `计时组数`：完整行组在正式计时区间内重复执行的次数，默认100组。",
        "6. `单rank单组字节`：`单行字节 × 总行数`。",
        "7. `Rank数`：同时参与环形双向搬运并产生结果的rank数量，不等于AIV Blocks。",
        "8. `单rank最小/中位/最大GB/s`：同一配置下各rank有效带宽的分布。",
        "9. `所有rank合计GB/s`：各rank带宽之和，用于表示并发总流量，不表示单条链路带宽。",
        "10. `校验`：所有rank目标数据均与预期一致时为`PASS`。",
        "",
        "每个AIV先执行`warmup_operations=100`次单行搬运，随后quiet一次；预热不进入计时。",
        "",
    ])
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["op"], int(row["row_bytes"]), int(row["total_rows"]),
            int(row["blocks"]), int(row["rows_per_block"]),
            int(row["measured_groups"]),
        )
        groups[key].append(row)

    for operation in ("read", "write"):
        lines.extend([
            f"### {operation.upper()} 双向",
            "",
            "| 单行字节 | 总行数 | 每AIV行数 | AIV Blocks | 计时组数 | 单rank单组字节 | Rank数 | 单rank最小GB/s | 单rank中位GB/s | 单rank最大GB/s | 所有rank合计GB/s | 校验 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for key in sorted(item for item in groups if item[0] == operation):
            group = groups[key]
            _, row_bytes, total_rows, blocks, rows_per_block, measured_groups = key
            values = [float(row["gbps"]) for row in group]
            status = (
                "PASS" if group and all(row.get("verify") == "PASS" for row in group)
                else "FAIL"
            )
            lines.append(
                f"| {row_bytes} | {total_rows} | {rows_per_block} | {blocks} | "
                f"{measured_groups} | {row_bytes * total_rows} | {len(values)} | {min(values):.3f} | "
                f"{statistics.median(values):.3f} | {max(values):.3f} | "
                f"{sum(values):.3f} | {status} |"
            )
        lines.append("")

    lines.extend([
        "## 95%扫描峰值对应的首个饱和行数",
        "",
        "在每个`操作 × 单行字节 × AIV Blocks`组合内，以所有rank双向合计带宽的扫描峰值为基准，只统计校验通过的测点。",
        "",
        "| 操作 | 单行字节 | AIV Blocks | 扫描峰值GB/s | 95%门槛GB/s | 首个达到门槛的总行数 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    saturation = defaultdict(list)
    for key, group in groups.items():
        operation, row_bytes, total_rows, blocks, _, _ = key
        if not group or not all(row.get("verify") == "PASS" for row in group):
            continue
        saturation[(operation, row_bytes, blocks)].append(
            (total_rows, sum(float(row["gbps"]) for row in group)))
    for key in sorted(saturation):
        operation, row_bytes, blocks = key
        points = sorted(saturation[key])
        peak = max(value for _, value in points)
        threshold = peak * 0.95
        first_rows = next(
            total_rows for total_rows, value in points if value >= threshold)
        lines.append(
            f"| {operation} | {row_bytes} | {blocks} | {peak:.3f} | "
            f"{threshold:.3f} | {first_rows} |"
        )
    lines.append("")


def main():
    args = parse_args()
    rows = parse_results(args.input_dir)
    if not rows:
        raise SystemExit(f"No Remote MTE BI row-sweep results found under {args.input_dir}")
    write_csv(args.output_csv, rows)
    lines = ["# DeepEP Remote MTE BI Token-row Sweep", ""]
    append_table(lines, rows)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
