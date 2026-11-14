#!/usr/bin/env python3

import argparse
import csv
import shlex
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
    for path in sorted(input_dir.glob("loop*_rank*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("[RESULT] "):
                continue
            row = {"source_log": path.name}
            for token in shlex.split(line[len("[RESULT] "):]):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                row[key] = value
            if (
                row.get("transport") == "remote_udma_bi"
                and row.get("mode") == "bandwidth"
                and row.get("bytes") == "65536"
            ):
                rows.append(row)
    return rows


def write_csv(path: Path, rows):
    fieldnames = [
        "rank", "transport", "op", "mode", "bytes", "blocks",
        "warmup", "loops", "batch", "pipe", "total_cycles",
        "min_cycles", "max_cycles", "gbps", "verify", "source_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows):
    lines = [
        "# DeepEP普通UDMA 64 KiB Loop Sweep",
        "",
        "两张卡同时使用1个AIV和普通QP 0向对端搬运；每个rank执行`loops`次普通UDMA NBI后只quiet一次：",
        "",
        "```text",
        "aclshmemx_udma_put_nbi/get_nbi × loops → aclshmemx_udma_quiet(peer)",
        "单rank GB/s = 65536 × loops / measured_seconds / 10^9",
        "双向合计GB/s = rank0 GB/s + rank1 GB/s",
        "```",
        "",
        "| 操作 | Loops | 每rank总字节 | rank0 GB/s | rank1 GB/s | 双向合计GB/s | 校验 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["op"], int(row["loops"]))].append(row)
    aggregate_points = defaultdict(list)
    for (operation, loops), group in sorted(groups.items()):
        by_rank = {int(row["rank"]): row for row in group}
        rank0 = by_rank.get(0)
        rank1 = by_rank.get(1)
        rank0_gbps = float(rank0["gbps"]) if rank0 is not None else 0.0
        rank1_gbps = float(rank1["gbps"]) if rank1 is not None else 0.0
        statuses = [row.get("verify", "FAIL") for row in group]
        status = "PASS" if len(group) == 2 and all(value == "PASS" for value in statuses) else "FAIL"
        total_bytes = 65536 * loops
        aggregate_gbps = rank0_gbps + rank1_gbps
        if status == "PASS":
            aggregate_points[operation].append((loops, aggregate_gbps))
        lines.append(
            f"| {operation} | {loops} | {total_bytes} | "
            f"{rank0_gbps:.3f} | {rank1_gbps:.3f} | "
            f"{aggregate_gbps:.3f} | {status} |"
        )

    lines.extend([
        "",
        "## 95%峰值判定",
        "",
        "`打满loops`定义为本次扫描中，第一个达到双向合计峰值95%的loops；该值只在当前扫描范围和设备环境内成立。",
        "",
        "| 操作 | 扫描峰值GB/s | 95%门槛GB/s | 首个达到门槛的Loops |",
        "|---|---:|---:|---:|",
    ])
    for operation in sorted(aggregate_points):
        points = sorted(aggregate_points[operation])
        peak = max(value for _, value in points)
        threshold = peak * 0.95
        saturation_loop = next(
            loops for loops, value in points if value >= threshold
        )
        lines.append(
            f"| {operation} | {peak:.3f} | {threshold:.3f} | "
            f"{saturation_loop} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    rows = parse_results(args.input_dir)
    if not rows:
        raise SystemExit("No 64 KiB remote_udma_bi loop-sweep records found")
    write_csv(args.output_csv, rows)
    write_markdown(args.output_markdown, rows)


if __name__ == "__main__":
    main()
