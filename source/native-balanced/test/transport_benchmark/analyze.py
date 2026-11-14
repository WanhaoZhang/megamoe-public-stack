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
    parser.add_argument("--output-parameters-csv", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--udma-loop-sweep-csv", type=Path)
    parser.add_argument("--remote-mte-bi-row-sweep-csv", type=Path)
    return parser.parse_args()


def parse_records(input_dir: Path):
    rows = []
    verifications = []
    skipped = []
    for path in sorted(input_dir.glob("rank*.log")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("[RESULT] "):
                prefix = "[RESULT] "
                target = rows
            elif line.startswith("[VERIFY] "):
                prefix = "[VERIFY] "
                target = verifications
            elif line.startswith("[SKIP] "):
                prefix = "[SKIP] "
                target = skipped
            else:
                continue
            row = {"source_log": path.name}
            for token in shlex.split(line[len(prefix):]):
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                row[key] = value
            target.append(row)
    return rows, verifications, skipped


def numeric(row, key):
    return float(row[key])


def write_csv(path: Path, rows):
    fieldnames = [
        "rank", "transport", "op", "mode", "bytes", "blocks",
        "warmup", "loops", "batch", "pipe", "total_cycles",
        "start_cycle", "end_cycle",
        "min_cycles", "max_cycles", "avg_ns", "min_ns", "max_ns",
        "gbps", "verify", "source_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def grouped_medians(rows, mode, value_key, extra_filter=None):
    groups = defaultdict(list)
    for row in rows:
        if row.get("mode") != mode:
            continue
        if row.get("verify") != "PASS":
            continue
        if extra_filter is not None and not extra_filter(row):
            continue
        key = (
            row["transport"], row["op"], int(row["bytes"]),
            int(row.get("blocks", "1")),
        )
        groups[key].append(numeric(row, value_key))
    return {
        key: statistics.median(values)
        for key, values in groups.items()
    }


def build_model_parameters(rows):
    bandwidth = grouped_medians(
        rows, "bandwidth", "gbps",
        extra_filter=lambda row: numeric(row, "gbps") > 0.0,
    )
    pairs = sorted({
        (row["transport"], row["op"])
        for row in rows
        if row.get("verify") == "PASS"
    })
    parameters = []
    for transport, operation in pairs:
        rtt_candidates = [
            numeric(row, "avg_ns")
            for row in rows
            if row.get("verify") == "PASS"
            and row.get("mode") == "rtt"
            and row.get("transport") == transport
            and row.get("op") == operation
            and int(row.get("bytes", 0)) == 64
            and numeric(row, "avg_ns") > 0.0
        ]
        bandwidth_candidates = [
            (value, size, blocks)
            for (row_transport, row_operation, size, blocks), value
            in bandwidth.items()
            if row_transport == transport and row_operation == operation
        ]
        if not rtt_candidates or not bandwidth_candidates:
            continue

        peak_gbps, peak_bytes, peak_blocks = max(bandwidth_candidates)
        parameters.append({
            "transport": transport,
            "op": operation,
            "rtt_bytes": 64,
            "rtt_avg_ns": max(rtt_candidates),
            "bandwidth_gbps": peak_gbps,
            "bandwidth_bytes": peak_bytes,
            "bandwidth_aiv_blocks": peak_blocks,
        })
    return parameters


def write_parameters_csv(path: Path, parameters):
    fieldnames = [
        "transport", "op", "rtt_bytes", "rtt_avg_ns",
        "bandwidth_gbps", "bandwidth_bytes", "bandwidth_aiv_blocks",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(parameters)


def read_udma_loop_sweep_rows(path):
    if path is None or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_remote_mte_bi_row_sweep_rows(path):
    if path is None or not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def append_udma_loop_sweep(lines, rows):
    if not rows:
        return

    lines.extend([
        "",
        "#### 64 KiB 双向 UDMA Loop Sweep",
        "",
        "两张卡各使用1个AIV和普通QP 0；每个rank连续提交`loops`次64 KiB NBI，随后只执行一次`aclshmemx_udma_quiet(peer)`。",
        "该表用于观察一次quiet前需要累计多少次NBI才能接近当前扫描峰值，不参与Cost Model峰值选择。",
        "",
        "| 操作 | Loops | 每rank总字节 | rank0 GB/s | rank1 GB/s | 双向合计GB/s | 校验 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
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
        status = (
            "PASS"
            if len(group) == 2 and all(row.get("verify") == "PASS" for row in group)
            else "FAIL"
        )
        aggregate_gbps = rank0_gbps + rank1_gbps
        if status == "PASS":
            aggregate_points[operation].append((loops, aggregate_gbps))
        lines.append(
            f"| {operation} | {loops} | {65536 * loops} | "
            f"{rank0_gbps:.3f} | {rank1_gbps:.3f} | "
            f"{aggregate_gbps:.3f} | {status} |"
        )

    lines.extend([
        "",
        "| 操作 | 扫描峰值GB/s | 95%门槛GB/s | 首个达到门槛的Loops |",
        "|---|---:|---:|---:|",
    ])
    for operation in sorted(aggregate_points):
        points = sorted(aggregate_points[operation])
        peak = max(value for _, value in points)
        threshold = peak * 0.95
        saturation_loop = next(loops for loops, value in points if value >= threshold)
        lines.append(
            f"| {operation} | {peak:.3f} | {threshold:.3f} | {saturation_loop} |"
        )


def append_remote_mte_bi_row_sweep(lines, rows):
    if not rows:
        return

    lines.extend([
        "",
        "#### Remote MTE BI Token-row Sweep",
        "",
        "每次MTE NBI只搬一行，并访问连续且互不重叠的行地址。每个AIV先预热100次单行搬运，然后quiet一次。",
        "1个AIV搬全部行；2个AIV平分固定总行数，因此两种配置的单rank总字节数相同。正式计时默认执行100个完整行组，每组结束时各AIV都执行一次`aclshmemx_mte_quiet()`。",
        "Host以所有参与AIV的最早开始到最晚结束作为本rank makespan，与UDMA QP口径一致。",
        "该表用于观察2/4/8 KiB Token行累计到16/64/256/1024行时的双向带宽，不参与Cost Model峰值选择。",
        "",
        "##### 参数含义",
        "",
        "1. `单行字节`：单次MTE GET/PUT搬运的数据量。",
        "2. `总行数`：一个完整测量组内，单个rank累计搬运的总行数。",
        "3. `每AIV行数`：`总行数 ÷ AIV Blocks`。",
        "4. `AIV Blocks`：实际参与搬运的AIV数量，当前测试1核和2核。",
        "5. `计时组数`：完整行组在正式计时区间内重复执行的次数，默认100组。",
        "6. `单rank单组字节`：`单行字节 × 总行数`。",
        "7. `Rank数`：同时参与环形双向搬运并产生结果的rank数量。",
        "8. `所有rank合计GB/s`：各rank带宽之和，不表示单条链路带宽。",
        "9. `校验`：所有rank目标数据均正确时为`PASS`。",
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
            "",
            f"##### {operation.upper()} 双向",
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

    lines.extend([
        "",
        "##### 95%扫描峰值对应的首个饱和行数",
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


def format_optional(value, digits=3):
    if value == "":
        return "N/A"
    return f"{float(value):.{digits}f}"


def render_markdown(
        rows, verifications, skipped, parameters, udma_loop_sweep_rows,
        remote_mte_bi_row_sweep_rows):
    lines = [
        "# DeepEP 搬运 Benchmark 汇总",
        "",
        "## 参数含义与计算口径",
        "",
        "### 通用字段",
        "",
        "| 字段 | 含义 | 使用说明 |",
        "|---|---|---|",
        "| `transport` | 搬运引擎/流量口径 | `remote_udma`为单向，"
        "`remote_udma_bi`为所有rank并发；`remote_mte`/`remote_mte_bi`"
        "同样区分单向和并发，其余为`local_sdma`或`local_mte` |",
        "| `op` | 数据方向 | `read`表示数据进入本地目标；`write`表示本地数据写向目标 |",
        "| `bytes` | 每次操作的消息大小 | 单个AIV核的消息大小 |",
        "| `AIV blocks`/`blocks` | 单rank同时参与搬运的AIV核数 | 远端MTE默认测1和2，本地MTE默认测1、4和16，其余路径默认固定为1 |",
        "| `Rank数` | 同时输出该组结果的rank数量 | 不等于AIV blocks |",
        "| `warmup` | 正式计时前的预热次数 | 不计入结果 |",
        "| `loops` | 每个AIV核在计时区间内执行的操作次数 | 单rank计时总字节数=`bytes × loops × blocks` |",
        "| `batch` | 两次完成动作之间连续提交的NBI数量 | 仅影响带宽模式；0表示全部loops提交完后只完成一次 |",
        "| `pipe` | UDMA WQE下发流水 | `pipe_s`或`pipe_mte3` |",
        "| `avg_ns/min_ns/max_ns` | 平均/最小/最大耗时 | 仅RTT使用；RTT采用总窗口平均，因此三列相等 |",
        "| `verify` | 数据正确性校验 | 只有`PASS`结果才进入Cost Model参数汇总 |",
        "",
        "### Blocks口径",
        "",
        "UDMA和SDMA默认使用`blocks=1`；远端MTE默认测`blocks=1,2`，本地MTE默认测`blocks=1,4,16`：",
        "",
        "```text",
        "每轮单rank搬运量 = bytes × blocks",
        "计时总搬运量     = bytes × loops × blocks",
        "```",
        "",
        "每个block使用独立地址并各搬一份`bytes`。远端MTE通过1和2 blocks对比单核与链路峰值；本地MTE通过1、4和16 blocks观察本地搬运并发扩展。",
        "",
        "### 带宽具体测量方式",
        "",
        "默认使用`warmup=100`和`loops=1000`；远端MTE带宽测`blocks=1,2`，本地MTE测`blocks=1,4,16`，其余路径测`blocks=1`；NBI路径使用`batch=0`。每次循环重复使用同一组源/目标地址；计时区间不包含预热、Host校验和kernel launch。",
        "",
        "| 路径 | READ | WRITE | 完成方式 | Rank流量口径 |",
        "|---|---|---|---|---|",
        "| `remote_udma` | `aclshmemx_udma_get_nbi`，远端GM→本地GM | `aclshmemx_udma_put_nbi`，本地GM→远端GM | `aclshmemx_udma_quiet(peer)` | 单向仅rank0活动；`_bi`为所有rank按环形peer并发 |",
        "| `local_sdma` | `aclshmemx_sdma_get_nbi`，本地GM→本地GM | Ascend950不支持，明确SKIP | `aclshmemx_sdma_quiet` | 每个rank各自在本卡搬运 |",
        "| `remote_mte` | `aclshmemx_mte_get_nbi`，远端GM→本地GM | `aclshmemx_mte_put_nbi`，本地GM→远端GM | `aclshmemx_mte_quiet()` | 单向仅rank0活动；`_bi`为所有rank按环形peer并发 |",
        "| `local_mte` | 本地GM→64 KiB UB | 64 KiB UB→本地GM | 每个不超过64 KiB的chunk执行`PipeBarrier<PIPE_ALL>` | 每个rank各自在本卡搬运 |",
        "",
        "UDMA、远端MTE和SDMA属于NBI路径。默认`batch=0`时，执行顺序为：",
        "",
        "```text",
        "warmup次NBI → quiet（不计时）",
        "开始计时 → loops次NBI → quiet → 结束计时",
        "```",
        "",
        "UDMA/SDMA按配置的完整`bytes`发起每次NBI；远端MTE每次API同样传入完整`bytes`，并提供64 KiB UB scratch，内部拆分由SHMEM实现决定。本地MTE不是NBI：一个`bytes`操作在函数内部按64 KiB分块并逐块等待完成，因此`batch`不改变本地MTE的完成粒度。",
        "",
        "默认UDMA/SDMA扫描256 B到1 MiB并额外测896 KiB；远端MTE固定测1 MiB，本地MTE测试1 MiB和16 MiB。带宽按以下公式计算：",
        "",
        "```text",
        "GB/s = bytes × loops × blocks / 计时秒数 / 10^9",
        "```",
        "",
        "`_bi`结果仍按每个rank分别输出；汇总中的所有rank合计只是并发流量之和，不表示单链路带宽。",
        "",
        "### model_parameters.csv字段",
        "",
        "| 字段 | 含义 |",
        "|---|---|",
        "| `rtt_bytes` | RTT探针大小，当前固定为64 B |",
        "| `rtt_avg_ns` | 已测有效64 B记录中最慢一次的平均完成时间；用于保守建模 |",
        "| `bandwidth_gbps` | 所有参与AIV核的单rank聚合峰值，十进制GB/s |",
        "| `bandwidth_bytes` | 出现峰值带宽时的单block消息大小；默认即单次消息大小 |",
        "| `bandwidth_aiv_blocks` | 出现峰值带宽时，单rank并发搬运的AIV核数 |",
        "",
        "## 原生 RTT",
        "",
        "RTT 使用64 B单次原生搬运并等待该引擎完成。Remote MTE READ为远端GM→本地UB，WRITE为本地UB→远端GM；UDMA/SDMA按其原生GM→GM语义测量。512 B结果作为诊断值保留。",
        "",
        "| 路径 | 操作 | 字节 | 平均(ns) | 最小(ns) | 最大(ns) | 校验 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    rtt_rows = [row for row in rows if row.get("mode") == "rtt"]
    for row in sorted(
        rtt_rows,
        key=lambda item: (item["transport"], item["op"], int(item["bytes"]))
    ):
        lines.append(
            f"| {row['transport']} | {row['op']} | {row['bytes']} | "
            f"{numeric(row, 'avg_ns'):.3f} | {numeric(row, 'min_ns'):.3f} | "
            f"{numeric(row, 'max_ns'):.3f} | {row['verify']} |"
        )

    lines.extend([
        "",
        "## 带宽",
        "",
        "`单 rank 聚合 GB/s` 是该 rank 上全部AIV block的并发总带宽；`所有 rank 合计`仅表示本次并发流量总和，不等于单链路带宽。",
    ])
    groups = defaultdict(list)
    for row in rows:
        if row.get("mode") != "bandwidth":
            continue
        key = (row["transport"], row["op"], row["bytes"], row["blocks"])
        groups[key].append(numeric(row, "gbps"))
    bandwidth_sections = [
        ("UDMA", {"remote_udma", "remote_udma_bi"}),
        ("SDMA", {"local_sdma"}),
        ("MTE", {"remote_mte", "remote_mte_bi", "local_mte"}),
    ]
    for section_name, transports in bandwidth_sections:
        lines.extend([
            "",
            f"### {section_name}",
            "",
            "| 路径 | 操作 | 字节 | AIV Blocks | Rank数 | 单rank聚合最小 | 单rank聚合中位 | 单rank聚合最大 | 所有rank合计(GB/s) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        section_keys = [
            key for key in groups
            if key[0] in transports
        ]
        for key in sorted(
            section_keys,
            key=lambda item: (item[0], item[1], int(item[2]), int(item[3])),
        ):
            values = groups[key]
            transport, operation, size, blocks = key
            lines.append(
                f"| {transport} | {operation} | {size} | {blocks} | {len(values)} | "
                f"{min(values):.3f} | {statistics.median(values):.3f} | "
                f"{max(values):.3f} | {sum(values):.3f} |"
            )
        if section_name == "UDMA":
            append_udma_loop_sweep(lines, udma_loop_sweep_rows)
        elif section_name == "MTE":
            append_remote_mte_bi_row_sweep(
                lines, remote_mte_bi_row_sweep_rows)

    lines.extend([
        "",
        "## Cost Model 参数候选",
        "",
        "带宽保持十进制`GB/s`，不转换为`GiB/s`。RTT参数取已测有效64 B记录中最慢一次的平均完成时间。",
        "",
        "| 路径 | 操作 | RTT(ns) | 峰值带宽(GB/s) | 峰值配置 |",
        "|---|---|---:|---:|---|",
    ])
    for parameter in parameters:
        lines.append(
            f"| {parameter['transport']} | {parameter['op']} | "
            f"{format_optional(parameter['rtt_avg_ns'])} | "
            f"{format_optional(parameter['bandwidth_gbps'])} | "
            f"{parameter['bandwidth_bytes']} B / "
            f"{parameter['bandwidth_aiv_blocks']} AIV blocks |"
        )

    lines.extend(["", "说明：远端MTE带宽默认测`AIV blocks=1,2`，本地MTE默认测`AIV blocks=1,4,16`，其余路径默认测`AIV blocks=1`。"])

    failed = [row for row in verifications if row.get("status") != "PASS"]
    unique_skipped = {
        (row.get("transport", "unknown"), row.get("op", "unknown"),
         row.get("reason", "unspecified"))
        for row in skipped
    }
    lines.extend([
        "",
        f"- 结果条数：`{len(rows)}`",
        f"- 搬运校验条数：`{len(verifications)}`",
        f"- 校验失败条数：`{len(failed)}`",
        f"- 跳过路径数：`{len(unique_skipped)}`",
        "- 原始 cycle、循环次数、batch 和每 rank 结果见 `results.csv`。",
        "",
    ])
    if unique_skipped:
        lines.extend(["## 跳过的路径", ""])
        for transport, operation, reason in sorted(unique_skipped):
            lines.append(f"- `{transport} {operation}`：`{reason}`")
        lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    rows, verifications, skipped = parse_records(args.input_dir)
    if not rows:
        raise SystemExit(f"No [RESULT] records found under {args.input_dir}")
    write_csv(args.output_csv, rows)
    parameters = build_model_parameters(rows)
    remote_mte_bi_row_sweep_rows = read_remote_mte_bi_row_sweep_rows(
        args.remote_mte_bi_row_sweep_csv)
    write_parameters_csv(args.output_parameters_csv, parameters)
    udma_loop_sweep_rows = read_udma_loop_sweep_rows(args.udma_loop_sweep_csv)
    args.output_markdown.write_text(
        render_markdown(
            rows, verifications, skipped, parameters, udma_loop_sweep_rows,
            remote_mte_bi_row_sweep_rows
        ),
        encoding="utf-8")


if __name__ == "__main__":
    main()
