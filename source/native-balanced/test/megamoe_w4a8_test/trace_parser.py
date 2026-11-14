import argparse
import csv
import json
import re
from collections import OrderedDict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

CYCLE_TO_TIME_BASE = 1000.0
TEST_DIR = Path(__file__).resolve().parent
DEFAULT_TIMER_DIR = TEST_DIR / "output" / "timer_registered" / "timer"
ITERATION_SUFFIX = re.compile(r"^(?P<stage>.+)_(?P<iteration>\d+)$")
DISPLAY_STAGES = (
    "AIV_DISPATCH",
    "AIC_GMM1",
    "AIV_SWIGLU",
    "AIC_GMM2",
    "AIV_COMBINE",
    "AIV_UNPERMUTE",
)
DISPLAY_PRECISION_US = Decimal("0.001")


def read_csv_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def collect_event_names(fieldnames):
    events = []
    fields = set(fieldnames)
    for col in fieldnames:
        if not col.endswith("_start_cycle"):
            continue
        name = col[: -len("_start_cycle")]
        if name == "KERNEL_TIMING":
            continue
        if not (name.startswith("AIC") or name.startswith("AIV")):
            continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path):
    """Return the minimum valid start cycle within a single rank's CSV.

    Each rank runs on a separate NPU whose hardware cycle counter has its own
    absolute origin, so every rank must be zeroed against its own minimum start
    rather than a shared global base; otherwise ranks with a higher counter
    origin appear to begin tens of milliseconds into the timeline.
    """
    if not csv_path.exists():
        return 0
    min_start = None
    fieldnames, rows = read_csv_rows(csv_path)
    for event in collect_event_names(fieldnames):
        start_col = f"{event}_start_cycle"
        end_col = f"{event}_end_cycle"
        for row in rows:
            start = int(row[start_col])
            end = int(row[end_col])
            if start > 0 and end > start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def generate_trace_json(csv_path, rank, base_cycle):
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames)
    trace_events = []

    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        tid = f"group_{group_id}_{core_type}_{sub_id}"
        pid = f"rank{rank}"

        for event in events:
            start_col = f"{event}_start_cycle"
            end_col = f"{event}_end_cycle"
            start = int(row[start_col])
            end = int(row[end_col])
            if start <= 0 or end <= start:
                continue

            ts = (start - base_cycle) / CYCLE_TO_TIME_BASE
            dur = (end - start) / CYCLE_TO_TIME_BASE
            trace_events.append({
                "ph": "X",
                "cat": "aic_op" if core_type == "AIC" else "aiv_op",
                "pid": pid,
                "tid": tid,
                "name": event,
                "ts": round(ts, 6),
                "dur": round(dur, 6),
                "args": {
                    "rank": rank,
                    "group_id": group_id,
                    "core_type": core_type,
                    "sub_id": sub_id,
                    "start_cycle": start,
                    "end_cycle": end,
                },
            })

    return trace_events


def summarize_stage_spans(trace_events):
    """Summarize each stage's cross-core wall-clock span.

    Each rank contributes one span per stage. The span includes all dynamic
    iterations and valid cores, starting at the earliest core start and ending
    at the latest core end. The overall stage average is calculated across
    ranks only.
    """
    stage_order = []
    intervals = {}

    for event in trace_events:
        event_name = event["name"]
        match = ITERATION_SUFFIX.match(event_name)
        if match:
            stage = match.group("stage")
        else:
            stage = event_name

        if stage not in stage_order:
            stage_order.append(stage)

        args = event["args"]
        key = (args["rank"], stage)
        start = args["start_cycle"]
        end = args["end_cycle"]
        if key not in intervals:
            intervals[key] = [start, end]
        else:
            intervals[key][0] = min(intervals[key][0], start)
            intervals[key][1] = max(intervals[key][1], end)

    durations_by_stage = OrderedDict((stage, []) for stage in stage_order)
    durations_by_rank_and_stage = OrderedDict()
    for (rank, stage), (start, end) in intervals.items():
        duration_us = Decimal(end - start) / Decimal(str(CYCLE_TO_TIME_BASE))
        durations_by_stage[stage].append(duration_us)
        durations_by_rank_and_stage[(rank, stage)] = duration_us

    return [
        {
            "stage": stage,
            "ranks": len(durations),
            "average_us": sum(durations) / len(durations),
            "rank_summaries": [
                {
                    "rank": rank,
                    "duration_us": rank_duration,
                }
                for (rank, rank_stage), rank_duration
                in durations_by_rank_and_stage.items()
                if rank_stage == stage
            ],
        }
        for stage, durations in durations_by_stage.items()
        if durations
    ]


def format_duration_us(duration_us):
    return format(
        Decimal(str(duration_us)).quantize(
            DISPLAY_PRECISION_US,
            rounding=ROUND_HALF_UP,
        ),
        "f",
    )


def print_stage_summary(trace_events):
    summaries_by_stage = {
        item["stage"]: item for item in summarize_stage_spans(trace_events)
    }
    summaries = [
        summaries_by_stage[stage]
        for stage in DISPLAY_STAGES
        if stage in summaries_by_stage
    ]
    if not summaries:
        print("[TIMER] No valid stage samples found.")
        return

    stage_width = max(len("Stage"), *(len(stage) for stage in DISPLAY_STAGES))
    rank_ids = sorted(
        {
            rank_summary["rank"]
            for item in summaries
            for rank_summary in item["rank_summaries"]
        }
    )

    print("Stage elapsed time (earliest start to latest end across all cores):")
    for rank in rank_ids:
        print()
        print(f"Rank {rank} / Card {rank}:")
        print(f"{'Stage':<{stage_width}}  {'Duration (us)':>13}")
        print(f"{'-' * stage_width}  {'-' * 13}")
        for item in summaries:
            rank_summary = next(
                (
                    summary
                    for summary in item["rank_summaries"]
                    if summary["rank"] == rank
                ),
                None,
            )
            if rank_summary is None:
                continue
            print(
                f"{item['stage']:<{stage_width}}  "
                f"{format_duration_us(rank_summary['duration_us']):>13}"
            )

    print()
    print("Average across cards:")
    print(f"{'Stage':<{stage_width}}  {'Duration (us)':>13}")
    print(f"{'-' * stage_width}  {'-' * 13}")
    for item in summaries:
        print(
            f"{item['stage']:<{stage_width}}  "
            f"{format_duration_us(item['average_us']):>13}"
        )


def main():
    parser = argparse.ArgumentParser(description="Generate Chrome trace JSON from parsed MegaMoe timer CSV files")
    parser.add_argument(
        "--timer-dir",
        type=Path,
        default=DEFAULT_TIMER_DIR,
        help="directory containing rank_X_npu_time_parsed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="trace JSON directory; defaults to <timer-dir>/../timer_json",
    )
    parser.add_argument("--rank-start", type=int, default=0, help="Start rank, inclusive")
    parser.add_argument("--rank-end", type=int, default=7, help="End rank, inclusive")
    parser.add_argument("--output", default="all_rank_trace.json", help="Merged trace JSON file name")
    args = parser.parse_args()

    timer_dir = args.timer_dir
    output_dir = args.output_dir or timer_dir.parent / "timer_json"
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_paths = [
        timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        for rank in range(args.rank_start, args.rank_end + 1)
    ]
    all_events = []

    for rank, csv_path in zip(range(args.rank_start, args.rank_end + 1), parsed_paths):
        if not csv_path.exists():
            print(f"[SKIP] rank_{rank}: 文件不存在 {csv_path}")
            continue
        base_cycle = find_min_start(csv_path)
        rank_events = generate_trace_json(csv_path, rank, base_cycle)
        all_events.extend(rank_events)
        rank_output = output_dir / f"rank_{rank}_npu_time.json"
        with open(rank_output, "w", encoding="utf-8") as f:
            json.dump(rank_events, f, indent=2, ensure_ascii=False, allow_nan=False)
        print(f"[INFO] rank_{rank}: {len(rank_events)} events -> {rank_output}")

    merged_output = output_dir / args.output
    with open(merged_output, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"[INFO] merged: {len(all_events)} events -> {merged_output}")
    print_stage_summary(all_events)


if __name__ == "__main__":
    main()
