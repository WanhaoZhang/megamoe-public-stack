#!/usr/bin/env python3
"""Generate Chrome/Perfetto trace JSON from parsed MegaMoe W8A8 CSV."""

import argparse
import csv
import json
from pathlib import Path


CYCLE_TO_TIME_BASE = 1000.0


def read_csv_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def collect_event_names(fieldnames):
    events = []
    fields = set(fieldnames)
    for column in fieldnames:
        if not column.endswith("_start_cycle"):
            continue
        name = column[: -len("_start_cycle")]
        if name == "KERNEL_TIMING" or "WAIT" in name.upper():
            continue
        if not (name.startswith("AIC") or name.startswith("AIV")):
            continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path: Path) -> int:
    min_start = None
    fieldnames, rows = read_csv_rows(csv_path)
    for event in collect_event_names(fieldnames):
        start_column = f"{event}_start_cycle"
        end_column = f"{event}_end_cycle"
        for row in rows:
            start = int(row[start_column])
            end = int(row[end_column])
            if start > 0 and end > start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def generate_rank_events(csv_path: Path, rank: int):
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames)
    base_cycle = find_min_start(csv_path)
    trace_events = []

    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        for event in events:
            start = int(row[f"{event}_start_cycle"])
            end = int(row[f"{event}_end_cycle"])
            if start <= 0 or end <= start:
                continue
            trace_events.append(
                {
                    "ph": "X",
                    "cat": "aic_op" if core_type == "AIC" else "aiv_op",
                    "pid": f"rank{rank}",
                    "tid": f"group_{group_id}_{core_type}_{sub_id}",
                    "name": event,
                    "ts": round((start - base_cycle) / CYCLE_TO_TIME_BASE, 6),
                    "dur": round((end - start) / CYCLE_TO_TIME_BASE, 6),
                    "args": {
                        "rank": rank,
                        "group_id": group_id,
                        "core_type": core_type,
                        "sub_id": sub_id,
                        "start_cycle": start,
                        "end_cycle": end,
                    },
                }
            )
    return trace_events


def generate_trace_files(
    timer_dir: Path,
    rank_start: int,
    rank_end: int,
    output_dir: Path | None = None,
    merged_name: str | None = None,
    name_suffix: str = "",
) -> Path:
    output_dir = output_dir or timer_dir.parent / "timer_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events = []
    suffix = f"_{name_suffix}" if name_suffix else ""

    for rank in range(rank_start, rank_end + 1):
        csv_path = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Timer CSV does not exist: {csv_path}")
        rank_events = generate_rank_events(csv_path, rank)
        all_events.extend(rank_events)
        rank_output = output_dir / f"rank_{rank}_npu_time{suffix}.json"
        with rank_output.open("w", encoding="utf-8") as stream:
            json.dump(rank_events, stream, indent=2, ensure_ascii=False)
        print(f"[TIMER] rank={rank} events={len(rank_events)} JSON={rank_output}")

    if merged_name is None:
        merged_name = f"all_rank_trace{suffix}.json"
    merged_output = output_dir / merged_name
    with merged_output.open("w", encoding="utf-8") as stream:
        json.dump(all_events, stream, indent=2, ensure_ascii=False)
    return merged_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, required=True)
    parser.add_argument("--output")
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args()
    output = generate_trace_files(
        args.timer_dir,
        args.rank_start,
        args.rank_end,
        args.output_dir,
        args.output,
        args.name_suffix,
    )
    print(f"[TIMER] Merged trace JSON: {output}")


if __name__ == "__main__":
    main()
