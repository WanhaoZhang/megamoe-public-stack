#!/usr/bin/env python3
"""Generate Chrome/Perfetto trace JSON from parsed MegaMoe BF16 Timer CSV.

Two phases are supported, selected with --phase:

  forward (default): reads parse_timer_csv.py output
      rank_<rank>_npu_time_parsed.csv, renders the AIC*/AIV* fixed slots,
      and omits WAIT events (they stay in the parsed CSV for diagnostics).
      JSON names carry the case shape, e.g.
      rank_0_npu_time_4096_6144_3584_16_2_bf16_udma.json.

  backward: reads parse_backward_timer.py output
      rank_<rank>_backward_time_parsed.csv, renders the AIC_*/AIV_* fixed
      slots plus the per-expert blocks (AIC_GMM2GRAD_0..N etc.), and KEEPS
      the wait bars (AIC_SPIN_* / AIV_XSYNC_*): those sync points are the
      exact stalls this view is meant to expose, so dropping them would hide
      the problem.  AIV_P1_WAIT (accumulated per-lane dHp-tile wait) nests
      under the AIV_SWIGLU_GRAD_* bars like AIC_DO_WAIT_* under
      AIC_GMM2GRAD_*: Perfetto auto-stacks the overlap onto a second row.
      JSON names carry a "backward" tag.

Both emit one all_<N>_rank_<phase>_trace JSON with exactly 96 core lanes
per rank; per-rank files are opt-in with --per-rank. Other instrumented stage
names (weight movement, owner reduction, etc.) are retained too. Openable in
Perfetto (https://ui.perfetto.dev) or chrome://tracing with zoom / pan /
click-to-inspect.
"""

import argparse
import csv
import json
from pathlib import Path


CYCLE_TO_TIME_BASE = 1000.0

# Backward stages that are pure waits: AIC spins on a GM flag, and the
# cross-rank barrier windows inside CrossRankSyncInWorldSize().  They are
# kept in the trace and tagged cat=wait so the timeline highlights the sync
# stalls being investigated.
BACKWARD_WAIT_NAMES = {
    "AIC_SPIN_AIV1",
    "AIC_SPIN_DZ",
    "AIC_SPIN_AIV0",
    "AIC_SPIN_HP",
    "AIC_P2_TAIL",
    "AIV_P1_WAIT",
    "AIV_XSYNC_GATHER",
    "AIV_XSYNC_UNPERMUTE",
    "AIV_XSYNC_DP",
    "AIV_XSYNC_TOPK",
}


def read_csv_rows(csv_path: Path):
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        return reader.fieldnames or [], list(reader)


def collect_event_names(fieldnames, phase: str):
    events = []
    fields = set(fieldnames)
    for column in fieldnames:
        if not column.endswith("_start_cycle"):
            continue
        name = column[: -len("_start_cycle")]
        if phase == "backward":
            # Backward writes its fixed slots and the per-expert dynamic
            # blocks (AIC_GMM2GRAD_0..N) into the shared AscendTimerV2
            # layout.  KERNEL is the whole-kernel span and would flatten
            # every other bar.
            if name == "KERNEL":
                continue
        else:
            if name == "KERNEL_TIMING":
                continue
            # WAIT events remain in the parsed CSV for diagnostics, but omitting
            # them keeps the Perfetto timeline focused on actual work.
            if "WAIT" in name.upper():
                continue
        if f"{name}_end_cycle" in fields:
            events.append(name)
    return events


def find_min_start(csv_path: Path, phase: str) -> int:
    min_start = None
    fieldnames, rows = read_csv_rows(csv_path)
    for event in collect_event_names(fieldnames, phase):
        start_column = f"{event}_start_cycle"
        end_column = f"{event}_end_cycle"
        for row in rows:
            start = int(row[start_column])
            end = int(row[end_column])
            if start > 0 and end > start:
                min_start = start if min_start is None else min(min_start, start)
    return min_start if min_start is not None else 0


def generate_rank_events(csv_path: Path, rank: int, phase: str):
    fieldnames, rows = read_csv_rows(csv_path)
    events = collect_event_names(fieldnames, phase)
    base_cycle = find_min_start(csv_path, phase)
    trace_events = []

    for row in rows:
        group_id = int(row["group_id"])
        core_type = row["core_type"]
        sub_id = int(row["sub_id"])
        if not 0 <= group_id < 32 or (core_type, sub_id) not in {("AIC", 0), ("AIV", 0), ("AIV", 1)}:
            raise ValueError(f"invalid AIC/AIV coordinate: {group_id}/{core_type}/{sub_id}")
        for event in events:
            start = int(row[f"{event}_start_cycle"])
            end = int(row[f"{event}_end_cycle"])
            if start <= 0 or end <= start:
                continue
            is_wait = phase == "backward" and (event in BACKWARD_WAIT_NAMES or
                event.startswith("AIC_DZ_WAIT_") or event.startswith("AIV_DX_WAIT_") or
                event.startswith("AIC_DO_REUSE_WAIT_"))
            trace_events.append(
                {
                    "ph": "X",
                    "cat": (
                        "wait" if is_wait
                        else ("aic_op" if core_type == "AIC" else "aiv_op")
                    ),
                    "pid": rank,
                    "tid": group_id * 3 + (0 if core_type == "AIC" else sub_id + 1),
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
                        "is_wait": is_wait,
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
    phase: str = "forward",
    per_rank: bool = False,
) -> Path:
    output_dir = output_dir or timer_dir.parent / "timer_json"
    output_dir.mkdir(parents=True, exist_ok=True)
    all_events = []

    for rank in range(rank_start, rank_end + 1):
        if phase == "backward":
            csv_path = timer_dir / f"rank_{rank}_backward_time_parsed.csv"
        else:
            csv_path = timer_dir / f"rank_{rank}_npu_time_parsed.csv"
        if not csv_path.exists():
            if phase == "backward":
                # profile_mega_moe_bf16.py only writes timer CSVs for the
                # ranks it was asked to profile, so skip the gaps.
                continue
            raise FileNotFoundError(f"Timer CSV does not exist: {csv_path}")
        rank_events = generate_rank_events(csv_path, rank, phase)
        metadata = [{"ph":"M", "pid":rank, "name":"process_name", "args":{"name":f"Rank{rank}"}}]
        for lane in range(96):
            group, sub = divmod(lane, 3)
            label = f"group{group}/AIC0" if sub == 0 else f"group{group}/AIV{sub-1}"
            metadata.extend([
                {"ph":"M", "pid":rank, "tid":lane, "name":"thread_name", "args":{"name":label}},
                {"ph":"M", "pid":rank, "tid":lane, "name":"thread_sort_index", "args":{"sort_index":lane}},
            ])
        rank_events.sort(key=lambda e:(e['tid'],e['ts'],-e['dur']))
        all_events.extend(metadata + rank_events)
        suffix = f"_{name_suffix}" if name_suffix else ""
        rank_output = output_dir / f"rank_{rank}_{phase}_trace{suffix}.json"
        if per_rank:
            with rank_output.open("w", encoding="utf-8") as stream:
                json.dump(metadata + rank_events, stream, indent=2, ensure_ascii=False)
        print(f"[TIMER] rank={rank} events={len(rank_events)} tracks=96")

    if merged_name is None:
        suffix = f"_{name_suffix}" if name_suffix else ""
        count = len({e['pid'] for e in all_events})
        merged_name = f"all_{count}_rank_{phase}_trace{suffix}.json"
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
    parser.add_argument("--per-rank", action="store_true", help="also emit optional per-rank files")
    parser.add_argument(
        "--phase",
        choices=("forward", "backward"),
        default="forward",
        help="forward (parse_timer_csv output) or backward (parse_backward_timer output)",
    )
    args = parser.parse_args()
    merged_output = generate_trace_files(
        args.timer_dir,
        args.rank_start,
        args.rank_end,
        args.output_dir,
        args.output,
        args.name_suffix,
        args.phase,
        args.per_rank,
    )
    print(f"[TIMER] Merged trace JSON: {merged_output}")


if __name__ == "__main__":
    main()
