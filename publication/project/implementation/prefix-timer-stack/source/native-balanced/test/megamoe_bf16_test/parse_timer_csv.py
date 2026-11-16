#!/usr/bin/env python3
"""Parse MegaMoe BF16 AscendTimerV2 raw per-rank CSV files."""

import argparse
import csv
from pathlib import Path


FIXED_NAMES = [
    "KERNEL_TIMING",
    "AIV_INIT_PARAM",
    "AIV_INIT_ROUTING",
    "AIV_WORKSPACE_INIT",
    "AIV_UDMA_COMBINE_DRAIN",
    "AIV_TOKEN_COUNT_ALLGATHER",
    "AIV_GMM2_COMBINE_END_WAIT",
    "AIV_REMOTE_COMBINE_WAIT",
    "AIV_UNPERMUTE",
    "AIC_RESERVED_0",
]
DYNAMIC_NAMES = [
    "AIV_DISPATCH",
    "AIV_SWIGLU_WAIT",
    "AIV_SWIGLU",
    "AIV_COMBINE_WAIT",
    "AIV_COMBINE",
    "AIC_GMM1_WAIT",
    "AIC_GMM1",
    "AIC_GMM2_WAIT",
    "AIC_GMM2",
    "AIV_GMM1_PROLOGUE",
    "AIV_GMM2_PROLOGUE",
    # Backward per-expert types occupy the shared dynamic region after the
    # forward entries (see AscendTimer::DynamicTimingType).  The forward
    # kernel never writes them, so they only pad the layout stride here;
    # parse_backward_timer.py owns their names.
    "AIV_BW_DISPATCH",
    "AIC_BW_RECOMPUTE",
    "AIC_BW_DO_WAIT",
    "AIC_BW_GMM2GRAD",
    "AIV_BW_SWIGLU_GRAD",
    "AIC_BW_GMM1GRAD",
    "AIV_BW_COMBINE",
    "AIC_BW_DW1",
    "AIC_BW_DW2",
]
MAX_DYNAMIC_ITER = 64
CORE_COUNT = 96
COUNTERS_PER_CORE = (
    (len(FIXED_NAMES) + len(DYNAMIC_NAMES) * MAX_DYNAMIC_ITER) * 2 + 15
) // 16 * 16
DYNAMIC_ITER_PER_CORE = (len(DYNAMIC_NAMES) + 15) // 16 * 16
BASE_COUNTERS = COUNTERS_PER_CORE * CORE_COUNT
TOTAL_SIZE = BASE_COUNTERS + DYNAMIC_ITER_PER_CORE * CORE_COUNT
CYCLE_TO_US = 1000.0


def load_raw(path: Path) -> list[int]:
    values = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.reader(stream):
            values.extend(int(item) for item in row if item.strip())
    if len(values) != TOTAL_SIZE:
        raise RuntimeError(
            f"{path} contains {len(values)} values; expected {TOTAL_SIZE}"
        )
    return values


def core_base(core_id: int) -> int:
    group_id, subgroup = divmod(core_id, 3)
    return group_id * COUNTERS_PER_CORE * 3 + subgroup * COUNTERS_PER_CORE


def event_pair(values: list[int], core_id: int, logical_index: int):
    index = core_base(core_id) + logical_index * 2
    start, end = values[index], values[index + 1]
    duration = (end - start) / CYCLE_TO_US if end > start else 0.0
    return start, end, duration


def parse_rank(raw_path: Path, parsed_path: Path) -> None:
    values = load_raw(raw_path)
    dynamic_counts = values[BASE_COUNTERS:]
    max_iters = [
        max(
            dynamic_counts[
                core_id * DYNAMIC_ITER_PER_CORE + type_index
            ]
            for core_id in range(CORE_COUNT)
        )
        for type_index in range(len(DYNAMIC_NAMES))
    ]

    events = [(name, index) for index, name in enumerate(FIXED_NAMES)]
    for type_index, name in enumerate(DYNAMIC_NAMES):
        for iteration in range(min(max_iters[type_index], MAX_DYNAMIC_ITER)):
            logical_index = (
                len(FIXED_NAMES) + type_index * MAX_DYNAMIC_ITER + iteration
            )
            events.append((f"{name}_{iteration}", logical_index))

    with parsed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name, _ in events:
            header.extend(
                [
                    f"{name}_start_cycle",
                    f"{name}_end_cycle",
                    f"{name}_duration(us)",
                ]
            )
        header.extend(f"{name}_IterCount" for name in DYNAMIC_NAMES)
        writer.writerow(header)

        for core_id in range(CORE_COUNT):
            group_id, subgroup = divmod(core_id, 3)
            core_type = "AIC" if subgroup == 0 else "AIV"
            sub_id = 0 if subgroup == 0 else subgroup - 1
            row = [core_id, group_id, core_type, sub_id]
            for _, logical_index in events:
                start, end, duration = event_pair(
                    values, core_id, logical_index
                )
                row.extend([start, end, f"{duration:.2f}"])
            dynamic_base = core_id * DYNAMIC_ITER_PER_CORE
            row.extend(
                dynamic_counts[dynamic_base + index]
                for index in range(len(DYNAMIC_NAMES))
            )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir or args.dir / "timer"
    output_dir.mkdir(parents=True, exist_ok=True)
    for rank in range(args.rank_start, args.rank_end + 1):
        raw_path = args.dir / f"rank_{rank}_npu_time.csv"
        parsed_path = output_dir / f"rank_{rank}_npu_time_parsed.csv"
        parse_rank(raw_path, parsed_path)
        print(f"[TIMER] parsed {raw_path} -> {parsed_path}")


if __name__ == "__main__":
    main()
