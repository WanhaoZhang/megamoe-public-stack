#!/usr/bin/env python3
"""Generate and compare data for the standalone C++ MegaMoe runner.

This utility intentionally does not import ascend_deepep or its Python extension.
It reuses the CPU W8A8 golden implementation shared with the registered test.
"""

import argparse
import csv
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_TEST_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_TEST_DIR))

from mega_moe_w8a8_golden import compare, generate  # noqa: E402


TOPK = 6
EXPERTS_PER_RANK = 16
LOCAL_MEM_BYTES = 1024 * 1024 * 1024


def read_shapes(csv_path: Path, ep: int) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != ["M", "N", "K"]:
            raise ValueError(
                f"{csv_path} must contain only M,N,K columns; "
                f"found {reader.fieldnames}"
            )
        cases = []
        for line_number, row in enumerate(reader, start=2):
            try:
                shape = {name: int(row[name]) for name in ("M", "N", "K")}
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{csv_path}:{line_number} contains a non-integer shape"
                ) from error
            if any(value <= 0 for value in shape.values()):
                raise ValueError(
                    f"{csv_path}:{line_number} requires positive M,N,K"
                )
            cases.append(
                {
                    **shape,
                    "TOPK": TOPK,
                    "expertPerRank": EXPERTS_PER_RANK,
                    "EP": ep,
                    "maxRecvTokenNum": (
                        shape["M"] * ep * min(EXPERTS_PER_RANK, TOPK)
                    ),
                    "localMemBytes": LOCAL_MEM_BYTES,
                }
            )
    if not cases:
        raise ValueError(f"{csv_path} does not contain any shapes")
    return cases


def shape_directory(case: dict) -> str:
    return f"M{case['M']}_N{case['N']}_K{case['K']}_EP{case['EP']}"


def write_runner_csv(
    path: Path,
    case: dict,
    case_id: str,
    warmup: int,
    loops: int,
) -> None:
    fields = [
        "case_id",
        "M",
        "K",
        "N",
        "TOPK",
        "expertPerRank",
        "EP",
        "maxRecvTokenNum",
        "localMemBytes",
        "warmup",
        "loops",
    ]
    row = {
        "case_id": case_id,
        **{name: case[name] for name in fields if name in case},
        "warmup": warmup,
        "loops": loops,
    }
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "compare"))
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=831)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--absolute-error-threshold", type=float, default=0.01)
    parser.add_argument("--max-error-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ep < 2:
        raise ValueError("MegaMoe requires at least two ranks")
    if args.warmup < 0 or args.loops <= 0:
        raise ValueError("warmup must be >= 0 and loops must be > 0")

    cases = read_shapes(args.csv_path, args.ep)
    for case in cases:
        case_dir = args.data_dir / shape_directory(case)
        case_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "generate":
            generate(case, case_dir, args.seed)
            write_runner_csv(
                case_dir / "runner_case.csv",
                case,
                "independent",
                args.warmup,
                args.loops,
            )
        else:
            compare(
                case,
                case_dir,
                deviation_gap_threshold=args.absolute_error_threshold,
                max_error_ratio=args.max_error_ratio,
                compare_cascade=False,
                absolute_error_threshold=args.absolute_error_threshold,
            )


if __name__ == "__main__":
    main()
