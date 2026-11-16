#!/usr/bin/env python3
"""Generate and compare data for the standalone C++ MegaMoe BF16 runner."""

import argparse
import csv
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_TEST_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PARENT_TEST_DIR))

from mega_moe_bf16_golden import compare, generate  # noqa: E402


TOPK = 6
EXPERTS_PER_RANK = 16
BUFFER_ALIGNMENT = 512
DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
BASE_LOCAL_MEM_BYTES = 1024 * 1024 * 1024


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def local_mem_bytes(use_udma_operator: bool, max_recv_tokens: int, k: int) -> int:
    if not use_udma_operator:
        return BASE_LOCAL_MEM_BYTES
    ptr_a_bytes = align_up(max_recv_tokens * k * 2, BUFFER_ALIGNMENT)
    return align_up(
        BASE_LOCAL_MEM_BYTES + ptr_a_bytes,
        DEVICE_LARGE_PAGE_SIZE,
    )


def read_shapes(csv_path: Path, ep: int) -> list[dict]:
    use_udma_operator = os.environ.get("MEGA_MOE_BF16_UDMA", "0") == "1"
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
            if shape["K"] % 256 != 0 or shape["N"] % 512 != 0:
                raise ValueError(
                    f"{csv_path}:{line_number} requires K % 256 == 0 and "
                    "N % 512 == 0"
                )
            # The standalone generator only produces the default balanced
            # round-robin routing pattern.
            max_recv_tokens = align_up(shape["M"] * TOPK, 128)
            cases.append(
                {
                    **shape,
                    "TOPK": TOPK,
                    "expertPerRank": EXPERTS_PER_RANK,
                    "EP": ep,
                    "maxRecvTokenNum": max_recv_tokens,
                    "validateRecvCapacity": True,
                    "localMemBytes": local_mem_bytes(
                        use_udma_operator, max_recv_tokens, shape["K"]
                    ),
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
    parser.add_argument("--absolute-tolerance", type=float, default=0.01)
    parser.add_argument("--relative-tolerance", type=float, default=0.01)
    parser.add_argument("--max-error-ratio", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ep < 2:
        raise ValueError("MegaMoe BF16 requires at least two ranks")
    if args.warmup < 0 or args.loops <= 0:
        raise ValueError("warmup must be >= 0 and loops must be > 0")

    cases = read_shapes(args.csv_path, args.ep)
    for case in cases:
        case_dir = args.data_dir / shape_directory(case)
        case_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "generate":
            generate(
                case,
                case_dir,
                args.seed,
                early_topk_weight=True,
            )
            write_runner_csv(
                case_dir / "runner_case.csv",
                case,
                "independent_bf16",
                args.warmup,
                args.loops,
            )
        else:
            compare(
                case,
                case_dir,
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
                max_error_ratio=args.max_error_ratio,
            )


if __name__ == "__main__":
    main()
