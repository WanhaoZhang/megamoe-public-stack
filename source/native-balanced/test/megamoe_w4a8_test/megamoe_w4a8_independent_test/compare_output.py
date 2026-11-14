#!/usr/bin/env python3
"""Compare standalone W4A8 BF16 output against the migrated CPU golden."""

import argparse
from pathlib import Path

import numpy as np


def bf16_to_fp32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--absolute-error-threshold", type=float, default=1e-3)
    parser.add_argument("--deviation-threshold", type=float, default=0.1)
    parser.add_argument("--max-error-ratio", type=float, default=0.05)
    args = parser.parse_args()

    suffix = f"w4a8_{args.m}_{args.k}_{args.n}_{args.ep}.bin"
    failures = []
    for rank in range(args.ep):
        actual_path = args.data_dir / f"rank{rank}_output_y_{suffix}"
        golden_path = args.data_dir / f"rank{rank}_combine_output_bf16_{suffix}"
        actual = bf16_to_fp32(np.fromfile(actual_path, dtype=np.uint16))
        golden = bf16_to_fp32(np.fromfile(golden_path, dtype=np.uint16))
        if actual.size != args.m * args.k or actual.shape != golden.shape:
            raise RuntimeError(
                f"rank {rank} output size mismatch: actual={actual.size}, "
                f"golden={golden.size}, expected={args.m * args.k}"
            )
        absolute = np.abs(actual - golden)
        deviation = absolute / (np.abs(golden) + 1e-12)
        incorrect = (absolute > args.absolute_error_threshold) & (
            deviation > args.deviation_threshold
        )
        error_ratio = float(np.mean(incorrect)) if incorrect.size else 0.0
        max_absolute = float(np.max(absolute)) if absolute.size else 0.0
        status = "PASS" if error_ratio <= args.max_error_ratio else "FAIL"
        print(
            f"[{status}] rank={rank} error_ratio={error_ratio:.6%} "
            f"max_abs_diff={max_absolute:.8g}"
        )
        if status == "FAIL":
            failures.append(rank)
    if failures:
        raise RuntimeError(f"W4A8 output comparison failed on ranks {failures}")


if __name__ == "__main__":
    main()
