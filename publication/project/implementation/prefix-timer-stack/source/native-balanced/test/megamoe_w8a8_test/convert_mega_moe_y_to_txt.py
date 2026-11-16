#!/usr/bin/env python3
"""Convert MegaMoe y.npy output data to readable text files."""

import argparse
import io
from pathlib import Path
import re
import zipfile

import numpy as np


RANK_Y_PATTERN = re.compile(r"^rank(\d+)/y\.npy$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a MegaMoe y.npy file or an output ZIP archive to text"
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="y.npy, ops_transformer_output.zip, or deepep_output.zip",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory; defaults to a directory beside the input",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=9,
        help="number of significant digits written for each value",
    )
    parser.add_argument(
        "--raw-bits",
        action="store_true",
        help="write uint16 BF16 storage values instead of decoded floats",
    )
    return parser.parse_args()


def bf16_bits_to_float32(data: np.ndarray) -> np.ndarray:
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def output_values(array: np.ndarray, raw_bits: bool) -> np.ndarray:
    if array.dtype == np.uint16:
        return array if raw_bits else bf16_bits_to_float32(array)
    if raw_bits:
        raise ValueError(
            f"--raw-bits requires uint16 BF16 storage, got {array.dtype}"
        )
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(
            f"expected uint16 BF16 storage or floating-point y, got {array.dtype}"
        )
    return array.astype(np.float32, copy=False)


def write_txt(
    array: np.ndarray,
    output_path: Path,
    precision: int,
    raw_bits: bool,
) -> None:
    values = output_values(array, raw_bits)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if raw_bits:
        np.savetxt(output_path, values, fmt="%d")
    else:
        np.savetxt(output_path, values, fmt=f"%.{precision}g")
    print(
        f"[CONVERT] shape={tuple(array.shape)} dtype={array.dtype} "
        f"output={output_path}"
    )


def convert_npy(
    input_path: Path,
    output_dir: Path,
    precision: int,
    raw_bits: bool,
) -> None:
    array = np.load(input_path, allow_pickle=False)
    write_txt(
        array,
        output_dir / f"{input_path.stem}.txt",
        precision,
        raw_bits,
    )


def convert_zip(
    input_path: Path,
    output_dir: Path,
    precision: int,
    raw_bits: bool,
) -> None:
    converted = 0
    with zipfile.ZipFile(input_path, "r") as archive:
        rank_entries = []
        for name in archive.namelist():
            match = RANK_Y_PATTERN.fullmatch(name)
            if match:
                rank_entries.append((int(match.group(1)), name))
        for rank, name in sorted(rank_entries):
            with archive.open(name, "r") as input_file:
                array = np.load(
                    io.BytesIO(input_file.read()),
                    allow_pickle=False,
                )
            write_txt(
                array,
                output_dir / f"rank{rank}_y.txt",
                precision,
                raw_bits,
            )
            converted += 1
    if converted == 0:
        raise RuntimeError(f"{input_path} does not contain rank*/y.npy")


def main():
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_txt"
    output_dir = output_dir.resolve()

    if input_path.suffix == ".npy":
        convert_npy(
            input_path,
            output_dir,
            args.precision,
            args.raw_bits,
        )
    elif input_path.suffix == ".zip":
        convert_zip(
            input_path,
            output_dir,
            args.precision,
            args.raw_bits,
        )
    else:
        raise ValueError("input must be a .npy file or .zip archive")


if __name__ == "__main__":
    main()
