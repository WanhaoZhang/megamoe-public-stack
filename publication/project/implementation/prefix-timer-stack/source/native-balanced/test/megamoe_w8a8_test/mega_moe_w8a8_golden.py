#!/usr/bin/env python3
"""Generate W8A8 MegaMoe inputs/golden data and compare runner outputs.

The numerical flow follows the original MegaMoe W8A8 CPU golden:
dispatch reorder + MXFP8 quantization, GMM1, SwiGLU, MXFP8 quantization,
GMM2, and BF16 weighted unpermute/combine.
"""

import argparse
import math
from pathlib import Path
import sys

try:
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "numpy is required by the MegaMoe precision test. "
        "Install it in the active Python environment."
    ) from error


def float32_to_bf16_bits(data: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(data, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(data: np.ndarray) -> np.ndarray:
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_round(data: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(data))


def make_fp8_e4m3fn_tables():
    decode = np.empty(256, dtype=np.float32)
    for raw in range(256):
        sign = -1.0 if raw & 0x80 else 1.0
        exponent = (raw >> 3) & 0xF
        mantissa = raw & 0x7
        if exponent == 0:
            value = mantissa * (2.0 ** -9)
        elif exponent == 0xF and mantissa == 0x7:
            value = math.nan
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        decode[raw] = sign * value
    positive_raw = np.array(
        [raw for raw in range(0x7F) if np.isfinite(decode[raw])],
        dtype=np.uint8,
    )
    return decode, positive_raw, decode[positive_raw]


FP8_DECODE, FP8_POSITIVE_RAW, FP8_POSITIVE_VALUES = make_fp8_e4m3fn_tables()


def encode_fp8_e4m3fn(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    absolute = np.minimum(np.abs(values), FP8_POSITIVE_VALUES[-1])
    upper_index = np.searchsorted(FP8_POSITIVE_VALUES, absolute, side="left")
    upper_index = np.clip(upper_index, 0, FP8_POSITIVE_VALUES.size - 1)
    lower_index = np.maximum(upper_index - 1, 0)
    lower_diff = absolute - FP8_POSITIVE_VALUES[lower_index]
    upper_diff = FP8_POSITIVE_VALUES[upper_index] - absolute
    choose_upper = upper_diff < lower_diff
    tie = upper_diff == lower_diff
    upper_even = (FP8_POSITIVE_RAW[upper_index] & 1) == 0
    choose_upper |= tie & upper_even
    selected = np.where(choose_upper, upper_index, lower_index)
    raw = FP8_POSITIVE_RAW[selected].copy()
    raw |= np.where(np.signbit(values), 0x80, 0).astype(np.uint8)
    raw = np.where(np.isnan(values), 0x7F, raw).astype(np.uint8)
    return raw


def decode_fp8_e4m3fn(raw: np.ndarray) -> np.ndarray:
    return FP8_DECODE[np.asarray(raw, dtype=np.uint8)]


def mxfp8_quantize(data: np.ndarray, axis: int):
    """Return E4M3 raw data, dequantized FP32 data, and unpacked E8M0 scales."""
    source = np.asarray(data, dtype=np.float32)
    axis = axis % source.ndim
    moved = np.moveaxis(source, axis, -1)
    original_length = moved.shape[-1]
    padded_length = ((original_length + 31) // 32) * 32
    if padded_length != original_length:
        moved = np.pad(
            moved,
            [(0, 0)] * (moved.ndim - 1) + [(0, padded_length - original_length)],
        )
    blocks = moved.reshape(*moved.shape[:-1], padded_length // 32, 32)
    max_abs = np.max(np.abs(blocks), axis=-1)
    with np.errstate(divide="ignore"):
        exponent = np.floor(np.log2(max_abs)).astype(np.float32) - 8.0
    exponent = np.where(max_abs == 0, -127.0, exponent)
    exponent = np.clip(exponent, -127.0, 127.0).astype(np.int16)
    scale = np.exp2(exponent.astype(np.float32))
    normalized = blocks / scale[..., None]
    raw_blocks = encode_fp8_e4m3fn(normalized)
    dequantized_blocks = decode_fp8_e4m3fn(raw_blocks) * scale[..., None]

    raw_moved = raw_blocks.reshape(*moved.shape)[..., :original_length]
    dequantized_moved = dequantized_blocks.reshape(*moved.shape)[
        ..., :original_length
    ]
    raw = np.moveaxis(raw_moved, -1, axis)
    dequantized = np.moveaxis(dequantized_moved, -1, axis)
    scale_raw = (exponent + 127).astype(np.uint8)
    scale_raw = np.moveaxis(scale_raw, -1, axis)
    scale_value = np.moveaxis(scale, -1, axis)
    return raw, dequantized, scale_raw, scale_value


def pack_transposed_weight_scales(scale_raw: np.ndarray) -> np.ndarray:
    """Pack adjacent logical-K scales for the transW=true NPU layout."""
    if scale_raw.shape[-1] % 2 != 0:
        raise ValueError("weight K scale count must be even")
    return scale_raw.reshape(
        *scale_raw.shape[:-1], scale_raw.shape[-1] // 2, 2
    )


def suffix(case: dict) -> str:
    return f"{case['M']}_{case['K']}_{case['N']}_{case['EP']}.bin"


def data_path(data_dir: Path, rank: int, name: str, case: dict) -> Path:
    return data_dir / f"rank{rank}_{name}_{suffix(case)}"


def write_array(path: Path, data: np.ndarray, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab" if append else "wb") as output:
        output.write(np.ascontiguousarray(data).tobytes())


def generate(case: dict, data_dir: Path, seed: int) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "torch is required to reproduce the ops-transformer-hs input RNG"
        ) from error

    m = case["M"]
    k = case["K"]
    n = case["N"]
    topk = case["TOPK"]
    local_experts = case["expertPerRank"]
    ep = case["EP"]
    total_experts = ep * local_experts
    global_m = ep * m
    if k % 64 != 0 or n % 128 != 0:
        raise ValueError("W8A8 golden requires K % 64 == 0 and N % 128 == 0")
    if topk > total_experts:
        raise ValueError("TOPK cannot exceed the total expert count")

    data_dir.mkdir(parents=True, exist_ok=True)

    # Keep the RNG order identical to GenInputAndGolden in
    # ops-transformer-hs: Torch generates x/topk weights while NumPy
    # generates fast random FP8 weights.
    np.random.seed(seed)
    torch.manual_seed(seed)
    x = (
        torch.empty((global_m, k), dtype=torch.bfloat16)
        .uniform_(-0.1, 0.1)
        .to(torch.float32)
        .numpy()
    )
    balanced_rounds = [
        (
            torch.arange(
                round_index,
                round_index + total_experts,
                dtype=torch.int32,
            )
            % total_experts
        )
        for round_index in range(
            (global_m * topk) // total_experts + 1
        )
    ]
    expert_ids = (
        torch.cat(balanced_rounds)[: global_m * topk]
        .view(global_m, topk)
        .numpy()
    )
    topk_weights_fp32 = (
        torch.empty((global_m, topk), dtype=torch.float32)
        .uniform_(-0.1, 0.1)
        .numpy()
    )
    topk_weights = bf16_round(topk_weights_fp32)
    active_mask = np.ones((global_m,), dtype=np.uint8)

    weight_file_names = (
        "weight1_w8_npu_w8a8",
        "weight2_w8_npu_w8a8",
        "weight1_w8_npu_scale_w8a8",
        "weight2_w8_npu_scale_w8a8",
    )
    for rank in range(ep):
        rank_slice = slice(rank * m, (rank + 1) * m)
        write_array(
            data_path(data_dir, rank, "input_x_w8a8", case),
            float32_to_bf16_bits(x[rank_slice]),
        )
        write_array(
            data_path(data_dir, rank, "input_expert_ids_w8a8", case),
            expert_ids[rank_slice].astype(np.int32),
        )
        write_array(
            data_path(data_dir, rank, "input_expert_scales_w8a8", case),
            float32_to_bf16_bits(topk_weights[rank_slice]),
        )
        write_array(
            data_path(data_dir, rank, "input_x_active_mask_w8a8", case),
            active_mask[rank_slice],
        )
        write_array(
            data_path(data_dir, rank, "scales", case),
            np.zeros(case["maxRecvTokenNum"], dtype=np.float32),
        )
        for name in weight_file_names:
            write_array(data_path(data_dir, rank, name, case), np.empty(0, np.uint8))

    route_outputs = np.zeros((global_m, topk, k), dtype=np.uint16)
    expert_counts = np.zeros(total_experts, dtype=np.int32)

    safe_fp8_values = np.array(
        [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        dtype=np.float32,
    )
    safe_fp8_codes = encode_fp8_e4m3fn(safe_fp8_values)
    scale1 = np.full((n, (k + 63) // 64, 2), 127, dtype=np.uint8)
    scale2 = np.full(
        (k, (n // 2 + 63) // 64, 2),
        127,
        dtype=np.uint8,
    )

    # ops-transformer-hs generates weights in chunks of 16 experts and
    # generates all GMM1 weights before all GMM2 weights in each chunk.
    weight_chunk_size = 16
    for chunk_start in range(0, total_experts, weight_chunk_size):
        chunk_end = min(chunk_start + weight_chunk_size, total_experts)
        chunk_experts = chunk_end - chunk_start
        weight1_indices = np.random.randint(
            0,
            len(safe_fp8_codes),
            size=(chunk_experts, n, k),
            dtype=np.uint8,
        )
        weight1_chunk = np.take(safe_fp8_codes, weight1_indices)
        del weight1_indices
        weight2_indices = np.random.randint(
            0,
            len(safe_fp8_codes),
            size=(chunk_experts, k, n // 2),
            dtype=np.uint8,
        )
        weight2_chunk = np.take(safe_fp8_codes, weight2_indices)
        del weight2_indices

        for chunk_offset, expert in enumerate(
            range(chunk_start, chunk_end)
        ):
            rank = expert // local_experts
            token_and_route = np.argwhere(expert_ids == expert)
            expert_counts[expert] = token_and_route.shape[0]
            weight1_raw = weight1_chunk[chunk_offset]
            weight2_raw = weight2_chunk[chunk_offset]
            weight1_dequant = decode_fp8_e4m3fn(weight1_raw)
            weight2_dequant = decode_fp8_e4m3fn(weight2_raw)
            write_array(
                data_path(
                    data_dir,
                    rank,
                    "weight1_w8_npu_w8a8",
                    case,
                ),
                weight1_raw,
                append=True,
            )
            write_array(
                data_path(
                    data_dir,
                    rank,
                    "weight1_w8_npu_scale_w8a8",
                    case,
                ),
                scale1,
                append=True,
            )
            write_array(
                data_path(
                    data_dir,
                    rank,
                    "weight2_w8_npu_w8a8",
                    case,
                ),
                weight2_raw,
                append=True,
            )
            write_array(
                data_path(
                    data_dir,
                    rank,
                    "weight2_w8_npu_scale_w8a8",
                    case,
                ),
                scale2,
                append=True,
            )

            if token_and_route.size != 0:
                token_indices = token_and_route[:, 0]
                _, dispatch_x, _, _ = mxfp8_quantize(
                    x[token_indices],
                    axis=1,
                )
                gmm1 = dispatch_x @ weight1_dequant.T
                gate_input, gate = np.split(gmm1, 2, axis=1)
                swiglu = gate_input / (1.0 + np.exp(-gate_input)) * gate
                swiglu = bf16_round(swiglu)
                _, swiglu_dequant, _, _ = mxfp8_quantize(
                    swiglu,
                    axis=1,
                )
                gmm2 = bf16_round(swiglu_dequant @ weight2_dequant.T)
                route_outputs[
                    token_and_route[:, 0],
                    token_and_route[:, 1],
                    :,
                ] = float32_to_bf16_bits(gmm2)

        del weight1_chunk, weight2_chunk

    combined = np.zeros((global_m, k), dtype=np.float32)
    for route in range(topk):
        route_value = bf16_bits_to_float32(route_outputs[:, route, :])
        contribution = bf16_round(
            route_value * topk_weights_fp32[:, route, None]
        )
        combined = contribution if route == 0 else bf16_round(combined + contribution)

    for rank in range(ep):
        rank_slice = slice(rank * m, (rank + 1) * m)
        expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
        write_array(
            data_path(data_dir, rank, "golden_y_w8a8", case),
            float32_to_bf16_bits(combined[rank_slice]),
        )
        write_array(
            data_path(
                data_dir, rank, "golden_expert_token_nums_w8a8", case
            ),
            expert_counts[expert_slice],
        )

    print(
        f"[GOLDEN] generated M={case['M']} N={case['N']} K={case['K']} EP={ep} "
        f"experts={total_experts} data_dir={data_dir}"
    )


def deviation_ratio(
    actual: np.ndarray,
    golden: np.ndarray,
    absolute_error_threshold: float,
) -> np.ndarray:
    """Return the W4A8-style protected per-element deviation."""
    absolute_diff = np.abs(actual - golden)
    denominator_floor = (
        (1.0 / (1 << 14)) / absolute_error_threshold
    )
    denominator = np.maximum(
        np.maximum(np.abs(actual), np.abs(golden)),
        denominator_floor,
    )
    relative_deviation = absolute_diff / (denominator + 1e-10)
    deviation = np.where(
        absolute_diff < absolute_error_threshold,
        absolute_diff,
        relative_deviation,
    )
    both_equal = actual == golden
    return np.where(
        np.isfinite(deviation),
        deviation,
        np.where(both_equal, 0.0, np.inf),
    )


def _load_y(path: Path) -> np.ndarray:
    return bf16_bits_to_float32(np.fromfile(path, np.uint16))


def _diagnostic_sections(size: int) -> list[tuple[str, np.ndarray]]:
    if size <= 0:
        return []
    middle_start = max(0, size // 2 - 5)
    windows = (
        ("first", 0),
        ("middle", middle_start),
        ("last", max(0, size - 10)),
    )
    sections = []
    used_ranges = set()
    for label, start in windows:
        stop = min(size, start + 10)
        key = (start, stop)
        if key not in used_ranges:
            sections.append((label, np.arange(start, stop)))
            used_ranges.add(key)
    return sections


def _print_triplet_diagnostics(
    rank: int,
    golden: np.ndarray,
    cascade: np.ndarray,
    fused: np.ndarray,
    cascade_deviation: np.ndarray,
    fused_deviation: np.ndarray,
) -> None:
    print(
        f"[PRECISION][TRIPLET] rank={rank} diagnostic values "
        "(flattened indices)"
    )
    if golden.size:
        deviation_gap = fused_deviation - cascade_deviation
        comparable_gap = np.nan_to_num(
            deviation_gap,
            nan=-np.inf,
            posinf=np.inf,
            neginf=-np.inf,
        )
        max_gap_index = int(np.argmax(comparable_gap))
        print(
            "  [MAX DEVIATION GAP] "
            f"index={max_gap_index} "
            f"golden={golden[max_gap_index]:.8g} "
            f"cascade={cascade[max_gap_index]:.8g} "
            f"fused={fused[max_gap_index]:.8g} "
            "cascade_deviation="
            f"{cascade_deviation[max_gap_index]:.6%} "
            f"fused_deviation={fused_deviation[max_gap_index]:.6%} "
            f"deviation_gap={deviation_gap[max_gap_index]:.6%}"
        )
    for label, indices in _diagnostic_sections(golden.size):
        print(f"  [{label.upper()} 10]")
        for index in indices:
            deviation_gap = fused_deviation[index] - cascade_deviation[index]
            print(
                f"  index={index} golden={golden[index]:.8g} "
                f"cascade={cascade[index]:.8g} fused={fused[index]:.8g} "
                f"cascade_deviation={cascade_deviation[index]:.6%} "
                f"fused_deviation={fused_deviation[index]:.6%} "
                f"deviation_gap={deviation_gap:.6%}"
            )


def _compare_y_triplet(
    rank: int,
    fused_path: Path,
    cascade_path: Path,
    golden_path: Path,
    deviation_gap_threshold: float,
    absolute_error_threshold: float,
    max_error_ratio: float,
    failures: list[str],
) -> None:
    named_paths = (
        ("fused", fused_path),
        ("cascade", cascade_path),
        ("golden", golden_path),
    )
    missing_paths = [
        f"{name}={path}" for name, path in named_paths if not path.is_file()
    ]
    if missing_paths:
        failures.extend(
            f"TRIPLET rank {rank}: missing {item}" for item in missing_paths
        )
        return

    fused_y = _load_y(fused_path)
    cascade_y = _load_y(cascade_path)
    golden_y = _load_y(golden_path)
    if not (fused_y.size == cascade_y.size == golden_y.size):
        failures.append(
            f"TRIPLET rank {rank}: y sizes fused={fused_y.size}, "
            f"cascade={cascade_y.size}, golden={golden_y.size}"
        )
        return

    cascade_deviation = deviation_ratio(
        cascade_y,
        golden_y,
        absolute_error_threshold,
    )
    fused_deviation = deviation_ratio(
        fused_y,
        golden_y,
        absolute_error_threshold,
    )
    incorrect = (
        fused_deviation > cascade_deviation + deviation_gap_threshold
    )
    incorrect_ratio = (
        float(np.count_nonzero(incorrect)) / max(1, incorrect.size)
    )
    result = "PASS" if incorrect_ratio <= max_error_ratio else "FAIL"
    cascade_max = (
        float(np.max(cascade_deviation)) if cascade_deviation.size else 0.0
    )
    fused_max = (
        float(np.max(fused_deviation)) if fused_deviation.size else 0.0
    )
    cascade_mean = (
        float(np.mean(cascade_deviation)) if cascade_deviation.size else 0.0
    )
    fused_mean = (
        float(np.mean(fused_deviation)) if fused_deviation.size else 0.0
    )
    legacy_incorrect = fused_deviation >= absolute_error_threshold
    legacy_error_ratio = (
        float(np.count_nonzero(legacy_incorrect))
        / max(1, legacy_incorrect.size)
    )
    legacy_result = (
        "PASS" if legacy_error_ratio <= max_error_ratio else "FAIL"
    )
    legacy_max_deviation = (
        float(np.max(fused_deviation)) if fused_deviation.size else 0.0
    )
    print(
        f"[PRECISION][TRIPLET] rank={rank} y={result} "
        f"incorrect_ratio={incorrect_ratio:.6%} "
        f"absolute_error_threshold={absolute_error_threshold:.6g} "
        f"allowed_deviation_gap={deviation_gap_threshold:.6%} "
        f"cascade_mean_deviation={cascade_mean:.6%} "
        f"fused_mean_deviation={fused_mean:.6%} "
        f"cascade_max_deviation={cascade_max:.6%} "
        f"fused_max_deviation={fused_max:.6%}"
    )
    print(
        f"[PRECISION][LEGACY-FUSED-GOLDEN] rank={rank} "
        f"y={legacy_result} error_ratio={legacy_error_ratio:.6%} "
        f"max_diff={legacy_max_deviation:.6g} "
        f"diff_threshold={absolute_error_threshold:.6g} "
        "(informational only)"
    )
    if result == "FAIL":
        _print_triplet_diagnostics(
            rank,
            golden_y,
            cascade_y,
            fused_y,
            cascade_deviation,
            fused_deviation,
        )
        failures.append(
            f"TRIPLET rank {rank}: {incorrect_ratio:.6%} of fused outputs "
            f"deviate from golden by more than cascade plus "
            f"{deviation_gap_threshold:.6%}; limit={max_error_ratio:.6%}"
        )


def _compare_y_pair(
    rank: int,
    label: str,
    actual_path: Path,
    expected_path: Path,
    diff_threshold: float,
    absolute_error_threshold: float,
    max_error_ratio: float,
    failures: list[str],
) -> None:
    missing_paths = [
        path for path in (actual_path, expected_path) if not path.is_file()
    ]
    if missing_paths:
        failures.extend(
            f"{label} rank {rank}: missing {path}" for path in missing_paths
        )
        return

    actual_y = bf16_bits_to_float32(
        np.fromfile(actual_path, np.uint16)
    )
    expected_y = bf16_bits_to_float32(
        np.fromfile(expected_path, np.uint16)
    )
    if actual_y.size != expected_y.size:
        failures.append(
            f"{label} rank {rank}: y size {actual_y.size} != "
            f"{expected_y.size}"
        )
        return

    diff = deviation_ratio(
        actual_y,
        expected_y,
        absolute_error_threshold,
    )
    bad = diff >= diff_threshold
    error_ratio = float(np.count_nonzero(bad)) / max(1, bad.size)
    max_diff = float(np.max(diff)) if diff.size else 0.0
    result = "PASS" if error_ratio <= max_error_ratio else "FAIL"
    print(
        f"[PRECISION][{label}] rank={rank} y={result} "
        f"error_ratio={error_ratio:.6%} max_diff={max_diff:.6g} "
        f"diff_threshold={diff_threshold}"
    )
    if diff.size:
        max_diff_index = int(np.argmax(diff))
        max_absolute_diff = float(
            np.abs(actual_y[max_diff_index] - expected_y[max_diff_index])
        )
        print(
            f"[PRECISION][{label}][MAX-DEVIATION] rank={rank} "
            f"index={max_diff_index} "
            f"golden={expected_y[max_diff_index]:.8g} "
            f"actual={actual_y[max_diff_index]:.8g} "
            f"absolute_diff={max_absolute_diff:.8g} "
            f"deviation={diff[max_diff_index]:.8g}"
        )
    if result == "FAIL":
        bad_indices = np.flatnonzero(bad)[:10]
        for index in bad_indices:
            print(
                f"  index={index} expected={expected_y[index]:.8g} "
                f"actual={actual_y[index]:.8g} diff={diff[index]:.8g}"
            )
        failures.append(
            f"{label} rank {rank}: y error ratio "
            f"{error_ratio:.6%} exceeds {max_error_ratio:.6%}"
        )


def _compare_token_pair(
    rank: int,
    label: str,
    actual_path: Path,
    expected_path: Path,
    failures: list[str],
) -> None:
    missing_paths = [
        path for path in (actual_path, expected_path) if not path.is_file()
    ]
    if missing_paths:
        failures.extend(
            f"{label} rank {rank}: missing {path}" for path in missing_paths
        )
        return

    actual_tokens = np.fromfile(actual_path, np.int32)
    expected_tokens = np.fromfile(expected_path, np.int32)
    tokens_equal = np.array_equal(actual_tokens, expected_tokens)
    print(
        f"[PRECISION][{label}] rank={rank} expert_token_nums="
        f"{'PASS' if tokens_equal else 'FAIL'}"
    )
    if not tokens_equal:
        failures.append(
            f"{label} rank {rank}: expert_token_nums "
            f"actual={actual_tokens.tolist()} "
            f"expected={expected_tokens.tolist()}"
        )


def compare(
    case: dict,
    data_dir: Path,
    deviation_gap_threshold: float,
    max_error_ratio: float,
    compare_cascade: bool = False,
    absolute_error_threshold: float = 0.01,
) -> None:
    if deviation_gap_threshold <= 0:
        raise ValueError("deviation_gap_threshold must be positive")
    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError("max_error_ratio must be in [0, 1]")
    if absolute_error_threshold <= 0:
        raise ValueError("absolute_error_threshold must be positive")
    failures = []

    token_comparisons = [
        (
            "FUSED-GOLDEN",
            "output_expert_token_nums_w8a8",
            "golden_expert_token_nums_w8a8",
        ),
    ]
    if compare_cascade:
        token_comparisons.extend(
            [
                (
                    "CASCADE-GOLDEN",
                    "output_cascade_expert_token_nums_w8a8",
                    "golden_expert_token_nums_w8a8",
                ),
                (
                    "FUSED-CASCADE",
                    "output_expert_token_nums_w8a8",
                    "output_cascade_expert_token_nums_w8a8",
                ),
            ]
        )

    for rank in range(case["EP"]):
        if compare_cascade:
            _compare_y_triplet(
                rank,
                data_path(data_dir, rank, "output_y_w8a8", case),
                data_path(
                    data_dir, rank, "output_cascade_y_w8a8", case
                ),
                data_path(data_dir, rank, "golden_y_w8a8", case),
                deviation_gap_threshold,
                absolute_error_threshold,
                max_error_ratio,
                failures,
            )
        else:
            _compare_y_pair(
                rank,
                "FUSED-GOLDEN",
                data_path(data_dir, rank, "output_y_w8a8", case),
                data_path(data_dir, rank, "golden_y_w8a8", case),
                absolute_error_threshold,
                absolute_error_threshold,
                max_error_ratio,
                failures,
            )
        for label, actual_name, expected_name in token_comparisons:
            _compare_token_pair(
                rank,
                label,
                data_path(data_dir, rank, actual_name, case),
                data_path(data_dir, rank, expected_name, case),
                failures,
            )

    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        raise SystemExit(1)
    suffix = " with cascade comparison" if compare_cascade else ""
    print(
        f"[PRECISION] MegaMoe W8A8 "
        f"M={case['M']} N={case['N']} K={case['K']} "
        f"passed{suffix}"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("generate", "compare"))
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--ep", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--absolute-error-threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--deviation-gap-threshold",
        "--diff-threshold",
        dest="deviation_gap_threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument("--max-error-ratio", type=float, default=0.05)
    parser.add_argument("--skip-cascade", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from test_mega_moe import read_cases, shape_directory_name

    for case in read_cases(args.csv_path, args.ep):
        data_dir = args.data_dir / shape_directory_name(case)
        data_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "generate":
            generate(case, data_dir, args.seed)
        else:
            compare(
                case,
                data_dir,
                args.deviation_gap_threshold,
                args.max_error_ratio,
                compare_cascade=not args.skip_cascade,
                absolute_error_threshold=args.absolute_error_threshold,
            )


if __name__ == "__main__":
    main()
