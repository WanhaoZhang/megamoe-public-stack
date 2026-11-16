#!/usr/bin/env python3
"""MegaMoe W8A8 Python performance and precision test.

This is the ascend_deepep adaptation of the timer MegaMoe test. It uses the
same torch.classes loading path as the other project operators. The CSV only
provides M/N/K shapes; all other test parameters are defined in this file.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time
import zipfile


TEST_DIR = Path(__file__).resolve().parent
DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
ACLSHMEM_DATA_OP_UDMA = 0x08
TOPK = 6
EXPERTS_PER_RANK = 16
LOCAL_MEM_BYTES = 1024 * 1024 * 1024
DEFAULT_SEED = 831
# The current ops-transformer-hs eager path passes warmup=0 to its profiler;
# its configured warmup=10 variable is not used by that execution branch.
DEFAULT_WARMUP = 0
DEFAULT_LOOPS = 110
PERFORMANCE_TAIL_LOOPS = 50
DEFAULT_ABSOLUTE_ERROR_THRESHOLD = 0.01
DEFAULT_DEVIATION_GAP_THRESHOLD = 0.01
DEFAULT_MAX_ERROR_RATIO = 0.05


def read_cases(csv_path: Path, ep: int) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_fields = ("M", "N", "K")
        if reader.fieldnames != list(expected_fields):
            raise ValueError(
                f"{csv_path} must contain only M,N,K columns; "
                f"found {reader.fieldnames}"
            )
        cases = []
        for line_number, row in enumerate(reader, start=2):
            try:
                shape = {name: int(row[name]) for name in expected_fields}
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
                        shape["M"]
                        * ep
                        * min(EXPERTS_PER_RANK, TOPK)
                    ),
                    "localMemBytes": LOCAL_MEM_BYTES,
                }
            )
    if not cases:
        raise ValueError(f"{csv_path} does not contain any M,N,K shapes")
    return cases


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "devices must be comma-separated integers"
        ) from error
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError(
            "devices must contain non-negative device IDs"
        )
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("device IDs must be unique")
    return devices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MegaMoe through ascend_deepep.MegaMoe"
    )
    parser.add_argument("devices", type=parse_devices, help="for example: 2,3")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=TEST_DIR / "mega_moe_cases.csv",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "precision", "performance"),
        default="both",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--interop-input",
        type=Path,
        help=(
            "read the shared W8A8 input archive exported by "
            "ops-transformer-hs"
        ),
    )
    parser.add_argument(
        "--interop-output",
        type=Path,
        help=(
            "deepep output archive path; defaults to "
            "<interop-input-dir>/deepep_output.zip"
        ),
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help=(
            "ops-transformer-hs output archive; defaults to "
            "<interop-input-dir>/ops_transformer_output.zip"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--loops", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--absolute-error-threshold",
        type=float,
        default=DEFAULT_ABSOLUTE_ERROR_THRESHOLD,
        help=(
            "use absolute error instead of relative deviation below this "
            "threshold (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--deviation-gap-threshold",
        "--diff-threshold",
        dest="deviation_gap_threshold",
        type=float,
        default=DEFAULT_DEVIATION_GAP_THRESHOLD,
        help=(
            "mark an element incorrect when fused deviation exceeds cascade "
            "deviation by more than this ratio (default: 0.01)"
        ),
    )
    parser.add_argument(
        "--max-error-ratio",
        type=float,
        default=DEFAULT_MAX_ERROR_RATIO,
    )
    parser.add_argument(
        "--skip-cascade",
        action="store_true",
        help="skip the fused-vs-small-operator cascade precision comparison",
    )
    parser.add_argument(
        "--cascade-gmm1-mode",
        choices=("fused", "split"),
        default="fused",
        help=(
            "fused uses grouped_matmul_swiglu_quant_v2; split uses "
            "grouped_matmul followed by cann_ops_nn.swiglu_group_quant"
        ),
    )
    parser.add_argument(
        "--hccl-init-method",
        default="tcp://127.0.0.1:50001",
        help="torch.distributed init method used by cascade small operators",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get(
            "ASCEND_DEEPEP_SHMEM_IP_PORT",
            "tcp://127.0.0.1:8899",
        ),
    )
    parser.add_argument(
        "--enable-timer",
        action="store_true",
        help=(
            "call compute_with_timer and automatically generate per-rank "
            "Timer CSV/JSON; the kernel must be built with --enable-timer"
        ),
    )
    return parser.parse_args()


def file_suffix(case: dict) -> str:
    return (
        f"{case['M']}_{case['K']}_{case['N']}_{case['EP']}.bin"
    )


def shape_label(case: dict) -> str:
    return f"M={case['M']} N={case['N']} K={case['K']}"


def shape_directory_name(case: dict) -> str:
    return f"M{case['M']}_N{case['N']}_K{case['K']}"


def timer_trace_name_suffix(case: dict) -> str:
    return (
        f"{case['M']}_{case['N']}_{case['K']}_"
        f"{case['TOPK']}_{case['EP']}_w8a8"
    )


def rank_file(data_dir: Path, rank: int, name: str, case: dict) -> Path:
    return data_dir / f"rank{rank}_{name}_{file_suffix(case)}"


AB_FORMAT_NAME = "mega_moe_w8a8_ab_v1"


def read_interop_metadata(input_path: Path) -> dict:
    with zipfile.ZipFile(input_path, "r") as archive:
        metadata = json.loads(archive.read("metadata.json"))
    if metadata.get("format") != AB_FORMAT_NAME:
        raise ValueError(
            f"{input_path} has unsupported format {metadata.get('format')!r}; "
            f"expected {AB_FORMAT_NAME!r}"
        )
    return metadata


def interop_case(metadata: dict) -> dict:
    required = (
        "M",
        "N",
        "K",
        "TOPK",
        "EP",
        "expertPerRank",
        "maxRecvTokenNum",
    )
    missing = [name for name in required if name not in metadata]
    if missing:
        raise ValueError(f"interop metadata is missing fields: {missing}")
    return {
        name: int(metadata[name])
        for name in required
    } | {"localMemBytes": LOCAL_MEM_BYTES}


def _load_npy_entry(archive, name):
    import numpy as np

    with archive.open(name, "r") as input_file:
        return np.load(input_file, allow_pickle=False).copy()


def load_interop_inputs(rank: int, case: dict, input_path: Path, device):
    import numpy as np
    import torch

    expected = {
        "x": (np.uint16, (case["M"], case["K"]), torch.bfloat16),
        "topk_ids": (np.int32, (case["M"], case["TOPK"]), None),
        "topk_weights": (
            np.uint16,
            (case["M"], case["TOPK"]),
            torch.bfloat16,
        ),
        "weight1": (
            np.uint8,
            (case["expertPerRank"], case["N"], case["K"]),
            None,
        ),
        "weight2": (
            np.uint8,
            (
                case["expertPerRank"],
                case["K"],
                case["N"] // 2,
            ),
            None,
        ),
        "weight_scales1": (
            np.uint8,
            (
                case["expertPerRank"],
                case["N"],
                (case["K"] + 63) // 64,
                2,
            ),
            None,
        ),
        "weight_scales2": (
            np.uint8,
            (
                case["expertPerRank"],
                case["K"],
                (case["N"] // 2 + 63) // 64,
                2,
            ),
            None,
        ),
    }
    tensors = []
    with zipfile.ZipFile(input_path, "r") as archive:
        for name, (dtype, shape, view_dtype) in expected.items():
            array = _load_npy_entry(archive, f"rank{rank}/{name}.npy")
            if array.dtype != dtype or tuple(array.shape) != tuple(shape):
                raise RuntimeError(
                    f"rank{rank}/{name}.npy has dtype={array.dtype}, "
                    f"shape={array.shape}; expected dtype={dtype}, shape={shape}"
                )
            tensor = torch.from_numpy(array)
            if view_dtype is not None:
                tensor = tensor.view(view_dtype)
            tensors.append(
                tensor.to(device=device, non_blocking=False).contiguous()
            )
    return tuple(tensors)


def load_numpy(path: Path, dtype, shape):
    import numpy as np

    array = np.fromfile(path, dtype=dtype)
    expected = 1
    for dim in shape:
        expected *= dim
    if array.size != expected:
        raise RuntimeError(
            f"{path} contains {array.size} elements, expected {expected}"
        )
    return array.reshape(shape).copy()


def load_precision_inputs(rank: int, case: dict, data_dir: Path, device):
    import numpy as np
    import torch

    m = case["M"]
    k = case["K"]
    n = case["N"]
    topk = case["TOPK"]
    experts = case["expertPerRank"]

    def npu_tensor(name, dtype, shape, torch_dtype=None):
        array = load_numpy(
            rank_file(data_dir, rank, name, case),
            dtype,
            shape,
        )
        tensor = torch.from_numpy(array)
        if torch_dtype is not None:
            tensor = tensor.view(torch_dtype)
        return tensor.to(device=device, non_blocking=False).contiguous()

    return (
        npu_tensor(
            "input_x_w8a8",
            np.uint16,
            (m, k),
            torch.bfloat16,
        ),
        npu_tensor(
            "input_expert_ids_w8a8",
            np.int32,
            (m, topk),
        ),
        npu_tensor(
            "input_expert_scales_w8a8",
            np.uint16,
            (m, topk),
            torch.bfloat16,
        ),
        npu_tensor(
            "weight1_w8_npu_w8a8",
            np.uint8,
            (experts, n, k),
        ),
        npu_tensor(
            "weight2_w8_npu_w8a8",
            np.uint8,
            (experts, k, n // 2),
        ),
        npu_tensor(
            "weight1_w8_npu_scale_w8a8",
            np.uint8,
            (experts, n, (k + 63) // 64, 2),
        ),
        npu_tensor(
            "weight2_w8_npu_scale_w8a8",
            np.uint8,
            (experts, k, (n // 2 + 63) // 64, 2),
        ),
    )


def create_npu_profiler(torch_npu, loops: int, output_dir: Path):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=loops,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(output_dir)
        ),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )


def run_cascade_rank(rank: int, case: dict, config: dict) -> None:
    """Run the HCCL small-operator cascade without initializing ACLSHMEM."""
    import numpy as np
    import torch
    import torch_npu

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(case["EP"])

    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    data_dir = Path(config["data_dir"])
    inputs = load_precision_inputs(rank, case, data_dir, device)

    from mega_moe_w8a8_cascade import (
        destroy_hccl_group,
        init_hccl_group,
        run_cascade,
    )

    try:
        _, cascade_comm_name = init_hccl_group(
            rank,
            case["EP"],
            config["hccl_init_method"],
            device,
        )
        print(
            f"[CASCADE] rank={rank} starting small-operator path",
            flush=True,
        )
        cascade_result = run_cascade(
            inputs,
            case,
            rank,
            cascade_comm_name,
            config["cascade_gmm1_mode"],
        )
        torch_npu.npu.synchronize()

        cascade_y_bits = (
            cascade_result.y.detach()
            .cpu()
            .contiguous()
            .view(torch.uint16)
            .numpy()
        )
        cascade_token_values = (
            cascade_result.expert_token_nums.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int32, copy=False)
        )
        cascade_y_bits.tofile(
            rank_file(data_dir, rank, "output_cascade_y_w8a8", case)
        )
        cascade_token_values.tofile(
            rank_file(
                data_dir,
                rank,
                "output_cascade_expert_token_nums_w8a8",
                case,
            )
        )
        print(
            f"[CASCADE] rank={rank} small-operator path finished",
            flush=True,
        )
    finally:
        destroy_hccl_group()


def run_rank(rank: int, case: dict, config: dict) -> None:
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    from ascend_deepep import MegaMoe, ShmemInstance

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(case["EP"])
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = config["shmem_ip_port"]

    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    data_dir = Path(config["data_dir"])
    result_dir = data_dir / "python_results"
    result_dir.mkdir(parents=True, exist_ok=True)

    if config.get("interop_input"):
        inputs = load_interop_inputs(
            rank,
            case,
            Path(config["interop_input"]),
            device,
        )
    else:
        inputs = load_precision_inputs(rank, case, data_dir, device)

    shmem = None
    op = None
    try:
        shmem = ShmemInstance(
            rank,
            case["EP"],
            config["shmem_ip_port"],
        )
        shmem.set_data_op_engine_type(ACLSHMEM_DATA_OP_UDMA)
        shmem.init(case["localMemBytes"])
        op = MegaMoe(case["localMemBytes"])

        for _ in range(config["warmup"]):
            op.compute(shmem, *inputs, case["maxRecvTokenNum"])
        torch_npu.npu.synchronize()

        y = None
        expert_token_nums = None
        timer = None
        for repeat_id in range(config["repeat"]):
            profile_context = None
            if config["enable_profiler"]:
                profile_context = create_npu_profiler(
                    torch_npu,
                    config["loops"],
                    result_dir
                    / "profiling"
                    / config["profile_session"]
                    / f"rank_{rank}"
                    / f"repeat_{repeat_id}",
                )
                profile_context.__enter__()
            try:
                for _ in range(config["loops"]):
                    if config["enable_timer"]:
                        y, expert_token_nums, timer = op.compute_with_timer(
                            shmem,
                            *inputs,
                            case["maxRecvTokenNum"],
                        )
                    else:
                        y, expert_token_nums = op.compute(
                            shmem,
                            *inputs,
                            case["maxRecvTokenNum"],
                        )
                    if profile_context is not None:
                        profile_context.step()
            finally:
                torch_npu.npu.synchronize()
                if profile_context is not None:
                    profile_context.__exit__(None, None, None)

            if profile_context is not None:
                print(
                    f"[PERF] MegaMoe W8A8 profiler collection completed "
                    f"{shape_label(case)} "
                    f"rank={rank} repeat={repeat_id} "
                    f"loops={config['loops']}",
                    flush=True,
                )

        y_bits = (
            y.detach()
            .cpu()
            .contiguous()
            .view(torch.uint16)
            .numpy()
        )
        token_values = (
            expert_token_nums.detach()
            .cpu()
            .contiguous()
            .numpy()
            .astype(np.int32, copy=False)
        )
        y_bits.tofile(
            rank_file(data_dir, rank, "output_y_w8a8", case)
        )
        token_values.tofile(
            rank_file(
                data_dir,
                rank,
                "output_expert_token_nums_w8a8",
                case,
            )
        )
        if timer is not None:
            np.savetxt(
                result_dir / f"rank_{rank}_npu_time.csv",
                timer.detach().cpu().contiguous().numpy(),
                delimiter=",",
                fmt="%d",
            )

        result = {
            "rank": rank,
            "device": config["devices"][rank],
        }
        with (result_dir / f"rank_{rank}.json").open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(result, output, indent=2)
        shmem.barrier()
    finally:
        if op is not None:
            op.destroy()
        if shmem is not None:
            shmem.destroy()


def write_performance_summary(
    case: dict,
    config: dict,
    data_dir: Path,
) -> None:
    def profiler_durations(rank: int) -> list[float]:
        exact_durations = []
        fallback_durations = []
        profile_root = (
            result_dir
            / "profiling"
            / config["profile_session"]
            / f"rank_{rank}"
        )
        if not profile_root.is_dir():
            return exact_durations
        for summary_path in sorted(profile_root.rglob("op_summary*.csv")):
            with summary_path.open(
                newline="",
                encoding="utf-8-sig",
            ) as summary_file:
                for row in csv.DictReader(summary_file):
                    op_type = row.get("OP Type", "")
                    op_text = " ".join(
                        (
                            op_type,
                            row.get("OP Name", ""),
                            row.get("Task Type", ""),
                        )
                    )
                    normalized = op_text.lower().replace("_", "")
                    if "megamoe" not in normalized:
                        continue
                    try:
                        duration = float(row["Task Duration(us)"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    normalized_type = op_type.lower().replace("_", "")
                    if normalized_type == "megamoe":
                        exact_durations.append(duration)
                    else:
                        fallback_durations.append(duration)
        durations = exact_durations or fallback_durations
        return durations[-PERFORMANCE_TAIL_LOOPS:]

    result_dir = data_dir / "python_results"
    rows = []
    for rank in range(case["EP"]):
        with (result_dir / f"rank_{rank}.json").open(
            encoding="utf-8"
        ) as input_file:
            row = json.load(input_file)
        task_durations = profiler_durations(rank)
        if not task_durations:
            raise RuntimeError(
                f"rank {rank}: MegaMoe profiler Task Duration(us) rows were "
                "not found under "
                f"{result_dir / 'profiling' / config['profile_session']}"
            )
        row.update(
            {
                "metric_source": "npu_profiler_task_duration",
                "tail_count": len(task_durations),
                "mean_us": statistics.fmean(task_durations),
                "median_us": statistics.median(task_durations),
                "min_us": min(task_durations),
                "max_us": max(task_durations),
            }
        )
        rows.append(row)

    output_path = result_dir / "performance.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "M",
                "N",
                "K",
                "rank",
                "device",
                "warmup",
                "loops",
                "repeat",
                "metric_source",
                "tail_count",
                "mean_us",
                "median_us",
                "min_us",
                "max_us",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    case["M"],
                    case["N"],
                    case["K"],
                    row["rank"],
                    row["device"],
                    config["warmup"],
                    config["loops"],
                    config["repeat"],
                    row["metric_source"],
                    row["tail_count"],
                    row["mean_us"],
                    row["median_us"],
                    row["min_us"],
                    row["max_us"],
                ]
            )

    rank_medians = [row["median_us"] for row in rows]
    rank_mins = [row["min_us"] for row in rows]
    print(
        f"[PERF] {shape_label(case)} "
        f"tail={PERFORMANCE_TAIL_LOOPS} "
        f"min_of_rank_medians_us={min(rank_medians):.3f} "
        f"max_of_rank_medians_us={max(rank_medians):.3f} "
        f"median_of_rank_medians_us={statistics.median(rank_medians):.3f} "
        f"min_of_rank_mins_us={min(rank_mins):.3f}"
    )
    print(f"[INFO] Performance CSV: {output_path}")


def _write_npy_entry(archive, name, array):
    import numpy as np

    with archive.open(name, "w") as output:
        np.lib.format.write_array(
            output,
            np.ascontiguousarray(array),
            allow_pickle=False,
        )


def export_interop_output(case: dict, data_dir: Path, output_path: Path) -> None:
    import numpy as np

    metadata = {
        "format": AB_FORMAT_NAME,
        "source": "deepep-ascend",
        "EP": case["EP"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata, indent=2))
        for rank in range(case["EP"]):
            y = load_numpy(
                rank_file(data_dir, rank, "output_y_w8a8", case),
                np.uint16,
                (case["M"], case["K"]),
            )
            expert_token_nums = load_numpy(
                rank_file(
                    data_dir,
                    rank,
                    "output_expert_token_nums_w8a8",
                    case,
                ),
                np.int32,
                (case["expertPerRank"],),
            )
            _write_npy_entry(archive, f"rank{rank}/y.npy", y)
            _write_npy_entry(
                archive,
                f"rank{rank}/expert_token_nums.npy",
                expert_token_nums,
            )
    print(f"[AB] deepep-ascend output saved to: {output_path}")


def compare_interop_outputs(
    case: dict,
    deepep_path: Path,
    reference_path: Path,
    golden_path: Path,
    deviation_threshold: float,
    max_error_ratio: float,
    absolute_error_threshold: float,
) -> None:
    import numpy as np

    if not reference_path.is_file():
        raise FileNotFoundError(
            f"ops-transformer-hs output archive not found: {reference_path}"
        )
    if not golden_path.is_file():
        raise FileNotFoundError(
            f"MegaMoe golden archive not found: {golden_path}"
        )

    from mega_moe_w8a8_golden import bf16_bits_to_float32, deviation_ratio

    def decode_y(array):
        if array.dtype == np.uint16:
            return bf16_bits_to_float32(array)
        if not np.issubdtype(array.dtype, np.floating):
            raise RuntimeError(
                f"y must contain uint16 BF16 storage or floats, got {array.dtype}"
            )
        return array.astype(np.float32, copy=False)

    def compare_y(
        rank,
        label,
        expected_name,
        actual_array,
        expected_array,
    ):
        if actual_array.shape != expected_array.shape:
            raise RuntimeError(
                f"rank {rank} {label} y shape mismatch: "
                f"actual={actual_array.shape}, expected={expected_array.shape}"
            )
        actual = decode_y(actual_array)
        expected = decode_y(expected_array)
        deviation = deviation_ratio(
            actual,
            expected,
            absolute_error_threshold,
        )
        incorrect = deviation > deviation_threshold
        error_ratio = float(np.mean(incorrect)) if incorrect.size else 0.0
        absolute_diff = np.abs(actual - expected)
        max_abs_diff = (
            float(np.max(absolute_diff)) if absolute_diff.size else 0.0
        )
        status = "PASS" if error_ratio <= max_error_ratio else "FAIL"
        print(
            f"[AB-COMPARE][{label}] rank={rank} y={status} "
            f"error_ratio={error_ratio:.6%} "
            f"max_abs_diff={max_abs_diff:.8g}"
        )

        max_record = None
        if deviation.size:
            comparable = np.nan_to_num(
                deviation,
                nan=-np.inf,
                posinf=np.inf,
                neginf=-np.inf,
            )
            flat_index = int(np.argmax(comparable))
            tensor_index = tuple(
                int(index)
                for index in np.unravel_index(flat_index, deviation.shape)
            )
            max_record = {
                "rank": rank,
                "flat_index": flat_index,
                "tensor_index": tensor_index,
                "expected": float(expected.reshape(-1)[flat_index]),
                "actual": float(actual.reshape(-1)[flat_index]),
                "absolute_diff": float(
                    absolute_diff.reshape(-1)[flat_index]
                ),
                "deviation": float(deviation.reshape(-1)[flat_index]),
            }
            print(
                f"[AB-COMPARE][{label}][MAX-DEVIATION] "
                f"rank={rank} flat_index={flat_index} "
                f"index={tensor_index} "
                f"{expected_name}={max_record['expected']:.8g} "
                f"actual={max_record['actual']:.8g} "
                f"absolute_diff={max_record['absolute_diff']:.8g} "
                f"deviation={max_record['deviation']:.6%}"
            )
        if status == "FAIL":
            for flat_index in np.flatnonzero(incorrect)[:10]:
                tensor_index = tuple(
                    int(index)
                    for index in np.unravel_index(
                        int(flat_index),
                        incorrect.shape,
                    )
                )
                print(
                    f"  index={tensor_index} "
                    f"{expected_name}="
                    f"{expected.reshape(-1)[flat_index]:.8g} "
                    f"actual={actual.reshape(-1)[flat_index]:.8g} "
                    "absolute_diff="
                    f"{absolute_diff.reshape(-1)[flat_index]:.8g} "
                    f"deviation={deviation.reshape(-1)[flat_index]:.6%}"
                )
        return status, error_ratio, max_record

    failed = []
    global_golden_max = None
    with (
        zipfile.ZipFile(deepep_path, "r") as actual_archive,
        zipfile.ZipFile(reference_path, "r") as reference_archive,
        zipfile.ZipFile(golden_path, "r") as golden_archive,
    ):
        golden_names = set(golden_archive.namelist())
        for rank in range(case["EP"]):
            actual_bits = _load_npy_entry(
                actual_archive, f"rank{rank}/y.npy"
            ).astype(np.uint16, copy=False)
            reference_bits = _load_npy_entry(
                reference_archive, f"rank{rank}/y.npy"
            ).astype(np.uint16, copy=False)
            status, error_ratio, _ = compare_y(
                rank,
                "DEEPEP-VS-OPS",
                "ops",
                actual_bits,
                reference_bits,
            )
            bit_mismatch_ratio = float(
                np.mean(actual_bits != reference_bits)
            )
            print(
                f"[AB-COMPARE][DEEPEP-VS-OPS] rank={rank} "
                f"bit_mismatch_ratio={bit_mismatch_ratio:.6%}"
            )
            if status == "FAIL":
                failed.append(
                    f"DEEPEP-VS-OPS rank {rank} y error ratio "
                    f"{error_ratio:.6%} exceeds {max_error_ratio:.6%}"
                )

            golden_y_name = f"rank{rank}/golden_y.npy"
            golden_tokens_name = (
                f"rank{rank}/golden_expert_token_nums.npy"
            )
            if (
                golden_y_name not in golden_names
                or golden_tokens_name not in golden_names
            ):
                raise RuntimeError(
                    f"{golden_path} does not contain rank {rank} golden data; "
                    "rerun the updated ops-transformer-hs test to regenerate "
                    "mega_moe_input.zip"
                )
            golden_y = _load_npy_entry(golden_archive, golden_y_name)
            _, _, golden_max = compare_y(
                rank,
                "DEEPEP-VS-GOLDEN",
                "golden",
                actual_bits,
                golden_y,
            )
            compare_y(
                rank,
                "OPS-VS-GOLDEN",
                "golden",
                reference_bits,
                golden_y,
            )
            if (
                golden_max is not None
                and (
                    global_golden_max is None
                    or golden_max["deviation"]
                    > global_golden_max["deviation"]
                )
            ):
                global_golden_max = golden_max
            # Golden comparisons are diagnostic only. The interop test result
            # is determined by whether Deepep and ops-transformer-hs agree.

            actual_tokens = _load_npy_entry(
                actual_archive,
                f"rank{rank}/expert_token_nums.npy",
            ).astype(np.int32, copy=False)
            reference_tokens = _load_npy_entry(
                reference_archive,
                f"rank{rank}/expert_token_nums.npy",
            ).astype(np.int32, copy=False)
            tokens_equal = np.array_equal(actual_tokens, reference_tokens)
            print(
                f"[AB-COMPARE][DEEPEP-VS-OPS] rank={rank} "
                "expert_token_nums="
                f"{'PASS' if tokens_equal else 'FAIL'}"
            )
            if not tokens_equal:
                failed.append(
                    f"DEEPEP-VS-OPS rank {rank} expert_token_nums differ"
                )

            golden_tokens = _load_npy_entry(
                golden_archive,
                golden_tokens_name,
            ).astype(np.int32, copy=False)
            golden_tokens_equal = np.array_equal(
                actual_tokens,
                golden_tokens,
            )
            ops_golden_tokens_equal = np.array_equal(
                reference_tokens,
                golden_tokens,
            )
            print(
                f"[AB-COMPARE][DEEPEP-VS-GOLDEN] rank={rank} "
                "expert_token_nums="
                f"{'PASS' if golden_tokens_equal else 'FAIL'}"
            )
            print(
                f"[AB-COMPARE][OPS-VS-GOLDEN] rank={rank} "
                "expert_token_nums="
                f"{'PASS' if ops_golden_tokens_equal else 'FAIL'}"
            )
    if global_golden_max is not None:
        print(
            "[AB-COMPARE][DEEPEP-VS-GOLDEN][GLOBAL-MAX-DEVIATION] "
            f"rank={global_golden_max['rank']} "
            f"flat_index={global_golden_max['flat_index']} "
            f"index={global_golden_max['tensor_index']} "
            f"golden={global_golden_max['expected']:.8g} "
            f"actual={global_golden_max['actual']:.8g} "
            "absolute_diff="
            f"{global_golden_max['absolute_diff']:.8g} "
            f"deviation={global_golden_max['deviation']:.6%}"
        )
    if failed:
        raise AssertionError("; ".join(failed))
    print(
        "[AB-COMPARE] deepep output passed the ops-transformer-hs comparison; "
        "CPU golden comparisons above are diagnostic only"
    )


def run_case(args, case: dict, data_dir: Path) -> None:
    if case["localMemBytes"] % DEVICE_LARGE_PAGE_SIZE != 0:
        raise ValueError(
            "localMemBytes must be aligned to DEVICE_LARGE_PAGE_SIZE"
        )

    performance_enabled = args.mode in ("both", "performance")
    warmup = (
        DEFAULT_WARMUP if performance_enabled else 0
    ) if args.warmup is None else args.warmup
    loops = (
        DEFAULT_LOOPS if performance_enabled else 1
    ) if args.loops is None else args.loops
    if warmup < 0 or loops <= 0 or args.repeat <= 0:
        raise ValueError("warmup must be >= 0; loops/repeat must be > 0")

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running MegaMoe W8A8 {shape_label(case)}")
    print(f"[INFO] Shape data directory: {data_dir}")

    if args.interop_input is None:
        from mega_moe_w8a8_golden import generate

        generate(case, data_dir, args.seed)

    import torch.multiprocessing as mp

    config = {
        "mode": args.mode,
        "devices": args.devices,
        "data_dir": str(data_dir),
        "warmup": warmup,
        "loops": loops,
        "repeat": args.repeat,
        "shmem_ip_port": args.shmem_ip_port,
        "interop_input": (
            str(args.interop_input.resolve())
            if args.interop_input is not None
            else None
        ),
        "enable_profiler": performance_enabled,
        "enable_timer": args.enable_timer,
        "profile_session": time.strftime("%Y%m%d_%H%M%S"),
        "compare_cascade": (
            args.interop_input is None
            and args.mode != "performance"
            and not args.skip_cascade
        ),
        "cascade_gmm1_mode": args.cascade_gmm1_mode,
        "hccl_init_method": args.hccl_init_method,
    }
    if config["compare_cascade"]:
        print(
            "[INFO] Cascade precision comparison enabled: "
            f"gmm1_mode={args.cascade_gmm1_mode} "
            f"hccl={args.hccl_init_method}"
        )
        print(
            "[INFO] Running cascade and fused paths in isolated processes"
        )
        mp.spawn(
            run_cascade_rank,
            args=(case, config),
            nprocs=case["EP"],
            join=True,
        )

    mp.spawn(
        run_rank,
        args=(case, config),
        nprocs=case["EP"],
        join=True,
    )

    if args.enable_timer:
        from parse_timer_csv import parse_rank
        from trace_parser import generate_trace_files

        timer_dir = data_dir / "python_results" / "timer"
        timer_dir.mkdir(parents=True, exist_ok=True)
        for rank in range(case["EP"]):
            parse_rank(
                data_dir / "python_results" / f"rank_{rank}_npu_time.csv",
                timer_dir / f"rank_{rank}_npu_time_parsed.csv",
            )
        print(f"[TIMER] Parsed Timer CSV files: {timer_dir}", flush=True)
        trace_path = generate_trace_files(
            timer_dir,
            rank_start=0,
            rank_end=case["EP"] - 1,
            name_suffix=timer_trace_name_suffix(case),
        )
        print(f"[TIMER] Merged trace JSON: {trace_path}", flush=True)

    if performance_enabled:
        write_performance_summary(case, config, data_dir)
    if args.interop_input is not None:
        output_path = args.interop_output
        if output_path is None:
            output_path = args.interop_input.parent / "deepep_output.zip"
        reference_path = args.reference_output
        if reference_path is None:
            reference_path = (
                args.interop_input.parent / "ops_transformer_output.zip"
            )
        export_interop_output(case, data_dir, output_path)
        compare_interop_outputs(
            case,
            output_path,
            reference_path,
            args.interop_input,
            args.deviation_gap_threshold,
            args.max_error_ratio,
            args.absolute_error_threshold,
        )
    elif args.mode != "performance":
        from mega_moe_w8a8_golden import compare

        compare(
            case,
            data_dir,
            args.deviation_gap_threshold,
            args.max_error_ratio,
            compare_cascade=config["compare_cascade"],
            absolute_error_threshold=args.absolute_error_threshold,
        )
    print(f"[SUCCESS] MegaMoe W8A8 {shape_label(case)} passed")


def main():
    args = parse_args()
    if args.interop_input is not None:
        if args.mode == "performance":
            raise ValueError("--interop-input does not support performance-only mode")
        metadata = read_interop_metadata(args.interop_input)
        cases = [interop_case(metadata)]
        if cases[0]["EP"] != len(args.devices):
            raise ValueError(
                f"interop input requires {cases[0]['EP']} devices, "
                f"but {len(args.devices)} were provided"
            )
    else:
        cases = read_cases(args.csv_path, len(args.devices))

    data_root = args.data_dir
    if data_root is None:
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        data_root = TEST_DIR / "output" / run_name
    data_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] MegaMoe Python data directory: {data_root}")

    sys.path.insert(0, str(TEST_DIR))
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(
        str(device) for device in args.devices
    )
    os.environ["SHMEM_UID_SESSION_ID"] = args.shmem_ip_port.removeprefix(
        "tcp://"
    )

    for case in cases:
        case_data_dir = (
            data_root / "deepep_runtime"
            if args.interop_input is not None
            else data_root / shape_directory_name(case)
        )
        run_case(
            args,
            case,
            case_data_dir,
        )
    print(f"[SUCCESS] MegaMoe Python test data retained in: {data_root}")


if __name__ == "__main__":
    main()
