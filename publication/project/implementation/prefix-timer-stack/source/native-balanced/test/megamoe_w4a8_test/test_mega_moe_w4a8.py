#!/usr/bin/env python3
"""Run migrated W4A8 through the DeepEP Torch class registration path.

用法示例：
    # 跑单个 case（例如 test_shapes.csv 第 4 行，0-based 即 M=512）
    # --dummy-data 默认 --balanced-ids：每个 expert token 数相等，性能数据稳定
    python test/megamoe_w4a8_test/test_mega_moe_w4a8.py 1,2,3,4 \
        --dummy-data --reuse-data --case-index 3

    # 默认 --case-index 0，跑 CSV 第一行
    python test/megamoe_w4a8_test/test_mega_moe_w4a8.py 1,2,3,4 --dummy-data

    # 复现不均衡分布（热点/温点/冷点 expert ids）
    python test/megamoe_w4a8_test/test_mega_moe_w4a8.py 1,2,3,4 \
        --dummy-data --no-balanced-ids --case-index 3

    # 用真实 W4 量化数据（慢，会跑 gen_input_and_golden_w4a8.py）
    python test/megamoe_w4a8_test/test_mega_moe_w4a8.py 1,2,3,4 \
        --case-index 3 --skip-golden
"""

import argparse
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time


TEST_DIR = Path(__file__).resolve().parent
GENERATOR_DIR = TEST_DIR / "megamoe_w4a8_independent_test"
ACLSHMEM_DATA_OP_UDMA = 0x08
ACLSHMEM_DATA_OP_SDMA = 0x02
ACLSHMEM_DATA_OP_UDMA_SDMA = ACLSHMEM_DATA_OP_UDMA | ACLSHMEM_DATA_OP_SDMA  # 0x0A
LOCAL_MEM_BYTES = 5 * 1024 * 1024 * 1024
# warmup 充分预热以消除前几次 cold-start 偏大值（warmup 不计入计时）。
# warmup/loops 均可通过 CLI --warmup/--loops 覆盖。
WARMUP = 10
LOOPS = 100


def read_case(path: Path, case_index: int = 0) -> dict:
    """读取 test_shapes.csv 的第 case_index 行（0-based）作为一个 case。

    支持多行 CSV：一次运行只处理选中的某一行，避免一次跑多个 case
    导致 shmem/processes 反复重建。多 case 由外层 run_perf.py 遍历调用。
    """
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    if case_index < 0 or case_index >= len(rows):
        raise ValueError(
            f"--case-index {case_index} out of range; CSV has {len(rows)} rows"
        )
    expected = ["M", "K", "N", "TOPK", "expertPerRank", "EP"]
    if list(rows[0]) != expected:
        raise ValueError(f"CSV columns must be {expected}")
    return {name: int(value) for name, value in rows[case_index].items()}


def rank_file(data_dir: Path, rank: int, name: str, case: dict) -> Path:
    suffix = f"w4a8_{case['M']}_{case['K']}_{case['N']}_{case['EP']}.bin"
    return data_dir / f"rank{rank}_{name}_{suffix}"


# run_rank 启动时必须加载的输入/权重文件名（不含 rank 前缀与 w4a8 后缀）。
REQUIRED_INPUT_NAMES = (
    "input_x",
    "input_expert_ids",
    "input_expert_scales",
    "weight1_w4_npu",
    "weight2_w4_npu",
    "weight1_w4_npu_scale",
    "weight2_w4_npu_scale",
)


def reusable_data_exists(
    data_dir: Path, case: dict, ignored_names=frozenset()
) -> bool:
    """检查 data_dir 下是否已存在所有 rank 的必需输入/权重文件。

    用于 --reuse-data：存在则复用，不存在则回退到生成。
    """
    for rank in range(case["EP"]):
        for name in REQUIRED_INPUT_NAMES:
            if name in ignored_names:
                continue
            if not rank_file(data_dir, rank, name, case).exists():
                return False
    return True


def _validate_external_expert_ids(ids, rank: int, case: dict, np):
    expected_shape = (case["M"], case["TOPK"])
    ids = np.asarray(ids)
    if ids.shape != expected_shape:
        raise ValueError(
            f"external expert ids for rank {rank} have shape {ids.shape}; "
            f"expected {expected_shape}"
        )
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError(
            f"external expert ids for rank {rank} must be integers, "
            f"got {ids.dtype}"
        )
    expert_num = case["EP"] * case["expertPerRank"]
    min_id = int(ids.min()) if ids.size else 0
    max_id = int(ids.max()) if ids.size else -1
    if min_id < 0 or max_id >= expert_num:
        raise ValueError(
            f"external expert ids for rank {rank} are outside "
            f"[0, {expert_num}): min={min_id}, max={max_id}"
        )
    return np.ascontiguousarray(ids, dtype=np.int32)


def _load_external_expert_ids(source: Path, case: dict):
    import numpy as np

    ep, m, topk = case["EP"], case["M"], case["TOPK"]
    if source.is_dir():
        arrays = []
        for rank in range(ep):
            exact = rank_file(source, rank, "input_expert_ids", case)
            if exact.is_file():
                candidates = [exact]
            else:
                candidates = sorted(
                    source.glob(f"rank{rank}_input_expert_ids*.bin")
                )
            if len(candidates) != 1:
                raise FileNotFoundError(
                    f"expected exactly one rank {rank} expert-id .bin under "
                    f"{source}, found {len(candidates)}"
                )
            ids = np.fromfile(candidates[0], dtype=np.int32)
            if ids.size != m * topk:
                raise ValueError(
                    f"{candidates[0]} has {ids.size} int32 values; "
                    f"expected {m * topk}"
                )
            arrays.append(ids.reshape(m, topk))
        return arrays

    if not source.is_file():
        raise FileNotFoundError(f"external expert-id source does not exist: {source}")
    if source.suffix == ".bin":
        ids = np.fromfile(source, dtype=np.int32)
        if ids.size != ep * m * topk:
            raise ValueError(
                f"{source} has {ids.size} int32 values; "
                f"expected {ep * m * topk} for shape {(ep, m, topk)}"
            )
        return list(ids.reshape(ep, m, topk))
    if source.suffix not in (".pt", ".pth"):
        raise ValueError(
            "--expert-ids-source accepts a .pt/.pth tensor, a single int32 "
            ".bin, or a directory of per-rank int32 .bin files"
        )

    import torch

    try:
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
    except Exception as safe_load_error:
        if source.name != "input_data.pt":
            raise RuntimeError(
                f"failed to load {source} as a tensor-only checkpoint"
            ) from safe_load_error
        module_dir = next(
            (
                parent
                for parent in source.parents
                if (parent / "gen_input_and_golden.py").is_file()
            ),
            None,
        )
        if module_dir is None:
            raise RuntimeError(
                f"cannot locate gen_input_and_golden.py above {source}; "
                "export torch.stack(input_data.expert_ids) to a separate .pt"
            ) from safe_load_error
        sys.path.insert(0, str(module_dir))
        try:
            print(
                f"[INFO] loading complete HaiSi input cache {source}; this "
                "loads every tensor in the cache"
            )
            payload = torch.load(
                source, map_location="cpu", weights_only=False
            )
        finally:
            sys.path.remove(str(module_dir))
    if hasattr(payload, "expert_ids"):
        payload = payload.expert_ids
    if isinstance(payload, dict):
        if "expert_ids" not in payload:
            raise KeyError(f"{source} dict does not contain an 'expert_ids' entry")
        payload = payload["expert_ids"]
    if isinstance(payload, (list, tuple)):
        if len(payload) != ep:
            raise ValueError(
                f"{source} contains {len(payload)} rank tensors; expected {ep}"
            )
        return [value.detach().cpu().numpy() for value in payload]
    if not isinstance(payload, torch.Tensor):
        raise TypeError(
            f"{source} must contain a tensor, list/tuple, or expert_ids dict; "
            f"got {type(payload).__name__}"
        )
    payload = payload.detach().cpu()
    if tuple(payload.shape) == (ep, m, topk):
        return [value.numpy() for value in payload.unbind(0)]
    if tuple(payload.shape) == (ep * m, topk):
        return [value.numpy() for value in payload.reshape(ep, m, topk)]
    raise ValueError(
        f"{source} tensor has shape {tuple(payload.shape)}; expected "
        f"{(ep, m, topk)} or {(ep * m, topk)}"
    )


def install_external_expert_ids(source: Path, data_dir: Path, case: dict) -> None:
    import numpy as np

    arrays = _load_external_expert_ids(source.resolve(), case)
    normalized = [
        _validate_external_expert_ids(ids, rank, case, np)
        for rank, ids in enumerate(arrays)
    ]
    counts = np.bincount(
        np.stack(normalized).reshape(-1),
        minlength=case["EP"] * case["expertPerRank"],
    )
    for rank, ids in enumerate(normalized):
        ids.tofile(rank_file(data_dir, rank, "input_expert_ids", case))
    print(
        f"[INFO] external expert ids installed from {source.resolve()}: "
        f"shape={(case['EP'], case['M'], case['TOPK'])}, dtype=int32, "
        f"min_count={int(counts.min())}, max_count={int(counts.max())}"
    )


def build_balanced_expert_ids(
    total: int, moe_expert_num: int, np
) -> "np.ndarray":
    """生成 total 长度的 int32 expert id 流，每个 id 出现次数相等(差不超过 1)。

    算法与 gen_input_and_golden_w4a8.py 的 gen_balanced_expert_ids_element 一致：
    每轮用 arange(r, r+N) % N 产生一个长度为 N 的排列，轮间错位 1 避免可预测性。
    拼接到 >= total 后截断到 total。这样全局每个 expert 次数均衡，
    MoE dispatch 后每个 expert 拿到的 token 数一致，性能数据稳定可比。
    """
    if moe_expert_num <= 0:
        raise ValueError("moe_expert_num must be positive")
    rounds = (total + moe_expert_num - 1) // moe_expert_num
    chunks = [
        np.arange(r, r + moe_expert_num, dtype=np.int32) % moe_expert_num
        for r in range(rounds)
    ]
    return np.concatenate(chunks)[:total]


def build_unbalanced_expert_ids(
    rank_token_counts: list[int], moe_expert_num: int, topk: int, torch
) -> "torch.Tensor":
    """按热点/温点/冷点权重模型生成不均衡 expert ids。"""
    if moe_expert_num < topk:
        raise ValueError(
            f"moe_expert_num ({moe_expert_num}) must be >= TOPK ({topk})"
        )

    hot_expert_count = min(8, moe_expert_num)
    warm_expert_count = min(16, max(moe_expert_num - hot_expert_count, 0))
    global_cold_expert_ratio = 0.005
    rank_cold_expert_ratio = 0.015
    expert_weights = torch.ones(moe_expert_num, dtype=torch.float32)
    routing_scale = moe_expert_num / max(topk, 1)
    expert_weights[0] = 20.0 * routing_scale
    if hot_expert_count > 1:
        expert_weights[1:hot_expert_count] = torch.logspace(
            math.log10(max(1.25 * routing_scale, 1.0)),
            math.log10(max(0.42 * routing_scale, 1.0)),
            hot_expert_count - 1,
        )
    warm_expert_start = hot_expert_count
    warm_expert_end = warm_expert_start + warm_expert_count
    expert_weights[warm_expert_start:warm_expert_end] = max(
        0.21 * routing_scale, 1.0
    )

    global_cold_expert_count = min(
        round(moe_expert_num * global_cold_expert_ratio),
        max(moe_expert_num - warm_expert_end, 0),
        max(moe_expert_num - topk, 0),
    )
    if global_cold_expert_count > 0:
        expert_weights[-global_cold_expert_count:] = 0

    ordinary_expert_end = moe_expert_num - global_cold_expert_count
    ordinary_expert_indices = torch.arange(warm_expert_end, ordinary_expert_end)
    rank_cold_expert_count = min(
        round(moe_expert_num * rank_cold_expert_ratio),
        ordinary_expert_indices.numel(),
        max(moe_expert_num - topk - global_cold_expert_count, 0),
    )
    expert_id_generator = torch.Generator().manual_seed(831)
    expert_permutation = torch.randperm(
        moe_expert_num, generator=expert_id_generator
    )
    rank_expert_ids = []
    sample_chunk_size = 4096
    for rank_token_num in rank_token_counts:
        rank_expert_weights = expert_weights.clone()
        if rank_cold_expert_count > 0:
            rank_cold_indices = ordinary_expert_indices[
                torch.randperm(
                    ordinary_expert_indices.numel(),
                    generator=expert_id_generator,
                )[
                    :rank_cold_expert_count
                ]
            ]
            rank_expert_weights[rank_cold_indices] = 0
        rank_expert_weights = rank_expert_weights[expert_permutation]
        sampled_chunks = []
        for start in range(0, rank_token_num, sample_chunk_size):
            current_bs = min(sample_chunk_size, rank_token_num - start)
            sampled_chunks.append(
                torch.multinomial(
                    rank_expert_weights.expand(current_bs, -1),
                    topk,
                    replacement=False,
                    generator=expert_id_generator,
                ).to(torch.int32)
            )
        rank_expert_ids.append(torch.cat(sampled_chunks, dim=0))
    return torch.cat(rank_expert_ids, dim=0)


def generate_dummy_data(
    data_dir: Path, case: dict, balanced_ids: bool = True
) -> None:
    """为性能测试快速生成随机数据（不经过 W4 量化，不计算 golden）。

    性能测试只测耗时、不验证数值精度，因此用随机字节填充即可，
    跳过 gen_input_and_golden_w4a8.py 里慢的 MXFP4 量化流程。
    input_expert_ids 取值约束在 [0, moe_expert_num) 内避免 kernel 越界。

    balanced_ids=True（默认，性能测试推荐）时，input_expert_ids 全局
    每个 expert 出现次数相等(差不超过 1)，保证各 expert 工作量一致，
    性能数据稳定可比。算法参考 gen_input_and_golden_w4a8.py 的
    is_topk_balance 路径：全局生成 (M*EP*TOPK,) 均匀 id 流再按 rank 切片。
    balanced_ids=False 时使用热点/温点/冷点带权采样，
    并保证同一 token 内的 TOPK expert id 不重复。

    写出的文件名与 rank_file 命名规则一致，run_rank 可直接加载。
    """
    import numpy as np
    import torch

    m, k, n = case["M"], case["K"], case["N"]
    topk, experts = case["TOPK"], case["expertPerRank"]
    ep = case["EP"]
    moe_expert_num = experts * ep
    rng = np.random.default_rng(0)

    # balanced 模式：全局 (M*EP*TOPK,) 均匀 id 流，reshape (EP, M, TOPK) 后按 rank 切。
    # 非均衡模式：每 rank 独立随机，下面循环里现生成。
    if balanced_ids:
        total = m * ep * topk
        ids_stream = build_balanced_expert_ids(total, moe_expert_num, np).reshape(ep, m, topk)
        print(f"[INFO] balanced expert_ids: total={total} experts={moe_expert_num} "
              f"each~={total // moe_expert_num}")
    else:
        ids_stream = build_unbalanced_expert_ids(
            [m] * ep, moe_expert_num, topk, torch
        ).view(ep, m, topk).numpy()
        expert_counts = np.bincount(ids_stream.reshape(-1), minlength=moe_expert_num)
        print(
            "[INFO] unbalanced expert_ids: "
            f"max={expert_counts.max()} average={expert_counts.mean():.2f} "
            f"max/average={expert_counts.max() / expert_counts.mean():.2f}"
        )

    distribution = "balanced" if balanced_ids else "hot/warm/cold unbalanced"
    print(f"[INFO] generating dummy {distribution} data "
          "for performance testing (no W4 quantization, no golden)")
    for rank in range(ep):
        # input_x: bf16 存为 uint16, shape (M, K)
        x = rng.integers(0, 0x10000, size=(m, k), dtype=np.uint16)
        x.tofile(rank_file(data_dir, rank, "input_x", case))
        # input_expert_ids: int32, 取值 [0, moe_expert_num), shape (M, TOPK)
        if balanced_ids:
            ids = ids_stream[rank]
        else:
            ids = ids_stream[rank]
        ids.tofile(rank_file(data_dir, rank, "input_expert_ids", case))
        # input_expert_scales: bf16 存为 uint16, shape (M, TOPK)
        scales = rng.integers(0, 0x10000, size=(m, topk), dtype=np.uint16)
        scales.tofile(rank_file(data_dir, rank, "input_expert_scales", case))
        # weight1_w4_npu: uint8 packed, size experts*N*K//2
        w1 = rng.integers(0, 0x100, size=(experts * n * k // 2,), dtype=np.uint8)
        w1.tofile(rank_file(data_dir, rank, "weight1_w4_npu", case))
        # weight2_w4_npu: uint8 packed, size experts*K*N//4
        w2 = rng.integers(0, 0x100, size=(experts * k * n // 4,), dtype=np.uint8)
        w2.tofile(rank_file(data_dir, rank, "weight2_w4_npu", case))
        # weight1_w4_npu_scale: uint8 (E8M0), shape (experts, N, K//64, 2)
        s1 = rng.integers(0, 0x100, size=(experts, n, k // 64, 2), dtype=np.uint8)
        s1.tofile(rank_file(data_dir, rank, "weight1_w4_npu_scale", case))
        # weight2_w4_npu_scale: uint8 (E8M0), shape (experts, K, N//128, 2)
        s2 = rng.integers(0, 0x100, size=(experts, k, n // 128, 2), dtype=np.uint8)
        s2.tofile(rank_file(data_dir, rank, "weight2_w4_npu_scale", case))
        print(f"[INFO] dummy data written for rank {rank}/{ep}")


def load_bits(path: Path, dtype, shape, torch, device, view_dtype=None):
    import numpy as np

    array = np.fromfile(path, dtype=dtype)
    expected = 1
    for extent in shape:
        expected *= extent
    if array.size != expected:
        raise RuntimeError(f"{path} has {array.size} values; expected {expected}")
    tensor = torch.from_numpy(array.copy()).reshape(shape)
    if view_dtype is not None:
        tensor = tensor.view(view_dtype)
    return tensor.to(device=device, non_blocking=False).contiguous()


def run_rank(
    rank: int, case: dict, data_dir: str, ip_port: str, enable_timer: bool,
    warmup: int, loops: int,
) -> None:
    import numpy as np
    import torch
    import torch_npu  # noqa: F401

    from ascend_deepep import MegaMoeW4A8, ShmemInstance

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(case["EP"])
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = ip_port
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    directory = Path(data_dir)
    m, k, n = case["M"], case["K"], case["N"]
    topk, experts = case["TOPK"], case["expertPerRank"]
    x = load_bits(rank_file(directory, rank, "input_x", case), np.uint16, (m, k), torch, device, torch.bfloat16)
    ids = load_bits(rank_file(directory, rank, "input_expert_ids", case), np.int32, (m, topk), torch, device)
    topk_weights = load_bits(rank_file(directory, rank, "input_expert_scales", case), np.uint16, (m, topk), torch, device, torch.bfloat16)
    weight1 = load_bits(rank_file(directory, rank, "weight1_w4_npu", case), np.uint8, (experts * n * k // 2,), torch, device)
    weight2 = load_bits(rank_file(directory, rank, "weight2_w4_npu", case), np.uint8, (experts * k * n // 4,), torch, device)
    scale1 = load_bits(rank_file(directory, rank, "weight1_w4_npu_scale", case), np.uint8, (experts, n, k // 64, 2), torch, device)
    scale2 = load_bits(rank_file(directory, rank, "weight2_w4_npu_scale", case), np.uint8, (experts, k, n // 128, 2), torch, device)
    max_recv = m * case["EP"] * min(topk, experts)

    shmem = None
    op = None
    try:
        shmem = ShmemInstance(rank, case["EP"], ip_port)
        shmem.set_data_op_engine_type(ACLSHMEM_DATA_OP_UDMA_SDMA)
        shmem.init(LOCAL_MEM_BYTES)
        op = MegaMoeW4A8(LOCAL_MEM_BYTES)

        def launch_once():
            if enable_timer:
                result_y, result_token_nums, result_timer = op.compute_with_timer(
                    shmem, x, ids, topk_weights, weight1, weight2, scale1, scale2,
                    max_recv,
                )
            else:
                result_y, result_token_nums = op.compute(
                    shmem, x, ids, topk_weights, weight1, weight2, scale1, scale2,
                    max_recv,
                )
                result_timer = None
            torch_npu.npu.synchronize()
            shmem.barrier()
            return result_y, result_token_nums, result_timer

        for _ in range(warmup):
            launch_once()

        y = None
        token_nums = None
        timer = None
        elapsed_seconds = 0.0
        for _ in range(loops):
            start_time = time.perf_counter()
            y, token_nums, timer = launch_once()
            elapsed_seconds += time.perf_counter() - start_time

        print(
            f"[PERF] registered rank={rank} warmup={warmup} loops={loops} "
            f"average_us={elapsed_seconds * 1e6 / loops:.3f}"
        )

        y.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(
            rank_file(directory, rank, "registered_output_y", case)
        )
        token_nums.detach().cpu().contiguous().numpy().astype(
            np.int32, copy=False
        ).tofile(rank_file(directory, rank, "registered_expert_token_nums", case))
        if timer is not None:
            np.savetxt(
                directory / f"rank_{rank}_npu_time.csv",
                timer.detach().cpu().contiguous().numpy(),
                delimiter=",",
                fmt="%d",
            )
        shmem.barrier()
    finally:
        if op is not None:
            op.destroy()
        if shmem is not None:
            shmem.destroy()


def compare_registered_outputs(case: dict, data_dir: Path) -> None:
    import numpy as np

    def decode(path: Path) -> np.ndarray:
        bits = np.fromfile(path, dtype=np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32)

    failures = []
    for rank in range(case["EP"]):
        actual = decode(rank_file(data_dir, rank, "registered_output_y", case))
        golden = decode(rank_file(data_dir, rank, "combine_output_bf16", case))
        if actual.shape != golden.shape or actual.size != case["M"] * case["K"]:
            raise RuntimeError(
                f"rank {rank} output size mismatch: actual={actual.size}, "
                f"golden={golden.size}"
            )
        absolute = np.abs(actual - golden)
        deviation = absolute / (np.abs(golden) + 1e-12)
        incorrect = (absolute > 1e-3) & (deviation > 0.1)
        error_ratio = float(np.mean(incorrect)) if incorrect.size else 0.0
        status = "PASS" if error_ratio <= 0.05 else "FAIL"
        print(f"[{status}] registered rank={rank} error_ratio={error_ratio:.6%}")
        incorrect_indices = np.flatnonzero(incorrect)
        if incorrect_indices.size:
            middle_start = max(0, (incorrect_indices.size - 10) // 2)
            samples = (
                ("first", incorrect_indices[:10]),
                ("middle", incorrect_indices[middle_start:middle_start + 10]),
                ("last", incorrect_indices[-10:]),
            )
            print(
                f"[DETAIL] registered rank={rank} "
                f"incorrect_count={incorrect_indices.size}"
            )
            for section, indices in samples:
                print(f"[DETAIL][{section}] rank={rank} count={indices.size}")
                for flat_index in indices:
                    token_index, hidden_index = divmod(
                        int(flat_index), case["K"]
                    )
                    actual_value = np.format_float_positional(
                        actual[flat_index], unique=True, trim="-"
                    )
                    golden_value = np.format_float_positional(
                        golden[flat_index], unique=True, trim="-"
                    )
                    absolute_error = np.format_float_positional(
                        absolute[flat_index], unique=True, trim="-"
                    )
                    print(
                        f"  flat_index={flat_index} token_index={token_index} "
                        f"hidden_index={hidden_index} "
                        f"actual={actual_value} "
                        f"golden={golden_value} "
                        f"absolute_error={absolute_error} "
                        f"relative_error={deviation[flat_index]:.6%}"
                    )
        if status == "FAIL":
            failures.append(rank)
    if failures:
        raise RuntimeError(f"registered W4A8 precision failed on ranks {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("devices", help="comma-separated local device IDs")
    parser.add_argument("--csv-path", type=Path, default=GENERATOR_DIR / "test_shapes.csv")
    parser.add_argument(
        "--case-index",
        type=int,
        default=0,
        help="0-based row index into --csv-path; one case per run (default 0)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "artifact directory; defaults to output/timer_registered when "
            "--enable-timer is set, otherwise output/registered"
        ),
    )
    parser.add_argument("--shmem-ip-port", default="tcp://127.0.0.1:18899")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument(
        "--reuse-data",
        action="store_true",
        help=(
            "reuse existing artifacts under --data-dir without regenerating; "
            "implies --skip-generate"
        ),
    )
    parser.add_argument(
        "--skip-golden",
        action="store_true",
        help="generate inputs and weights, but skip CPU golden calculation",
    )
    parser.add_argument(
        "--expert-ids-source",
        type=Path,
        default=None,
        help=(
            "replace generated input_expert_ids with external IDs from a "
            ".pt/.pth tensor, one EP*M*TOPK int32 .bin, or a directory of "
            "per-rank int32 .bin files; skips CPU golden comparison"
        ),
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help=(
            "generate random dummy data for performance testing instead of "
            "running the slow W4 quantization generator; implies --skip-golden "
            "and skips precision comparison"
        ),
    )
    parser.add_argument(
        "--balanced-ids",
        dest="balanced_ids",
        action="store_true",
        default=True,
        help=(
            "with --dummy-data: generate input_expert_ids so each expert gets "
            "an equal share of tokens (counts differ by at most 1), giving "
            "stable comparable perf numbers (default: enabled)"
        ),
    )
    parser.add_argument(
        "--no-balanced-ids",
        dest="balanced_ids",
        action="store_false",
        help=(
            "with --dummy-data: use a hot/warm/cold weighted expert-id "
            "distribution"
        ),
    )
    parser.add_argument(
        "--enable-timer",
        action="store_true",
        help="collect AscendTimerV2 data into rank_<rank>_npu_time.csv",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help=f"warmup iterations before timing (default {WARMUP})",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=LOOPS,
        help=f"timed iterations (default {LOOPS})",
    )
    args = parser.parse_args()
    if args.data_dir is None:
        output_name = "timer_registered" if args.enable_timer else "registered"
        args.data_dir = TEST_DIR / "output" / output_name
    devices = [int(value) for value in args.devices.split(",")]
    case = read_case(args.csv_path, args.case_index)
    print(f"[INFO] case_index={args.case_index} case={case}")
    if len(devices) != case["EP"] or len(set(devices)) != len(devices):
        raise ValueError("device count must equal CSV EP and device IDs must be unique")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    ignored_inputs = frozenset(
        {"input_expert_ids"} if args.expert_ids_source is not None else ()
    )
    if args.expert_ids_source is not None and not args.skip_golden:
        args.skip_golden = True
        print(
            "[INFO] --expert-ids-source changes routing; CPU golden generation "
            "and precision comparison are disabled"
        )
    # --reuse-data：数据存在则复用，不存在则回退到生成。
    # --skip-generate：无条件跳过生成（调用方保证数据已存在）。
    if args.reuse_data and not reusable_data_exists(
        args.data_dir, case, ignored_inputs
    ):
        print("[INFO] --reuse-data specified but required input files are "
              "missing; falling back to data generation")
        args.reuse_data = False
    skip_generate = args.skip_generate or args.reuse_data
    if args.dummy_data:
        # 性能测试专用：随机数据，跳过慢的 W4 量化与 golden。
        # dummy 数据可被 --reuse-data 复用，故仅在缺失时生成。
        if not reusable_data_exists(args.data_dir, case, ignored_inputs):
            generate_dummy_data(args.data_dir, case, args.balanced_ids)
        else:
            print("[INFO] --dummy-data: required files already exist, reusing")
    elif not skip_generate:
        environment = os.environ.copy()
        environment["MEGA_MOE_W4A8_CSV_PATH"] = str(args.csv_path.resolve())
        environment["MEGA_MOE_W4A8_DATA_DIR"] = str(args.data_dir.resolve())
        # 非复用模式强制清空旧数据，保持原行为
        environment["MEGA_MOE_W4A8_FORCE_CLEAN"] = "1"
        generator_command = [
            sys.executable,
            str(GENERATOR_DIR / "gen_input_and_golden_w4a8.py"),
        ]
        if args.skip_golden:
            generator_command.append("--skip-golden")
        subprocess.run(
            generator_command,
            cwd=GENERATOR_DIR,
            env=environment,
            check=True,
        )
    if args.expert_ids_source is not None:
        install_external_expert_ids(args.expert_ids_source, args.data_dir, case)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(value) for value in devices)
    context = mp.get_context("spawn")
    processes = [
        context.Process(
            target=run_rank,
            args=(
                rank,
                case,
                str(args.data_dir.resolve()),
                args.shmem_ip_port,
                args.enable_timer,
                args.warmup,
                args.loops,
            ),
        )
        for rank in range(case["EP"])
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    failed = [index for index, process in enumerate(processes) if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"registered W4A8 path failed on local ranks {failed}")
    if args.skip_golden or args.dummy_data:
        print("[INFO] CPU golden calculation and precision comparison skipped")
    else:
        compare_registered_outputs(case, args.data_dir)
    print(
        f"[INFO] warmup={args.warmup} loops={args.loops}; "
        "outputs and timer data are from the last measured loop"
    )
    print(f"[SUCCESS] MegaMoeW4A8 registered path completed; outputs: {args.data_dir}")


if __name__ == "__main__":
    main()
