#!/usr/bin/env python3
"""Minimal MegaMoe BF16 launcher for ``msprof op`` and its simulator.

The script intentionally contains no CPU golden, precision comparison,
intermediate dump, timer instrumentation, or auxiliary PyTorch operators.
For the backward phase, the training forward is required to create the saved
context; ``torch.autograd.backward(y, grad_y)`` then launches the custom
MegaMoe backward directly without constructing a scalar loss.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


LOCAL_MEM_BYTES = 1024 * 1024 * 1024
ACLSHMEM_DATA_OP_UDMA = 0x08


def create_npu_profiler(
    torch_npu,
    loops: int,
    output_dir: Path,
    enable_aic_metrics: bool,
):
    profile_args = {}
    if enable_aic_metrics:
        # Keep the first bring-up timeline-only.  AIC PMU collection changes
        # device execution conditions and is enabled separately after the
        # application-level trace is known to work.
        profile_args["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
            export_type=torch_npu.profiler.ExportType.Text,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            l2_cache=False,
            data_simplification=False,
        )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=loops, repeat=1, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
        profile_memory=False,
        record_shapes=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        **profile_args,
    )


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "devices must be comma-separated integers"
        ) from exc
    if len(devices) < 2:
        raise argparse.ArgumentTypeError("MegaMoe BF16 requires at least two EP ranks")
    if any(device < 0 for device in devices) or len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("devices must be unique non-negative integers")
    return devices


def parse_rank_list(value: str) -> list[int]:
    try:
        ranks = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "profile ranks must be comma-separated integers"
        ) from exc
    if not ranks or any(rank < 0 for rank in ranks) or len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError(
            "profile ranks must be unique non-negative integers"
        )
    return ranks


def make_inputs(rank: int, config: dict, device):
    import torch
    import numpy as np

    m = config["m"]
    k = config["k"]
    n = config["n"]
    topk = config["topk"]
    expert_per_rank = config["expert_per_rank"]
    total_experts = config["ep"] * expert_per_rank

    # Generate on CPU so the only AI Core kernels launched by this runner are
    # MegaMoe forward/backward.  Host-to-device copies are runtime DMA tasks.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config["seed"] + rank)
    x = (
        torch.empty((m, k), dtype=torch.float32)
        .uniform_(-0.1, 0.1, generator=generator)
        .to(dtype=torch.bfloat16, device=device)
        .contiguous()
    )
    # Routing control: default "balanced" matches the old round-robin exactly
    # (every expert gets the same token count); "random"/"imbalanced" reuse the
    # golden generator so a skewed routing run can be compared against it to
    # tell load imbalance from GEMM-scheduler imbalance.
    from mega_moe_bf16_golden import _generate_expert_ids

    rng = np.random.default_rng(config["seed"] + rank)
    expert_ids = _generate_expert_ids(
        rng, m, topk, total_experts,
        config.get("routing_mode", "balanced"),
        config.get("routing_skew", 1.0),
    )
    topk_ids = (
        torch.from_numpy(expert_ids).to(device).contiguous()
    )
    topk_weights = torch.full(
        (m, topk), 1.0 / topk, dtype=torch.bfloat16, device=device
    ).contiguous()
    w1 = (
        torch.empty((expert_per_rank, n, k), dtype=torch.float32)
        .uniform_(-1.0, 1.0, generator=generator)
        .to(dtype=torch.bfloat16, device=device)
        .contiguous()
    )
    w2 = (
        torch.empty((expert_per_rank, k, n // 2), dtype=torch.float32)
        .uniform_(-1.0, 1.0, generator=generator)
        .to(dtype=torch.bfloat16, device=device)
        .contiguous()
    )
    grad_y = (
        torch.empty((m, k), dtype=torch.float32)
        .uniform_(-0.1, 0.1, generator=generator)
        .to(dtype=torch.bfloat16, device=device)
        .contiguous()
    )
    return x, topk_ids, topk_weights, w1, w2, grad_y


def run_rank(rank: int, config: dict) -> None:
    import torch
    import torch_npu

    from ascend_deepep import MegaMoeBF16, MegaMoeBF16Train, ShmemInstance

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(config["ep"])
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = config["shmem_ip_port"]

    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    x, topk_ids, topk_weights, w1, w2, grad_y = make_inputs(rank, config, device)
    max_recv = config["m"] * config["ep"] * min(
        config["expert_per_rank"], config["topk"]
    )
    import numpy as np

    counts = np.bincount(
        topk_ids.detach().cpu().contiguous().numpy().reshape(-1).astype(np.int64),
        minlength=config["ep"] * config["expert_per_rank"],
    )
    print(
        f"[ROUTING] rank={rank} mode={config.get('routing_mode', 'balanced')} "
        f"expert_tokens={counts.tolist()} "
        f"(min={counts.min()} max={counts.max()} ratio={counts.max() / max(counts.min(), 1):.2f}x)",
        flush=True,
    )

    shmem = ShmemInstance(rank, config["ep"], config["shmem_ip_port"])
    shmem.set_data_op_engine_type(ACLSHMEM_DATA_OP_UDMA)
    shmem.init(config["local_mem_bytes"])
    op = None
    try:
        if config["phase"] == "forward":
            op = MegaMoeBF16(config["local_mem_bytes"])

            def launch(profile_regions: bool = False) -> None:
                if profile_regions:
                    with torch.profiler.record_function("megamoe_bf16.forward"):
                        op.compute(
                            shmem, x, topk_ids, topk_weights, w1, w2, max_recv
                        )
                else:
                    op.compute(
                        shmem, x, topk_ids, topk_weights, w1, w2, max_recv
                    )

        else:
            x.requires_grad_(True)
            w1.requires_grad_(True)
            w2.requires_grad_(True)
            op = MegaMoeBF16Train(shmem, config["local_mem_bytes"])
            holder = {}
            if config["capture_backward_timer"]:
                holder["capture_backward_timer"] = True

            def launch(profile_regions: bool = False) -> None:
                if profile_regions:
                    with torch.profiler.record_function(
                        "megamoe_bf16.forward_for_backward"
                    ):
                        y, _ = op(
                            x, topk_ids, topk_weights, w1, w2, max_recv,
                            diagnostic_holder=holder,
                        )
                    with torch.profiler.record_function("megamoe_bf16.backward"):
                        torch.autograd.backward(y, grad_y)
                else:
                    y, _ = op(
                        x, topk_ids, topk_weights, w1, w2, max_recv,
                        diagnostic_holder=holder,
                    )
                    torch.autograd.backward(y, grad_y)
                x.grad = None
                w1.grad = None
                w2.grad = None

        for _ in range(config["warmup"]):
            launch()
        torch_npu.npu.synchronize()
        shmem.barrier()

        profile_context = None
        profile_output = None
        if config["profile"] and rank in config["profile_ranks"]:
            profile_output = Path(config["profile_dir"]) / f"rank_{rank}"
            profile_output.mkdir(parents=True, exist_ok=True)
            profile_context = create_npu_profiler(
                torch_npu,
                config["loops"],
                profile_output,
                config["profile_aic_metrics"],
            )
            profile_context.__enter__()
        try:
            for loop_idx in range(config["loops"]):
                launch(profile_regions=profile_context is not None)
                if profile_context is not None:
                    # Ensure the final asynchronous kernel completes inside
                    # the active profiler window before exporting the trace.
                    if loop_idx + 1 == config["loops"]:
                        torch_npu.npu.synchronize()
                    profile_context.step()
        finally:
            torch_npu.npu.synchronize()
            if profile_context is not None:
                profile_context.__exit__(None, None, None)
        torch_npu.npu.synchronize()
        shmem.barrier()
        if (
            config["phase"] == "backward"
            and config["capture_backward_timer"]
            and rank in config["profile_ranks"]
        ):
            timer = holder.get("backward_timer")
            if timer is None:
                print(
                    "[WARN] backward timer requested but no timer tensor was "
                    "captured (kernel may be built without --enable-timer)",
                    flush=True,
                )
            else:
                import numpy as np

                timer_csv = Path(config["profile_dir"]) / f"rank_{rank}_backward_time.csv"
                timer_csv.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(
                    timer_csv,
                    timer.detach().cpu().contiguous().numpy(),
                    delimiter=",",
                    fmt="%d",
                )
                print(f"[TIMER] rank={rank} backward timer -> {timer_csv}", flush=True)
        print(
            f"[PROFILE] rank={rank} phase={config['phase']} "
            f"warmup={config['warmup']} loops={config['loops']} complete",
            flush=True,
        )
        if profile_output is not None:
            print(f"[PROFILE] rank={rank} output={profile_output}", flush=True)
    finally:
        if op is not None:
            if config["phase"] == "forward":
                op.destroy()
            else:
                op.close()
        shmem.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", type=parse_devices, nargs="?", default="0,1")
    parser.add_argument("--phase", choices=("forward", "backward"), default="backward")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--expert-per-rank", type=int, default=2)
    parser.add_argument(
        "--routing-mode",
        choices=("balanced", "random", "imbalanced"),
        default="balanced",
        help=(
            "routing distribution: balanced (round-robin, every expert gets "
            "the same token count), random (uniform), imbalanced (seeded "
            "skewed).  Default balanced matches the original round-robin."
        ),
    )
    parser.add_argument(
        "--routing-skew",
        type=float,
        default=1.0,
        help="hot-expert log-weight range for --routing-mode imbalanced",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--local-mem-bytes", type=int, default=LOCAL_MEM_BYTES)
    parser.add_argument("--shmem-ip-port", default="tcp://127.0.0.1:8899")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--profile-ranks",
        type=parse_rank_list,
        default=[0],
        help="comma-separated local ranks that enable torch_npu.profiler",
    )
    parser.add_argument(
        "--profile-aic-metrics",
        action="store_true",
        help="add PipeUtilization PMU metrics after timeline-only profiling works",
    )
    parser.add_argument(
        "--profile-dir", default="/tmp/megamoe_bf16_torch_profile"
    )
    parser.add_argument(
        "--capture-backward-timer",
        action="store_true",
        help=(
            "run each backward with the AscendTimerV2 instrumentation enabled "
            "(kernel must be built with --enable-timer) and write "
            "rank_<rank>_backward_time.csv per profiled rank"
        ),
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        help=(
            "after profiling, write the Perfetto/Chrome trace JSON here "
            "(default <profile-dir>/timer_json).  Only meaningful together "
            "with --capture-backward-timer."
        ),
    )
    parser.add_argument(
        "--timer-groups",
        type=int,
        default=32,
        help="AIC groups in the timer buffer (block_dim / GetCoreNumAic)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.m <= 0 or args.k <= 0 or args.n <= 0:
        raise ValueError("m, k, and n must be positive")
    if args.k % 256 != 0 or args.n % 512 != 0:
        raise ValueError("MegaMoe BF16 requires k % 256 == 0 and n % 512 == 0")
    if args.topk <= 0 or args.expert_per_rank <= 0:
        raise ValueError("topk and expert-per-rank must be positive")

    devices = args.devices
    if any(rank >= len(devices) for rank in args.profile_ranks):
        raise ValueError("profile-ranks must be local rank indices within devices")
    if args.topk > len(devices) * args.expert_per_rank:
        raise ValueError("topk exceeds the total number of experts")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, devices))
    os.environ["SHMEM_UID_SESSION_ID"] = args.shmem_ip_port.removeprefix("tcp://")

    config = vars(args).copy()
    config["ep"] = len(devices)
    import torch.multiprocessing as mp

    mp.spawn(run_rank, args=(config,), nprocs=config["ep"], join=True)

    if args.capture_backward_timer:
        from parse_backward_timer import parse_rank as parse_backward_rank
        from trace_parser import generate_trace_files

        timer_dir = Path(config["profile_dir"]) / "timer"
        timer_dir.mkdir(parents=True, exist_ok=True)
        ranks_with_timer = []
        for rank in range(config["ep"]):
            raw_path = Path(config["profile_dir"]) / f"rank_{rank}_backward_time.csv"
            if not raw_path.exists():
                continue
            parsed_path = timer_dir / f"rank_{rank}_backward_time_parsed.csv"
            parse_backward_rank(raw_path, parsed_path, config["timer_groups"])
            ranks_with_timer.append(rank)
        if not ranks_with_timer:
            print(
                "[WARN] capture-backward-timer requested but no rank wrote a "
                "timer CSV (kernel may be built without --enable-timer)",
                flush=True,
            )
        else:
            output_dir = args.trace_dir or Path(config["profile_dir"]) / "timer_json"
            output_dir.mkdir(parents=True, exist_ok=True)
            merged = generate_trace_files(
                timer_dir,
                rank_start=min(ranks_with_timer),
                rank_end=max(ranks_with_timer),
                output_dir=output_dir,
                name_suffix="backward",
                phase="backward",
            )
            print(
                f"[TRACE] backward pipeline JSON: {merged} "
                "(open in Perfetto or chrome://tracing)",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
