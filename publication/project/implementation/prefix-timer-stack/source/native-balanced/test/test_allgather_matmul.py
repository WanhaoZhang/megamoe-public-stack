import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from ascend_deepep import AllGatherMatmul, ShmemInstance


DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
AGMM_MIN_NUM_BYTES = 1024 * 1024 * 1024


def make_input_a(rank: int, m: int, k: int) -> torch.Tensor:
    base = torch.arange(m * k, device="npu", dtype=torch.float16).reshape(m, k)
    return base / 1000 + rank


def make_input_b(k: int, n: int) -> torch.Tensor:
    base = torch.arange(k * n, device="npu", dtype=torch.float16).reshape(k, n)
    return base / 1000


def save_rank_tensor(tensor: torch.Tensor, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch_npu.npu.synchronize()
    torch.save(tensor.detach().cpu(), tmp_path)
    os.replace(tmp_path, path)


def make_golden_from_files(world_size: int, exchange_dir: Path, b: torch.Tensor) -> torch.Tensor:
    a_blocks = []
    for src_rank in range(world_size):
        a_path = exchange_dir / f"a_rank_{src_rank}.pt"
        if not a_path.exists():
            raise FileNotFoundError(f"missing rank input file: {a_path}")
        a_blocks.append(torch.load(a_path, map_location="cpu").to(device="npu", dtype=torch.float16))
    a_gather = torch.cat(a_blocks, dim=0)
    return a_gather @ b


def run_eager_allgather_matmul(agmm_op, shmem, a: torch.Tensor, b: torch.Tensor, args):
    return agmm_op.compute(shmem, a, b), a


def run_graph_allgather_matmul(agmm_op, shmem, a: torch.Tensor, b: torch.Tensor, args):
    out = torch.empty((a.shape[0] * args.ep_size, b.shape[1]), device=a.device, dtype=a.dtype)
    for _ in range(args.graph_warmup_iters):
        agmm_op.compute_out(shmem, a, b, out)
    torch_npu.npu.synchronize()
    shmem.barrier()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, capture_error_mode=args.graph_capture_error_mode):
        agmm_op.compute_out(shmem, a, b, out)
    torch_npu.npu.synchronize()
    shmem.barrier()

    # Replay reads the same tensor addresses, so mutate the static input before replay.
    replay_a = a + args.graph_replay_input_offset
    a.copy_(replay_a)
    torch_npu.npu.synchronize()
    shmem.barrier()

    for _ in range(args.graph_replay_iters):
        graph.replay()
    torch_npu.npu.synchronize()

    return out, a


def run_rank(rank: int, args) -> None:
    world_size = int(args.ep_size)
    device_id = int(args.first_device) + rank

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = args.shmem_ip_port

    torch.npu.set_device(device_id)

    shmem = None
    agmm_op = None
    try:
        shmem = ShmemInstance(rank, world_size, args.shmem_ip_port)
        shmem.init(args.num_bytes)
        agmm_op = AllGatherMatmul(args.num_bytes)

        exchange_dir = Path(args.exchange_dir)
        if rank == 0:
            if exchange_dir.exists():
                shutil.rmtree(exchange_dir)
            exchange_dir.mkdir(parents=True, exist_ok=True)
        shmem.barrier()

        a = make_input_a(rank, args.m, args.k)
        b = make_input_b(args.k, args.n)
        shmem.barrier()

        if args.mode == "graph":
            out, golden_a = run_graph_allgather_matmul(agmm_op, shmem, a, b, args)
        else:
            out, golden_a = run_eager_allgather_matmul(agmm_op, shmem, a, b, args)
            torch_npu.npu.synchronize()

        save_rank_tensor(golden_a, exchange_dir / f"a_rank_{rank}.pt")
        shmem.barrier()

        if not args.skip_check_output:
            golden = make_golden_from_files(world_size, exchange_dir, b)
            torch.testing.assert_close(
                out,
                golden,
                rtol=args.rtol,
                atol=args.atol,
            )
            if args.save_outputs:
                save_rank_tensor(out, exchange_dir / f"out_rank_{rank}.pt")
                save_rank_tensor(golden, exchange_dir / f"golden_rank_{rank}.pt")

        shmem.barrier()
        if rank == 0 and not args.keep_exchange_files:
            shutil.rmtree(exchange_dir)

        print(
            f"rank {rank}/{world_size} allgather_matmul {args.mode} completed: "
            f"a={tuple(a.shape)} b={tuple(b.shape)} out={tuple(out.shape)}"
        )
    finally:
        if agmm_op is not None:
            agmm_op.destroy()
            agmm_op = None
        if shmem is not None:
            shmem.destroy()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ep-size",
        type=int,
        default=4,
        help="Number of worker processes/ranks to launch.",
    )
    parser.add_argument(
        "--first-device",
        type=int,
        default=0,
        help="First NPU device id used by rank 0.",
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", "tcp://127.0.0.1:8899"),
        help="Bootstrap address shared by all ranks.",
    )
    parser.add_argument(
        "--num-bytes",
        type=int,
        default=AGMM_MIN_NUM_BYTES,
        help="Local SHMEM memory bytes allocated by the allgather-matmul op.",
    )
    parser.add_argument("--m", type=int, default=128, help="M dimension.")
    parser.add_argument(
        "--n",
        type=int,
        default=256,
        help="N dimension. The minimal wrapper requires n %% 256 == 0.",
    )
    parser.add_argument("--k", type=int, default=256, help="K dimension.")
    parser.add_argument(
        "--mode",
        choices=("eager", "graph"),
        default="eager",
        help="Run allgather_matmul eagerly or capture/replay it with torch.npu.NPUGraph.",
    )
    parser.add_argument(
        "--graph-warmup-iters",
        type=int,
        default=1,
        help="Warmup iterations before NPU graph capture.",
    )
    parser.add_argument(
        "--graph-replay-iters",
        type=int,
        default=1,
        help="Replay iterations after NPU graph capture.",
    )
    parser.add_argument(
        "--graph-replay-input-offset",
        type=float,
        default=1.0,
        help="Value added to static input a before graph replay, so validation checks replay output.",
    )
    parser.add_argument(
        "--graph-capture-error-mode",
        choices=("global", "thread_local", "relaxed"),
        default="global",
        help="Capture error mode passed to torch.npu.graph.",
    )
    parser.add_argument(
        "--exchange-dir",
        default="test/.agmm_exchange",
        help="Directory used to exchange per-rank inputs for golden generation.",
    )
    parser.add_argument(
        "--keep-exchange-files",
        action="store_true",
        help="Keep exchanged input files after the test.",
    )
    parser.add_argument(
        "--save-outputs",
        action="store_true",
        help="Save per-rank output and golden tensors into --exchange-dir.",
    )
    parser.add_argument(
        "--skip-check-output",
        action="store_true",
        help="Skip golden output precision check.",
    )
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for golden check.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for golden check.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ep_size < 1:
        raise ValueError("--ep-size must be greater than 0")
    if args.num_bytes < AGMM_MIN_NUM_BYTES:
        raise ValueError(
            f"--num-bytes must be >= {AGMM_MIN_NUM_BYTES} for the default catccos symmetric workspace"
        )
    if args.num_bytes % DEVICE_LARGE_PAGE_SIZE != 0:
        raise ValueError(f"--num-bytes must be aligned to {DEVICE_LARGE_PAGE_SIZE} bytes")
    if args.n % 256 != 0:
        raise ValueError("--n must be divisible by 256")
    if args.mode == "graph":
        if args.graph_warmup_iters < 0:
            raise ValueError("--graph-warmup-iters must be greater than or equal to 0")
        if args.graph_replay_iters < 1:
            raise ValueError("--graph-replay-iters must be greater than 0")

    print(
        f"Launching AGMM {args.mode} with {args.ep_size} ranks, first device {args.first_device}, "
        f"shape m/n/k={args.m}/{args.n}/{args.k}, shmem ip/port {args.shmem_ip_port}"
    )
    if args.ep_size == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()

# python test/test_allgather_matmul.py --ep-size 4 --num-bytes 1073741824 --m 128 --n 256 --k 256
# python test/test_allgather_matmul.py --mode graph --ep-size 4 --num-bytes 1073741824 --m 128 --n 256 --k 256
