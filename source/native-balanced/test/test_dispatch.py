import argparse
import os

import torch
import torch.multiprocessing as mp
import torch_npu  # noqa: F401

from ascend_deepep import Dispatch, ShmemInstance


DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024


def run_rank(rank: int, args) -> None:
    world_size = int(args.ep_size)
    device_id = int(args.first_device) + rank

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = args.shmem_ip_port

    torch.npu.set_device(device_id)

    shmem = None
    dispatch_op = None
    try:
        shmem = ShmemInstance(rank, world_size, args.shmem_ip_port)
        shmem.init(args.num_bytes)
        dispatch_op = Dispatch(args.num_bytes)

        x = torch.randn(8, 16, device="npu", dtype=torch.float16)
        topk_idx = torch.full((8, 1), rank, device="npu", dtype=torch.int32)
        topk_weights = torch.ones(8, 1, device="npu", dtype=torch.float32)

        recv_x, recv_topk_idx, recv_topk_weights, handle = dispatch_op.compute(
            shmem,
            x,
            topk_idx,
            topk_weights,
            world_size,
            8,
            1,
        )

        # combine_op = Combine(args.num_bytes)
        # (out,) = combine_op.compute(shmem, recv_x, handle, topk_weights)

        torch_npu.npu.synchronize()

        shmem.barrier()

        # if args.check_outputs:
        #     torch.testing.assert_close(out, x)
        print(f"rank {rank}/{world_size} dispatch + barrier call chain completed")
    finally:
        if dispatch_op is not None:
            dispatch_op.destroy()
            dispatch_op = None
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
        default=64 * 1024 * 1024,
        help="Local SHMEM memory bytes allocated by the dispatch op.",
    )
    parser.add_argument(
        "--mode",
        choices=("init", "call-chain"),
        default="init",
        help="Test scope. Use call-chain only after dispatch/combine kernels are implemented.",
    )
    parser.add_argument(
        "--check-outputs",
        action="store_true",
        help="Assert dispatch/combine output correctness. Intended for implemented kernels.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.ep_size < 1:
        raise ValueError("--ep-size must be greater than 0")
    if args.check_outputs and args.mode != "call-chain":
        raise ValueError("--check-outputs requires --mode call-chain")
    if args.num_bytes % DEVICE_LARGE_PAGE_SIZE != 0:
        raise ValueError(f"--num-bytes must be aligned to {DEVICE_LARGE_PAGE_SIZE} bytes")

    print(f"Launching {args.ep_size} ranks with first device {args.first_device} and shmem ip/port {args.shmem_ip_port}")
    if args.ep_size == 1:
        run_rank(0, args)
    else:
        mp.spawn(run_rank, args=(args,), nprocs=args.ep_size, join=True)


if __name__ == "__main__":
    main()

# python test/test_dispatch.py --ep-size 4 --mode call-chain
