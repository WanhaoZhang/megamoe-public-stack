#!/usr/bin/env python3
"""Minimal forward-only smoke for MegaMoeBF16.compute_train (swiglu-limit path).

SMOKE_CHECK_CLIP_MASKS=1 selects a small exact-GMM1 case and checks all valid
MXFP8 values, scales and mask words against original BF16 inputs. It also
compares y/counts with BF16 saving. This does not validate backward gradients.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.multiprocessing as mp

MASTER_PORT = "tcp://127.0.0.1:8899"
EP = int(os.environ.get("SMOKE_EP", "2"))
VISIBLE = os.environ.get("SMOKE_DEVICES", "0,1")
LOCAL_MEM_BYTES = 2 * 1024 * 1024 * 1024
DATA_OP_UDMA = 0x08


def check_exact_saved(rank, saved, dispatch_output, counts, w1, m, k, n,
                      topk, local_e, capacity, limit):
    import numpy as np
    from saved_gate_up_reference import compare_mxfp8_blocks, pack_clip_masks
    from test_workspace_layout import training_saved_state_size, align_bytes
    f, groups = n // 2, n // 64
    base = training_saved_state_size(
        m, k, topk, EP, local_e, capacity, n, save_gate_up=False,
        embed_dispatch_output=False)
    data_bytes, scale_bytes = align_bytes(capacity*f), align_bytes(capacity*groups)
    mask_base = base + 2*data_bytes + 2*scale_bytes
    mask_bytes = align_bytes(capacity*groups*4)
    row_base = 0
    checked = 0
    for expert, rows in enumerate(counts.cpu().tolist()):
        rows = int(rows)
        if not rows:
            continue
        # Sparse test X has only column 0 nonzero (0 or +/-1). GMM1 is an
        # exact sign change of BF16 W, avoiding matmul rounding at boundaries.
        x = dispatch_output.narrow(
            0, row_base*k*2, rows*k*2).view(torch.bfloat16).reshape(rows,k)
        expected = (x[:, :1].float() * w1[expert, :, 0].float()).cpu().numpy()
        gate, up = expected[:, :f], expected[:, f:]
        for plane, target in enumerate((gate, up)):
            raw = saved.narrow(0, base + plane*data_bytes + row_base*f, rows*f).cpu().numpy()
            scales = saved.narrow(0, base + 2*data_bytes + plane*scale_bytes + row_base*groups,
                                  rows*groups).cpu().numpy()
            compare_mxfp8_blocks(raw.reshape(rows,groups,32), scales.reshape(rows,groups),
                                 target.reshape(rows,groups,32), context=f"rank={rank} expert={expert} plane={plane}")
        if limit > 0:
            expected_masks = pack_clip_masks(gate, up, limit)
            for plane, target in enumerate(expected_masks):
                actual = saved.narrow(0, mask_base + plane*mask_bytes + row_base*groups*4,
                                      rows*groups*4).cpu().numpy().view('<u4').reshape(rows,groups)
                np.testing.assert_array_equal(actual, target,
                    err_msg=f"rank={rank} expert={expert} mask={plane} limit={limit}")
        checked += rows
        row_base += rows
    print(f"[SAVED_CLIP_MASK] rank={rank} limit={limit} rows={checked} "
          f"format_version={2 if limit > 0 else 1} masks={'three_bitplanes_v1' if limit > 0 else 'none'} "
          "all_valid_values_scales_and_bits=PASS", flush=True)


def run_rank(local_rank: int) -> None:
    import torch_npu  # noqa: F401

    from ascend_deepep import MegaMoeBF16, ShmemInstance

    rank = local_rank
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(EP)
    os.environ["ASCEND_DEEPEP_SHMEM_IP_PORT"] = MASTER_PORT

    device = torch.device(f"npu:{local_rank}")
    torch.npu.set_device(device)
    torch.manual_seed(42 + rank)

    shmem = ShmemInstance(rank, EP, MASTER_PORT)
    shmem.set_data_op_engine_type(DATA_OP_UDMA)
    shmem.init(LOCAL_MEM_BYTES)
    op = MegaMoeBF16(LOCAL_MEM_BYTES)

    check_masks = os.environ.get("SMOKE_CHECK_CLIP_MASKS", "0") == "1"
    m, k, n, topk, local_e = (129, 256, 512, 4, 4) if check_masks else (4096, 7168, 8192, 16, 4)
    x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(
        0, EP * local_e, (m, topk), dtype=torch.int32, device=device)
    topk_w = torch.randn(m, topk, dtype=torch.bfloat16, device=device)
    w1 = torch.randn(local_e, n, k, dtype=torch.bfloat16, device=device) * 0.1
    w2 = torch.randn(local_e, k, n // 2, dtype=torch.bfloat16, device=device) * 0.1
    if check_masks:
        x.zero_()
        x[:, 0] = (1 - 2*(torch.arange(m, device=device) % 2)).to(torch.bfloat16)
        # Leave local experts 2/3 empty and cover every source rank.
        ids = torch.arange(m*topk, device=device).reshape(m,topk) % (EP*2)
        topk_ids = ((ids//2)*4 + ids%2).to(torch.int32)
        topk_w.fill_(1.0/topk)
        values = torch.tensor([-16,-10.0625,-10,-9.9375,-1.0078125,-1,-0.99609375,
                               -0.0,0,0.99609375,1,1.0078125,9.9375,10,10.0625,16],
                              dtype=torch.bfloat16, device=device)
        w1.zero_()
        w1[:, :, 0] = values.repeat(n//values.numel())
    max_recv = m * EP * min(local_e, topk)

    from test_workspace_layout import dispatch_output_size, training_saved_state_size
    save_mode = os.environ.get("SMOKE_SAVE_GMM1_OUTPUT", "bf16")
    if check_masks and save_mode != "mxfp8":
        raise ValueError("SMOKE_CHECK_CLIP_MASKS=1 requires SMOKE_SAVE_GMM1_OUTPUT=mxfp8")
    limits = (1.0, 10.0, 1.001, 0.0, 1.0) if check_masks else (0.0, 1.0, 10.0)
    for iteration, swiglu_limit in enumerate(limits):
        if check_masks and iteration == len(limits)-1:
            x.zero_()  # All words must be overwritten with zero, not left stale.
        y, counts, saved_state, dispatch_output, transient = op.compute_train(
            shmem, x, topk_ids, topk_w, w1, w2, max_recv, float(swiglu_limit), save_mode)
        torch.npu.synchronize()
        compressed = save_mode == "mxfp8" or (save_mode == "auto" and swiglu_limit == 0.0)
        expected_bytes = training_saved_state_size(
            m, k, topk, EP, local_e, max_recv, n,
            save_mxfp8=compressed, save_gate_up=save_mode != "none",
            save_clip_masks=compressed and swiglu_limit > 0,
            embed_dispatch_output=False)
        assert saved_state.numel() == expected_bytes, (
            f"rank {rank}: saved allocation mismatch for limit={swiglu_limit}: "
            f"{saved_state.numel()} != {expected_bytes}")
        expected_dispatch_bytes = dispatch_output_size(k, max_recv)
        assert dispatch_output.dtype == torch.uint8 and (
            dispatch_output.numel() == expected_dispatch_bytes), (
            f"rank {rank}: dispatch allocation mismatch: "
            f"dtype={dispatch_output.dtype} bytes={dispatch_output.numel()} "
            f"expected={expected_dispatch_bytes}")
        assert torch.isfinite(y).all().item(), (
            f"rank {rank}: non-finite y with swiglu_limit={swiglu_limit}")
        if check_masks:
            check_exact_saved(
                rank, saved_state, dispatch_output, counts, w1, m, k, n,
                topk, local_e, max_recv, swiglu_limit)
            (baseline_y, baseline_counts, baseline_saved,
             baseline_dispatch, baseline_transient) = op.compute_train(
                shmem, x, topk_ids, topk_w, w1, w2, max_recv, float(swiglu_limit), "bf16")
            torch.npu.synchronize()
            torch.testing.assert_close(counts, baseline_counts, rtol=0, atol=0)
            torch.testing.assert_close(y, baseline_y, rtol=0.01, atol=0.01)
            print(f"[CLIP_FORWARD] rank={rank} limit={swiglu_limit} y_counts_vs_bf16=PASS", flush=True)
            del (baseline_y, baseline_counts, baseline_saved,
                 baseline_dispatch, baseline_transient)
        print(
            f"[SMOKE] rank={rank} swiglu_limit={swiglu_limit} "
            f"save_mode={save_mode} saved_format={'NONE' if save_mode == 'none' else 'MXFP8' if compressed else 'BF16'} "
            f"y_finite=True y_absmax={y.abs().max().item():.6f} "
            f"counts_sum={int(counts.sum().item())} "
            f"saved_bytes={saved_state.numel()} "
            f"dispatch_bytes={dispatch_output.numel()} "
            f"transient_bytes={transient.numel()}",
            flush=True,
        )
        del y, counts, saved_state, dispatch_output, transient

    op.destroy()
    shmem.destroy()
    print(f"[SMOKE] rank={rank} OK", flush=True)


if __name__ == "__main__":
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = VISIBLE
    os.environ["SHMEM_UID_SESSION_ID"] = MASTER_PORT.removeprefix("tcp://")
    mp.spawn(run_rank, nprocs=EP, join=True)
    print("[SMOKE] compute_train forward passed", flush=True)
