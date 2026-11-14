# MegaMoe BF16 UDMA forward path

`MegaMoeBF16UDMA` is the UDMA transport variant of the BF16 forward operator.
It has an independent kernel orchestration class, device workspace/peer-memory
layout, CATLASS row-group Combine implementation, and Host-side extra-buffer
calculation. The legacy BF16 kernel remains in `kernels/mega_moe_bf16`.

The current AIV ownership and scheduling are intentionally unchanged by this
refactor:

```text
AIV0: SwiGLU
AIV1: UDMA Dispatch and row-group Combine
AIC : GMM1 and GMM2
```

Routed TopK weights use the shared BF16 batched Gather when `M * TopK <= 65536`.
Its temporary UB (at most 140 KiB) is released after MTE3 publication and before
transport buffer allocation; larger tables use the existing scalar fallback.
This does not change UDMA/SDMA submission, readiness signals or saved-state
ownership. The UDMA integration still requires device compilation, precision
and performance validation. Existing Timer boundaries are retained: this weight
gather belongs to `AIV_INITROUTING_EXPERT_TOKEN_COUNT`, after the SORT interval.

Build and test:

Packed TopK weights now use receiver-side MTE (`DataCopyPad`) directly into
`training_saved_state`, in the same expert/source/row order as packed X.
They no longer use remote UDMA or local SDMA. The existing pre-compute local
barrier waits for MTE weight completion; transport signals still guard the
activation payload. Only packed X and token counts need a later snapshot.
The old SHMEM packed-weight receive slot is removed, moving `ptrD`; rebuild
Host and kernel together. Saved state retains the weights for backward.

```bash
bash scripts/build_shmem.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16_udma.sh -soc_type Ascend950
bash test/megamoe_bf16_test/test_mega_moe_udma.sh 6,7
```

The operator sizes one shared symmetric activation area from the caller-provided
`maxRecvTokenNum`. The area is reused sequentially as InitRouting scratch,
Dispatch receive/GMM1 input, and Combine staging, so the UDMA path does not
retain a duplicate ordinary-workspace `ptrA`. The end-to-end BF16 test uses the
balanced capacity `align128(M * TopK)` by default and a dropless worst-case
capacity for random or imbalanced routing. The test initializes SHMEM once
with a 3 GiB UDMA floor, or a larger 2 MiB-aligned size when the calculated
peer-memory layout requires it, and reuses that allocation across calls.
