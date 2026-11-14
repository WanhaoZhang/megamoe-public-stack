# MegaMoe BF16 early-topk backward design

> 2026-09-07 集成说明：本文保留 train 旧训练接口的设计/实现记录。
> 当前 raw 前向已切换三输出 saved-state，旧训练入口暂时明确报错。
> 反向适配请以 [新持久状态与迁移清单](MegaMoe_BF16_前向持久Workspace第一版与反向适配.md) 为准；
> 下文旧 workspace 偏移、接口示例和历史验证结论不能直接用于新 ABI。

## Contract

This document is the source of truth for the training path where the route
weight is applied after SwiGLU and before GMM2.

For one packed route, let `p` be its dispatched scalar top-k weight:

```text
Z  = Xpacked @ W1.T
H  = SwiGLU(Z)
Hp = p * H
O  = Hp @ W2.T
```

The corresponding backward is:

```text
dO  = dispatch(grad_y)             # no p here
dHp = dO @ W2
dp  = ReduceSum_F(dHp * H)
dH  = p * dHp
dZ  = SwiGLUBackward(dH, Z)
dX  = dZ @ W1
dW1 = dZ.T @ Xpacked
dW2 = dO.T @ Hp
```

`topk_ids` is discrete and has no gradient. `topk_weights` is a differentiable
training input; Python autograd must return a tensor of the same shape, dtype
and device rather than silently returning `None`.

## Saved forward state

The forward workspace persists until backward and contains the existing packed
input/routing metadata plus:

- `ptrDispatchedTopkWeights`: one BF16 `p` for every packed route;
- `ptrSavedTokenPerExpert`: routing-count snapshot used to restore the SHMEM
  table;
- `ptrBwProgress`: debug progress only.

The original source-rank `topk_weights` tensor is still saved by autograd for
the input contract and output allocation. The kernel uses the packed weights
for numerical work.

## Backward workspace

The backward-only workspace is ordered as follows:

```text
ptrDO             [R, K] BF16
ptrRecomputeGate  [R, F] BF16
ptrRecomputeUp    [R, F] BF16
ptrGradZ          [R, N] BF16
ptrHp             [R, F] BF16
ptrDpProducts     [R, F] BF16
flags
```

`R=maxOutputSize`, `F=N/2`. `ptrDpProducts` intentionally stores the
per-feature product rather than a single `dp` per route. F is tiled in blocks
of 256 and different tiles of one row may run on different AIVs. A direct
`dp[row] += partial` implementation would therefore require a cross-core
atomic and a precisely defined accumulation dtype. The correctness-first
implementation assigns one AIV to each packed row in P6 and reduces F in FP32.

## Phase ownership

| Phase | Producer | Result |
|---|---|---|
| P0-A | all AIV | raw routed `grad_y` in peer `ptrA0` |
| P0-B | AIV0 | packed `dO` |
| P0-C | AIC + AIV1 | recomputed gate/up planes |
| P1 | AIC + AIV1 | `dHp`, `Hp`, `dHp*H`, `dH=p*dHp`, `dZ` |
| P2 | AIC + AIV0 | packed `dX`, reverse-routed through `ptrD` |
| P3 | AIC | `dW1` |
| P4 | AIC | `dW2` using `ptrHp` |
| P5 | all AIV | unpermuted `grad_x` |
| P6 | all AIV | reduced, reverse-routed and unpermuted `grad_topk_weights` |

P6 reuses peer `ptrD` only after every rank has completed all P5 reads.

## Pipeline synchronization audit

Hard events below are intra-core pipeline dependencies. They do not replace
`SyncAll` or the explicit cross-rank handshake.

| Buffer / state | Producer | Consumer or overwrite | Required ordering |
|---|---|---|---|
| P1 tile0 `dHp` | AIC Fixpipe | AIV PIPE_V | AIC `NotifyVector`; AIV `WaitForCube` |
| gate/up UB | PIPE_MTE2 | PIPE_V | `MTE2_V` after each load |
| H UB / weighted dH UB | PIPE_V | dependent PIPE_V operation | `PipeBarrier<PIPE_V>` |
| H/dHp tile before row weight | PIPE_V | PIPE_S reads packed p | `V_S` |
| packed p scalar | PIPE_S | row-wise PIPE_V `Muls` | `S_V` for each row |
| tile1 `dHp*H`, tile2 `Hp` | PIPE_V | PIPE_MTE3 | `V_MTE3` |
| tile1/tile2 after auxiliary stores | PIPE_MTE3 | next PIPE_MTE2 reload | `MTE3_MTE2` |
| tile2 up after SwiGLU grad | PIPE_V | next PIPE_MTE2 load | `V_MTE2` |
| tile0/tile1 `dZ` | PIPE_V | PIPE_MTE3 | `V_MTE3` |
| tile1 after dZ store | PIPE_MTE3 | next PIPE_MTE2 load | `MTE3_MTE2` |
| final `dZ` tile | PIPE_MTE3 | caller's PIPE_V acknowledgement | `MTE3_V`, then `NotifyCube` |
| all P1 GM outputs | local pipelines | P2/P3/P4 AIC readers | `PIPE_ALL`, per-AIV1 completion counters |
| peer `ptrD` dX writes | all source ranks | P5 source reads | local drain + `SyncAll` + cross-rank sync |
| P5 reads of peer `ptrD` | all source ranks | P6 overwrite with dp | local drain + `SyncAll` + cross-rank sync |
| peer `ptrD` dp writes | all expert ranks | P6 source reads | local drain + `SyncAll` + cross-rank sync |

Audit rule: for every reused UB region, record the producing pipeline, reading
pipeline and next overwriting pipeline separately. A producer-to-consumer event
does not automatically protect a later overwrite (WAR), and `SetFlag/WaitFlag`
must name the actual pipelines on the two sides.

## First-version performance boundary

P6 currently performs a scalar FP32 reduction over the BF16 product matrix.
This is a correctness baseline, not the final performance form. A later
optimization may vector-reduce per-row tiles and then combine partials, but it
must preserve single ownership of the final packed `dp` and the two cross-rank
barriers around `ptrD` reuse.

## Validation checkpoints

The backward test should compare, in order:

1. packed `dO` (raw `grad_y`);
2. gate and up;
3. `Hp` and `dHp*H`;
4. `dZ`;
5. `grad_x`, `grad_topk_weights`, `grad_w1`, `grad_w2`.

Use shapes that exercise multiple F tiles (`F>256`), multiple route slots,
multiple experts per rank, non-256 token counts and a larger token count such
as 1024. A small single-tile case cannot validate cross-tile ownership or the
phase-tail synchronization.
