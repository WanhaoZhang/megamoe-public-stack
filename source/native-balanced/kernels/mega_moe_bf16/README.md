# MegaMoe BF16 forward and training migration

`MegaMoeBF16` is an independent Ascend950/A5 BF16 forward operator. It reuses
DeepEP MegaMoe's routing, expert scheduling, dispatch, combine, and unpermute
pipeline. It does not call CATCCOS `DispatchFFNCombine`, and it does not change
the W8A8 or W4A8 execution path.

PR92 is used only as a reference for the A5 BF16 `MmadPingpong` configuration,
FP32 accumulation, and FP32 SwiGLU arithmetic.

The BF16 kernel no longer instantiates the quantized `MegaMoe` template. Its
orchestration, BF16 workspace layout, peer-memory layout, and CATLASS stages
live under `kernels/mega_moe_bf16`. The UDMA variant has a separate
`MegaMoeBF16UDMA` class and UDMA-only layout/transport implementation under
`kernels/mega_moe_bf16_udma`; neither path is selected through a BF16/UDMA
compile-time switch in the generic MegaMoe kernel.

Data flow:

```text
BF16 x
  -> BF16 dispatch communication
  -> BF16 x BF16 GMM1 (FP32 L0C accumulation, BF16 output)
  -> SwiGLU (FP32 internal arithmetic, BF16 output)
  -> BF16 x BF16 GMM2 (FP32 L0C accumulation, BF16 output)
  -> BF16 combine communication
  -> BF16 y
```

Build:

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
```

普通 BF16 前向的 GMM1 统一使用 M192：L1 tile 为 `192×256×256`，
L0 tile 为 `192×256×64`。GMM2 和反向仍使用 M128；MGroup、UDMA
也保留原有 M128 / Ready256 路径。`none`、`bf16`、`mxfp8`、`auto`
保存模式均使用普通前向的 M192，不按保存模式回退 M128。

普通前向 Dispatch 固定每次搬运 1 行，仍使用两个 UB 槽；仅普通前向
入口启用该粒度，反向、MGroup、UDMA 保持原配置。K7168 时搬运 UB
合计为 28 KiB，原 2 行为 56 KiB。GM workspace、保存格式及 Ready
发布单位不变；Host 仍按 2 行检查 UB 容量，不放宽已有形状约束。

默认 `adaptive-auto` 对所有专家使用按实际行数、N tile 数和 AIC 数
选择宽度的 N 条带调度。每卡专家数小于 8，且均衡预估每专家行数
`M×TopK/expertPerRank >= 768` 时，才成为 Ready384 候选；每张卡还需
确认实际首个非空专家至少有 768 行。该专家按 384 行发布 Dispatch 就绪，
后续专家整专家就绪。实际首专家不足 768 行时，本次调用全部专家整专家就绪。
一个完整 Ready384 块对应两个 M192 tile；末尾块按实际有效行数处理。

Ready 优先的生产者轮次重排只对足够大且能按 384 行对齐的生产者分片启用；
其余分片按顺序发布，不把不规则分片强行套入原 M128 的重排条件。
Ready 计数器仍按 `ceil(maxRecvTokens/256)` 预留，保持 Host 和反向的
GM 偏移不变，Ready384 使用的空间不会更多。

保留原有构建期调度模式用于对照；它们均不改变普通前向的 M192 分块：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule adaptive-ready-first

bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule adaptive-expert

bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule legacy
```

`adaptive-ready-first` 放开默认专家数和预估行数限制，但仍检查实际首专家
至少 768 行；`adaptive-expert` 使用 N 条带并等待整专家；`legacy`
使用 `GemmIdentityBlockSwizzle<3, 1>` 并等待整专家。
构建脚本每次都传递调度模式；运行测试不会改变已经编好的 SO。
计时图仍可用 `--enable-timer`，正式性能对比使用关闭 Timer 的构建。

完整方案、容量核算及 train 对照数据见
[普通 BF16 M192 前向优化](../../docs/MegaMoe_BF16_M192前向优化.md)。

See [BF16 performance follow-up and open issues](../../docs/megamoe_bf16_performance_followup_20260905.md)
for the measurements, swizzle analysis, and InitRouting/Unpermute candidates.

BF16, MGroup and UDMA batch the routed TopK-weight permutation in a separate UB phase
when `M * TopK <= 65536`. It gathers raw 16-bit values, publishes through MTE3,
and releases at most 140 KiB of scratch before allocating Dispatch buffers.
Larger route tables retain each variant's scalar gather path. The MGroup and
UDMA integrations still require their own device validation. For ordinary BF16,
after the device-annotation fix, the user reported
successful CANN compilation and precision for the default EP8 shape, with
31856 us end-to-end and 662 us InitRouting. The small end-to-end improvement
still needs to be distinguished from measurement variation; this result does
not validate other shapes or variants.

End-to-end precision and performance test on two devices:

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1
```

The Python performance report uses the same NPU Profiler task-duration method
as the W8A8 MegaMoe test: 110 loops by default and the last 50 matching rows.

To validate the exported BF16 kernel ABI without Python registration:

```bash
cd test/megamoe_bf16_test/megamoe_bf16_independent_test
bash build.sh
bash test_mega_moe.sh 0,1
```

Use `--mode precision` for bring-up without performance loops, or
`--performance-only` to skip CPU golden calculation and comparison. See
`test/megamoe_bf16_test/README.md` for the complete test interface.

Python API:

Ordinary `compute_train` writes packed X and packed TopK weights into one
independent `dispatch_output`. The framework may retain that tensor for the
current backward or release it for a future redispatching backward. Compact
`training_saved_state` keeps routing metadata and optional gate/up state. See
[optional Dispatch persistence](../../docs/MegaMoe_BF16_Dispatch输出可选持久化.md).

The raw `MegaMoeBF16.compute` returns three tensors:
`y, expert_token_nums, training_saved_state`; `compute_with_timer` adds a timer.
`compute_train` returns `y, expert_token_nums, training_saved_state,
dispatch_output, forward_transient_workspace`. The public `MegaMoeBF16Train`
wrapper still returns only `y, expert_token_nums`; its default retains the
Dispatch output for the current backward.

Ordinary BF16 accepts either canonical weights `W1 [E,N,K], W2 [E,K,F]` or
Megatron legacy weights `W1 [E,K,N], W2 [E,F,K]`. The Host identifies the pair
from both shapes and selects the matching compile-time CATLASS B layout. The
legacy training path therefore passes parameter views directly, without a full
transpose-and-copy. See
[dual weight layouts](../../docs/MegaMoe_BF16_双权重布局与免转置说明.md)
for the forward and backward matrix mapping.

## Buffer sizing

Use the shared `TrainingSavedStateLayout` and the variant's transient layout in
`mega_moe_training_workspace.h`. Persistent state includes packed X, routing
metadata, the token-count snapshot, packed BF16 TopK weights, and 1024 bytes of
backward progress. Transient memory holds SwiGLU output and forward readiness
flags; UDMA additionally retains its SHMEM activation receive region. Ordinary
BF16 allocates this transient payload with `torch::empty`; the kernel resets all
of its control flags at the start of every invocation before Dispatch begins.

For balanced `M=4096, K=7168, N=8192, TopK=16, EP=8, E=4, R=65536`:

| Per-rank allocation | Ordinary BF16 / MGroup | UDMA |
|---|---:|---:|
| Persistent saved-state | 940,185,600 bytes | 940,185,600 bytes |
| Forward transient including readiness | 536,936,960 bytes | 536,879,616 bytes |
| Required symmetric memory | 1,879,244,800 bytes | 2,818,769,920 bytes |
| Test script symmetric allocation floor | 2 GiB | 3 GiB |

Inputs, outputs, expert weights and runtime allocations are additional. For
other shapes or unbalanced routing, use the actual receive capacity and the
shared layout formulas in the [migration document](../../docs/MegaMoe_BF16_前向持久Workspace第一版与反向适配.md),
rather than reusing these numbers.
