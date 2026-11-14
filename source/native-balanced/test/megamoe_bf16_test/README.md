# MegaMoe BF16 end-to-end test

The integration currently enables raw BF16/MGroup/UDMA forward only. Training
and backward scripts are retained, but the Host rejects their old training ABI
until [saved-state migration](../../docs/MegaMoe_BF16_前向持久Workspace第一版与反向适配.md)
is complete. Rebuild the libraries after pulling: the shared timer layout also
changed; do not mix an old SO with the new Python extension or parsers.

This directory contains the BF16 counterpart of the W8A8 MegaMoe test:

- multi-rank ACLSHMEM execution through `ascend_deepep.MegaMoeBF16`;
- CPU golden for Dispatch, GMM1, SwiGLU, GMM2, weighted unpermute, and expert counts;
- NPU Profiler `Task Duration(us)` performance measurements matching W8A8;
- shape selection through `mega_moe_cases.csv`;
- host-only workspace and UB capacity unit tests.

All three BF16 variants share the batched routed TopK-weight Gather for tables
of at most 65536 routes (`M * TopK`); larger tables retain the scalar fallback.
The helper uses a separate UB phase before Dispatch/transport buffers are
allocated. `test_topk_weight_gather.cpp` checks capacity, route coverage, tails
and bit patterns on the host. Device compilation/precision has been reported
for ordinary BF16's default EP8 shape; the MGroup/UDMA integrations remain
pending device validation. No new build or runtime flag is required.

## Default forward shapes

All three Python runners (BF16, BF16 UDMA, and BF16 MGroup) use
`mega_moe_cases.csv` and run the following case by default:

| M (tokens per rank) | K (hidden size H) | FFN width after SwiGLU | CSV N (GMM1 output width) |
| ---: | ---: | ---: | ---: |
| 4096 | 7168 | 4096 | 8192 |

The CSV column order is `M,N,K`, not `M,K,N`. Its `N` is twice the
post-SwiGLU FFN width: `W1=[experts,N,K]`, `W2=[experts,K,N/2]`.
These dimensions follow the training shapes in
`megamoe训练新Shape多机EP通信计算掩盖分析.md`.
TopK remains 16; EP is selected by the device list and experts per rank are
4. This matches the analysis document's 4-experts-per-rank assumption; its
EP=128/256 and 32-AIC topology remain analytical targets. The independent C++ test
keeps its separate CSV.

All three BF16 Python APIs now return
`(y, expert_token_nums, training_saved_state)`. Dispatch writes packed X
directly to `training_saved_state.ptrA` in ordinary BF16 and MGroup. UDMA must
first receive packed X in symmetric `ptrA1`, then snapshots its valid prefix
into `training_saved_state.ptrA` before Combine reuses `ptrA1`. All three
paths produce routing metadata directly in `training_saved_state`; UDMA
only snapshots packed X and the mutable SHMEM token-count table. UDMA's
transient workspace contains only GMM2 input and synchronization/readiness
flags. Host allocation and device offsets use the same compact layout;
the compact transient layout saves 513 KiB per rank for the default EP8 shape.
All three variants now use MTE to copy dispatched TopK weights directly into
saved state, aligned with packed X. Ordinary BF16/MGroup remove the old transient
weight slot; UDMA removes its SHMEM receive-weight slot and no longer sends
weights through UDMA/SDMA. UDMA activations retain their original transport.
For the default shape, saved state grows by 128 KiB to 940,185,600 bytes and
UDMA's required peer layout shrinks by 128 KiB; its 3 GiB allocation floor stays.
Rebuild Host and kernels together because the progress and UDMA ptrD offsets move.
The retained allocation contains packed X, packed TopK weights, routing metadata, the token-count snapshot, and
backward progress state. Forward-only activations, flags, and communication
scratch live in separate internal transient allocations that can be released
after the synchronized forward launch returns. Ordinary BF16 allocates the
complete transient payload with `torch::empty`; each kernel invocation resets
the SwiGLU-to-GMM2, Dispatch-to-GMM1, and row-readiness flags before
Dispatch begins, so correctness does not depend on allocator-provided zeros.

In `precision` / `both` mode the test also compares saved packed-weight BF16
bits against source input weights and saved routing permutations across ranks.
It prints `[SAVED_STATE] rank=... packed_topk_weights PASS rows=...`; this check
runs outside profiling and does not copy the large packed-X saved allocation.
Use random TopK weights to detect missing or incorrectly ordered state.

Like the W8A8 precision test, the CPU golden uses deterministic dense weights.
BF16 weights are generated four experts at a time and each expert is converted
to FP32 only while computing its golden output, which bounds host peak memory
without weakening the numerical stimulus.

Build the base package and BF16 kernel first:

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
```

Timer instrumentation is compile-time optional and is disabled by default.
Build and run the BF16 path with Timer V2 enabled as follows:

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1 \
  --mode precision --warmup 0 --loops 1 --enable-timer
```

The independent UDMA operator is built and run separately, so the legacy BF16
entry remains unchanged:

```bash
bash scripts/build_mega_moe_bf16_udma.sh -soc_type Ascend950
bash test/megamoe_bf16_test/test_mega_moe_udma.sh 0,1
```

The UDMA Timer build uses the same switch:

```bash
bash scripts/build_mega_moe_bf16_udma.sh -soc_type Ascend950 --enable-timer
bash test/megamoe_bf16_test/test_mega_moe_udma.sh 0,1 \
  --mode precision --warmup 0 --loops 1 --enable-timer
```

Raw files are written to `python_results/rank_<rank>_npu_time.csv`; parsed
per-core, per-expert files are written to `python_results/timer/`. The same
test command automatically writes per-rank trace JSON and a merged trace under
`python_results/timer_json/`. WAIT events remain available in the parsed CSV
but are omitted from trace JSON. JSON names include M/N/K, TopK, EP and the
operator path, for example
`rank_0_npu_time_4096_6144_3584_16_2_bf16_udma.json` and
`all_rank_trace_4096_6144_3584_16_2_bf16_udma.json`. These files can be opened
with Perfetto or `chrome://tracing`. The BF16-local `trace_parser.py` performs
this conversion and does not depend on the W4A8 test directory. Rebuilding
without `--enable-timer` explicitly resets the CMake option to OFF, so normal BF16
builds contain no cycle reads, Timer barriers, or Timer GM writes.

The same `trace_parser.py` also renders the backward pipeline timeline, via
`profile_mega_moe_bf16.py --capture-backward-timer`. After the profile finishes
it parses each `rank_<rank>_backward_time.csv` and writes
`rank_<rank>_npu_time_backward.json` plus `all_rank_trace_backward.json` under
`<profile-dir>/timer_json/` (override with `--trace-dir`). Unlike the forward
view, the backward trace keeps the wait bars — `AIC_SPIN_*` and the
`AIV_XSYNC_*` cross-rank barriers (tagged `cat=wait`) — because those sync
stalls are what a backward timeline exists to
show. Use `--timer-groups` if the kernel was launched with fewer AIC groups
than the default 32.

Run precision and performance on two devices:

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1
```

Without `--data-dir`, each run keeps its inputs, outputs, golden data, and
profiler files under
`test/megamoe_bf16_test/output/run_<timestamp>_<pid>/`.

### Reuse inputs and golden across runs

Input and golden caching is enabled by default for the ordinary BF16, MGroup
and UDMA Python test launchers. The first run generates the data; subsequent
runs with the same stimulus skip generation. Each invocation still launches
the current kernel and performs the existing precision and saved-state checks
in `precision` / `both` mode. NPU outputs and profiler results are not cached.

Repeat the same command after rebuilding a kernel:

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode both --warmup 10 --loops 100 --enable-timer
```

The terminal prints `[DATA_CACHE] MISS ... reason=...` on regeneration,
`[DATA_CACHE] READY ...` only after all expected files have been generated,
and `[DATA_CACHE] HIT ...` when reusing data. Existing `[PRECISION]`,
`[SAVED_STATE]` and `[PERF]` reporting stays in place.

The persistent cache defaults to `test/megamoe_bf16_test/output/data_cache/`.
Use `--data-cache-dir /path/on/local/disk` to change it, or append
`--refresh-data` to regenerate the current case even if its cache is valid.
`--data-dir` still selects the results root; it is not required for cache reuse.

Cache identity includes M/N/K, EP, TopK, experts per rank, seed, routing mode
and skew, TopK weight mode and application stage, receive-capacity validation,
NumPy version and the golden generator source fingerprint. Changing only a
kernel, Timer, loops, tolerances or SHMEM allocation size does not invalidate
it. A precision cache also serves performance-only runs; an input-only cache
must be regenerated with golden when precision is first requested. Historical
data without a cache manifest is not silently adopted.

Cache files are checked for exact expected byte lengths and recorded mtimes;
multi-GB weight files are not rehashed on each hit. Treat cached inputs and
their result-directory links as read-only. Missing, truncated or normally
modified files cause regeneration. Interrupted generation has no completed
manifest and cannot be reused. Per-entry OS locking serializes concurrent
generation and is released automatically if the generating process exits.

On the same filesystem, each result directory holds hard links to cached
inputs and golden, avoiding large copies. Refresh unlinks the old cache files
before generating replacements, so earlier result links keep their original
data. If hard links are unavailable, the launcher reports the fallback and
copies the files; place the cache on the results filesystem to avoid that
cost. Different stimulus keys and retained old runs can consume additional
disk space; the launcher does not delete historical results automatically.

Physical-server runs use the same preparation phase and existing cross-server
configuration check. Each host may have its own local cache path; a shared
cache requires filesystem support for the OS file locks. Each server still
checks its current local outputs. The standalone independent C++ test scripts
are separate and do not use this cache.

CPU cache regression (no NPU required):

```bash
python3 -m unittest discover -s test/megamoe_bf16_test -p test_bf16_data_cache.py -v
```

Precision-only bring-up:

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1 \
  --mode precision --data-dir /tmp/megamoe_bf16_precision
```

Print the first, middle, and last 10 flattened output values even when
precision passes:

```bash
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1 \
  --mode precision --print-diagnostics
```

The default routing pattern is exactly balanced so performance measurements
remain comparable. To test non-balanced routed-token counts, use the seeded
skewed generator:

```bash
bash test/megamoe_bf16_test/test_mega_moe_udma.sh 0,1 \
  --mode precision --warmup 0 --loops 1 \
  --routing-mode imbalanced --seed 831 \
  --print-diagnostics
```

The test prints a line such as
`[ROUTING] mode=imbalanced rank_rows=[...] expert_min=... expert_max=...`
before launching the NPU kernel. `--routing-mode random` samples distinct
experts uniformly and is generally only mildly imbalanced at large M;
`--routing-mode imbalanced` is the recommended capacity and correctness test.
Its default `--routing-skew` is `1.0`; increase it for a hotter distribution.
Every token still selects
`TopK` distinct experts, matching the normal MoE routing contract.

Performance defaults match the W8A8 test: zero warmup calls, 110 profiled
loops, and statistics over the last 50 matching MegaMoe BF16 task-duration
rows. Use `--warmup`, `--loops`, and `--repeat` to override them.

Precision continues to use the BF16 CPU golden. The W8A8-only small-operator
cascade and ops-transformer-hs archive interoperability are not run because
there is no equivalent BF16 reference path or shared archive format yet.
The fused-vs-golden comparison uses the same protected-deviation rule as
W8A8, checks expert counts exactly, and prints maximum deviation plus
first/middle/last diagnostic windows when a rank fails.

## BF16 routed-token receive capacity and SHMEM

The existing `--routing-mode` selects both the generated routing pattern and
its receive-capacity policy. The default balanced test reserves the aligned
per-rank route count; random and imbalanced tests retain the dropless
worst-case bound:

```text
balanced:          maxRecvTokenNum = align128(M * TopK)
random/imbalanced: maxRecvTokenNum =
                       align128(M * EP * min(expertPerRank, TopK))
```

The UDMA path reuses one symmetric activation area sequentially as InitRouting
scratch, Dispatch receive/GMM1 input, and Combine staging. The test computes
the required SHMEM bytes from the kernel's peer-memory layout (control area,
token counters, ptrA0, routed TopK weights, ptrD, and UDMA-only buffers).
SHMEM is selected once before `ShmemInstance::Init()` and reused by every
warmup and measured kernel invocation. Allocation uses separate floors for the
basic/MGroup and UDMA variants, and still grows if a future shape requires more:

```text
basic/MGroup floor = 2 GiB
UDMA floor         = 3 GiB
localMemBytes      = align_up(max(floor, requiredPeerBytes), 2 MiB)
```

For the default `M=4096,K=7168,TopK=16` shape, balanced capacity is 65,536
rows. `ptrA0` and `ptrD` together use 1.75 GiB; UDMA additionally uses a
896 MiB shared `ptrA1` area. The corresponding peer layouts fit the 2 GiB
basic/MGroup and 3 GiB UDMA floors. This is a Host/test allocation-policy
change only; the kernel communication and compute algorithms are unchanged.

In particular, the current UDMA Host check requires
`maxRecvTokenNum * K * sizeof(BF16) <= UINT32_MAX`. Balanced capacity for the
default shape remains below this limit as EP grows. The random/imbalanced
worst-case capacity with 4 experts per rank remains within it through EP=16
and exceeds it at EP=32. Increasing SHMEM does not bypass this length check.

During input generation, the golden utility counts the routed rows owned by
every rank and fails before launching any BF16 variant if a count exceeds
`maxRecvTokenNum`. The check runs on the CPU outside the measured NPU kernel
and therefore does not affect performance timings.

Use a separate CSV to test smaller or additional shapes. Every BF16 case must
satisfy `K % 256 == 0` and `N % 512 == 0`.

## Host Unpermute regression

The BF16, MGroup and UDMA paths share a batched-index Unpermute helper. It
preserves route order and rounds to BF16 after each addition. It uses two
input rows, two output rows and 4 KiB of route indices (60 KiB total for
K=7168). Shapes that exceed the UB budget or index capacity use the existing
three-row implementation. No build or runtime switch is needed.

The following host-only regression executes the production helper with a
randomized FIFO/event model for MTE2, Vector and MTE3. It checks BF16 output
bits, tail batches, odd TopK, inactive cores, memory bounds and event drain.
The stub include path is **only** for this test; do not add it to NPU builds.
This is not CANN compilation or hardware validation.

```bash
c++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -Itest/megamoe_bf16_test/unpermute_host_stub \
  test/megamoe_bf16_test/test_unpermute_pipeline.cpp \
  -o /tmp/test_unpermute_pipeline
/tmp/test_unpermute_pipeline
```

## Independent C++ test

The independent test bypasses `ascend_deepep`, `_C.so`, `torch_npu`, and the
Torch custom-class registration. It links the BF16 kernel library directly:

```bash
cd test/megamoe_bf16_test/megamoe_bf16_independent_test
bash build.sh
bash test_mega_moe.sh 0,1
```

For the standalone UDMA executable, use `build_udma.sh` and
`test_mega_moe_udma.sh` in the same directory.

See `megamoe_bf16_independent_test/README.md` for isolated build and data
options.

## 普通 BF16 M192 前向验证

普通前向统一使用 M192；默认每卡专家数 <8 且预估、实际首专家至少
768 行时启用首专家 Ready384，其他专家整专家就绪。MGroup、UDMA
保留 M128。保存模式不改变矩阵分块或框架接口。

普通前向的 Dispatch 每次搬运 1 行，仍使用两个 UB 槽；反向、MGroup、
UDMA 的搬运粒度不变。K7168 时，两槽合计从 56 KiB 降为 28 KiB。
GM workspace、保存偏移、控制信息和 Ready384 发布单位不变；Host
仍按原 2 行检查 UB 容量，属于保守上界。

`test_dispatch_tile_ready.cpp` 同时检查旧 M128/Ready256 和新 M192/Ready384
协议、部分尾块、重复调用清零、计数器容量及 1/2 行 Dispatch UB 布局；`test_saved_gate_up_mxfp8.cpp`
检查 M192 的量化暂存区不会覆盖原始 Gate/Up，含 mask 占用为 237 KiB。
下面命令从仓库根目录执行，属于 Host 检查，不替代 NPU 精度验证：

```bash
c++ -std=c++17 -O2 test/megamoe_bf16_test/test_dispatch_tile_ready.cpp -o /tmp/ready128
/tmp/ready128
c++ -std=c++17 -O2 -DDEEPEP_BF16_M192_FORWARD=1 \
  test/megamoe_bf16_test/test_dispatch_tile_ready.cpp -o /tmp/ready192
/tmp/ready192
c++ -std=c++17 -O2 -I. test/megamoe_bf16_test/test_saved_gate_up_mxfp8.cpp -o /tmp/saved_mxfp8
/tmp/saved_mxfp8
```

测试默认仍为 TopK16、每卡 4 专家，可用 `--topk` 和 `--expert-per-rank`
覆盖以复测不同专家数。设备参数按 rank 顺序给出物理卡号；脚本将运行时可见列表排序，
再映射到相应逻辑卡，支持 `0,1,2,3,4,7,6,5` 这样的顺序。均衡路由
容量按每专家向上取整后求和再对齐，覆盖专家数不能整除路由行数的情况。

正式性能对比必须使用同一套数据、Host、CANN 和
CATLASS 提交，分别构建原版 train 和 train 加优化，复用本 PR 的同一套测试脚本和 Host，
仅更换被测 Kernel SO。原版脚本没有新增的专家数/TopK 参数，不能把更换
测试入口造成的配置差异计入收益。关闭 Timer，先检查 NPU 空闲：

```bash
npu-smi info
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode performance --warmup 10 --loops 110 \
  --topk 16 --expert-per-rank 16 --routing-mode balanced \
  --save-gmm1-output bf16
```

`performance` 只采集性能，不代表通过 CPU Golden；精度另用 `--mode precision`
验证，MXFP8 可指定 `--save-gmm1-output mxfp8`。性能汇总兼容
`op_summary*.csv` 和 `kernel_details.csv`，只接受 NPU kernel 时间，
不使用 PyTorch `operator_details.csv` 或诊断 Timer 回退作为性能结果。

最终方案、精度范围与完整性能表见
[普通 BF16 M192 前向优化](../../docs/MegaMoe_BF16_M192前向优化.md)。


有限幅 MXFP8 的反向 mask 修复及反向测试解析修正由独立反向 PR 提供。
本前向 PR 不修改反向 Kernel；显式 MXFP8 保存且有限幅时，完整梯度验证
需配合该反向修复；PR38 已合入 train。

反向测试的保存 Gate/Up 参考使用限幅前原始值（MXFP8 使用反量化值），
梯度参考仍应用限幅。该修正只影响 `--check-intermediates` 的参考比较，
不改变 Kernel、容差或保存格式。可用下面命令验证有限幅保存和梯度：

```bash
npu-smi info
for SAVE in bf16 mxfp8; do
  bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3,4,5,6,7 \
    --m 513 --k 512 --n 1024 --expert-per-rank 3 --topk 6 \
    --routing-mode random --save-gmm1-output "$SAVE" \
    --swiglu-limit 0.5 --check-intermediates --watchdog-secs 120
done
```
