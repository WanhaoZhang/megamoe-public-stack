# 普通 BF16 前向：单 AIV SwiGLU 解耦与 EP4/EP8 验证

## 行为与开关

普通 BF16 forward 增加单 AIV 的 GM 解耦路径：GMM1 写出原始 BF16 Gate/Up，AIV0 从 GM 读取并执行原有 SwiGLU、乘 TopK 权重和 Hp 写回。GMM1 不再因逐 tile 的 UB 握手而等待 AIV0。AIV1 的 Dispatch/Combine、GMM2、backward 算法均保持原流程。

默认 `auto`：**H<3584开启；H=3584仅mxfp8开启、none/bf16关闭；H>3584关闭**。off/on仍可强制选择。此规则针对新版直接解包＋批量乘权SwiGLU；只决定执行路径，不改变保存模式、舍入顺序或输出精度。它是已测场景下的工程默认值，不是所有M/FFN/路由分布的理论交叉点。完整前后向组合说明见[向量与路由搬运优化](BF16_VECTOR_ROUTING_OPTIMIZATION.md)。

构建时可强制选择，test/run 无需额外参数：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --swiglu-decouple off
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --swiglu-decouple on
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --swiglu-decouple auto --enable-timer
```

对应 CMake 参数 `ASCEND_DEEPEP_BF16_SWIGLU_DECOUPLE=off|on|auto`。不引入 Python 接口参数，不改变 kernel 参数 ABI、权重布局或 saved-state 格式。请通过构建脚本一起更新 kernel SO 和 Host 扩展；不要把新 kernel SO 单独配给旧 Host 扩展。

## 核心路径与同步

1. `mega_moe_training_workspace.h` 的 `ForwardUsesGmSwiglu()` 按hidden与已解析保存模式固定off/on/auto策略及3584边界。普通 forward 在调用开始选择一次模板路径，没有任务遍历、设备同步或按专家重新选择策略。
2. `mega_moe_bf16_catlass_impl.h` 的 `ForwardGmTileCopy` / `GroupMatmulSwigluGm()` 保留现有 GMM1 分核、输入 ready 和 Gate/Up 成对计算，仅把输出从配对 AIV0 的 UB 改为 GM。该路径不发布或等待逐 tile 的 Cube/Vector UB 握手。
3. 每个 AIC 在当前专家的 GM 写入完成后 `PipeBarrier<PIPE_ALL>()`，再增加 `groupFlagList[1]`。没有当前专家任务的 AIC 也参加发布。AIV0 等待该计数达到实际 AIC 数，随后只处理原调度分给自己的 tile。
4. `groupFlagList[0]` 仍是原有 Hp 完成计数。`ResetWorkSpaceFlagList()` 每次调用复位整条控制 cache line；已有 Dispatch 前的 `SyncAll<false>()` 确保复位先于任何发布/消费。无需新增 flag 内存或 Host 清零。
5. `LoadRawTile()` 等待上一 tile 的向量计算和 MTE3 完成，再把当前 Gate/Up 完整搬到 UB。原 `Run<SaveMode>()` 的数值操作、SwiGLU BF16 舍入、乘权及 Hp 写回顺序不变。mxfp8 的保存、scale 与 clip masks 在修改 UB 原始值之前生成。
6. GMM2 仍按对应专家的 Hp 计数等待。AIC 完成自己的全部 GMM1 任务、配对 AIV1 释放 Dispatch UB 后，可以推进 GMM2；不新增“等全部 SwiGLU 结束”的同步。AIV0 无需协助 Dispatch 或 Combine。

这不是全核同步方案，也没有引入 dual AIV 或 flat 向量循环实验。

## 临时内存与生命周期

| 保存模式 | raw Gate | raw Up | 新增 transient |
|---|---|---|---|
| bf16 | 已有持久化 Gate 平面 | 已有持久化 Up 平面 | 0 |
| none | 复用 Hp 平面 | 追加一个 BF16 Up 平面 | `align512(max_recv_tokens × FFN × 2)` |
| mxfp8 | 复用 Hp 平面 | 追加一个 BF16 Up 平面 | 同 none；压缩保存区不变 |

追加区从原 `ForwardDispatchTileReadyLayout.totalBytes` 开始，不移动 Hp、控制 flags 或 ready counters。回退到原路径时不分配该追加区，bf16 模式也不新增大块内存。Host 保留原总容量检查，并检查追加区与原总量相加不超过 int64 容量。

复用 Hp 的前提是当前专家所有 AIC 已完成 raw Gate 写入；每个消费者完整载入自己独占的 tile 后，才把 Hp 写回同一位置。其他消费者的 tile 不重叠，没有在原始 Gate 被覆盖后重新读取它。raw Up 独立，避免与 Hp 写回竞争。即使内存复用，GM 写出和读回 Gate/Up 的流量仍存在，性能收益已经包括其代价。

Host通过`MegaMoeBF16SwigluDecoupled(H,savedMode)`优先查询实际SO的`mega_moe_bf16_swiglu_decoupled_for_saved_mode`，再调用`ForwardSwigluExtraBytes()`，与kernel选择一致。兼容旧SO时回退旧hidden-only查询；两种符号都没有时按原UB路径分配。新SO保留旧hidden-only查询并返回各保存模式容量需求的并集，旧Host在边界最多多分配，不会少分配。

UDMA、MGroup、backward 和 SHMEM 布局不采用这项新策略。

## PR41旧SwiGLU的历史性能证据

以下为 2026-10-20 的实验数据，源基线 `train@0bfd1f24be57e89f2fb95a6b2f3d835e7304e9a3`。当时PR41仅保留验证过的单AIV路径；以下是旧SwiGLU数据，不代表新版SwiGLU的解耦差异或当前auto阈值。指标是每卡 50 次 forward Event 中位数的最大值，warmup10/loops50、无 Timer、`swiglu_limit=0`，包括 forward 接入范围；不是单纯 GMM 时间，也不是逐次 global max 的 p50。

K3参考：M8192/H3584/FFN3072、EP4、每卡7专家、TopK16、32 AIC、严格均衡 max_recv131072、保留 dispatch_output。改变 H 时其他参数不变。

| H | 保存模式 | 原路径 ms | 解耦 ms | 收益 | 新增临时区 |
|---:|---|---:|---:|---:|---:|
| 3584 | none | 29.022 | 24.318 | 16.21% | 768 MiB |
| 3584 | bf16 | 29.345 | 24.334 | 17.08% | 0 |
| 3584 | mxfp8 | 30.618 | 24.328 | 20.54% | 768 MiB |
| 4096 | bf16 | 31.684 | 27.491 | 13.24% | 0 |
| 5120 | none | 35.882 | 33.658 | 6.20% | 768 MiB |
| 5120 | bf16 | 36.081 | 33.673 | 6.67% | 0 |
| 5120 | mxfp8 | 37.462 | 33.705 | 10.03% | 768 MiB |
| 6144 | bf16 | 40.460 | 39.721 | 1.83% | 0 |

H4096/H5120 bf16 为两轮同机对照；H5120 none 按用户要求复用了另一台同型号设备的有效原路径，明确属于跨机对照。新增 mxfp8 与 H6144 bf16 为单对同机对照，不能声称独立重复实验下的收益保证。H6144 两条路径统一预留4 GiB SHMEM，因为两块通信 payload 已占满3 GiB，另有控制区；初始化在计时外。

EP8常测场景：M4096/H7168/FFN4096、每卡4专家、TopK16、max_recv65536。auto 回退原路径，各保存模式相对原路径变化小于0.05%，无额外临时区。强制on时 none 约退化0.57%、bf16约退化0.64%；mxfp8单对改善0.47%，但 none/mxfp8 会额外占用512 MiB。

当时基于旧SwiGLU，统一默认阈值先取5120；当前新版SwiGLU已改用文首3584规则。H6144 bf16 的1.83%收益保留，可强制on，但不把它外推为尚未测过的6K none/mxfp8同样值得开启。小M、极端不均衡和其他M/FFN组合的最优分界尚未穷举。

## 历史PR41验证范围

已有实验完成三种保存模式的 CPU golden 前反向、transient 整体填0xFF，以及改变输入的多次调用回归。小shape比较 Y/dX/dTopK/dW1/dW2；大shape核对Y逐位一致、专家计数及梯度finite，不声称完整大shape梯度数值golden。带clamp的精度检查使用 `swiglu_limit=.05`，上述性能使用0，二者口径不同。

正式补丁已用清理后的源码重新验证：auto/on/off 和 auto+Timer 四种 Ascend950 构建、Host 扩展均通过；实际 SO 在H3584/5120/5121/6144/7168的策略查询符合预期。none/bf16/mxfp8各通过四卡CPU golden前反向精度检查（H256、M1792、每卡7专家、TopK16、clamp=.05）。

正式auto的H5120 mxfp8代表场景为33.677ms（无Timer）/33.798ms（有Timer），与此前实验版33.705ms一致，未发现性能回退；两次均通过每卡6轮改变输入的小shape全输出/梯度逐位检查，以及大shapeY逐位、计数和梯度finite检查。Timer版保存了全部四卡图。没有重跑既有完整性能矩阵，也未新增backward性能结论。

本地Python workspace/schedule共32项通过；C++策略/容量测试按off/on/auto分别通过，原ready协议测试按M128/M192分别通过，`git diff --check`通过。临时控制器、debug.md、私有实验宏、设备清单与原始日志不随PR提交。

Host-only 回归：`test_forward_swiglu_policy.cpp` 验证 off/on/auto 边界、三种保存模式与不同容量/专家数的追加区；`test_dispatch_tile_ready.cpp` 验证原控制布局、尾块、到达顺序和复位；Python workspace/schedule 回归保留。它们不替代 NPU 内存顺序验证。

## 本轮组合验证

新版直接解包和批量乘权同时作用于解耦及UB路径，保留原来的数学公式、clamp与BF16舍入；原有同步和临时区复用仍如上。正式组合还包含路由Gather和反向向量优化，见[核心代码契约与验收口径](BF16_VECTOR_ROUTING_OPTIMIZATION.md)。历史PR41的auto/on/off、Timer及0xFF检查不冒充本轮组合验证；本轮独立记录测试与性能。
