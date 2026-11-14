# MegaMoe BF16 前向训练状态拆分与反向适配

> 2026-09-10 当前状态：PR12 已恢复训练入口；三版前向已追加原始 BF16 gate/up 保存区，
> 当前还支持 none/mxfp8/auto，默认仍为 bf16；显式 MXFP8＋截断已追加三个掩码 plane。
> 原字段偏移不变，当前反向继续重算 GMM1。下文 9 月 7 日的入口保护和布局是历史迁移说明。
> 当前保存格式和设备反馈以 [MXFP8 saved-state](MegaMoe_BF16_MXFP8_saved_state第一版.md) 为准。
> 当前布局、初始化范围、内存增量及反向读取切换以
> [gate/up 持久化与反向接入](MegaMoe_BF16_gate_up持久化与反向接入.md) 和
> [反向适配与双机测试参考](MegaMoe_BF16_反向适配与双机测试参考.md) 顶部为准。

更新日期：2026-09-07。本文覆盖普通 BF16、MGroup 和 UDMA 当前实现。
反向开发交接先读 [反向适配与双机测试参考](MegaMoe_BF16_反向适配与双机测试参考.md)，
其中记录最新双机前向反馈和反向测试可复用的入口；本文保留详细布局与逐文件迁移契约。
相关提交：`9627e93` 统一三输出和 saved-state；`8323352` 让 UDMA 路由元数据
直接写持久区；`268af98` 删除 UDMA 临时区中的旧元数据空位。

适配核对基线：前向分支 `deepep_moe_xx` 的 saved-state 实现，以及
`origin/train@56462ad2bdda3fd2cd244509e15569c4c67a5fbb`（2026-09-07 集成前重新拉取确认）。
普通 BF16 TopK weight gather 的 `66d81a1` / `cc0a446` 已收到用户验证通过反馈，
不改变本文持久区 ABI。后续合入若 train 再前进，应重新检查第 4 节的调用点。

**交付状态：本次合并整合 `deepep_moe_xx@8d4d821` 与 `train@56462ad`，
代码、适配文档一起交付。前向接口已更新；反向代码保留，saved-state 适配待完成。
过渡期只开放 raw 前向，不代表新接口下的训练已经验证。**

过渡期入口：

| 入口 | 本次集成状态 |
|---|---|
| 普通 BF16 / MGroup / UDMA `compute` | 三输出 `(y, counts, training_saved_state)` |
| 三个版本 `compute_with_timer` | 上述三输出后附加 timer |
| 普通 BF16 `compute_train` / `MegaMoeBF16Train` | Host 明确抛出 `training is temporarily unavailable`，不 launch 旧训练前向 |
| 普通 BF16 `backward` / `backward_debug` / `backward_with_timer` | 同样拒绝，不能向其传入新 saved-state |
| MGroup / UDMA 训练封装 | 没有新增可用反向入口；共享 saved-state 不等于完成反向接入 |

`csrc/ops/mega_moe_bf16.cpp` 的两个训练 Host 入口为过渡保护点。
旧 `MegaMoeBF16Forward` 的重复分配/launch 实现已替换成报错入口；
`MegaMoeBF16Backward` 的原实现和 kernel 优化保留在保护之后，供后续适配。
不能只删除报错：必须先完成第 4 节的整条调用链迁移和第 7 节验收。

`WorkspaceInfo(saved, transient, tiling)` 服务新前向；
`ForwardWorkspaceInfo(base, tiling)` 仅保留旧连续 workspace 的反向地址视图，
它继承公共字段以兼容现有 `Params`，**不支持解析新 saved-state**。
`BackwardWorkspaceInfo` 保留 train 的临时区、partial reduction 和 flags 布局。
旧前向的重复计数快照未叠加到新实现上：计数已由 `SaveForwardState()` 保存。

本次源码修正已补齐 train 反向 epilogue 必需的 packed TopK 权重：三个版本
均由 MTE 直接搬入 saved-state；UDMA 激活仍走原 UDMA/SDMA 通路。反向直接
读取持久权重，不增加重排或跨 rank 重建。用户已反馈合并前的修正验证无明显问题；本次合并后的目标设备编译和前向回归仍需重新执行。

## 1. 目标与边界

普通 BF16 前向不再返回独立 `permutedInput`，也不再执行
`workspace.ptrA -> permutedInput` 的整块复制。前向内存物理拆成：

```text
training_saved_state          forward_transient_workspace
  ptrA / Xpacked                ptrA2
  cumsum                        GMM1 -> GMM2 flags
  expandedRowIdx                Dispatch -> GMM1 flags
  expandedRowIdxGather
  sumBeforeRank
  token-count snapshot
  dispatched TopK weights
  backward progress
```

普通 BF16、MGroup 和 UDMA 的 Python `compute()` 均返回：

```text
(y, expertTokenNums, training_saved_state)
```

`compute_with_timer()` 在上述结果后额外返回 `timer`。saved-state 为一维
UInt8 Tensor，内部按共享布局解释为 BF16 输入与 Int32 元数据。

`training_saved_state` 必须由训练框架持有到对应 backward 完成；内部
`forward_transient_workspace` 只服务本次 forward。Host launcher 在返回前同步
stream，因此临时 Tensor 可以在算子返回后释放。MGroup 同样让 Dispatch 直接写
持久输入；UDMA 的激活接收区属于对称 SHMEM，仍需在复用前快照，见下文。

本次保留 train 的反向算法实现并增加 Host 过渡保护。下文的反向接口和布局适配仍待实现。

## 2. 前向数据生命周期

```text
Host预先分别分配saved与transient两块allocation
        |
        v
Dispatch直接写training_saved_state.ptrA（Xpacked）和packed TopK权重
        |
        v
GMM1直接读取同一ptrA
        |
        v
SwiGLU产生forward_transient_workspace.ptrA2
        |
        v
SaveForwardState只快照可变SHMEM token-count表
        |
        v
GMM2读取ptrA2，执行Combine
        |
        v
forward返回training_saved_state，transient结束生命周期
        |
        v
backward按共享布局解析training_saved_state.ptrA等字段
```

普通 BF16 后续阶段不会覆盖 `ptrA`，因此 forward 与 backward 共用这一份
Xpacked，不再需要第二个 `[maxRecvTokens,K]` BF16 Tensor。

以上是普通 BF16 的保存顺序。MGroup 有自己的 Wave 调度，在 Wave 计算结束后、
重置 SHMEM token-count 表之前保存计数快照；它同样复用普通 BF16 的持久布局。

UDMA 的特殊生命周期为：

```text
对称SHMEM ptrA1：InitRouting scratch → Dispatch接收/GMM1输入
                                              |
                                              v
                            有效输入快照到training_saved_state.ptrA
                                              |
                                              v
                                ptrA1复用于Combine发送暂存
```

UDMA 远端写入依赖对称 SHMEM，且 Combine 会覆盖 `ptrA1`，所以当前方案仍需
保留 packed X 快照。它的四类路由元数据、packed TopK 权重已经直接写入
saved-state，不再复制；
`SaveForwardState()` 目前只复制有效 packed X 和可变 SHMEM token-count 表。

## 3. 布局定义

Host 和 device 共用：

```text
kernels/mega_moe_bf16/include/mega_moe_training_workspace.h
```

其中 `TrainingSavedStateLayout` 定义三条路径共用的持久区。
`ForwardTransientWorkspaceLayout` 定义普通 BF16/MGroup 的基础临时区，
`UDMAForwardTransientWorkspaceLayout` 定义 UDMA 的紧凑临时区。
Host、Device 和 backward 应使用同一份布局定义，避免分别手写偏移。

### 3.1 training_saved_state

- `ptrA`：Dispatch 后的 Xpacked；
- `ptrcumsumMM`：各本地 expert 按来源 rank 累计的接收行数；
- `expandedRowIdx`、`expandedRowIdxGather`：正反路由索引；
- `ptrSumBeforeRank`：rank 前缀；
- `ptrSavedTokenPerExpert`：从可变 SHMEM 快照的完整 token-count 表；
- `ptrDispatchedTopkWeights`：与 `ptrA` 同一行序的 BF16 packed TopK 权重；
- `ptrBwProgress`：反向诊断进度区，前向分配时清零；真正的反向握手 flags
  仍位于 backward 临时区，不应与该进度区混用。

此前代码将 `ptrDispatchedTopkWeights` 留在前向临时区（UDMA 在 SHMEM），
本次已修正这个持久状态缺口。**当前契约保存目标 rank 的 packed TopK 权重，
供反向直接读取；原始 `topkWeights` 仍由 autograd 单独持有。**
原始本地权重不能替代已经跨 rank Dispatch 后、与 Xpacked 对齐的权重向量。

packed X 按 `maxRecvTokens` 分配容量，有效行数需按路由计数解释。UDMA 仅
复制有效输入前缀，不能把未使用容量当作有效数据。当前 UInt8 保存区没有独立的
自描述版本头，forward/backward 必须使用一致的布局与 shape 参数。

#### 当前字段与偏移契约

定义 `M` 为本 rank 输入 token 数，`K` 为 hidden size，`T=TopK`，`P=EP`，
`E` 为每 rank expert 数，`R=maxRecvTokens`，`S=AlignUp(P*E,128)`。
`A_b(x)=AlignUp(x,b)`；下表大小、offset 均以**字节**计，`S` 本身以 Int32
元素计。实际代码必须调用 `TrainingSavedStateLayout(M,K,T,P,E,R)`。

| offset 成员 / device 字段 | 类型与逻辑范围 | 预留字节数 | offset 计算 |
|---|---|---:|---|
| `ptrAOffset` / `ptrA` | BF16 `[R,K]`，仅有效 packed 行可读 | `A_512(2*R*K)` | `0` |
| `cumsumOffset` / `ptrcumsumMM` | Int32 `[P,E]` | `A_512(4*P*E)` | 上一段末尾 |
| `expandedRowIdxOffset` / `expandedRowIdx` | Int32，前 `M*T` 项有效 | `4*A_256(M)*T` | 上一段末尾 |
| `expandedRowIdxGatherOffset` / `expandedRowIdxGather` | Int32，前 `M*T` 项有效 | `4*A_256(M)*T` | 上一段末尾 |
| `sumBeforeRankOffset` / `ptrSumBeforeRank` | Int32 `[P,E]` | `4*S` | 上一段末尾 |
| `savedTokenCountsOffset` / `ptrSavedTokenPerExpert` | Int32 `[P,S]`，每来源行前 `P*E` 项为计数 | `A_512(4*P*S)` | 上一段末尾 |
| `dispatchedTopkWeightsOffset` / `ptrDispatchedTopkWeights` | BF16 `[R]`，与有效 packed X 行一一对应 | `A_512(2*R)` | 上一段末尾 |
| `backwardProgressOffset` / `ptrBwProgress` | Int32 `[256]`，诊断阶段码 | `A_128(4*256)=1024` | 上一段末尾 |
| `totalBytes` | UInt8 Tensor 总长度 | — | progress 段末尾 |

不要把所有段都重新按 512 字节对齐；上表刻意保留了各段不同的 padding 规则。
`N` 不参与 saved-state 大小计算，但仍参与 forward/backward 临时区大小计算。

#### packed TopK 权重保存方式

新权重段放在 `savedTokenCounts` 之后、progress 之前，已有输入和路由字段的
offset 保持不变，progress 向后移动 `AlignUp(2*R,512)` 字节。默认 `R=65536`
时 saved 增加 131,072 字节（128 KiB），由 940,054,528 变为 940,185,600 字节。

- 普通 BF16 / MGroup：Dispatch 的目标权重指针直接映射到 saved 段，前向
  SwiGLU 与反向共用同一份。原 transient 权重槽已移除，追加 ready 区通过
  公共布局自动前移；saved + transient 总容量不增加。
- UDMA：`DispatchTopkWeightsMTE()` 在接收端从各来源 rank 的
  `ptrRoutedTopkWeights` 通过 `DataCopyPad` 拉取小块权重，直接写 saved，
  按 `[expert][source rank][row]` 排列。激活仍远端 UDMA、本地 SDMA；
  权重不再进入 UDMA/SDMA 队列，也不需要保存后再复制。
- UDMA 原 SHMEM packed 权重接收槽已删除，`ptrD` 前移同样的 128 KiB；
  Host `CalcExtraBufferSize()`、Device `PeermemInfo`、测试容量公式一致更新。
  默认 SHMEM 初始化额度仍为 3 GiB，不能把额度不变误解为 device 偏移未变。
- UDMA 本地所有 AIV1 完成 MTE 写入后，经 `MTE3_S` 和计算前已有的
  `SyncAll<false>()` 保证 SwiGLU 可读。UDMA/SDMA signal 此后只保证激活
  到达；源权重先于计数交换发布，来源 SHMEM 的下次复用仍遵守尾部同步。

普通 BF16/MGroup 的 Host 和独立 stub 原本已调用公共 layout，无需再各写
一套大小公式。外部仍返回三 Tensor；旧 Host/SO 必须配套重编译。

路由解释以当前 kernel 的 `TokenPerExpertLayout()`、`GetCumsumForMMAIV()`、
`CrossRankSyncCntPerExpert()` 和 `GatherRoutedTopkWeights()` 为依据：

- 计数 `C[src,dst,e]` 位于 `src*S + dst*E + e`，表示来源 rank 的 token
  路由到目标 rank 上第 e 个 expert 的行数。不能用本地 `expertTokenNums[E]`
  替代完整表，因为那会丢失来源边界。
- 本 rank 为 `r` 时，`cumsum[s,e] = sum(C[u,r,e], u=0..s)`。
  `groupList_` 从最后一个来源行 `(P-1)*E` 开始。实际 expert 分组解释须继续
  遵守 tiling 的 `groupListType`；不要擅自把 per-expert count 改成 expert 前缀。
- `expandedRowIdx[sortedRoute]` 是本地按路由排序的位置到原始
  `token*T+slot` 的映射；`expandedRowIdxGather[token*T+slot]` 是逆映射。
  这两个索引属于**来源 rank 的本地路由序列**，不是目标 rank 的 packed X 索引。
- `sumBeforeRank[src,e]` 给出来源 rank 的排序序列中、本 rank 第 e 个 expert
  的起点；它用于跨 rank slice 寻址，不是单个全局标量前缀。
- 当前 raw 输出不再单独返回 `actualRecvTokens`；有效行数从同一 tiling 约定
  下的 counts/groupList 推导，并按实际容量处理。未使用容量、padding 和空
  expert 不能进入 GEMM、权重重排或梯度归约。

#### 生命周期与所有权

| 数据 | 持有者与可用时间 | backward 可否修改 |
|---|---|---|
| saved `ptrA`、路由/计数内容及 packed 权重 | 每次 forward 独立 Tensor，由对应 autograd ctx 持有到 backward 完成 | 按只读使用，不能当重计算 scratch |
| saved progress 1024 字节 | 同一 saved Tensor；每次 backward launch 前复位 | 允许写诊断码，详见 4.5 |
| 原始 `topk_weights/w1/w2/counts` | autograd 独立保存，与 saved-state 对应同一次 forward | 不得在对应 backward 前更新权重或覆盖 Tensor |
| forward transient | 本次 forward 内部持有，launcher 返回前同步 stream | 返回后不可访问，不能把裸指针留给 backward |
| backward workspace | 每次 backward 分配/正确复位，持续到该调用完成 | 存储重计算、梯度暂存和握手 flags；不承担 packed 权重重建 |
| SHMEM | 调用者维持实例存活；多个 forward/backward 复用 | 依照既有通信和恢复/清零协议使用，不作为持久状态替代品 |

多 micro-batch 独立保存状态不等于支持同一 SHMEM 上并发执行内核；仍须维持
所有 rank 一致的调用顺序和现有 stream/barrier 约束。

### 3.2 forward_transient_workspace

普通 BF16/MGroup 的基础临时区包含：

- `ptrA2`：加权 SwiGLU 输出/GMM2 输入；
- `ptrFlagSwiGluToGmm2`；
- `ptrFlagDispatchToGmm1`。

普通 BF16 在基础临时区后通过 `ForwardDispatchTileReadyLayout` 追加 256 行
粒度的 Dispatch-ready 计数。MGroup 追加自己的 M-group ready 区；两者不能
混用追加区的大小和偏移。

backward 不再长期持有 `ptrA2` 和 forward flags；packed weight 已按 3.1 节
搬入持久区，不随 transient 释放。
基础拆分前后的容量守恒检查不包含后来追加的 ready 区，不能直接当成当前总量。

### 3.3 UDMA 紧凑临时区

`UDMAForwardTransientWorkspaceLayout` 仅包含 `ptrA2`、两类同步 flags 和
Combine-ready 计数。路由元数据和 dispatched TopK weights 均在 saved-state
中分配；来源排序暂存 routed weights 仍在 SHMEM，供 MTE 拉取。

Torch Host、独立 CSV stub 与 Device `WorkspaceInfo` 共用这份紧凑布局。
原四类元数据空位已删除，后续 flags 和 Combine-ready 偏移同步前移。

默认 `M4096/N8192/K7168/EP8/TopK16/每rank 4 experts`，均衡接收容量
65,536 行时，每 rank 的容量如下：

| 项目 | 字节数 |
|---|---:|
| training saved-state，补齐 packed 权重前 | 940,054,528 |
| training saved-state，当前 | 940,185,600 |
| UDMA transient，缩减前 | 537,404,928 |
| UDMA transient，当前 | 536,879,616 |
| 删除的空位 | 525,312（513 KiB） |
| SHMEM 算子布局需要，原权重接收槽保留时 | 2,818,900,992 |
| SHMEM 算子布局需要，当前 | 2,818,769,920 |
| 默认 SHMEM 额度 | 3,221,225,472（3 GiB） |

`268af98` 当时只缩减 UDMA transient，saved-state 和 SHMEM 容量不变。
本次 packed 权重持久化是另一项改动，已同步移除 SHMEM 中的旧接收槽。
表中不包含输入、
权重、输出和运行时的全部设备内存。其他 shape 和路由模式必须使用实际传入的
`maxRecvTokens` 重新计算，不能写死上述大小。

## 4. 反向分支配套修改

本节左侧描述合并前 train 的旧行为，右侧为待实现契约。集成版本保留这些训练入口但明确报错，
`compute_train/backward` 尚未迁移；不能将下面伪代码当成可直接运行的 API。

### 4.1 新旧接口总表

| 入口 | train 旧接口 | 迁移目标 |
|---|---|---|
| raw `compute` | `(y,counts,permuted_input,actual_recv_tokens,saved_meta)` | 当前前向已实现 `(y,counts,training_saved_state)` |
| raw `compute_with_timer` | 上述五项再加 `timer` | 当前前向已实现三项再加 `timer` |
| `compute_train` / Host `MegaMoeBF16Forward` | `(y,counts,permuted_input,forward_workspace)` | 接到新前向三输出；不再保留旧整块 workspace |
| `backward` | 输入包含 `counts,permuted_input,forward_workspace,max_recv_tokens` | 改为 `counts,training_saved_state,max_recv_tokens`；其余输入不变 |
| `backward_debug` / `backward_with_timer` | 同一套旧状态参数 | 同步切换；仍分别额外返回 backward workspace / timer |
| `MegaMoeBF16Train.forward` | 对用户返回 `(y,counts)` | 对用户返回值保持不变，状态只由 ctx 持有 |

普通 backward 的返回顺序仍为 `(dx,dtopk,dw1,dw2)`。原始 `saved_meta`
和旧 `forward_workspace` 均不能仅改名后冒充 `training_saved_state`。

### 4.2 Python autograd

将：

```python
y, counts, permuted_input, forward_workspace = op.compute_train(...)
ctx.save_for_backward(
    topk_weights, w1, w2, counts, permuted_input, forward_workspace
)
```

改为：

```python
y, counts, training_saved_state = op.compute_train(...)
ctx.save_for_backward(
    topk_weights, w1, w2, counts, training_saved_state
)
```

必须同时修改 `ascend_deepep/__init__.py` 的 `_MegaMoeBF16TrainFunction`：

- forward 解包、`ctx.save_for_backward`、backward 的 `ctx.saved_tensors` 解包；
- `backward`、`backward_debug`、`backward_with_timer` 三条分支的实参；
- `diagnostic_holder` 删除 `permuted_input`，将旧 `forward_workspace` 键改为
  `training_saved_state`，并同步修改读取它的 watchdog；
- 保留 `ctx.mark_non_differentiable(counts)`、原始 TopK weights、SHMEM 引用、
  `max_recv` 及 autograd 输入梯度位置。内部状态由六项变五项，不改变 Function
  原本九个输入对应的梯度返回槽。

诊断 holder 的 `backward_workspace`、`backward_timer` 属于另一类输出，不要
随前向字段一起删除。对外训练 wrapper 仍返回 `y, counts`。

### 4.3 Host C++、Torch class 与 launcher

- `compute_train()` 适配所调用 BF16 变体的三 Tensor 结果；
- 删除独立 `permutedInput` 的分配和参数传递；
- backward 参数改为一个 `training_saved_state`；
- 校验与输入为同一 NPU、连续、一维、dtype 为 UInt8，以及字节数等于相同 shape 下的
  `TrainingSavedStateLayout::totalBytes`；
- forward/backward 都 include 公共布局头，不复制偏移公式。

合并前 train 的 `MegaMoeBF16Forward()` 独立分配并返回旧 `forwardWorkspace`；
本次已替换为明确报错的过渡入口。**只改 raw `MegaMoeBF16()` 不会自动改变 compute_train**。两条 Host 前向入口
应统一使用新三输出实现，避免留下第二套旧生命周期。

删除旧 `MegaMoeBF16ForwardWorkspaceLayout(forwardBytes)` 对持久状态的大小校验
和 progress 寻址；保留并更新 `MegaMoeBF16BackwardWorkspaceLayout`。字节数
相同只能验证容量，不能鉴别无版本头的旧/新 ABI；Python、Host、launcher 和
device 必须作为同一版本重新编译加载。

launcher 的声明、定义、动态 kernel 函数指针类型、导出 `_kernel_do` 签名和
device `Init` 都要一起把两条旧状态指针换成一条新状态指针。保留
`backward_workspace`、timer、tiling、通信上下文和现有前后 stream 同步/rank
barrier。使用旧 SO 配新 Host 不能靠 Tensor 检查补救。

### 4.4 Backward device 视图与 packed TopK 权重

删除：

```cpp
forwardWs_.ptrA = permutedInput;
```

同时替换旧 `ForwardWorkspaceInfo(forwardWorkspace,tiling)` 构造。它仍按
`ptrA + ptrA2 + metadata + flags + packed weights` 的旧连续布局推算地址，
不能拿新 saved-state base 调这个构造函数。

从公共 layout 显式建立只包含持久字段的 backward 视图，例如：

```cpp
// 迁移伪代码：stateBase 为 UInt8/GM_ADDR 字节地址，t 为本次 tiling。
const MegaMoeBF16Training::TrainingSavedStateLayout saved(
    t.m, t.k, t.topK, t.epWorldSize, t.expertPerRank, t.maxOutputSize);
forwardWs_.ptrA = stateBase + saved.ptrAOffset;
forwardWs_.ptrcumsumMM = stateBase + saved.cumsumOffset;
forwardWs_.expandedRowIdx = stateBase + saved.expandedRowIdxOffset;
forwardWs_.expandedRowIdxGather = stateBase + saved.expandedRowIdxGatherOffset;
forwardWs_.ptrSumBeforeRank = stateBase + saved.sumBeforeRankOffset;
forwardWs_.ptrSavedTokenPerExpert = stateBase + saved.savedTokenCountsOffset;
forwardWs_.ptrDispatchedTopkWeights = stateBase + saved.dispatchedTopkWeightsOffset;
forwardWs_.ptrBwProgress = stateBase + saved.backwardProgressOffset;
```

若仍把该视图赋给 `params_.workspaceInfo`，应同步检查结构体类型兼容和所有
被访问字段；不能保留未初始化的 transient 指针。之后：

- GMM1 重计算和 `dW1` 继续读取 saved state 内的 `ptrA`；
- token count 从 `ptrSavedTokenPerExpert` 恢复；
- cumsum、两套路由索引和 rank prefix 均从 saved state 读取；
- 原始 `topkWeights` 仍由 autograd 单独保存并传给 backward；
- epilogue 的 `packedTopkWeightsGmAddr` 读取 saved 内的 BF16 权重段；
- 不得访问已经释放的 `forward_transient_workspace`。

**不能遗漏 packed TopK 权重：这不是单纯的字段改名。** train 当前
`MegaMoeBF16Backward::Init()` 将 `forwardWs_.ptrDispatchedTopkWeights`
传给 `BlockEpilogueSwigluBackwardBF16`。后者在 P1 读取该 BF16 向量，计算
`Hp=p*H` 和 `dH=p*dHp`。本次前向已将它移入 saved-state，反向必须使用
新 offset。`params_.probsGmAddr=topkWeights` 本身不会替代这段 packed 数据。

反向直接映射 3.1 节新增字段，保留 epilogue 的 early-TopK 数学语义与乘权位置；不要求 backward
增加重排、权重通信或额外乘权。验证必须使用各 rank 不同的随机权重，检查
saved packed 权重的位值和行序；全 1 权重不能检出遗漏/错序。

### 4.5 progress 清零、诊断与重复调用

train Host 目前清零旧 `forwardLayout.progressOffset`，device 将
`forwardWs_.ptrBwProgress` 解释为 Int32 `[256]`，按 `GetBlockIdx()%256`
记录阶段码。迁移后每次 backward launch 前只清零新 progress 段：

```cpp
// Host 迁移示例：trainingSavedState 是一维 UInt8，所以 narrow 参数为字节。
const uint64_t progressBytes = MegaMoeBF16Training::AlignUp(
    MegaMoeBF16Training::kBackwardProgressElements * sizeof(int32_t), 128);
trainingSavedState.narrow(0, savedLayout.backwardProgressOffset,
                         progressBytes).zero_();
```

以上延续当前 train 的 Host 清零策略，不是对新 ABI 的已完成验证。清零必须
位于 kernel 可见的 stream 顺序中；不能清零整个 saved-state，否则输入和
路由全部丢失。backward 临时区中的 `ptrBwFlagDispatchToGmm2Grad`、
`ptrBwFlagAiv1Ready`（包含 AIV0 ready）和 `ptrBwFlagHDone` 仍分别初始化。

`test_mega_moe_backward.py` watchdog 的 holder 键及旧 workspace 注释需要
更新；按 `backwardProgressOffset` 读取 1024 字节并解释为 Int32。当前版本
progress 恰好仍在最后 1024 字节，但新实现不要用旧 forwardBytes 公式寻址。

Host 对保存 Tensor 的原地清零涉及 autograd version counter。train 的
`test/bigop_fused/test_speed.py` 已为此使用每轮新 forward 的计时方式；适配
时保留该语义，不宣称同一图 `retain_graph=True` 多次 backward 自动可用。
若要支持重复 backward，应另行设计并验证 progress 的独立所有权/写入策略。

### 4.6 文件级适配清单

以下文件已随 train 合并进入当前代码树；后续在这些仓内路径完成反向适配。

| 文件 / 入口 | 必须完成的修改 |
|---|---|
| `ascend_deepep/__init__.py` | autograd forward/backward 解包、三个 backward 调用分支、ctx 和 holder |
| `csrc/torch_classes.cpp` | `compute_train/backward/backward_debug/backward_with_timer` 参数与返回值 |
| `csrc/ops/mega_moe_bf16_result.h`、`mega_moe_bf16.h/.cpp` | 三输出类型、统一训练前向入口、含 packed 权重的 saved 容量校验、新 progress offset |
| `csrc/launcher.h/.cpp`、`include/ascend_deepep/kernels/launch.h` | Host 到 kernel 的声明、函数指针与实参顺序 |
| `kernels/mega_moe_bf16/mega_moe_bf16_backward.cpp` | kernel 导出与 launch wrapper、`Init` 状态指针 |
| `kernels/mega_moe_bf16/include/mega_moe_workspace_info.h` | 新持久视图和 backward 临时区；保留 train 现有反向 scratch/flags |
| `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h` | saved 字段映射、计数恢复、epilogue 直接读取 saved packed 权重 |
| `kernels/mega_moe_bf16/include/block_epilogue_swiglu_backward_bf16.h` | 核对权重消费者，保持 `Hp=p*H` / `dH=p*dHp` 与 BF16 行序；不顺带改算法 |
| `test/megamoe_bf16_test/test_mega_moe_backward.py`、`test_workspace_layout.py` | watchdog、布局回归、saved 内容及梯度检查 |
| `test/bigop_fused/fused.py`、`test_speed.py`、precision runner | 检查 wrapper 对接及复用方式，维持每 shape/每次调用正确初始化 |

不要整体覆盖 train 的 workspace/反向头文件：其中已有 gate/up 直接写 GM、
双 AIV P1、TopK 梯度批量归约、timer 和 SHMEM table reset 等改动，需要保留。
旧 `ForwardWorkspaceInfo` 也不能仅因同名结构冲突而原样留下解析新状态。

迁移顺序建议为：验证前向 packed 权重持久化 → 公共布局与接口 →
Host/launcher/device 同步编译 → progress/诊断 → 第 7 节验证。前向代码及 MD 合入后，旧训练入口
若尚未完成适配，应明确报“不支持新 saved-state ABI”，或保留完整隔离的旧
入口；不能通过改 tuple 解包或传旧指针让它看似可运行。具体过渡处理随集成评审。

## 5. 当前同步边界

| 路径 | 当前保存位置与同步 |
|---|---|
| 普通 BF16 | 保留保存前 `SyncAll<false>()`；保存后第二个全核同步已删除。AIV1 完成 `PipeBarrier + savePipe.Destroy()` 后，通过 `NotifyCube/WaitForVector` 通知配对 AIC |
| MGroup | Wave 计算结束后、重置 token count 前保存；保存前后保留 `SyncAll<true>()` |
| UDMA | GMM1 后、GMM2/Combine 复用 ptrA1 前保存；保存前后保留 `SyncAll<false>()`，并保留现有定向通知和通信 quiet |

UDMA 保存后同步不能照搬普通 BF16 删除：输入快照按连续字节分核，Combine
按计算 tile 复用 ptrA1，快核可能覆盖其他核尚未保存的区间。MGroup 保存后
同步则保证所有计数快照完成，最后一个 AIV 才能清零整张 SHMEM 计数表。
UDMA 新的 MTE 权重搬运还受 3.1 节计算前 barrier 保护。

删除元数据复制或压缩 transient 不意味着可以继续删除同步。保存、UB 复用、
通信完成和 GMM2-ready 各自的依赖仍需满足。普通 BF16 的 Dispatch/GMM1
性能实验独立推进，不应顺带修改其他路径的保存和同步协议。

## 6. 不在本次范围内

- 不迁移训练 backward 的旧状态接口，也不修改其数学算法；Host 保护已加入；
- 不缩减 `max_recv_tokens`；
- 不修改 GEMM、SwiGLU、TopK weight 前乘或重计算算法；
- 不取消 UDMA 的 ptrA1 激活快照；SHMEM 仅删除已不用的 packed 权重接收槽，
  不顺带调整激活通信协议；
- 不将 saved-state/workspace 改造视为 UDMA 长循环卡死的修复。

## 7. 验证要求

本地已通过 16 项现有 Python workspace/layout 测试和 UDMA 紧凑布局的原生
C++ 回归测试。还用最小 Host shim 检查过实际 Device `WorkspaceInfo` 的地址、
边界及持久/临时区隔离；该检查不是 Ascend kernel 编译或执行。

本次 packed 权重修正后，18 项 Python 布局/权重参考检查通过；原生 C++
共享布局、UDMA 容量、Dispatch-ready 回归通过。Host shim 对普通/MGroup
共享 `WorkspaceInfo` 和 UDMA `WorkspaceInfo` 验证了权重指向 saved 段、
奇数行容量边界、progress 不被权重覆盖及清空 transient 不影响持久权重。
原生检查不是 CANN 编译，也不能模拟 MTE/UDMA 的设备完成时序。

普通 BF16 曾收到精度 PASS 的设备结果；最新 UDMA 改动仍需独立设备验收。
MGroup 设备验证暂缓，UDMA 长循环问题尚未定位。

普通 BF16 TopK gather 优化已有用户验证通过反馈，但不覆盖本次后续的权重
持久化和 UDMA MTE 改造。更新后的 Python precision/both 测试会在各 rank
执行结束后，根据来源输入权重与 saved 排序索引核对目标 packed 权重的 BF16
位值，屏幕输出 `[SAVED_STATE] ... PASS`。该检查在 profiler 区间之外；
不能替代三条路径的设备运行或后续 backward 梯度验收。

前向仓需要验证：

1. 三条路径的 Host 扩展、kernel 和所使用的独立 stub 配套重编译，且带
   Timer/不带 Timer 均可构建；
2. precision 通过，并检查 saved-state 的有效输入及路由元数据内容，不能只
   检查输出 y 或 Tensor 大小；
3. 普通 BF16/MGroup 不再复制 Xpacked；UDMA 保留必需快照。结合 Timer 检查
   实际保存开销和总耗时，不预设三条路径采用相同的保存位置；
4. 连续两次 forward 后，第一次返回的 `training_saved_state` 不被覆盖；
5. UDMA 验证 ptrA1 被 Combine 和后续调用复用后，返回的持久状态仍正确；
6. 分别验证短循环正确性和长循环稳定性。

反向分支完成上述适配后，还需要验证：

1. `dx/dtopk/dw1/dw2` 全部通过；
2. 多个未完成 backward 的 micro-batch 各自持有独立 saved state；
3. backward 不读取 transient 地址；
4. forward 与 backward 的布局大小、偏移和版本完全一致。

## 8. 源码入口

- [共享布局](../kernels/mega_moe_bf16/include/mega_moe_training_workspace.h)
- [普通 BF16 workspace 映射](../kernels/mega_moe_bf16/include/mega_moe_workspace_info.h)
- [普通 BF16 保存与同步](../kernels/mega_moe_bf16/include/mega_moe_bf16.h)
- [MGroup 调度与保存](../kernels/mega_moe_bf16_mgroup/include/mega_moe_bf16_mgroup.h)
- [UDMA workspace 映射](../kernels/mega_moe_bf16_udma/include/mega_moe_workspace_info.h)
- [UDMA 保存与同步](../kernels/mega_moe_bf16_udma/include/mega_moe_bf16_udma.h)
- [UDMA Host 分配](../csrc/ops/mega_moe_bf16_udma.cpp)
- [独立 BF16/UDMA stub](../kernels/mega_moe_bf16/mega_moe_bf16_stub.cpp)
- [Python 布局测试](../test/megamoe_bf16_test/test_workspace_layout.py)
- [UDMA C++ 布局测试](../test/megamoe_bf16_test/test_udma_workspace_layout.cpp)

## 9. 2026-09-07 内存与持久状态复核

重新拉取后 train 仍为 `56462ad2bdda3fd2cd244509e15569c4c67a5fbb`。
按该提交的 backward 实际读取点复核，补齐 packed TopK 权重后，未发现其他
需要新增的前向持久数据。此结论针对当前不含 dropout 的实现及其重计算策略，
不代表三个变体的训练设备验证已经完成。

### 9.1 反向读取点逐项对应

| 反向读取数据 | 当前持久来源 | 用途 / 源码入口 |
|---|---|---|
| Xpacked | saved `ptrA` | `UpdateGlobalBuffer()` 中 GMM1 重计算及 dW1 |
| packed TopK 权重 | saved `ptrDispatchedTopkWeights` | `Init()` 传给 SwiGLU backward epilogue，P1 计算 Hp 和 dH |
| cumsum / groupList | saved `ptrcumsumMM`，groupList 为最后一段视图 | expert 行数、来源边界与分组 |
| sorted route → original route | saved `expandedRowIdx` | `GatherRoutes()` 重排 dY |
| original route → sorted route | saved `expandedRowIdxGather` | `UnpermuteBackward()` 和 `UnpermuteTopkGrad()` |
| rank/expert 路由起点 | saved `ptrSumBeforeRank` | backward Dispatch / Combine / TopK 梯度传输寻址 |
| 完整 count table | saved `ptrSavedTokenPerExpert` | `RestoreTokenPerExpert()`；每 expert 输出 counts 不能替代它 |
| progress | saved `ptrBwProgress` | 诊断码；每次 launch 前仅复位本段，不是数学所需输入 |
| W1、W2、原始 TopK weights、counts | Python ctx 单独保存 Tensor | 权重计算、Host shape/dtype 校验和输出分配；不再复制进 saved 区 |
| max_recv / SHMEM 对象等上下文 | Python ctx / 调用者 | 重建相同 tiling 和内存视图，保证通信实例存活 |

检查了 `mega_moe_bf16_backward.h` 和 backward CATLASS 实现，没有发现 backward
继续读取前向 `ptrA2`、scale 别名或 forward-ready flags 的路径。GMM1 gate/up
由重计算得到，Hp、dZ、dO 和 dpProducts 属于独立 backward workspace；它们
不需要由 forward 保存。当前 backward 不直接读取原始 x / topk_ids，所需
信息已由 packed X、双向路由索引和完整计数表覆盖。不要因此删除现有 ctx
中的原始 TopK weights：Host 仍使用其 shape、dtype 及输出模板。

### 9.2 分配与偏移复核

默认 `M4096/N8192/K7168/TopK16/EP8/E4/R65536`，字节数如下。SHMEM 列为
算子布局实际需要量，初始化额度普通/MGroup 为 2 GiB、UDMA 为 3 GiB。

| 路径 | saved | transient（含追加 ready 区） | SHMEM 布局需要量 |
|---|---:|---:|---:|
| 普通 BF16 | 940,185,600 | 536,936,960 | 1,879,244,800 |
| MGroup | 940,185,600 | 536,936,960 | 1,879,244,800 |
| UDMA | 940,185,600 | 536,879,616 | 2,818,769,920 |

以上不含输入、模型权重、输出、allocator 预留和完整训练峰值，不作为整卡
不会 OOM 的保证。多个未反传 micro-batch 各自持有一份 saved，不能共用覆盖。

复核结果：

- 三个 Host 及 BF16/UDMA 独立 stub 通过公共 `TrainingSavedStateLayout`
  分配新持久容量；普通/MGroup 的 saved+基础 transient 容量守恒。
- 普通 BF16 `ForwardDispatchTileReadyLayout` 与 MGroup `ReadyFlagBase()`
  都从新的 transient 末尾计算追加区，移除临时权重槽后同步前移。
- UDMA Host `CalcExtraBufferSize()`、Device `PeermemInfo` 和 Python SHMEM
  预算均已删除旧权重接收槽，`ptrD` 前移一致；source routed 权重仍保留在
  SHMEM，服务前向 MTE 拉取。
- 三个 launcher 返回前均同步 stream 并执行既有 rank barrier，内部临时
  allocation 的结束时机与当前调用方式匹配。
- train 仍需完成第 4 节的旧 workspace 视图替换、持久容量校验、packed
  权重读取、progress offset / 1024 字节清零及 watchdog 适配；这些不是
  在当前前向分支再添加持久字段就能自动完成的。

本轮另修复一个大 shape 下的算术问题：普通/MGroup 共用 `PeermemInfo` 和
UDMA `PeermemInfo` 中，`M*TopK*K` 的 uint32 中间乘积可能先溢出，再乘
`sizeof(bfloat16_t)`。现在第一项显式转为 uint64，保持与 Host 分配公式一致。
默认 shape 不触发该问题；不将它作为之前 UDMA 卡死的根因结论。

验证从实际源码提取 Host 容量函数及 Device `PeermemInfo` 构造函数，使用
原生 C++ 做地址算术检查，共 324 组容量/对齐组合通过；其中旧表达式在 36
组大尺寸组合中与 Host 不一致。该检查使用整数地址，不分配大块设备内存，
包括故意超出部分算子运行限制的算术边界；它不等同于这些 shape 可上卡运行，
也不替代 CANN 编译或设备精度检查。

## 10. 本次分支集成与设备复验

保留双方提交历史。先将 `train@56462ad` 合入 `deepep_moe_xx@8d4d821`，
同一合并结果同步到 train，文档与代码一起进入两条分支。后续个人分支继续 merge train。
集成时保留 MGroup Combine 的 `slice.rowBegin + mLoc`：普通 BF16 传入 0，
MGroup 传入 expert 内的切片起点，不能把它误作全局 expert 前缀而忽略。

train 同时扩展了共享 timer 格式并更新解析器。拉取后应重编基础库、实际使用的
三个 BF16 库及 Python 扩展，避免混用旧 SO 与新布局/计时格式：

```bash
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer
bash scripts/build_mega_moe_bf16_mgroup.sh -soc_type Ascend950 --enable-timer
bash scripts/build_mega_moe_bf16_udma.sh -soc_type Ascend950 --enable-timer
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 --mode precision --warmup 0 --loops 1 --repeat 1
```

随后按需要对 MGroup/UDMA 跑同形状的短精度回归，检查输出精度和
`[SAVED_STATE] ... packed_topk_weights PASS`。短跑不证明 UDMA 长循环卡死已修复；
该问题继续暂停。反向测试脚本当前应得到明确的过渡报错，完成适配后再跑梯度验收。
本地 CPU 布局/解析器检查不能替代 CANN 编译与上述设备回归。

本次集成本地检查结果：24 项 Python 布局用例通过；C++17 布局/ready 用例通过；
用 Host shim 编译实际 Device 地址布局，核对新持久区、临时区隔离与保留的旧
反向视图/临时区通过；反向 timer 自测通过，前后向解析器均使用 251,904 个
Int64 的共享 timer buffer。修改的 Python/Shell 语法与 diff 检查通过。
源码核对确认三个前向 Host 主体与 `8d4d821` 一致，四个反向 kernel/epilogue
源文件与 `56462ad` 一致。Host shim 检查不等于 AscendC kernel 编译。
