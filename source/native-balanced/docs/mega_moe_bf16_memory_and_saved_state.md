# MegaMoe BF16 训练内存布局与跨反向保存状态

> 2026-09-07 集成说明：本文保留 train 旧训练接口的设计/实现记录。
> 当前 raw 前向已切换三输出 saved-state，旧训练入口暂时明确报错。
> 反向适配请以 [新持久状态与迁移清单](MegaMoe_BF16_前向持久Workspace第一版与反向适配.md) 为准；
> 下文旧 workspace 偏移、接口示例和历史验证结论不能直接用于新 ABI。

本文记录 MegaMoe BF16 训练前向、反向与 SHMEM peermem 之间的内存关系，重点解释
`savedTokenPerExpert` 的真实含义和生命周期。它既是当前实现的阅读笔记，也是后续排查
反向精度问题、重构 workspace 接口时的设计依据。

## 1. 先给结论

重构后的实现中有三类概念不同的内存：

1. **saved context**：训练前向产生并由 autograd 持有，包含当前完整的 forward workspace、
   `savedTokenPerExpert` 和 1 KiB 诊断进度区。
2. **backward workspace**：backward 调用时才申请，仅包含反向 scratch 和同步 flag。
3. **peermem**：各 EP rank 都能访问的 SHMEM 对称窗口，承担跨 rank 路由数据和计数通信；
   它既不属于 `forward_workspace`，也不属于 `backward_workspace`。

`savedTokenPerExpert` 对应的正是 `PeermemInfo::ptrTokenPerExpert` 计数表的持久化快照：

```text
forward 中的 ptrTokenPerExpert
        │
        │ SaveTokenPerExpert()，在清零前复制
        ▼
forward_workspace 中的 savedTokenPerExpert
        │
        │ RestoreTokenPerExpert()，backward 开始时复制
        ▼
backward 使用的 ptrTokenPerExpert
```

这个名字容易让人误以为它保存了 token 数据。实际上它保存的是**路由 token 数量表**，
更准确的名字可以是 `savedTokenPerExpertTable` 或 `savedRouteCountTable`。

## 2. 当前持久状态与临时 workspace 的物理布局

训练前向申请并返回一个 UInt8 `forward_workspace`：

```text
forward_workspace base
│
├─ ForwardWorkspaceInfo region
│  ├─ ptrA                  [R, K]，expert-packed X
│  ├─ ptrA2                 [R, F]，前向 SwiGLU 输出 H
│  ├─ ptrcumsumMM           路由累计计数/group list
│  ├─ expandedRowIdx        packed route -> 原 token/slot
│  ├─ expandedRowIdxGather  原 token/slot -> packed/返回位置
│  ├─ ptrSumBeforeRank      rank 前缀计数
│  └─ forward flags
│
├─ ptrSavedTokenPerExpert    peermem 计数表快照
└─ ptrBwProgress             256 x int32 调试阶段进度
```

backward 被调用时，host 再单独申请：

```text
backward_workspace base
├─ ptrDO                     [R, K]，expert-packed weighted dO
├─ ptrRecomputeGate          [R, F]
├─ ptrRecomputeUp            [R, F]
├─ ptrGradZ                  [R, N]
├─ ptrH                      [R, F]
└─ backward flags
```

其中：

- `R = maxOutputSize`
- `K = hidden size`
- `N = weight1` 的输出宽度
- `F = N / 2`

device 侧分别由 `ForwardWorkspaceInfo` 和 `BackwardWorkspaceInfo` 解释这些内存；
`ForwardWorkspaceInfo` 同时包含前向工作区、count table 快照和进度区。host 侧有对应的大小计算。任何一侧修改
字段顺序、对齐或尺寸公式而另一侧没有同步，都会导致地址错位。因此 host/device 重复维护
layout 仍是一个长期风险点。

## 3. peermem 是另一块内存

`PeermemInfo` 从当前 rank 的 SHMEM 对称窗口基址解析：

```text
rank 的 peermem base
│
├─ 60 KiB reserved control/synchronization area
├─ ptrTokenPerExpert  跨 rank 的路由计数表
├─ ptrA0              dispatch 发送槽，[M * topK, K] BF16
└─ ptrD               combine 返回槽，[M * topK, K] BF16
```

每个 rank 的窗口使用相同偏移布局。远端访问本质上是：

```text
remote_rank_base + identical_field_offset
```

所以 `forward_workspace` / `backward_workspace` 与 peermem 不能混为一谈：前两者是普通 NPU
Tensor，后者是通信层管理、会被后续阶段复用和清理的对称通信窗口。

## 4. `ptrTokenPerExpert` 这张表保存什么

它的逻辑索引是：

```text
[source rank][destination rank][local expert]
```

含义是：某个 source rank 有多少条 route 要发往某个 destination rank 上的某个本地 expert。
单个元素可写成：

```text
count[source][destination][local_expert]
```

例如 `EP = 2`、每个 rank 有 `E = 2` 个本地 expert 时，有效信息可以理解为：

```text
source rank 0:
  -> rank 0: [expert 0 count, expert 1 count]
  -> rank 1: [expert 0 count, expert 1 count]

source rank 1:
  -> rank 0: [expert 0 count, expert 1 count]
  -> rank 1: [expert 0 count, expert 1 count]
```

物理布局中，每个 source rank 的一行会按 128 个 `int32` 元素对齐：

```text
perRankStride = align(EP * E, 128)              // int32 元素数
index = sourceRank * perRankStride
      + destinationRank * E
      + localExpert
```

整张表实际复制的大小为：

```text
tableElems  = EP * align(EP * E, 128)
tableBytes  = tableElems * sizeof(int32)
```

这里保存的是包含 padding 的**完整物理表**，不是仅复制 `EP * EP * E` 个有效元素。这使
forward 保存和 backward 恢复时可以直接逐行 `DataCopy`，并保持所有既有索引公式不变。

## 5. 为什么 `expertTokenNums` 不足以替代它

前向返回的 `expertTokenNums` 只有本 rank 各本地 expert 的总 token 数，逻辑形状为 `[E]`。
它回答的是：

```text
本 rank 的 expert e 总共收到多少 route？
```

而反向 dispatch/combine 还需要知道这些 route 分别来自哪个 source rank，从而确定每段数据
的边界、远端地址与累计偏移。也就是需要：

```text
每个 source rank 给本 rank 的 expert e 发了多少 route？
```

对 source 维求和会丢失边界信息：

```text
expertTokenNums[e]
  = sum_source count[source][this_destination][e]
```

因此 `[E]` 的总数不能还原完整的 `[source][destination][expert]` 表，反向必须保存这张
更细粒度的路由计数表，或者重新计算/重新通信得到它。

## 6. forward 到 backward 的完整生命周期

### 6.1 forward 构造全局计数表

1. routing 初始化先写当前 source rank 对各 destination/expert 的计数。
2. `CrossRankSyncCntPerExpert()` 把每个 source rank 的一行分发到所有 rank 的 peermem。
3. 此时各 rank 的 `ptrTokenPerExpert` 都拥有完整的跨 rank 计数表。
4. forward dispatch/combine 使用这张表计算各 expert 的分段和通信地址。

### 6.2 forward 结束前保存并清理

训练模式下，代码执行顺序是：

```text
SaveTokenPerExpert();   // peermem -> persistent workspace
ResetTokenPerExpert();  // 清零 peermem，供后续通信复用
```

这里的顺序非常关键。如果先 reset 再 save，反向恢复到的只会是一张零表。

### 6.3 autograd 保持 Tensor 存活

host 返回：

```text
(y, expertTokenNums, forwardWorkspace)
```

Python 对外仍只返回正常结果，但通过：

```python
ctx.save_for_backward(topk_weights, w1, w2, counts, forward_workspace)
```

让 `forward_workspace` 的底层 NPU storage 跨越 forward 到 backward 保持存活。这里不是 kernel
内部 workspace 自动跨调用存活，而是 PyTorch autograd context 持有了这个 Tensor。

### 6.4 backward 恢复计数表

backward host 首先申请独立的 `backward_workspace`，kernel 分别接收 `forward_workspace` 与
`backward_workspace`。随后由 `RestoreTokenPerExpert()` 执行：

```text
savedTokenPerExpert -> PeermemInfo::ptrTokenPerExpert
```

完整物理表按 source-rank 行划分给各 AIV；每个 AIV 使用自己的 UB 和 MTE 事件恢复互不
重叠的行。紧随其后的全 AIV 同步保持原有阶段边界，确保任何反向 Dispatch 开始前整张表
已经恢复完成。

恢复完成后，反向 dispatch/combine 可以继续复用原本基于 peermem 计数表实现的地址计算逻辑。

## 7. 当前重构后的设计

### 已解决的问题

- peermem 是可复用通信区，forward 结束前保存必要状态、backward 开始时恢复，是合理的生命周期处理。
- 保存完整物理布局可以避免反向重新组织计数表。
- 反向复用前向产生的 routing 映射和 packed X，可以避免重跑 routing。
- BF16 被定义为训练算子，统一使用 `MegaMoeBF16Forward()` host 实现和
  `mega_moe_bf16_kernel_do` kernel ABI，并固定保存 backward context；不再维护训练模式开关
  或重复的训练前向 launcher/symbol。
- 原有 `MegaMoeBF16` TorchBind 类继续保留 `compute()` / `compute_train()`，兼容已有调用；
  新增公开 `MegaMoeBF16Train(nn.Module)`，通过标准 `forward()` 接入 autograd。
- backward scratch 不再在 forward 时申请，也不再跨 autograd 生命周期占用。
- backward kernel ABI 显式区分 `forward_workspace` 和 `backward_workspace`，不再通过
  前向 workspace 大小推导 backward scratch 基址。
- backward flags 位于每次新建并清零的 `backward_workspace`，不会复用持久状态里的旧值。
- 1 KiB `bwProgress` 留在 `forward_workspace` 尾部，使现有 watchdog 在同步 backward 阻塞时仍可观察。

### 仍可继续优化的部分

第一阶段重构为了降低改动风险，`forward_workspace` 仍保留了完整前向状态：

```text
forward_workspace：完整 ForwardWorkspaceInfo
```

其中 `ptrA2` 和部分 forward flags 未必需要跨 forward/backward 存活。后续可以把真正需要的
packed X、routing metadata、累计计数和 count table 写入更紧凑的独立 layout。当前 host/device
仍各自维护一份 offset 公式，也需要持续保证一致。

## 8. 后续压缩 `forward_workspace` 的方向

当前已经实现：

```text
forward:
  forward_workspace  // 完整前向 workspace + 保存状态

backward:
  forward_workspace  // forward 产生、autograd 保存
  backward_workspace  // backward 调用时才申请
```

下一阶段可以进一步拆出纯前向 scratch，只保存反向真正需要的内容：

```text
forward_workspace:
  packed X
  routing metadata
  cumulative counts / rank prefixes
  savedTokenPerExpert
```

逐项确认字段后，将 `ptrA2` 和纯 forward flags 等排除出去。

需要特别注意：

- 仅把原大 workspace 的若干区间作为 Tensor view 返回并不能释放其余区域。
- 要获得真实的内存生命周期收益，应使用独立 allocation，并由 forward 直接写入或显式复制。
- 这一步会要求 forward 直接写入独立 saved layout，或对必要字段执行显式复制。

## 9. 对当前精度排查的启示

当 `grad_x`、`grad_w1`、`grad_w2` 全部大范围错误时，应优先检查它们共有的上游信息，而不是
只盯着某一个最终写回阶段。内存和路由侧建议按以下顺序观察：

1. forward 保存前的 `ptrTokenPerExpert` 是否符合 CPU 路由统计。
2. `savedTokenPerExpert` 是否与保存前的 peermem 表逐元素一致，包括 padding。
3. backward restore 后的 `ptrTokenPerExpert` 是否与 saved 表一致。
4. `ptrcumsumMM`、`ptrSumBeforeRank` 与计数表推导出的分段是否一致。
5. `expandedRowIdx` / `expandedRowIdxGather` 是否互为预期映射。
6. P0 dispatch 后 `ptrDO` 的 expert 分段与计数表是否一致。

如果第 1～3 步已经错误，三个梯度共同失败是自然结果；如果它们完全正确，再继续进入 CATLASS
重计算、SwiGLU backward 和 dW/dX 的 layout 检查。

## 10. 对应代码入口

- `kernels/mega_moe_bf16/include/mega_moe_base.h`
  - `PeermemInfo`
- `kernels/mega_moe_bf16/include/mega_moe_workspace_info.h`
  - `ForwardWorkspaceInfo`
  - `BackwardWorkspaceInfo`
- `kernels/mega_moe_bf16/include/mega_moe_bf16.h`
  - `CrossRankSyncCntPerExpert()`
  - `SaveTokenPerExpert()`
  - `ResetTokenPerExpert()`
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h`
  - `RestoreTokenPerExpert()`
- `csrc/ops/mega_moe_bf16.cpp`
  - host 侧 workspace 申请和 layout 镜像
- `ascend_deepep/__init__.py`
  - autograd `save_for_backward` 与 backward 参数传递
