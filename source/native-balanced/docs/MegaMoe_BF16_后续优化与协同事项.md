# MegaMoe BF16 后续优化与协同事项

## 1. 文档目的

本文记录普通 MegaMoe BF16 在当前稳定版本之后仍可评估的显存和性能优化，
并区分纯前向修改、需要反向共同适配的修改以及框架侧修改。本文中的待办均不
表示已经实现；生产代码以对应分支和提交为准。

当前开发分支为 `train-bf16-forward-stable`。截至 2026-09-22，已经完成并验证：

- 前向输出和 transient workspace 去除 Host 全量预清零；
- 反向空专家权重梯度精确清零；
- 反向 token-count 表多 AIV 并行恢复；
- 大 EP compact token-count exchange，并保留 `auto/dense/compact` 运行时选择；
- split training state 路径将 token-count 快照提前，恢复 GMM1 到 GMM2 的本核
  AIV1-to-AIC 通知，取消两阶段之间的全核同步；
- 同步 `train` 中的持久化通信 context，避免每次前后向重新扫描 peer 和执行
  context H2D。

8 卡常测 shape 的最新前向结果为：balanced 约 `30526 us`，random 约
`30978 us`；`run_precision.sh both train` 已通过。小 EP 的几十微秒差异可能
包含运行波动，大 EP 和非均衡场景仍需设备验证。

compact token-count exchange 从 EP512 起由 `auto` 默认启用，当前 64 字节
packet 支持每卡 1～8 个专家。compact 协议本身不限制总专家数；普通 BF16
入口当前仍统一限制 EP 不超过 1024、每卡专家数不超过 32、总专家数不超过
2048。以后放宽普通 BF16 的总专家数边界时，无需再次修改 compact 的资格判断。

本分支为评估大 EP 曾增加 token-table 和 token-count exchange 微基准。它们已
完成方案判断，不进入最终 `train`：正式 PR 只保留生产实现、布局测试、Python
参数传递以及 compact 模式的运行时回退开关。

## 2. 纯前向候选

### 2.1 删除前向末尾重复的 token 表清零

当前 Host 在每次 Kernel launch 前精确清理本 rank 的 SHMEM 控制区，Kernel
入口又通过设备侧跨 rank 同步保证所有 rank 清理完成。前向结束时仍会调用
`ResetTokenPerExpert()` 清理同一 token 表。

可以评估删除结束清零，只保留结束时保护 Combine 读者的同步以及下一次调用前
的 Host 精确清零。该项不改变反向接口，但必须验证连续前向、前向后反向、不同
路由模式以及多 stream 调用。EP8 收益预计不可见；EP1024 的历史微基准显示完整
8 MiB 表清零约为几十微秒，因此不作为当前高优先级修改。

### 2.2 gate/up 保存流水

前向 epilogue 已直接把 gate/up 写入持久区。进一步使用额外 UB 或改变保存与
SwiGLU 的重叠关系，理论上可以减少等待，但已有小 EP 数据没有显示稳定的大幅
收益。该方向容易增加 UB 压力并改变 MTE/V 流水，只在 timer 明确显示保存位于
关键路径时继续。

### 2.3 InitRouting、Dispatch 和 Ready256

现有策略保留 ExpertReady，并只在满足条件时对首个非空专家使用 Ready256。
后续重点是 EP512/1024、每卡 1～8 专家以及非均衡路由下的首块启动时间。没有
大 EP 环境前，不继续增加调度模式；任何修改都应保持对外只有 ExpertReady 和
Ready256 两类策略。

## 3. 需要反向共同适配

### 3.1 Token count 持久状态紧凑化

当前前向把 SHMEM 中完整的
`tokenPerExpert[sourceRank][destinationRank][localExpert]` 保存到
`training_saved_state`，反向再由 `RestoreTokenPerExpert()` 恢复。容量为：

```text
EP * AlignUp(EP * expertPerRank, 128) * sizeof(int32_t)
```

EP512、每卡 4 专家时约 4 MiB；EP1024、每卡 2 专家时约 8 MiB。反向当前的
有效读取只需要本 rank 对应的 destination 列，因此可改为：

```text
compactCounts[sourceRank][localExpert]
```

容量为 `AlignUp(EP * expertPerRank * sizeof(int32_t), 512)`，上述两个大 EP
场景均约为 8 KiB。实施时需要前后向同时修改：

1. 前向从完整 SHMEM 表抽取本 rank 的 destination 列并写入紧凑持久区；
2. 反向 `DispatchTokensBackward()` 和 `ReduceAndRouteTopkGrad()` 直接读取紧凑表；
3. 删除完整表恢复以及仅服务于恢复表的结束清零；
4. 更新 saved-state layout/version、Host 容量检查和 Python/C++ 布局镜像。

该项主要减少多层、多 micro-batch 下的持久显存。单次 EP1024 反向的恢复和清零
收益不足约 0.1%，不应只为性能实施。

### 3.2 反向重新 Dispatch 后释放前向 Dispatch 输出

当前 `dispatch_output` 保存 packed token 和 packed TopK weight，常测 shape 约
896 MiB。若反向改为根据保存的路由信息重新 Dispatch，则框架可在前向结束后
释放这块数据。这是显存收益最大的协同项之一，但会增加反向通信和调度开销，
需要反向实现、显存峰值和端到端步时一起评估。

### 3.3 BF16 saved gate 与 `ptrHp` 复用

BF16 saved gate 与反向 `ptrHp` 的布局均为 `[maxRecvTokens, F]` BF16。常测
shape 下各约 512 MiB。P1 读完一个 gate tile 后，可把对应地址覆盖为 Hp，从而
取消独立 `ptrHp` scratch。

该方案与 EP 没有直接比例关系，主要由 `maxRecvTokens * F` 决定。它会消费
saved-state，因此默认不兼容重复 backward、`retain_graph=True` 和重入反向；
MXFP8 保存区也不足以容纳 BF16 Hp。必须先与反向和框架确认生命周期契约。

### 3.4 P1 gate/up 单次读取与 SwiGLU 梯度融合

当前 P1 对同一 gate/up tile 读取两次，并分别计算 forward auxiliary 和
SwiGLU gradient。可以在一个寄存器循环中同时计算 H、dp、Hp、dH、dGate 和
dUp，减少第二次 GM 读取、MXFP8 重解码以及重复的 Exp/Div。

实施时必须保留现有 BF16 舍入边界、截断 mask 语义和 dP reduce 顺序。该项最有
希望获得可测的反向性能收益，但应先确认 AIV P1 位于关键路径，再由反向维护者
共同评审和验证。

## 4. 需要框架侧讨论

### 4.1 SHMEM workspace 的对象生命周期

每创建一个 MegaMoe BF16 算子对象，当前实现会申请并首次清零一块 symmetric
SHMEM workspace。框架需要确认是每设备共享一个对象，还是每个 MoE 层分别创建。
若每层独立持有约 2 GiB workspace，显存浪费会远大于 Kernel 内的小块优化。

推荐在同一设备、同一串行执行域内共享 workspace；多 stream 并发时需要显式
workspace pool 和所有权管理，不能直接无条件复用。

### 4.2 按实际 recv token 缩小容量

大部分前向持久区和反向 workspace 都随 `maxRecvTokens` 线性增长。提前获得
actual recv tokens 可以缩小内存，但需要额外计数通信、Host 可见结果或分桶式
workspace pool，也会影响异步下发和静态图。只有整网确认实际接收量长期明显低于
容量上限时才值得设计。

### 4.3 前后向保存策略

框架需要明确每层、每个 micro-batch 对以下数据的持有时间：

- BF16 或 MXFP8 gate/up；
- Dispatch 输出；
- token-count 和路由元数据；
- forward/backward transient workspace。

显存评估必须按同时存活的层数和 micro-batch 数计算，不能只看单次算子的局部
workspace 大小。

## 5. 后续修改的验证边界

涉及 saved-state 或 workspace 生命周期的修改，至少覆盖：

- balanced、random 和空专家路由；
- BF16、MXFP8、MXFP8+截断以及 `save=none`；
- 连续 forward-backward；
- 多个 forward 后逆序 backward；
- compact exchange 的 `auto/dense/compact`；
- EP8 小规模回归，以及具备环境后的 EP512/1024；
- 前向最终输出、所有反向梯度和显存峰值。

优化结论应分别标记为源码分析、Host/CPU 检查、AscendC 编译、单机设备验证和
整网验证，不能用微基准结果替代端到端结论。
