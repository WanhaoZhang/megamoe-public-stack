# 2026-11-16：先消除公共 dW 路径的执行排布混杂

## 已有证据

同一主形状、same 路由、MaxVio10、最后一轮诊断，八 rank 各自区间合计的中位数：

| 路径 | MoE 前向 | MoE 重计算 | MoE 反向 |
|---|---:|---:|---:|
| fixed home |235.631ms|238.006ms|528.500ms|
| Native 原执行器 anchor |212.988ms|214.773ms|895.936ms|
| B0 公共 raw dW |216.667ms|218.54ms|1249.72ms|

这是带原始 Timer 和诊断事件的单次短运行，不能作为 TimerOFF 性能成绩或稳定收益证据。方向上与此前33次试验的负结果一致，但定量差值可能被打点扰动。

## 已确认的源码事实

`mega_moe_bf16_backward.h:EnableLifetime` 当前通过 `gradient_.ready=0` 选择 raw dW 输出。该字段还控制：

- `ExpertAt` 是否采用已经编译出的 gradientOrder（复制专家优先）。
- 非顺序 expert 的输入行前缀恢复。
- replica dW 写入原 SHMEM partial 还是返回的 torch-owned 输出。
- 空 owner 清零与原 early-reduce 的关系。
- 依赖该字段的 early DW1 / cooperative backward 资格（实际是否启用还受冻结构建和形状限制）。

因此，B0–B3 之间仍是公共路径下的生命周期对照，但 B0 与 anchor 的差距包含了计算排布和原生重叠变化，不能仅解释成 FP32 网络归并成本。单个字段同时承担这些语义，是当前实验底座需要修正的混杂。

## 下一项独立实现

从冻结主线6e7dd58建立另一独立 child clone；当前 causal-v1 cohort 保持只读。增加 raw-output 状态，保留 gradient readiness、专家顺序及可用计算流水；单独关闭原生 owner Reduce，并明确选择 raw replica dW 输出位置。检查所有共享该字段的路径，不能只改一行后假定等价。

首先通过空 owner/空副本、强制复制、checkpoint replay、W1/W2 和至少3次更新的逐参数 FP32 梯度比较；验证没有隐藏的 native reduction，EP 缩放仍一次。以原始B0/B1/B3及 anchor 为匹配对照，保持计划、SO构建选项、资源和框架路径一致。诊断 SO 与 TimerOFF SO 分开冻结。

只有修正后的公共路径通过这些门禁，才能继续解释“驻留或延迟归并为什么无效”，再决定是否做 PlanTicket 元数据复用和新的联合策略。此项是执行器公平性修正，不是论文创新点，也不改变此前负结果记录。
