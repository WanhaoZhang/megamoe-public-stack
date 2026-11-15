# 完整更新生命周期研究：第二阶段诊断报告

日期：2026-11-16。项目：`D:\megamoe_train\planner-executor-ep8-lab\step-lifetime-study`。

## 当前结论

第一阶段的 A18/B15 共33次首轮运行没有建立稳定收益，因此没有启动108次矩阵，也没有宣称新的联合 planner 有效。本阶段完成五组小形状资格验证、五组主形状三更新诊断，并找到一个需要先排除的执行器混杂：公共 raw dW 模式同时关闭了依赖 `gradient_.ready` 的专家处理顺序和部分流水资格。

完整目标仍然是：在一次 optimizer 更新内联合选择就绪任务、复制位置、驻留时间和梯度交付。当前实现尚未达到该目标。本阶段的贡献是把负结果拆成可验证的问题，而不是将缓存命中或通信减少当作训练加速。

## 工程与验证

- 主线冻结在 `research/step-lifetime@6e7dd58`；本阶段独立 clone 为 `research/step-lifetime-causal`，诊断运行源码基于 `abaf8e9`，配置和完整字节指纹见每个 evidence 目录。
- 原 Timer 私有 ABI1 worker：`ep8-cap64-lifetime-timer-causal-v1`，SHA256 `e0e18541b7faecb6a99d27a6e2ab3587bc8a93ed7d24768a40fbbbd7cf1fb49c`。前后向 worker flags 均为0；CAP64，panel restore2，原 Timer schema9。该 SO 仅用于诊断。
- 五个模式：fixed_home、原生核内归并 anchor、每调用准备/立即公共交付 B0、驻留/立即交付 B1、驻留/更新末优先交付 B3。B2 已有第一阶段性能记录，本阶段未重复。
- 小形状 L2/H512/F512/S256，GA4/W2，使用 MoonEP 规划作为正副本覆盖刺激；与同输入、参数和路由的 TimerOFF fixed-home 参考比较完整 step0 未裁剪 FP32 owner 梯度。五组逐参数最大相对 L2 分别为0.005874、0.005936、0.009932、0.004860、0.009466，均小于0.02。每组均完成三次更新。此处不把完整模型 step0 梯度比较冒称为所有更新逐参数轨迹完全相同。
- 主形状保持 L4、EP8/E256/Top8、H4096/F1024、每卡完整16K序列×GA4、W2、固定 owner、FP32持久状态/BF16专家、recv262144，无CPU offload。主诊断只测 MaxVio10、same模式，每组3更新。各组八 rank 完成、源码未变化、独占检查通过，整卡峰值未超过90GiB。
- 小形状的独立 FFN 输出/dx/dp/dW 门槛沿用前阶段资格；本阶段新增完整模型梯度与原 Timer 覆盖资格。自然训练分布、完整模型 W1 控制和错误计划的集体恢复仍未新增验收。

## 看到了什么

以下取最后一个诊断更新，分别计算各 rank 的同类设备 Event 区间并集，再列八 rank 中位数。它们包含流等待和 host 供给空隙，不是纯 kernel active time；父子区间相互重叠，不能将各列简单相加。只存在一次短诊断，不是重复实验的置信区间。

| 模式 | MoE前向 ms | MoE重计算 ms | MoE反向 ms | 公共梯度服务区间 ms | finish的host包络 ms |
|---|---:|---:|---:|---:|---:|
| fixed_home |235.631|238.006|528.500|—|—|
| anchor |212.988|214.773|895.936|核内路径，另见原Timer|0.123|
| B0 |216.67|218.54|1249.72|364.54|0.154|
| B1 |214.98|215.25|1244.18|364.24|0.151|
| B3 |214.68|216.04|814.65|466.04|153.670|

Native 复制的前向与重计算合计比 fixed_home 少约46ms，而反向多约367ms；规划及公共校验另有约75ms区间。这给出了“前向看起来有利，完整更新却没有受益”的具体候选机制。正式结论仍以第一阶段 TimerOFF 完整更新为准：带原 Timer 的各阶段数字可能被打点扰动，不能直接当作生产路径的损失分解。

B3 的 native 反向区间比 B1 短，但梯度服务区间更长，并把等待放到了更新末。不能据此宣称 B3 更快，也不能简单相加推算加速；它提示交付时机可能同时影响算子执行、通信争用和更新尾部。需要后续匹配消融验证原因。

所有方法仍有约25.7ms owner BF16物化、23.5ms计算池发布。它们是可向所有基线开放的公共工程优化，不能作为新策略独占的优势。

## 真实复用与归并边界

对最后一个更新的全部八 rank，B1与B3均记录318次缓存条目访问：48次miss、148次同组不同 FFN wave 命中、92次同一 invocation 的重计算/反向命中、30次不同 microbatch 任务组之间的命中。因此270次命中里，真正跨 microbatch 任务组的只有30次；不能用总命中率85%证明大量跨 microbatch 复用。

原始捕获中的 `kind` 分类误把不同 wave 归为 cross_group。本报告根据真实 prefix key `(step, group, wave, 1)` 和 `temporal-groups` 任务清单离线重算，原始记录保留不覆盖。计数是逻辑缓存条目访问，不是硬件链路流量。

B3 在该更新有八 rank 合计16条容量触发归并、8条更新末归并记录，说明八条目预算下并非所有梯度都留到最后。容量回收、权重驻留和梯度银行不能混为同一种生命周期。

## 两个需要先修正的底座问题

### 1. raw dW 交付与计算排布耦合

已确认 `EnableLifetime` 用 `gradient_.ready=0` 选择 raw dW。同一字段还控制 `ExpertAt`、非顺序专家的行前缀恢复、复制梯度输出地址、空owner清零，以及特定构建/形状下的 early/cooperative 优化资格。它使 B0–B3 与 anchor 之间多了一个执行排布变量。B0–B3之间的同路径对照仍保留意义，但不能把与 anchor 的全部差距归因于“延迟归并策略本身”。

独立修正分支位于 `..\raw-control-20261116\stack`，分支 `research/step-lifetime-raw-control`。私有 ABI2 在同一 SO 内保留 legacy=1 与 preserve=2 控制，保留原计算顺序，单独选择 raw 输出与是否核内归并。原33次结果与本阶段源码均不改写。构建、数值和性能验收状态以该子目录的记录为准；本报告不预先认定修正有效。

### 2. 原计划已经包含梯度贡献者信息

利用 copies 与保留来源维度的 source quota，可以推导所有 rank 的物理执行计数和非零梯度义务。此前576份正副本记录及本阶段已收集的4032份完整记录逐项精确匹配；其余主诊断的全量计数审计见各 `collection-scope.json`，不是只检查本地抽样片段。

这为 PlanTicket 在原始在线编译时一次生成义务、重放/反向复用提供语义依据。但尚未实现该运行时优化。host计数读回的约0.8–1.15秒包络包含等待前序设备工作，不能声称全部是可消除的CPU开销。需通过后续匹配实验检查是否真正缩短更新。

## 下一步与停止条件

1. 先完成 ABI2 的强制副本、空专家、W1/W2、checkpoint、输出/dx/dp/dW 数值门禁；再做至少三次完整模型更新的逐参数验证。
2. 同一 ABI2 SO 对比 legacy/preserve，隔离计算排布变量；保持输入、计划、资源、框架、初始参数一致。不能拿不同 checkpoint 路径作控制。
3. 再比较修正后的 anchor/B0/B1/B3，正式计时关闭 Timer。若仍不能胜过最强适用基线，就记录原因，不启动108次矩阵。
4. 只有证明任务选择或生命周期存在独立收益，且简单组合不足，才实现新联合策略。PlanTicket元数据和公共权重发布优化均向可用基线开放。

仍未完成：新的在线联合策略、设备端生命周期决策、自然/独立训练路由验证、五轮稳定收益和完整关键路径依赖闭环。当前不作论文创新性或投稿竞争力承诺。

## 产物与重放

- 运行登记：`causal-v1.json`；每组 `evidence/<tag>/status.json` 和 `source-identity.json`。
- 完整数值资格：`diagnostics/causal-v1-math-*-comparison.json`，`stack/configs/causal-v1-qualification.json`。
- 全量计划比较：`diagnostics/main-all-plan-equality.json`（完成状态以文件为准）。
- 数字与图：`analysis/causal-v1/summary.json`、`scope-metrics.csv`、各模式 `*-rank0.png/svg/pdf`。各图只使用同一设备的 Event 原点；空白不代表已证实的设备空闲。
- 源码、构建与SO：`stack`、`implementation/build-causal-timer.sh`、`implementation/worker-build.json`、`implementation/causal-timer-worker.so`。
- 运行脚本：`run_causal.py causal-v1`；已有标签不会覆盖，重新执行必须选新prefix并重新生成匹配资格。
- 全量原始 NPZ、完整FP32梯度张量留在独立远端 `/home/z00956592/megaflux/step-lifetime-causal-20261116/stack/evidence`。本地 main fixed_home 包含全量NPZ；其余主诊断本地只回收32份注明范围的原始片段，`collection-scope.json`保存全量清单、SHA、计数审计和覆盖，不能冒称全量已回收。

第一阶段完整性能图仍在父目录 `figures`；本阶段不新增性能加速曲线。
