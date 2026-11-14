# MegaMoe BF16 gate/up 持久化与反向接入

> 最新工作区已增加 [MXFP8 saved-state 第一版](MegaMoe_BF16_MXFP8_saved_state第一版.md)：
> 新增运行态 `save_gmm1_output`，默认 bf16 保存，none 关闭保存，bf16/mxfp8 指定格式。
> 显式 auto 下 limit=0 保存 FP8 数据和 E8M0 scale，开启截断时仍保存 BF16。
> 显式 mxfp8 开启截断时已实现前向三组 bit 掩码；反向消费仍未接入，见
> [MXFP8 截断掩码方案](MegaMoe_BF16_MXFP8截断掩码与反向免GMM1重算方案.md)。
> 以下保留串行 BF16 保存基准及其反向接入背景，当前格式、容量、验证状态以新文档为准。

本轮普通 BF16（raw 和 compute_train）、MGroup、UDMA 前向均保存 GMM1 的原始
BF16 gate/up。当前反向仍保留 GMM1 重计算，不读取这两块新数据；消除重计算由反向负责人接入。
本节以下为最初 BF16 保存基准说明；普通前向后续设备反馈已通过，当前模式布局
与验证记录以页首新文档为准。读取保存值替代反向重算尚未实现。

## 数据与布局

共享定义位于 `kernels/mega_moe_bf16/include/mega_moe_training_workspace.h`。
当前 BF16 保存对应构造参数
`TrainingSavedStateLayout(M, K, TopK, EP, expertPerRank, R, N, false, true, false)`，
最后的 N 是 GMM1 完整输出宽度，F=N/2，R=maxRecvTokens；不能把 F 作为 N 传入。
原有 packed X、路由、完整计数、packed TopK 权重、progress 的偏移全部保持原值。

| 新字段 | 字节偏移／容量 | 内容 |
|---|---|---|
| `legacyBytes` | 原 saved-state 的 totalBytes | 原初始化和元数据区域的结束位置 |
| `gateOffset` | AlignUp(legacyBytes,512) | BF16 `[R,F]` 行主序 gate |
| `upOffset` | gateOffset+gateUpPlaneBytes | BF16 `[R,F]` 行主序 up |
| `gateUpPlaneBytes` | AlignUp(R×F×2,512) | 每个 plane 的容量 |
| `totalBytes` | upOffset+gateUpPlaneBytes | 新分配大小 |

Device 的 `WorkspaceInfo` 提供 `ptrSavedGate`、`ptrSavedUp`。有效行与 packed X 完全相同：
本卡专家依次连续排列，每个专家内按来源 rank 和原有 Dispatch 顺序排列。
专家起始行为前面有效专家行数之和，第 r 行第 c 列地址为 `planeBase+(r*F+c)*2`。
只读取由 counts/cumsum 和接收容量确定的有效行；剩余容量和对齐 padding 内容未定义。

保存值来自 GMM1 Fixpipe 已输出到 UB 的 BF16 值，**尚未做截断、SiLU 或 TopK 乘权**。
gate 对应 W1 前 F 行，up 对应后 F 行；即使启用 swiglu_limit，保存的仍是截断前值，
反向必须使用同一 swiglu_limit 计算截断和梯度掩码。无需为此新增路由计数或控制表。

## 写入、同步和初始化

共用 `BlockEpilogueSwigluBF16::SaveGateUp()` 在 `ComputeSwiglu()` 前执行两次 UB→GM
DataCopyPad，直接写到 saved-state；没有先写临时 GM 再做整块复制，也没有额外 UB 缓冲。
调用方的 WaitForCube 保证两块 Fixpipe 输入就绪。当前基准恢复为：
保存 gate → 保存 up → 等待两块 MTE3 写出全部完成 → SwiGLU。
这保证 MTE3 读完原始 gate/up 后，向量计算才覆盖 UB。
原来的输出写回、完成等待、计数发布和 NotifyCube 保留。
普通前向按专家基址寻址；MGroup 使用 slice 的全局 packed 行基址，不能把每个 slice 从 0 写起。

UDMA 的 gate/up 也由本地 MTE3 写到普通 GM saved-state，**不走 UDMA 或 SHMEM**。
UDMA 的 packed X 接收快照及原同步仍保留，它与此次 gate/up 保存是不同的数据路径。

Host 使用 `empty(totalBytes)` 分配，随后只对 `[0,legacyBytes)` 清零；新增两块 plane 不清零，
每个有效 tile 都由 GMM1 覆盖。原 packed X/元数据的初始化范围保持原样，避免同时引入另一项初始化优化。
独立 C++ stub 同样扩大分配、保持原 memset 字节数。第一次分配和首次写入仍可能有开销，
不应把“没有新增 memset”理解为“保存没有成本”。

原 transient 的 `ptrA2` 已是 BF16 `[R,F]`，存的是 GMM2 输入 Hp，仍需保留。
此次不扩大或缩小其容量，也不改变 ready/Combine 标志的初始化。SHMEM 大小和偏移不变。

M4096/K7168/N8192、EP8、每卡4专家、R65536 时：

- 每个 gate/up plane 为 536,870,912 字节（512 MiB），合计新增 1 GiB／卡。
- saved-state 从 940,185,600 增至 2,013,927,424 字节，原字段和 progress 偏移不变。
- 同时持有多份 forward 状态时，每份均增加这部分内存，生命周期到对应 backward 完成。
- 现有反向还会分配原 gate/up 重计算区，所以过渡期峰值也增加约 1 GiB／卡；此时不能先删原 scratch。

## 反向负责人如何切换（尚未实施）

2026-09-11 已同步 PR13/15，后续应以当前 train 的 BF16 中间舍入为基准；
不能从旧版整段覆盖 `ComputeForwardAux/ComputeSwigluGrad/ComputeSwigluRecompute`。

当前反向只同步了新布局的容量计算，仍读取原 packed X 等字段并执行 GMM1RecomputePhase，
因此新增保存不要求算法切换同时完成。更新后的 Host、Device、测试代码必须一起构建，不能混用旧 SO。
raw 的三输出、compute_train 的四输出和 Python 训练封装返回值均未改变。

切换时需一起处理：

1. `MegaMoeBF16Backward::Init` 中 epilogue 的前两个输入由
   `bwWs_.ptrRecomputeGate/Up` 改为 `savedState_.ptrSavedGate/Up`；核对有效行和专家偏移。
2. 仅在已适配的保存值消费路径停止 `GMM1RecomputePhase()`；none 路径仍需重算。
   同步区分 timer 和阶段参数，保留其余路径使用的进度标记。
   P0-A 的 dY gather、P0-B 的计数恢复/Dispatch、所有 AIV UB 释放，以及 P1 所需的
   AIV0/AIV1 ready 等待仍有依赖，不能删除整个 P0 或直接省掉等待。
3. 同步缩减 Host `MegaMoeBF16BackwardWorkspaceLayout` 和 Device `BackwardWorkspaceInfo`，
   保存值消费路径去掉两个重计算 plane，none 路径仍保留；重新计算后续
   ptrGradZ、ptrHp、ptrDpProducts 和 flags 的偏移。
   原布局中它们在 ptrDO 后各占 AlignUp(R×F×2,512)。Host 仍用 zeros 时，缩减后也会减少这部分清零成本。
4. 保留 ptrDO、ptrGradZ、ptrHp、TopK partial reduction 和通信/同步区；gate/up 不能替代这些数据。
   backward 只清 saved 的 `backwardProgressOffset` 段，禁止清零整个 saved-state。
5. 使用相同输入比较“原 GMM1 重计算”与“读取保存值”的四类梯度，分别覆盖 limit=0、实际触发截断、
   空专家、非整 tile 行数、多轮 forward→backward。只跑前向不能证明该切换正确。

## 验证

原构建／sh 入口保持不变。`test_mega_moe.py` 已更新分配大小和 packed 权重偏移校验，
precision/both 在计时结束后增加 `[SAVED_GATE_UP] ... PASS`：每个非空专家抽查首/中/末行，
每个 plane 抽查首/中/末列，用保存的 packed X 和输入 W1 的 FP32 CPU matmul、BF16 舍入作参考。
此项只读取少量行和 W1 列对应的数据，不下载整个 plane；performance 模式跳过。
它能检查保存数据和 packed X/W1 的对应关系，不能单独证明全部路由或所有元素正确，仍需原最终 y 精度校验。
compute_train 默认大 shape smoke 检查容量及有限值；新增严格小 shape smoke
检查 MXFP8 掩码/保存值和 BF16 对照输出。两者均不替代真实训练梯度验证。

本地验证包括：36 组新旧布局偏移对照；实际 SaveGateUp 方法的 24 组 CPU tile 模拟
（尾行/列、stride、专家基址和边界哨兵）；两项现有 C++ 布局测试；24 项 Python workspace 检查；
实际 Python 抽查函数的正常、空专家、未初始化尾部及 gate/up 破坏检出检查。
CPU 模拟不验证 CANN 指令编译和设备同步。完整梯度和 EP512 仍待设备验证；
普通前向的设备反馈见下面的基准记录。

gate/up 串行保存版本 `d2977cf` 已由设备反馈精度通过，前向两次为 31,782 / 31,705 μs。
重叠保存实验 `62f34a5` 的两次耗时为 31,751 / 31,904 μs，均值约 31,828 μs，
未观察到相对串行保存均值约 31,744 μs 的收益。样本不足以证明稳定退化，但不保留该同步调整。
当前共用前向 epilogue 已恢复为 `d2977cf` 的串行保存实现，作为后续量化优化基准。
gate/up 持久化、原始 BF16 数值、内存布局、初始化范围和 L2 策略保持不变。
FP8 保存及前向截断掩码现已实现，本文其余内容保留 BF16 基准背景；压缩数据和掩码的
反向消费、取消 GMM1 重算仍待适配，当前状态以页首链接文档为准。

重叠保存实验曾提取实际 `operator()` / `SaveGateUp()` 方法，使用延迟读取 UB 的 MTE3 队列模拟，
通过 288 次 tile 检查，覆盖尾块、地址边界、连续三轮事件复用、截断/乘权模式下的缓冲访问，
并模拟返回后 Cube 立即覆盖两块 UB，确认保存的原始值未被破坏。
SwiGLU 和 TopK 乘权方法与串行版本逐字对照一致；模拟中的计算替身只检验读写依赖，
不代表真实向量指令数值精度或 CANN 同步已验证。
