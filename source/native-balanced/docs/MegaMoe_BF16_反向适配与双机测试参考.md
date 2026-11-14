# MegaMoe BF16 反向适配与双机测试参考

> 2026-09-17：普通 training 前向已将 packed token 和 packed TopK weight 从
> `training_saved_state` 拆成独立 `dispatch_output`。最新接口、内存结果和
> `retain_dispatch_output` 行为见
> [Dispatch 输出可选持久化](MegaMoe_BF16_Dispatch输出可选持久化.md)。

> 最新格式变更见 [MXFP8 saved-state 第一版](MegaMoe_BF16_MXFP8_saved_state第一版.md)：
> 新增每次调用的 `save_gmm1_output=auto/none/bf16/mxfp8`，默认 bf16 保存；
> backward 必须使用对应 forward 的模式，autograd ctx 已保存并传递。显式 auto 下
> limit=0 的 gate/up 为 FP8+scale，开启截断仍为 BF16。反向尚未切换读取保存值；
> 显式 mxfp8＋正 limit 已追加前向 gateHigh/upLow/upHigh 三个 bit plane，
> 反向需按 [掩码与免重算方案](MegaMoe_BF16_MXFP8截断掩码与反向免GMM1重算方案.md) 接入。
> 不可把新的 FP8 plane 当成 BF16，旧字段偏移和 progress 清零范围不变。

当前接口核对：2026-09-11，稳定分支同步 `origin/train@cc57ed2`（PR13/15）。
详见 [本轮集成与回归](MegaMoe_BF16_稳定前向优化迁入说明.md#2026-09-11同步-train-的-pr13--pr15)。
反向持久视图现为 `BackwardSavedStateInfo savedState_`，复用支持各保存模式的共享布局。
后半部分 9 月 7 日内容仅为迁移历史，不要照历史伪代码接入。

## PR12 集成后的当前接口（以本节为准）

稳定分支已 merge `origin/train@cc50455`。PR12 恢复了 compute_train/backward 入口，
本次把 EP512/2048 专家、直接 Dispatch 和 UB 容量检查同步接入 raw 与 training 前向。
`compute_train(..., max_recv_tokens, swiglu_limit, save_gmm1_output="bf16")` 当前返回五项：
`(y, expert_token_nums, training_saved_state, dispatch_output, forward_transient_workspace)`。
Python wrapper 持有 saved-state 到反向完成。后续修正已删除 backward 的 dummy transient，
并从 Python、绑定、Host、launcher 到 Device 入口移除其参数；反向只初始化 saved-state 地址。
`backward`、`backward_debug`、`backward_with_timer` 的输入统一为
`(shmem, grad_y, topk_weights, w1, w2, counts, training_saved_state, dispatch_output, max_recv_tokens, swiglu_limit, save_gmm1_output="bf16")`。
显式保存 none/mxfp8/auto 的 raw 接入必须把同次 forward 的模式传给 backward；
参数省略只代表 bf16 默认，不会从 Tensor 推断格式。
timer 绑定已补齐 `swiglu_limit` 并显式传给 Host，反向算法及同步保持原样。
显式使用 `diagnostic_holder` 时可持有 Dispatch 输出和 transient 用于诊断。
反向 ABI 已变化，接入方需删除旧 transient 实参，并一起重编 Host 扩展和 BF16 SO。
本地参数链和 workspace 检查通过，设备反向梯度回归待完成。
用户后续反馈普通前向精度通过、31310 us，compute_train smoke 通过。
在此基础上，前后向 SwiGLU epilogue 已新增关闭截断时跳过 Min/Max 的路径，
反向同时跳过截断梯度指示值及其乘法，启用截断时保留原运算顺序和边界行为。
反向 GMM、通信和同步保持原样。当前前向设备 smoke 已反馈通过，
不能用该结果代替反向验证；当前反向仍不消费新增 gate/up 或掩码。

CATLASS 保持 `48f7461`，SHMEM 保持 `f4a4dfd`。当前 CANN 环境必须重新编译、
验证普通前向精度和性能；EP512 尚未设备验证，不能当作整网训练支持结论。
[完整集成检查与回归命令](MegaMoe_BF16_稳定前向优化迁入说明.md#当前集成状态已合入-train-的-pr12)。

下文 9 月 7 日及 PR12 合入前的 ABI 清单保留作为迁移历史，其中“入口受保护”
和“compute_train 改三输出”的建议不再描述当前实现；接入应使用上面的四输出接口。

## 2026-09-10：EP512 普通前向接入补充

`train-bf16-forward-stable` 已对普通 BF16 raw 前向完成 EP512、每卡 4 专家的代码适配，
包括 Dispatch 直接来源偏移、UB 容量修正、分块清零及 2048 专家的 Host 检查。
接口仍是 `MegaMoeBF16.compute(shmem, x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens)`，
返回 `(y, expert_token_nums, training_saved_state)`；WorldSize/Rank 从 SHMEM 实例传入。
详细内存数值、接入前提和本地验证见
[稳定前向迁入说明的 EP512 部分](MegaMoe_BF16_稳定前向优化迁入说明.md#ep512每卡-4-专家的普通-bf16-前向适配)。

**这不是 512 卡设备通过记录。训练 wrapper 的当前状态见上方 PR12 集成说明。**
后续普通反向已删除固定来源缓存，改为读取 saved-state 中的来源偏移；计数表清零改为
16 KiB 分块，恢复仍逐来源行搬运。Host 接受最多 2048 专家并校验 TopK 和实际 UB 容量，
普通反向不再受 `EP <= AIC数 * 10` 限制，梯度公式和同步协议保持原样。
本地 CPU 地址/资源探测通过，EP512 前后向均仍待设备验证。
当前测试脚本没有扩展，整网可使用自己的 PyTorch 入口。

gate/up 持久化已在三版前向实现，新字段追加到 saved-state 末尾、保留原字段偏移；
现有反向继续重算 GMM1、不读取新字段，原有重计算 scratch 暂时保留。
待反向负责人接入保存值后，再移除重计算及不再使用的 scratch，并审查 P0 的剩余依赖。
布局构造末尾为 `N, saveMxFp8, saveGateUp, saveClipMasks`，Host/Device 须一起构建；
典型配置 BF16 保存增加 1 GiB，无掩码 MXFP8 增加 528 MiB，有掩码增加 624 MiB；
none 不增加扩展，新增 plane 不清零。
详见 [gate/up 持久化与反向接入](MegaMoe_BF16_gate_up持久化与反向接入.md)，
包括 raw 保存值语义、有效行、初始化、UB 复用和完整接入步骤。

## 1. 历史基线（2026-09-07）

本次同步前，远端 `train` 为 `0d3f3fa`，已包含前向 saved-state 重构和原有反向代码。
当前分支在其上新增 `f628484`（普通 BF16 物理机双机测试）和 `503c837`
（TCPStore 旧后端）。本次将这些提交与本文一起同步到 train。

2026-09-07，用户反馈切换 TCPStore 后双机测试可以跑通。本轮上下文为普通 BF16，
两机各 8 卡、EP16、64 个专家、TopK16，`M=512, N=8192, K=7168`，均衡路由。
这是用户在远端设备的运行反馈；当前会话没有补充最终逐 rank 校验日志或性能数值。
不将该结果扩展为反向、UDMA 长循环、MGroup 或更大 shape 的验证结论。

**以下为 2026-09-07 的历史状态：只开放 raw 前向。`compute_train` 和三个 backward 入口仍受 Host 保护，
saved-state 反向适配尚未完成。双机前向跑通不等于训练可以直接运行。**

阅读顺序：

1. 本文：确定接口迁移顺序、应复用的测试逻辑和验证边界。
2. [前向持久状态与反向适配详表](MegaMoe_BF16_前向持久Workspace第一版与反向适配.md)：
   第 3 节查字段/偏移/生命周期，第 4 节查逐文件修改点，第 7 节查验收。
3. [物理机前向运行说明](../test/megamoe_bf16_test/README_physical.md)：查主从配置和运行命令。
4. [旧训练内存说明](mega_moe_bf16_memory_and_saved_state.md)：仅用于理解历史设计，
   其中旧 `forward_workspace` 布局和接口不能直接用于新 saved-state。

## 2. 先接通状态接口，再迁移双机测试

按 `Python autograd → TorchBind → Host C++ → launcher → device` 顺序检查，
Host/device 的状态 ABI 必须一起修改、一起构建。迁移目标如下，属于待实现契约：

| 接口 | 当前状态 | 反向适配目标 |
|---|---|---|
| raw `compute` | 已返回 `(y, counts, training_saved_state)` | 复用现有前向实现 |
| `compute_train` | 旧四输出声明，Host 明确拒绝调用 | 接到新三输出实现 |
| autograd 保存项 | `topk_weights, w1, w2, counts, permuted_input, forward_workspace` | 最后两项合并为 `training_saved_state` |
| `backward/debug/with_timer` 输入 | 两条旧状态参数 | 改成一条 saved-state 参数，保留其余输入和诊断输出 |
| 对外 `MegaMoeBF16Train.forward` | 过渡期不可运行 | 仍对用户返回 `(y, counts)` |
| backward 梯度输出 | `(dx, dtopk, dw1, dw2)` | 保持顺序和 autograd 输入槽对应关系 |

主要入口：

- [Python autograd](../ascend_deepep/__init__.py)：`_MegaMoeBF16TrainFunction`，
  同步修改 forward/backward 解包、三个调用分支、`diagnostic_holder` 和 watchdog。
- [TorchBind](../csrc/torch_classes.cpp) 与 [Host](../csrc/ops/mega_moe_bf16.cpp)：
  统一训练前向到现有实现；保持反向保护，直至整条迁移完成。
- [launcher](../csrc/launcher.cpp) 与 [反向 kernel 导出](../kernels/mega_moe_bf16/mega_moe_bf16_backward.cpp)：
  同步声明、函数指针类型、参数顺序和 `Init`，不能混用旧 SO。
- [公共布局](../kernels/mega_moe_bf16/include/mega_moe_training_workspace.h)：
  统一调用 `TrainingSavedStateLayout(M,K,TopK,EP,expertPerRank,maxRecvTokens)`，不另写偏移公式。
- [workspace 视图](../kernels/mega_moe_bf16/include/mega_moe_workspace_info.h) 与
  [反向实现](../kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h)：
  替换旧 `ForwardWorkspaceInfo(forwardWorkspace, tiling)` 和 `ptrA=permutedInput`。

## 3. 反向具体读取哪些前向数据

`training_saved_state` 是独立的一维 UInt8 Tensor，需由对应 autograd ctx 持有到反向完成。
除诊断 progress 外按只读处理；前向 transient 已结束生命周期，不能留下其裸指针给反向。

| saved 字段 | 反向用途 |
|---|---|
| `ptrA` | 同行序的 BF16 packed X；供 GMM1 重计算和 dW1 使用 |
| `ptrDispatchedTopkWeights` | 目标 rank 收到的 BF16 packed 权重；供 SwiGLU/TopK epilogue 使用 |
| `ptrcumsumMM`、`ptrSumBeforeRank` | expert/source-rank 分段与寻址 |
| `expandedRowIdx`、`expandedRowIdxGather` | 来源 rank 的本地路由正逆映射 |
| `ptrSavedTokenPerExpert` | 完整来源/目标/expert 计数快照；反向恢复到可复用 SHMEM 计数表 |
| `ptrBwProgress` | 每次 backward 前按新 offset 清零的 1024 字节诊断区 |

不要用本地 `counts` 替代完整计数表，也不要用来源 rank 原始 `topk_weights`
替代目标 rank 的 packed 权重。原始权重仍单独保存在 ctx 中，packed 权重必须使用
`dispatchedTopkWeightsOffset`，只读取有效 packed 行。

普通 BF16/MGroup 的 Dispatch 直接写持久输入与 packed 权重；UDMA 的激活先进入
对称 SHMEM，必须在 Combine 复用前快照到 saved-state。三个版本的 packed 权重均由
MTE 写入持久段。共用持久布局不代表三者已有可用反向，也不能据此删掉 UDMA 的快照同步。

反向 workspace 另行分配并初始化，保留 train 已有 scratch、partial reduction 和握手 flags。
只清零 saved 的 progress 段，不能清零整块状态。反向使用完恢复的 SHMEM 计数表后，
保留既有 reset/barrier 协议；多次 forward 的状态持有不等于允许同一 SHMEM 并发运行。

数学阅读顺序为 `Combine BW → GMM2 BW → SwiGLU/TopK BW → GMM1 BW → Dispatch BW`。
源码中的 P0-A/P0-B 将原始 dY 路由到专家；P0-C 是 GMM1 **重计算**，不是 GMM1 反向。
令 `H=SwiGLU(Xpacked @ W1.T)`、`Hp=p*H`，则：

```text
dHp = dO @ W2             dp = sum(dHp * H, hidden)
dH  = p * dHp            dZ = SwiGLUBackward(dH, Z)
dXroute = dZ @ W1        dW1 = dZ.T @ Xpacked
dW2 = dO.T @ Hp
```

保持上述乘权位置；不要在路由 dY 时再乘一次 p。先完成接口迁移，保留 train 中已有
gate/up 写 GM、双 AIV epilogue、TopK 梯度归约、timer 和同步优化。

## 4. 双机反向测试应参考哪些前向逻辑

已通过的入口是 [test_mega_moe.sh](../test/megamoe_bf16_test/test_mega_moe.sh)
→ [test_mega_moe.py](../test/megamoe_bf16_test/test_mega_moe.py)。
[physical_server_support.py](../test/megamoe_bf16_test/physical_server_support.py)
是内部辅助模块。反向迁移应保留已有
[反向 sh](../test/megamoe_bf16_test/test_mega_moe_backward.sh) / [反向 py](../test/megamoe_bf16_test/test_mega_moe_backward.py)
作为其单机和双机共用入口，不再增加专用 physical launcher。

| 复用点 | 反向适配要求 |
|---|---|
| 顶部 `SERVER_NUM/SERVER_INDEX/MASTER_IP` | 两机手动配置，shape/seed/TopK/循环参数保持一致 |
| 全局与本地 rank 分离 | `EP=SERVER_NUM*本机设备数`；`global_rank=SERVER_INDEX*本机设备数+local_rank`；每机只 spawn 本机进程 |
| 设备编号 | 按可见设备后的本地逻辑编号选 NPU；从机仍传本机 0–7，不能用全局 8–15 选设备 |
| SHMEM | 用 global rank、全局 EP、相同主机 bootstrap 地址；初始化额度按实际前向/反向需要计算 |
| `Coordinator.check_manifest/phase` | 保留参数核对、各阶段错误汇总和最后读完确认；反向 manifest 还要覆盖反向测试/golden 源码与梯度校验配置 |
| 全局输入与 golden | 两机使用同一算法/seed 生成全局 x、路由、TopK 权重、专家权重和 dY，再按 global rank 切片 |
| 本地梯度检查 | dx/dtopk 检查本机来源 token；dw1/dw2 检查本机所属专家，参考值必须包含所有来源 rank 的贡献 |
| 性能汇总 | 保留逐 global rank 结果与全局汇总；不能只用主机 8 个 rank 代表 EP16 |

现有反向测试的 `reference_backward()` 已按全局输入计算四类梯度，可参考其数学语义，
但 `run_rank/main` 仍按单机组织：`LOCAL_RANK=rank`、按 rank 选设备、`nprocs=EP`。
不能只把 EP 改成 16。它的默认 TopK 是 8、SHMEM 额度固定为 1 GiB，也不能直接当作
本次前向 TopK16/双机容量配置；需显式对齐配置并重新计算容量。

当前前向 `run_multinode()` 和 `make_manifest()` 仍依赖前向 runner 接口，
`exchange_orders()` 依赖前向输出文件名和 shape。参考并提取通用逻辑时需适配这些依赖，
不能直接将它们当作现成的反向启动器。原始排序索引交换仅服务 CPU 校验 packed 权重，
反向 kernel 直接读取本机 saved-state，不需要通过 TCPStore 传 saved-state、激活或梯度。

TCPStore 协调端口与 SHMEM bootstrap 分开，显式保留 `use_libuv=False`。
实测主机 PyTorch `2.7.1+cpu` 的 libuv 本机 1 KiB 通过、256 KiB 超时；
两机旧后端均通过两种大小，随后用户反馈双机前向跑通。
该结果支持使用旧后端，但没有确定主机 libuv 异常的底层原因。

## 5. 交付反向前的验证顺序

1. 先完成普通 BF16 单机的新 ABI 构建和前向回归，再开启 `compute_train/backward`。
   仅修改本次 Python 多机协调无需重编；真正迁移 C++/kernel ABI 必须重编配套扩展与 SO。
2. 单机检查四类梯度及 packed 权重行序，使用各 rank 不同的随机 TopK 权重。
   保留小 shape、空 expert、不均衡容量等覆盖，不能靠全 1 权重或放宽阈值判断正确。
3. 按第 4 节接入双机，先跑少量调用，检查全部 16 个 rank 的四类梯度和全局成功标记。
   SHMEM/barrier 必须包含全局 rank；不能让某台独自多跑或提前退出。
4. 再验证多轮新的 forward→backward 和多份 saved-state 的生命周期。
   不把同一图 `retain_graph=True` 重复反向当作默认已支持；progress 原地写入涉及 autograd 版本计数。
5. 精度稳定后再启用 timer/profiler 和长循环，分别记录 shape、EP、容量、循环与构建版本。

本次只交付参考说明和已验证的前向测试基础，不在同步分支时解除反向保护。
UDMA 长循环问题仍暂停，MGroup 设备验证另行推进。
