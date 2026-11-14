# 真正 EP8：M4096 均衡 random 精度、完整前反向与阶段诊断

日期：2026-10-28。**四组精度全部逐位一致；S9 比本仓同计划串行慢约 1.70%，比原 MoonEP 串行快约 2.18%～2.21%。额外开销主要出现在执行器管理链路，未看到本次无副本场景的前反向计算 kernel 明显退化。** 测试源码提交 `dd2808a`，分支 `work/planner-megamoe-bench`。四组精度、八组正式性能、三组独立诊断共 15 次运行、120 份 rank 回执全部验收，最终容器只剩 tail 保活进程。

## 固定输入与计时口径

真正单机 WORLD=LOCAL_WORLD=EP=8；M4096/H7168/F3072/TopK6，E16、home=slots=2，BF16，seed42，统一接收容量 R=27776。专家数沿用之前对照设置，不将 EP8 子组或 world16 混入本次结果。

`balanced_random` 每 E 个 token 使用随机专家排列和 K 个互异偏移，再打乱 token 与 TopK 列顺序；每个 token 的专家唯一，各 rank 使用不同 seed。它是计数严格均衡的随机输入，不是独立同分布路由。每 rank 每专家 1536 条，全局每专家 12288 条、每卡 24576 行。设备回执确认三条 MoonEP 链路副本数均为 0，四条链路执行行数均为八个 24576，因此本次没有实际副本搬移或副本梯度归主负载。

四条链路为原 MegaMoe、原 MoonEP 串行、本仓 MoonEP 串行、S9。输入、初始权重、概率、容量一致；本仓串行与 S9 使用同 planner、同计划。每次完整 F/B 包含所需 planner、home 发布、恢复/归主、返回结果所有权复制与最终完成等待；排除输入生成、进程初始化、审计/hash、迭代开始的对齐。不包含 autograd 引擎、FSDP AG/RS、checkpoint、SGD 或整网训练。

## 精度

四组共 32 份八卡回执全部正常退出并通过后置空闲检查。Y/dX/dP/dW1/dW2 与原 MegaMoe 全部逐位一致，最大绝对误差和 relative L2 均为 0；F/B 之间覆盖权重池后的恢复检查通过。原 MegaMoe 是已有 BF16 对照，不是独立 FP32 golden；单层结果不替代整网门禁。

性能/诊断各进程的首次审计输出也全部与精度对照 hash 相同；host/fenced 诊断末次采样的五输出另行在计时外核验。精度门槛未放宽。local-serial/S9 实际消耗的计划指纹一致；原 MoonEP 的无副本表示保留 home+slots stride，不要求其原始 physical-id 字节与本仓 owner-only 表示相同。

## 正式完整前反向性能

每个进程 warmup10、采样30，顺序为 original/reference/local-serial/S9/S9/local-serial/reference/original。每次迭代先取八 rank 最大值，再计算中位数、P95 和标准差。每个后端两轮共 60 次采样；下表不带 profiler、阶段探针或额外分段 fence。

| 顺序 | 后端 | 中位 ms | P95 ms | 标准差 ms |
|---|---|---:|---:|---:|
| 正序 | 原 MegaMoe | 27.150600 | 27.483470 | 0.119226 |
| 正序 | 原 MoonEP 串行 | 30.226595 | 30.687390 | 0.166763 |
| 正序 | 本仓 MoonEP 串行 | 29.064940 | 29.432040 | 0.141645 |
| 正序 | S9 | 29.558095 | 29.949230 | 0.167308 |
| 反序 | S9 | 29.526335 | 30.018440 | 0.226464 |
| 反序 | 本仓 MoonEP 串行 | 29.032480 | 29.158580 | 0.082806 |
| 反序 | 原 MoonEP 串行 | 30.183155 | 31.491970 | 0.430150 |
| 反序 | 原 MegaMoe | 27.375565 | 27.443570 | 0.035423 |

| 对照 | S9 正序耗时变化 | S9 反序耗时变化 |
|---|---:|---:|
| 原 MegaMoe | +8.867% | +7.857% |
| 原 MoonEP 串行 | −2.212% | −2.176% |
| 本仓同计划串行 | +1.697%（+0.493155 ms） | +1.701%（+0.493855 ms） |

正数表示更慢。S9 两轮中位数差约 0.11%；同计划串行差值在正反顺序下均约 0.494 ms。原 MoonEP 反序 P95/标准差更大，保留该波动，不将中位数收益扩展为所有尾延迟或其他形状的结论。相比之前小形状 hot 的 15.61%/17.57% 差值，本场景百分比明显减小；两轮的形状和路由同时变化，不能将差异单独归因于 M 增大。

## S9 慢在哪里

### 同步阶段诊断

original/local-serial/S9 分别另起进程，host/fenced 各 warmup2、采样5；随后另采一轮 profiler（warmup1、active2）。下表来自 **fenced 诊断**：每次选择整步最慢的 rank，再取该 rank 的各阶段 exclusive 时间，最后对 5 次求平均。这样不会把不同 rank 的阶段最大值拼成一条不存在的时间线；父子 inclusive 时间不重复累加。

| 阶段，ms | 本仓串行 | S9 |
|---|---:|---:|
| 前后 home 发布合计 | 1.079648 | 1.051546 |
| 原生 MoonEP policy 调用 | 0.301102 | 0.290210 |
| 通用计划校验 | 1.375484 | 1.367094 |
| 前向计划适配/管理其余部分 | 0.192194 | 0.223512 |
| 前向 kernel 调用（完成后） | 9.208104 | 9.092468 |
| 反向 kernel 调用（完成后） | 17.376510 | 17.381126 |
| 完成校验及其 Python 包装合计 | 0.173252 | 0.539438 |
| 反向保存计划重建 | 0 | 0.151016 |
| 前向执行器包装/描述符与检查 | 0 | 0.144418 |
| 计划保存、结果复制、pool/step 包装及其他剩余 | 0.302052 | 0.539204 |
| **诊断整步均值** | **30.008346** | **30.780032** |

分段同步改变排队和重叠，且 S9 有更多被同步的管理调用，因此诊断整步与正式计时不同。表中约 0.772 ms 的差值 **不能替代正式测到的约 0.494 ms**，也不能把三项额外管理成本的和直接当作可实现收益。`kernel_call` 是调用边界的完成时间，包含包装、分配、同步和设备执行，不是纯 AIC 时间。

这组证据支持以下优先级：

1. **S9 相对本仓串行的差距主要在执行器管理。** 两者 native policy / generic validation 相近，前反向 kernel 合计为 26.585 / 26.474 ms，没有观察到 S9 更慢。S9 增加前向计划租约/描述符检查、保存计划、反向重建和前后 finish；串行是在计算前检查一次计划，再执行普通 F/B。
2. **相对无 planner 原算子的绝对开销中，通用校验比 planner 本体更大。** 本次 generic validation 约 1.37 ms，原生 policy 调用约 0.29～0.30 ms。只优化 planner 决策计算，不会消除这部分校验和执行器管理开销。
3. **本场景不支持“副本权重搬移/归主慢”的判断。** 实际 active copies=0；S9 使用 home tiling，前向改回逻辑 route，副本 pipeline/反向 panel restore 在该分支不启用。原始算子中常规 token 通信仍然存在。

源码对应：`ascend_deepep/fsdp2_bridge.py:199` 保存前向 placement 并 finish；`:227` 反向重建计划、执行并再次 finish；`:34` 的 `_SavedPlacementReplay.plan` 每次生成新 epoch/header。`ascend_deepep/megaflux_device.py:122` 创建租约并核验计划；`:181` finish。`ascend_deepep/execution_plan_device.py:33` 通用校验包含 route/count 检查与 all-gather。`kernels/mega_moe_bf16/megaflux_full_bf16.cpp:55`、`megaflux_full_bf16_backward.cpp:67` 是 active=0 分支。

### 不把等待重复算成开销

不加阶段 fence 的 host 诊断中，S9 前向/反向 `control_finish` 的均值为 **8.965 / 17.302 ms**；提前完成设备计算后，同一调用只有 **0.220 / 0.176 ms**。前两个大数主要包含对已提交计算的等待，不能据此说“finish 本身消耗 26 ms”。类似地，串行的 backward home_publish 为 9.521 ms，分段同步后为 0.562 ms：前者等待了前向计算。

host 诊断整步均值 local-serial=29.353 ms、S9=29.790 ms，差约 0.437 ms，与无探针正式差值同方向。它仍带 Python 测试包装，仅用于检查归因方向。

### 八卡 profiler 的交叉检查

每个后端八 rank 各采两个 active step。以下为实际设备前向 kernel 时间与启动跨度，所有数值均来自同一 step 的八 rank。

| 后端 / step | 最大 kernel 时长 ms | 八 rank 启动跨度 ms | 最晚启动到最晚结束窗口 ms |
|---|---:|---:|---:|
| 原 MegaMoe / 1 | 8.885441 | 0.085804 | 8.802927 |
| 原 MegaMoe / 2 | 8.900337 | 0.098145 | 8.804015 |
| 本仓串行 / 1 | 9.280468 | 0.469654 | 8.818464 |
| 本仓串行 / 2 | 9.854013 | 1.024557 | 8.835233 |
| S9 / 1 | 10.220205 | 1.410047 | 8.815403 |
| S9 / 2 | 10.101154 | 1.304354 | 8.801444 |

S9 各 rank 的前向结束跨度仅约 0.006 ms，较长的早启动 kernel 包含等待晚启动 peer 的影响；最晚启动后的窗口三组均约 8.8 ms。因此不能将 profiler 中 S9 的 10.1～10.2 ms 与原组 8.9 ms 直接相减，宣称 GEMM 退化 1.3 ms。该窗口也不是纯计算时间，只是启动错位的交叉证据。反向各 step 的最大 kernel 时长：原组 17.270/17.268 ms、本仓串行 17.223/17.262 ms、S9 17.375/17.301 ms。

profiler 会增加 host 开销，只有两个采样；这些 trace 不代替正式 30×2 次完整 F/B 计时。本轮 Timer OFF，没有 kernel 内部 Dispatch/GMM1/GMM2/Combine 的分段数据，不能进一步将内部阶段的差值称为已证实。

后续若优化，应先针对上述执行器管理成本设计单独 A/B，再评估已有 fused validation 候选；保留计划检查、错误传播和资源生命周期语义。本轮没有启用这些候选，也没有修改算子。

## 本轮修改

仅增加或调整测试脚本与文档，没有改变算子、planner 算法或生产桥接逻辑，也没有重新构建/替换 SO。

- `inputs.py`：增加 rank 独立、TopK 唯一、计数严格均衡的 random 路由；容量按八个 rank 的实际路由计数校准。
- `stage_probe.py` 与测试入口：增加独立 diagnostic 模式，包括不加阶段 fence 的 host 延迟、加同步的完成阶段耗时、CPU/NPU profiler 标记；正式性能保持无探针。诊断采样末次五输出 hash 在计时外核验。
- `campaign.py`：限制离线精度比较的 CPU BLAS 线程，避免大 tensor 比较无界展开线程。
- CPU 检查新增路由不重复/均衡/rank 独立/容量，以及阶段嵌套计时不重复累加；本地和容器共 15 项通过。

前一轮已完成四种 runner、同输入/同容量/同 SO 证书、真正八 rank 检查、正反顺序完整计时，以及用户授权的空闲 Warning/Alarm 门禁。本轮复用这些基础设施。

## 环境与证据

机器 `177.2.25.1`，自有容器 `moonep-megamoe-ep16-20261027`。运行前每次检查两次空闲，结束后再检查；按用户授权使用 0/1 卡 Warning、其余六卡 OK 的设备，保留告警，没有清错/复位或停止其他任务。

算子基线 `1d08ec8`，同事最新已合入 `b61d367` 仅增加 P4 数值失败说明；本地生产代码与该版本一致。Timer OFF、generic validation、worker flags=0、row-ready/P4=0、owner compact=0、owner rows=16、panel restore=2、operand-ready=1、delay=0。worker SO SHA256：`b9ca378491df0f358cb9be6da3975b394cb0f87cadcf9630c95db7717d6a80b4`。

远端原始结果：`/project/optimization/planner-bench-20261028-a1/test/megamoe_planner_test/output/ep8-m4096-balanced-random-a1/`。大 tensor `.pt` 保留在自有目录；本地仅收集紧凑回执、日志、统计和 profiler 证据。

本地证据：`test/megamoe_planner_test/output/remote/ep8-m4096-balanced-random-a1/evidence/`，1191 个文件逐一核验 SHA256；紧凑包 `evidence.tar.gz` 大小 5055628 字节，SHA256=`2e8003cdcfe2c8d671b314b22c312c9083195ad244e9c205d3d72d769f7f36c7`。每次前两次、后一次门禁保留在对应 run 目录，15 次均通过。当前归档不包含大 `.pt` 和原始 profiler 二进制，原件仍在远端自有目录。

统计与阶段原始采样摘要保存在同名 JSON；正式性能样本、诊断关键 rank 的 exclusive/inclusive 时间、各 rank kernel 时间及启动跨度可独立复算。测试源码指纹=`1c0ec48baeea32a8806ae144158f2bb1d875417872c9c5a33e6536b0967245a6`；外部依赖指纹=`6b866fad3e683571f4d01aec79890c2f92ecac460329d7a13bb72bc9689e8ace`。
