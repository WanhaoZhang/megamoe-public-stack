# 真正 EP8 热点场景：串行与 S9 完整前反向对照

日期：2026-10-28。**精度与完整前反向性能流程已跑通；本场景的 S9 没有性能收益。** S9 相比同计划本仓串行慢 15.61% / 17.57%，相比原 MoonEP 串行慢 11.93% / 12.71%。这是一次小形状微基准结论，不是大形状、自然路由或整网训练结论。

## 场景、机器与版本

- 真正单机 `WORLD=LOCAL_WORLD=EP=8`，E16、home=slots=2；M2049、H512、F1024、TopK2，BF16，seed42。
- 路由为固定 `hot`：每个 token 选择逻辑专家 0/1。这里 CLI 的 rho 默认字段未参与路由生成，**不是原整网报告的 rho=0 均衡场景**。
- 四组输入、各 rank 初始权重、概率和容量完全相同；八个 rank 各有不同的 X/权重。R=37248，包含原始 owner-only 对照需求，规划前分配固定。
- 原始组实际执行行数 `[32784,0,0,0,0,0,0,0]`；三条 MoonEP 链路均有 7 个副本，执行行数 `[4098,4098,4098,4098,4098,4098,4098,4098]`，planner required=4224。
- 机器 `177.2.25.1`，本任务容器 `moonep-megamoe-ep16-20261027`。用户明确允许空闲 Warning/Alarm 设备；12 次运行的前两次/后一次快照都是 0/1 卡 Warning、其余六卡 OK。每次八卡进程/计算设备 holder 检查通过，仅 manager 上的 npu-smi 查询进程允许通过。未复位/清错、未动其他任务；最终自有容器只剩原 tail 保活进程。
- 测试实现提交 `ec00a8b`，算子构建来自 `1d08ec8`（通过 `93a67db` 合入）。测试结束前同事新增 `b61d367`，仅更新 P4 整网数值失败文档，已合入；生产 Python/C++/kernel 内容未因此改变。
- FSDPTurbo=`c34a5e3`，原 MoonEP=`5f53a44`；CANN 9.2.0，Torch 2.10.0+cpu / torch_npu 2.10.0.post6。没有 pip 替换或旧 SO 覆盖。
- Timer OFF、generic plan validation、worker flags=0、row-ready/P4=0、owner compact=0、owner rows=16、panel restore=2、operand-ready=1、delay=0。
- worker SO SHA256：`b9ca378491df0f358cb9be6da3975b394cb0f87cadcf9630c95db7717d6a80b4`。各回执另存所有实际加载 native 库路径与 SHA，性能阶段绑定相同精度证书。

## 精度

四组各八 rank，共 32 份精度回执，退出与运行后门禁均通过。Y/dX/dP/dW1/dW2 有限性、冷专家零梯度及 F/B 之间覆盖权重池后的恢复检查通过。

**S9 与原 MoonEP 串行、本仓串行：八 rank 的计划相同，五输出全部 bitwise 一致。** 对照无 planner 的原 MegaMoe，三条链路的 Y/dX/dP 完全一致；dW1/dW2 最大相对 L2 分别约 0.26637% / 0.26682%，通过原五输出门槛和额外 relative L2≤5% 检查，没有放宽原门槛。

这里的 original 是已有 BF16 算子对照，不是独立 FP32 数学证明。单层五输出一致不替代整网 loss/梯度/最终参数门禁。本轮也没有测试默认关闭的 P4 候选；同事最新文档明确记录 P4 的整网确定性数值失败。

## 完整前反向性能

每次先 warmup10，再采样30；按 original/reference/local-serial/S9/S9/local-serial/reference/original 顺序独立启动。每个采样先取八 rank 最大完整耗时，再算中位数、P95、标准差。每个后端总共两轮、60 次有效采样，完整保留两轮，未选择更快一轮。

计时包含 planner、需要的权重发布同步、复制/恢复、前反向、owner SUM、返回所有权复制与完成等待。输入构造、第一次 poison/计划诊断、hash/保存、迭代开始对齐以及进程初始化/销毁不在计时内。没有 autograd 引擎、FSDP AG/RS、checkpoint、SGD 或整网。

| 方向 | 后端 | 完整 F/B 中位 ms | P95 ms | 标准差 ms |
|---|---|---:|---:|---:|
| 正序 | 原 MegaMoe（无 planner） | 4.237215 | 4.284970 | 0.021247 |
| 正序 | 原 MoonEP 串行 | 4.110020 | 4.918750 | 0.356778 |
| 正序 | 本仓 MoonEP 串行 | 3.979275 | 4.084630 | 0.077181 |
| 正序 | S9 融合链路 | 4.600260 | 5.198870 | 0.182829 |
| 反序 | S9 融合链路 | 4.613020 | 4.797980 | 0.096378 |
| 反序 | 本仓 MoonEP 串行 | 3.923620 | 4.277630 | 0.142331 |
| 反序 | 原 MoonEP 串行 | 4.092845 | 4.157710 | 0.044483 |
| 反序 | 原 MegaMoe（无 planner） | 4.223985 | 4.308800 | 0.047686 |

| 对照 | S9 正序耗时变化 | S9 反序耗时变化 |
|---|---:|---:|
| 原 MegaMoe（无 planner） | +8.57% | +9.21% |
| 原 MoonEP 串行 | +11.93% | +12.71% |
| 本仓 MoonEP 串行 | +15.61% | +17.57% |

正数表示 S9 更慢。S9 两轮自身中位数只差约 0.28%；原 MoonEP 中位数也接近，但其首轮 P95 波动较大，因此不把单轮尾延迟当作稳定结论。同计划串行/S9 的方向在正反两轮一致；**本小形状不支持“融合更快”的结论**。未做独立分段诊断，不能将整体差距直接归因到 planner 或某个 GEMM kernel。

## 交付与后续边界

本轮实际覆盖 MoonEP + hot + 小形状；Native、balanced/rotate、大形状、rho 矩阵、EP>8、replay 和整网均不由此宣称通过。后续可先补大形状同输入对照，再单独测 generic validation、计划保存/finish、发布/恢复等开销；本轮没有修改算子逻辑或启用候选优化。

复现入口见 `test/megamoe_planner_test/README.md`。源分支为 `work/planner-megamoe-bench`，没有推送或创建新 PR。

原始设备证据位于容器 `/project/optimization/planner-bench-20261028-a1/test/megamoe_planner_test/output/ep8-hot-moonep-a3/`（含五输出 pt、plan、plog、96 份 rank 回执与所有门禁）。本地紧凑证据在 `test/megamoe_planner_test/output/remote/ep8-hot-moonep-a3/evidence/`；输出目录由 Git 忽略。

紧凑证据包 SHA256：`4f962ab195870b56dddbe0954f6ef731caf51c94727502f6ed81e45af330b989`。仓库内另保留 `docs/planner-megamoe-ep8-hot-results-20261028.json`，包含完整统计与配对比例，可独立检查报告数字。

测试源码内容指纹：`f97c0503e9181956723539c4b9fe1bb19e88dabf7c79e5be226c8c0142c6dcba`。外部依赖内容指纹：`6b866fad3e683571f4d01aec79890c2f92ecac460329d7a13bb72bc9689e8ace`。
