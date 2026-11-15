# 普通MegaMoE训练版跨微批次调度

本分支在普通MegaMoE上实现真实就绪前沿和同层/跨层FFN分组，模型shared expert仍由框架计算。它是训练适配和实验底座，没有实现Afore设备副本缓存或新的联合planner，正式矩阵保持关闭。

## 共同条件

L4、EP8/E256/Top8、H4096、routed/shared FFN1024，每序列完整16384 token。每卡microbatch1、GA4，每更新全局32序列/524288有效标签；FSDP8、Expert-FSDP1、TP/CP1。FP32参数/梯度/Adam状态、BF16专家计算、AdamW恒定LR1e-5、betas(0.9,0.95)、eps1e-8、weight decay0.01、clip1，无CPU offload。

仅W2完成当前整网资格，接收容量262144，整卡90GiB门槛。W4旧路径存在数值失败和主形状OOM，设备选择器支持W4不表示整网W4已验收。输入为合法合成token和next-token标签，路由为受控银行；不是预训练收敛或自然路由结果。

## 两种MaxUtil选择规则

- `work_tie`：保留早期适配，平局优先更多工作，强制任务先进入约束。默认不变，便于重放旧结果。
- `upstream_first`：按实际ready到达顺序枚举数值mask，保留首个最大值，再补入强制任务。核心选择集合直接对照EasyInfra作者冻结版本`6327f7bf24086547de8f622f2a00fd763024b3e9`；执行仍保留训练所需的规范TaskKey顺序及双层域上限，不冒称作者完整CUDA/NCCL运行时复现。

新规则须显式提供：

```text
--temporal-policy easy_home --maxutil-rule upstream_first
--temporal-prefix-reuse --hold-fsdp-forward
--temporal-scheduler ascendc
--temporal-scheduler-library /path/to/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so
```

`fixed_home`为同层固定分组＋固定home控制，同样使用prefix复用和前向驻留；`legacy`保留原Native快路径及整层checkpoint划分，作额外参照。两条分组路径的checkpoint配方一致，不能将它们相对legacy的全部差异归为复制成本。Native单调用规划固定768/2/4。

ABI4选择器SHA为`871c167357d1d6e1575d331b3e73c7b36bcc115fb41988e02d4d8a3bb029013e`，原ABI3/f13c3eec不替换。普通性能worker仍为46772ebc，需`EP8_WORKER_VARIANT=ep8-cap64-aligned`；TimerV2诊断worker不能混入正常成绩。

## 源码边界

- [temporal_scheduler.py](../tools/temporal_scheduler.py)：无参数所有权的CPU选择参考和波次编译。
- [device_temporal_scheduler.py](../tools/device_temporal_scheduler.py)：AscendC在线选择，实际当前路由，无未来信息。
- [temporal_model.py](../tools/temporal_model.py)、[temporal_prefix_reuse.py](../tools/temporal_prefix_reuse.py)：框架就绪前沿、公开FSDP生命周期、prefix/finish重放。
- [grouped_experts.py](../tools/grouped_experts.py)：保持owner的虚拟(layer,expert)域；最多两层64home，16K Native行上限下分波。
- [gate_upstream_maxutil_cpu.py](../tools/gate_upstream_maxutil_cpu.py)：固定作者blob核心函数的CPU选择集合对照，不导入作者完整模块。

保留每个原始assignment一次执行、路由权重、固定参数/优化器owner；checkpoint和反向使用同一不可变计划。池仍按真实同步边界复用，不假设多个persistent kernel自动并行。前向驻留不是远端专家副本缓存。

## 已完成的验收与观察

作者核心规则560个CPU集合用例、八rank各674个设备选择/拒绝用例通过；三条新路径完成三次Adam和逐参数梯度相对L2≤0.02，主形状门禁及九点screen通过。

MaxVio10单轮，每点新进程25更新、5预热/20测量，取八rank最大完整wall时间：

| 热点模式 | Native驻留(s) | 固定同层分组＋home(s) | 作者选择＋home(s) |
|---|---:|---:|---:|
| same | 4.3329 | 4.1861 | 4.2120 |
| alternating | 4.3372 | 4.0580 | 4.0717 |
| random | 4.3366 | 4.0954 | 4.1580 |

作者规则在same确实产生跨层波次，但本轮三档均未胜过固定分组；不据此宣称普遍无效或统计显著性。跨层虚拟域/波次实现不同于作者推理执行器，结构尺寸不是梯度通信或关键路径测量。没有新策略收益或论文创新性结论。

本地完整报告、原始状态、源码与SO指纹、PNG/SVG/PDF和CSV位于`D:/megamoe_train/planner-executor-ep8-lab/temporal-replica-study`。运行器和重放工具位于该项目的RUNBOOK及replay-tools归档，不把其原始数据提交到代码仓库。大梯度PT仍有项目专属远端数据，RAW-DATA.md明确索引；不声称全部已下载。
