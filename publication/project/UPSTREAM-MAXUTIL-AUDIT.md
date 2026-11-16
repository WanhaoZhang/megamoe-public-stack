# 作者MaxUtil规则复核与独立重新验收

2026-11-15。此前实现和报告使用“EasyBalance-style”，**不能据其负结果否定作者原始的选择规则**。本次发现了会改变就绪前沿的适配差异，保留旧数据并另立基线验收，不改路由或梯度门槛来获得正结果。

## 核实的原始代码

作者仓库冻结为[`yize-wu/EasyInfra@6327f7bf24086547de8f622f2a00fd763024b3e9`](https://github.com/yize-wu/EasyInfra/tree/6327f7bf24086547de8f622f2a00fd763024b3e9)。重点核对[cross_layer_balancer.py](https://github.com/yize-wu/EasyInfra/blob/6327f7bf24086547de8f622f2a00fd763024b3e9/easyinfra/schedule/cross_layer_balancer.py)：`_prepare_mask`、`_enumerate_max_util`及同步选择流程。

源文件及Apache2.0 LICENSE保存在`references/easyinfra-6327f7bf/`，Git blob与SHA256在其manifest.json。原始文件没有末尾换行；第一次文本导出多加一个换行，已通过固定Git blob SHA识别并修正，未放宽源身份检查。未执行作者CUDA运行时或导入其完整模块。

## 实际差异

| 项目 | 已完成v1/v3的work_tie适配 | 本次upstream_first |
|---|---|---|
| 目标 | 最大化合组token利用率 | 相同目标 |
| 平局 | 优先更多总工作，再按逻辑身份 | 按就绪到达顺序的数值mask，保留首个最大值 |
| 防饥饿任务 | 先把强制任务加入候选约束 | 先选核心MaxUtil集合，再加入强制任务 |

例如两个已就绪任务的负载都为`(8,2)`。选一个或两个的线性利用率都是0.625。旧适配会同时执行两个，它们可能始终保持同层；作者首个mask只推进一个，另一个就可能与其后续层的任务相遇。这个例子只说明选择集合和未来前沿不同，不是训练加速证明。

旧v1的实际八卡审计显示：same和alternating两档的Easy-home均为400个32home原始波次、0个64home跨层波次；random有104个64home波次。same中层间热点原本独立，不能把未形成跨层组直接解释成没有互补机会。

## 保留的训练适配边界

- 仍是固定GA4/W2、完整16K序列；最小集合为ceil(W/2)，防饥饿阈值2。
- 仍只允许两个层域，保持现有64home上限；强制任务合并后超域则拒绝候选。
- 选择集合对齐作者核心规则，执行顺序仍用规范TaskKey，遵守本训练栈的collective和反向边界；未复制作者的CUDA/NCCL流水或其deferred-first重排。
- 仍使用真实原始prefix输出观察当前路由，不读取未来层/步；FP32参数/梯度/Adam、BF16专家、固定owner、共同90GiB预算不变。
- 空总负载仅为设备边界测试约定；实际16K/Top8合法任务总工作为正。不能把此项边界说成完整作者运行时的逐bit复现。

## 已实现与验收状态

`--maxutil-rule upstream_first`单独开启新规则，默认仍为work_tie，旧标签不重命名。CPU参考通过560个选择集合用例：240个触发已审阅且固定blob身份的作者两个核心函数，320个走作者流程中的就绪集合不足时全选分支；另有3个相同负载/反转到达顺序反例。回执：`diagnostics/upstream-maxutil-cpu.json`。

AscendC选择器ABI4增加独立模式3，新SO为`libascend_deepep_temporal_scheduler_v4.so`，SHA256 `871c167357d1d6e1575d331b3e73c7b36bcc115fb41988e02d4d8a3bb029013e`。旧ABI3/f13c3eec原SO和普通worker/46772ebc都未替换。真实新SO本地保存在`implementation/upstream-selector-v4.so`，编译日志是`build-temporal-scheduler-v4.log`。**编译成功不等于NPU或训练资格通过。**

登记顺序为：八rank各674个设备选择/拒绝检查→两条作者选择路径及固定home控制的三Adam/逐参数梯度→same主形状三Adam/90GiB→三时间相关性×Native/固定home/作者选择共9个新进程screen。每点25更新、5预热/20测量，单轮，仍不是108次正式矩阵。任何数值、协议、显存或准入失败保留并停止；数据只以`upstream-maxutil-results.json`及`upstream-maxutil-summary.json`为准。

## 旧确认批次的停止

固定home第二轮已完成，均值4.095234473s。下一组Native在最终准入时发现外部设备作业，没有启动训练，自动停止；其失败日志保留。复核作者规则后取消余下4个尚未准入的旧适配重复。未发出任何进程终止信号，不把不完整批次说成三个独立重复。状态为`resident-v3-confirmation.json`。

当前没有新联合planner、Afore设备缓存或论文创新性结论。旧work_tie结果与新作者核心选择规则的实验将分别呈现。
