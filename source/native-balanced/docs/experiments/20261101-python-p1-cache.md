# OLMoE Python新采样与八卡反向优化

> 2026-11-02 合入状态：本报告对应实现及最终说明已合入 `megamoe-balanced`；文中“未push”描述2026-11-01原实验交付时状态。默认P1cache0、owner1:1不变，本次不新增设备验收。相对node/analysis等原始证据链接属于文中技术项目归档，大文件不随Git分发；当前可复用入口见[合入说明](20261102-balanced-python-integration.md)。

本轮固定案例验收通过：P1 completed-panel acquire缓存使完整B耗时四组配对降幅中位 **7.0930%**，直接FB **4.6805%**；四组B/FB及P90全部改善。正确性与生命周期通过。新开关默认仍0，代码在独立实验分支，未push共同开发分支；这不是整网或其他负载验收。

## 场景来源与不均衡

按用户要求通过ModelScope下载官方 `allenai/OLMoE-1B-7B-0924`，三分片SHA与固定HF revision逐一相同。文本来自固定官方 `OLMoE-mix-0924/data/starcoder/python-0000.json.gz` 前2048文档，原顺序pack+EOS，不按内容筛选。固定预训练权重、BF16、train、梯度开启、checkpoint、无optimizer更新；选中输入另做实际causal loss backward。来源及完整指纹见 [SCENARIO](D:/megamoe_train/paper-python-v4pro-20261101/SCENARIO.md)、[MODEL-ASSETS](D:/megamoe_train/paper-python-v4pro-20261101/node71/MODEL-ASSETS.json)。

首事件8×4096=32768tokens，第10个MoE层（layer9）的自然路由 **κ=3.0070735886693**。定义 `κ = E·Σnₑ²/(Σnₑ)²`，不是max/mean。最热E46占28054条，平均4096，max/mean6.8491；16层κ范围1.6194–3.2373。选取规则在采样前固定：首个存在κ∈[2.7,3.3]层的事件，取最接近3、再取层号小者，最多32事件。

这项证据是**新采样**：不是MegaFlux作者原始Python P95，不是作者训练轨迹复现，也不是DeepSeek自然router采样。原始source counts、IDs、prob和训练验证均保留。a2与未接受a1的路由逐位一致，a2通过严格设备前后门禁；失败a1未被抹掉或转为成功。

目标沿用当前八卡算子配置：world8、两个EP4组、E8/Top2/M8192/H7168/F3072/home2/slots2/cap20480，canonical权重、BF16 saved、clip10。64专家直方图降序CDF积分到8专家，并满足Top2互异的单专家条目上限、最大余数取整，再seed17置换专家/打散token。目标 **κ=2.481662403792143**，没有为得到3而改造router。token共选及X/dY/权重/p为受控合成输入；这一步改变了原生64专家/Top8的形状和共选关系。

![来源、映射与planner负载](D:/megamoe_train/paper-python-v4pro-20261101/analysis/workload-and-planner.png)

| 项目 | EP rank0（全局0/4） | rank1（1/5） | rank2（2/6） | rank3（3/7） |
|---|---:|---:|---:|---:|
| 初始owner逻辑路由条目 | 44819 | 4067 | 6672 | 9978 |
| planner后实际执行行数 | 16384 | 16384 | 16384 | 16384 |
| home物理专家行数 | 5645 / 10739 | 3495 / 572 | 5497 / 1175 | 8107 / 1871 |
| replica专家/行数 | 无 | E1 / 12317 | E1 / 9712 | E0 / 6406 |

初始最重owner为均值2.7355倍；每EP组3副本，每个完整副本的权重或梯度为126MiB。planner已完全拉平执行行数，但rank0/4仍承担E0/E1的归主工作。两个EP组使用相同直方图，不能把rank0与4视作两份独立自然热点观测。完整原始计划见 [PLAN-FACTS](D:/megamoe_train/paper-python-v4pro-20261101/node71/PLAN-FACTS.json)。

## 基线瓶颈与本轮实现

本报告C0是已有最佳C1：replicated dW1 early=1、DX completed-panel acquire cache=1。C1是在此基础上**只增加P1 W2 completed-panel acquire cache**。没有移植旧Finance的owner2:1，也没有叠加scan、双缓冲或更改消费顺序。

基线TimerOFF单轮筛选：完整B中位18.808575ms、P90 18.964631ms；直接FB中位29.123420ms、P90 29.287595ms。该筛选不能代替下面的正式配对。

基线原TimerV2观察到：副本rank1/5的P1 GMM2GRAD跨度约6.04/6.06ms，而owner本地两个专家各约1.36–1.50ms；rank1/5随后还有本地非复制专家dW1尾部，AIC约18.30/18.37ms结束。owner0/4早AIV1约9.2–12.5ms做dW1，晚AIV0约14.8/15.1–17.3/17.6ms，归约负担等额却开始很晚。TimerON仅解释依赖链，各rank独立归零，不能跨rank相减估计通信延迟，也不能把这些数字直接当OFF可节省时间。

源码发现P1的实际B GM→L1 load没有接入已有cache：同一AIC/专家的不同output tile会重复对相同W2 panel执行所有producer的精确epoch检查及DSB，目标W2共28panel。DX已有同类缓存，本轮只把它接入P1。

新开关 **MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=0/1，默认0**，进入CMake/构建脚本/manifest/只读SO getter。canonical RowMajor W2、panel restore2、有效slot与ready、panel数1–64时按每AIC/专家调用创建私有缓存；其他条件仍用原P1路径。第一次实际operand load，包括预取，仍完整WaitPanel+DSB，返回后记录；复用key包括ready地址、slot、projection、W1/W2 panel数和精确epoch，缓存不跨调用。

保留外层WaitPanel、operand-ready、专家遍历、完整K累加、BF16 partial、同元素贡献者求和顺序、dW1排程、owner任务分工、所有workspace/UB和最终barrier；Timer schema、Python API/张量形状不变。构建限制为既有隔离earlyDW1/DX cache路径，不改变生产默认。

![P1缩短与owner剩余尾部](D:/megamoe_train/paper-python-v4pro-20261101/analysis/pipeline-summary.png)

副本P1在各rank局部图缩短23.5%–37.3%；rank1/5的AIC末尾分别18.299→15.987ms、18.367→16.122ms。owner晚AIV0的dW1却仍约17.25/17.59ms结束。完整数值、同步等待为何增长及字节量推导见 [流水分析](D:/megamoe_train/paper-python-v4pro-20261101/PIPELINE-ANALYSIS.md)。这是ON描述性证据，最终整体收益仍由OFF配对判定。

## 正确性与构建

| 检查 | 当前证据 |
|---|---|
| 官方模型与文本固定 | ModelScope权重三片与HF SHA一致；固定文本revision/原顺序/输入SHA |
| 真实训练模式验证 | 选中8上下文均causal loss backward，179个梯度张量有限，路由逐位复核 |
| C0三次重复 | 八rank、Y/dX/dp/dW1/dW2逐位 |
| 独立原MegaMoE参考 | Y/dX/dp最大数值差0（该参考只记录误差，未记录存储SHA）；dW沿用原BF16 partial判据，最坏relative L2约0.2756%，未放宽 |
| 新开关0与C0 | 八rank×三次五输出逐位 |
| C1 TimerOFF与C0 | 八rank×三次五输出逐位 |
| C1 TimerON与C0 | 八rank×三次五输出逐位 |
| 生命周期 | EP2/4/8×开关0/1均通过；两层、M257尾块、balanced/hot/rotate、checkpoint、pool权重投毒、SGD、独立参考；C0/C1五输出和SGD更新hash逐位一致 |
| CPU及构建身份 | 真实cache key/epoch/panel边界测试通过；9项CMake检查通过；每个新库155个编译输入及getter核验通过 |

旧参考cap65536避免初始owner热点截断，C0/C1均cap20480，planner守恒且每rank实际16384。C0/C1五输出逐位要求始终不变。当前设备验收仅canonical BF16上述范围，不能扩称legacy layout、整网、UAM、真实FSDP或更大EP验证。

源码分支 `codex/paper-python-v4pro-20261101`，base `7c41ebf938363995838056677e73e5ea6a289d02`，未push。新库完整SHA见 [P1-BUILD-VERIFICATION](D:/megamoe_train/paper-python-v4pro-20261101/node71/P1-BUILD-VERIFICATION.json)，核验覆盖构建清单的155项源码（不等于所有外部SDK/第三方传递依赖）；环境/SDK另固定。本轮新开关默认关闭，回退只需构建0并核加载SO getter；旧base-off-a1冻结可直接对照。

## 正式TimerOFF结果

生命周期全部通过、CPU打点导出结束后，依次完成两轮C0自重复与AB/BA/AB/BA四对；每进程B和FB各10预热30测量，逐样本取八rank最大wall。所有hash在计时外；计时期间关闭Timer/profiler/设备诊断回读，F仍在线MoonEP/home刷新，B含对应恢复/计算/归主/完成/autograd梯度成本。没有真实整网FSDP/router/shared/optimizer成本。10个进程均真实exit0、pre/post门禁通过、计时前后五输出SHA与原capture相同。

单位ms；降幅为 `1-C1/C0`，四组降幅取中位，未混为各轮样本池化结果。

| 组 / 顺序 | B中位 C0→C1 | B降幅 | B P90 C0→C1 | FB中位 C0→C1 | FB降幅 | FB P90 C0→C1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / AB | 18.7253→17.7906 | 4.9914% | 18.9088→17.8859 | 29.0959→27.8811 | 4.1753% | 29.1681→28.0862 |
| 2 / BA | 18.8215→17.4808 | 7.1229% | 18.9916→17.5699 | 29.2579→27.8139 | 4.9354% | 29.3081→27.9005 |
| 3 / AB | 18.8569→17.5250 | 7.0631% | 18.9460→17.7027 | 29.1398→27.8502 | 4.4256% | 29.2115→27.9569 |
| 4 / BA | 18.8493→17.4674 | 7.3313% | 19.0070→17.6375 | 29.1885→27.7468 | 4.9393% | 29.4762→27.8593 |

![完整B与直接FB四组配对](D:/megamoe_train/paper-python-v4pro-20261101/analysis/paired-performance.png)

两轮C0自重复B中位18.7020/18.6993，FB29.0186/29.0494。六轮有效C0的控制波动：B中位 **0.8395%**、FB中位 **0.8218%**，B P90 **1.0960%**、FB P90 **1.1645%**。B接受门槛 `max(2%,2×0.8395%)=2%`；配对中位降幅7.0930%超过门槛，四组全正，FB与两阶段P90均无回退，预先固定的全部条件通过。

结果与所有30样本/逐rank中位/最大wall的rank分布见 [P1-PERFORMANCE-RESULTS](D:/megamoe_train/paper-python-v4pro-20261101/node71/P1-PERFORMANCE-RESULTS.json)。C0最大wall主要落在rank1/5；C1分散到第二EP组多个rank。最大wall的rank可能只是更早进入最终依赖等待，不能直接当成瓶颈生产者；本轮没有OFF分段计时，不以ON阶段时长给OFF owner服务精确预算。

因此，本轮确认P1冗余acquire检查影响了OFF完整性能；局部生产提前沿后续dW1和完成链转化为整体收益。owner晚加入/归约服务和最终依赖汇合是ON图显示的下一处优先研究点，尚需独立OFF候选验证其可节省量；未自动叠加owner2:1等其他优化。接受只针对上述固定直方图派生算子案例。

## 流水图与复现入口

- [C0反向原JSON，无wait块](D:/megamoe_train/paper-python-v4pro-20261101/node71/analysis/base-no-waits/all_8_rank_backward_trace.json)
- [C0前向原JSON，无wait块](D:/megamoe_train/paper-python-v4pro-20261101/node71/analysis/base-no-waits/all_8_rank_forward_trace.json)
- [C1反向原JSON，无wait块](D:/megamoe_train/paper-python-v4pro-20261101/node71/analysis/candidate-no-waits/all_8_rank_backward_trace.json)
- [C1前向原JSON，无wait块](D:/megamoe_train/paper-python-v4pro-20261101/node71/analysis/candidate-no-waits/all_8_rank_forward_trace.json)
- [复现说明](D:/megamoe_train/paper-python-v4pro-20261101/REPRODUCTION.md)、[固定场景](D:/megamoe_train/paper-python-v4pro-20261101/SCENARIO.md)、[单项方案与判据](D:/megamoe_train/paper-python-v4pro-20261101/OPTIMIZATION-PLAN.md)。

每图8rank×96轨道，逐原始区间、同轨零重叠；原NPY及导出audit保留。本地node71/统一标识rack07os1证据；大模型/PT/完整依赖原件在远端同名技术项目，以最终交付清单实际范围为准。原rack08因外部占用停止，自有idle容器与临时公开模型复制服务已关闭；不干预他人任务。

本轮采纳运行均通过严格pre/post门禁，结束后再次检查八卡健康/进程/root-fuser，证明自有容器仅余idle进程后停止，停止后门禁仍通过；见cleanup-idle-node71-a1.json。没有干预外部任务、重置设备或覆盖原采样。所有失败编号保留并排除；不把构建/采样/单算子接受扩称完整DeepSeek训练接受。


配套采样/映射/运行/统计脚本交付于 `D:/megamoe_train/paper-python-v4pro-20261101/code/reproduction-scripts.zip`；大模型、PT与原始trace不进Git，位置及范围见技术项目DELIVERY.json。
