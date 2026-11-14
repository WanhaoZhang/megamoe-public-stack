# 本轮实际执行依赖

下述结论来自冻结worker源码，不是从kernel耗时排行榜猜测。正式计算SO保持不变。

## 模型与计划

每层普通MoE先执行routed experts，再执行框架shared MLP并相加。原前向在线规划；整层checkpoint重放相同计划并重新执行前向。反向使用保存的BF16 gate/up，所以不进入save=none的内部GMM1重算分支。二者不能重复计费。

## 前向执行顺序与ready边界

当前普通forward入口通过megaflux_full_bf16.cpp显式启用M192：GMM1及对应SwiGLU的行tile为192，GMM2和反向GEMM行tile为128。Planner的128行配额粒度不能当成所有计算的tile大小。

当前普通forward先遍历所有物理专家的GMM1，再遍历GMM2；AIV SwiGLU和combine与对应AIC阶段流水。它不是GMM1/GMM2逐专家交替，也没有两套独立AIC资源。

冻结worker的自动输入tile-ready布局限定物理专家数不超过7；本轮有副本64或无副本32均不满足，所以按完整专家输入ready进入GMM1。不能把未来更细粒度的输入流式启动当成当前策略免费可用的能力。权重条带ready、AIV/AIC配对和实际预取仍按源码保留。

正式worker与Timer9的CMakeCache均核对：OWNER_AIV1_TAIL=0、OWNER_EARLY_RATIO=1、DISPATCH_READY_LARGEST=0。奇/偶owner使用既有静态64 worker拆分，不能套用别的分支中AIV1接管成对tile的优化。

## GM/BF16反向

1. P0准备路由梯度、恢复/重新派发输入及专家权重。来源和目标的ready依赖保留。
2. P1按GradientKey确定的专家顺序计算dH和SwiGLU梯度，产生dZ、Hp及路由权重梯度所需中间值。
3. 奇数AIV1在自身P1结束后进入固定owner服务队列，可能等待尚未产出的远端梯度。
4. P2的`GMM1GradReuseDOPhase`对每个专家先做dW2，再等待dO读者退休，随后做dX/回传；不能把dW2提前算成P1已经完成。
5. AIC在P2后执行尾部dW1。当前有副本时为64物理专家，`earlyDW1Enabled_`的≤32条件不成立。
6. 偶数AIV0在dX回传/空梯度清理后进入owner服务，可与AIC尾部dW1重叠。两类owner worker均按dW2→dW1、专家和静态tile顺序服务。
7. owner完整交付与本地/跨rank同步完成后，才能继续后续反向输出和框架梯度处理。没有免费跨层延期归并。

有副本时，反向的`ExpertAt`采用既有`GradientKey`顺序：所有有副本的逻辑专家优先并按全局逻辑ID排序，然后是本地无副本专家，空槽最后。不能把反向顺序假设成物理slot顺序。新增或移除某专家的最后一个副本会改变这个顺序，即使总行数变化很小，也可能改变后续专家的梯度释放时间。

本轮`routed_input_mode=redispatch`在同一个反向kernel内执行原始X/dY/概率的gather、来源发布、重新派发和源读退休，另有原始[4,128,2] transport Timer。整个kernel Timer已包含这些阶段，不能在其上重复累加。AIC可按专家dO/权重ready进入P1，与后续派发重叠，不能假设所有派发结束后才开始计算。

源码锚点（相对`source/native-balanced/kernels/mega_moe_bf16`）：

- `include/mega_moe_bf16_backward.h:1640`：P2逐专家dW2/dX及dO生命周期。
- `include/mega_moe_bf16_backward.h:196`和`:285`：既有梯度优先级与反向专家顺序。
- `include/mega_moe_bf16_backward.h:1042`：原X/dY redispatch与四段原始Timer。
- `include/mega_moe_bf16_backward.h:2260`：early dW1实际开关。
- `include/mega_moe_bf16_backward.h:2344`：BF16保存态跳过内部GMM1重算。
- `include/mega_moe_bf16_backward.h:2437`：奇数AIV1开始owner服务。
- `include/mega_moe_bf16_backward.h:2494`：AIC尾部dW1与AIV工作重叠。
- `include/mega_moe_bf16_backward.h:2530`：偶数worker归并及后续同步。
- `include/mega_moe_gradient_pipe.h:156`：固定投影/专家/tile顺序和贡献者等待。
- `include/mega_moe_owner_schedule.h`：静态worker任务分配。

## 无副本分支与观测限制

全局没有任何副本时，入口使用home32 tiling并关闭`gradient.ready`；即使专家数为32，也不满足early dW1的ready条件。该分支没有副本owner归并，需另用均衡、无副本校准输入覆盖。

Timer9保留原始每核/每专家区间，同时采样梯度column0就绪和owner tile等待/服务。当前两种投影的column数均为偶数，所以column0 owner样本落在偶数静态任务余数；不能冒称它直接测到了奇数worker的逐tile服务。奇/偶服务的重叠还需结合原始每lane归并phase包络，模型误差中保留这种采样限制。

`worker_owner_rows=16`是owner归并内部的16行子块，不是本地专家数上限。
`mega_moe_gradient_pipe.h:27`定义外层ready/task tile为128×256；`:237`的
OWNER_TILE_SERVICE包围完整的`offset+=REDUCE_ROWS`循环，计入全部8个16行子块。
因此本形状每个完整tile仍为64KiB，W1/W2为896/448个外层tile；不能再把
该计时乘8。原home≤16来自旧32物理槽与home=slots的组合，本轮64槽已验证home32。

每个rank的原始时钟独立归档。成本模型使用阶段时长、逻辑依赖和同步边界，不拼接各rank独立零点。嵌套或重叠Timer/Event区间不直接相加。

另外保留同一宿主机的monotonic_ns调用开始/返回时间，表示主机提交包络，不冒称设备完成时间。诊断不逐层同步：在当前流复制原始Timer缓冲、在训练步退休时统一完成和导出。额外复制及导出开销单独记录，诊断wall time不进入正式成绩。设备GetSystemCycle和Event轴分别保存，不能仅减去各rank自己的起点后贴到主机轴。
