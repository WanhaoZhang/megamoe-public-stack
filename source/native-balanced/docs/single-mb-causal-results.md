# 单 microbatch 训练：EPLB 因果干预与剩余空间（阶段结果）

实验项目：`D:/megamoe_train/single-mb-fbw-planner-study`；原实验只读：
`D:/megamoe_train/planner-executor-ep8-lab`。本轮执行在隔离的 pod5-8 容器，
不是其他任务所在的 pod6-1。研究分支 `research/single-mb-fbw-planner`。

## 1. 当前能够回答什么

EPLB-fixed 的主要问题不能用“在线 planner 太慢”解释。保持原分配规则，
只收紧实例预算，已明显减少整步时间；完整计划回放没有进一步改善满预算
EPLB。这里研究的是当前固定 owner、同卡合并的适配，结论不外推到官方
EPLB 的全部实现。预算改变还会影响配额和计算碎片，不能把差值全部记作
梯度通信时间。

Native、UltraEP 是否已到上限，当前仍不能回答。早归并和 fan-in 排序没有
达到工程门槛，但排序干预在强基线上的实际覆盖很少。新在线 planner 尚未
验收，也没有稳定的整网增量成绩。

## 2. 先核验适配和原始时间线

对五策略 × 三步 × 两层 × 八 rank 的 240 份原始快照，逐 assignment 检查
固定 owner 解释、逻辑/物理专家映射、source 配额、实际执行计数与跨 rank
计划一致性，全部通过。它排除了这些调用里的所检查映射错误，不是整个
执行器正确性的证明。

MaxVio7 诊断中，Native/EPLB 的全八卡活跃远端副本均值为 8.33/220.83；
128 行 tile 数仅相差约 1.28%，而非零物理专家输出的完整 dW tile 任务为
352128/640864。后者不能解释为 FLOPs 增加 82%，因为每个 GEMM 的归约 K
也改变了。额外完整 partial、恢复、GEMM 启停/尾部和归并必须分别看。

原始 dH/dX 区间包含等待与并发干扰；跨 rank 保留逻辑依赖边和各自时钟，
没有直接对不同 rank 的绝对 tick 取 max，也没有把重叠区间相加。
详见项目 `analysis/original-contributor-dependencies.json` 和原始打点图。

## 3. 同计划在线/配额/完整 body 三臂诊断

每格独立启动一次八卡进程组，25 次完整 Adam 更新，去掉前五次后取每步
八 rank 最大 wall time，再求平均。初始参数、输入、路由身份及同策略的
50 份冻结计划完全一致；设备端实际加载 SO 与源码前后哈希可追溯。

所有臂都保存同样的预载存储，并支付完整路由值检查的同步成本。
`quota` 仍执行公共在线统计、编译/assignment/校验；`body` 跳过整条在线
计划路径。以下是诊断分数，不是普通在线 planner 成绩，也不能把两个
进程均值之差当成隔离的 kernel duration。

|偏斜|策略|online ms|quota ms|body ms|
|---:|---|---:|---:|---:|
|20|Native|1202.36|1199.25|1193.21|
|20|EPLB-budget32|1238.68|1238.42|1232.25|
|20|EPLB-fixed 满预算|1566.20|1568.11|1579.18|
|0.2|Native|1082.32|1080.66|1072.77|
|0.2|EPLB-budget32|1202.94|1198.29|1199.39|
|0.2|EPLB-fixed 满预算|1612.76|1611.06|1640.23|

从满预算到 32 的 online 整步差为约 328/410 ms；Native 完整 body 回放
仅改善约 9 ms，预算32改善约 4–6 ms；满预算回放反而更慢。当前只能说
没有观察到“去掉在线生成即可释放满预算 EPLB 收益”。负的回放差值可能
含缓存、排队、编译输出的预热效果与进程变化，尚未单独归因。

预算32仍慢于 Native，因此它是必要的额外强基线，不是新算法。这里只
完成了满预算/32两个预算层级的计时；完整预算扫描、五次进程重复仍待做。

原件：项目 `evidence/intervention-*-v2`；核验与统计：
`analysis/prepared-plan-pilot-v2-summary.json`，脚本 `tools/analyze_interventions.py`。

## 4. 为什么不能只优化远端字节

固定 budget32 的副本和每逻辑专家目标任务量，改变来源配额，高偏斜时
远端 token 行数从 917516 降到 759478（约少17.2%），非空来源片段也从
2271 降到2164，但最大单向 source→destination 行数从16857升到21636。
body 整步从1232.25变成1235.76 ms，没有收益。低偏斜仅约0.15%差异。

这不是“局部性无效”的证明；它说明更少的总字节不保证关键链路、生产者
就绪或计算流水更快。尚不能将高偏斜的变化归因于单一链路。逆转副本
槽位在两档也未带来收益，且反向按共享逻辑专家排序，不能称为任意反向
顺序搜索。后续必须显式区分前向物理顺序与反向生产/服务顺序。

500份重建计划的 CPU copies/quota 哈希均核对到实际计时的全部八rank。
结构特征在 `analysis/prepared-plan-features.json`；它们不是时间预测。
特别是 AIC 的 start cursor 跨专家轮转，不能简单把每专家 ceil-wave
相加当作严格执行下界。

这里必须区分host分配和device执行：host描述保留home32+slots32，但实际
forward kernel 和 routed-only backward ABI3 都会在header无活跃副本时
覆盖为homeTiling32，且跳过EnableGradientPipe/Restore。只检查host的64
专家形状会得到错误结论。旧模型empty-expert系数为0，并不代表它建模了
整个执行模式开关的成本；第一个副本的代价需要单独实测。

## 5. owner 窗口和已有排序的负结果边界

高偏斜正常在线整步：Native 原窗口/late-only 为1210.25/1210.14 ms；
满预算 EPLB 为1638.58/1642.34 ms。所有初始身份已配对核验，各格一次
进程，均未达到2%工程门槛。late-only同时将原奇/偶 share 的并集交给
偶数 AIV，改变了服务并行度；它不是单独改变时间点的理想干预。负结果
不能证明通信争用不存在。

原优先级/64槽early-dW1的20格筛选同样没有2%增量。但实际50份fixture的
顺序重建表明：低偏斜 Native/UltraEP 的400个rank-call均未改变活跃
生产顺序；高偏斜仅33/400、35/400改变，EPLB则400/400改变。因而不能
据低覆盖筛选否定一般的就绪调度。记录见 `analysis/priority-order-coverage.json`。

## 6. 下一轮为何先切实际依赖

当前 routed-only redispatch ABI3 会在 dH 前等待全部 X/P 搬运和所有源
读租约结束。旧模型关于逐专家 dO 与 P1 重叠的假设不完整，不能直接用
其47ms误差的排序决定上线。

正在构建的窄干预只提前 dHp=dY@W2 的 AIC GEMM：AIV epilogue 仍等待 P，
任何 dW/dX 前仍保留原全局读租约，保护被 X 源借用的 ptrD。进一步比较
home-first P1、W2恢复前置/W1后置。此处不新增激活拷贝，不改变数值契约。
四版同源构建、29项设备门禁和8格整步筛选已完成。高偏斜下 mode3 对
matched mode0 的 Native/UltraEP 降幅约1.712%/1.497%，均未达到2%扩展门槛；
低偏斜 Native 无活跃副本，mode3 是不生效控制，不能用于否定 no-copy 优化。
每格仅一次进程，不能声称稳定收益。mode4/5 的非对称 P1 顺序与 no-copy
扩展仅通过288项CPU覆盖/依赖检查，尚未编译或通过NPU验收。

只有这类 native 依赖干预测出空间后，才建立当前 source×expert、真实
tile/恢复、完整梯度贡献者释放与固定 owner 服务的联合成本和搜索器。
复制预算、字节最少或“考虑反向”本身都不是足够的论文创新。

当前仍缺：完整六档强基线、预算扫描、五次独立交织重复、四格消融、
held-out 路由/模型形状及自然训练质量。原聚合/专家梯度门槛通过，但严格
小参数误差与旧 MoonEP 长轨迹告警尚未全部解决，不能写成全面数值验收。

## 7. 元数据初始化解释了 EPLB 差距的大部分

保持同一计划、同一梯度生产/服务顺序，只缓存精确整数排序键，14项设备
正确性门禁全部通过。16格正常在线整步已完成并独立下载审计。高偏斜
EPLB-fixed 从1639.60降到1308.72ms（20.18%）；Native从1210.34降到1171.82ms
（3.18%），UltraEP从1193.59降到1157.11ms（3.06%）。低偏斜 EPLB 降21.03%，
没有活跃副本的 Native 控制仅波动-0.015%。

旧排序在比较专家时反复从uncached全局plan扫描贡献者集合，EPLB触发次数
更多。这是可修复的执行器成本，不能误称策略必然代价。给两边同时修复后，
高偏斜EPLB与UltraEP差距从446.01缩小到151.61ms，约消除旧差距的66.0%。
剩余差距仍需隔离完整dW贡献、恢复和流水，不能直接全部归到通信。

后续30个新进程已完成并独立下载审计：高偏斜五对确认的Native/Ultra/EPLB
整步降幅为3.158%/3.117%/20.191%，配对log-t95下界均超过2%；未对多重
比较调整，统计单位是进程对，不能把20个postwarmup更新当20次重复。
低偏斜/budget32仍只有筛选。精确表和审计见
`docs/gradient-key-cache-diagnostic.md`及`research/single_mb_fbw/analysis/gradient-key-cache-screen-audit.json`。
该修复必须用于全部强基线，不充当新planner创新。
