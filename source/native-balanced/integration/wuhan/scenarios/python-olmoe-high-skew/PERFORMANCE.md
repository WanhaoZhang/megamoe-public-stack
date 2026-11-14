# 此固定场景的历史性能、流水和当前共同分支代码边界

> 共享专家分支也包含本例及两项优化；本页历史性能与build-reference属于balanced-only实验，shared组合及同事新异步bridge尚无本轮新设备验收。参见[合入边界](../../../../docs/experiments/20261102-shared-python-integration.md)。

2026-11-02，P1-cache、owner2:1及固定场景工具已合入 `megamoe-balanced`，原实验提交历史保留。**本次合入没有重新测 B/FB。** 本文记录同一固定场景已经完成的 routed 算子实验；历史SO/原始编译输入与当前合入源码比对见 [build-reference.json](build-reference.json)，不将TimerON或其他场景结果冒充新测量。

## 1. 代码版本和比较对象

| 名称 | 源码 / 配置 | 本场景证据 |
|---|---|---|
| 历史原基线 | `7c41ebf938363995838056677e73e5ea6a289d02`；复制专家 dW1 前移=1、DX panel acquire cache=1、P1 cache=0、owner1:1 | 作为下节 P1 实验 C0 |
| 历史 P1-cache 方案 | `b9147416010cc3c1e72b260dd22ba938b9247b34`；只增加 P1 completed-panel acquire cache=1 | 四组完整 B/FB，通过当时固定案例判据；开关默认0；现已合入共同分支 |
| 历史 owner2:1 方案 | 实现 `847a1f9178793886ae7bbd7abadd936b8e1af4da`，含最终报告 HEAD `ea98005f876b0ba5544533f8c22797de4bef2293`；基于 P1-cache=1，再将 owner early:late 从1:1改2:1 | 三组有效完整 B/FB；缺第四组，未正式接受；ratio生产默认1 |
| 本次共同开发分支 | `7c41ebf` 起点，合入上述两项优化及场景工具/文档；不合入shared-expert分支 | 13项CPU检查、历史编译输入/SO指纹核对；**本次未新增NPU性能或共享融合验收** |

当前分支已有DX/P1 cache和owner early ratio开关，实际配置以manifest/getter为准；**P1默认0、owner ratio默认1**，代码合入不代表候选已经启用。显式构建候选/基线/回退见 [合入说明](../../../../docs/experiments/20261102-balanced-python-integration.md)，场景数据与开关解耦。

这里两轮 C0 都是已有复制/排程优化路径，**不是完全原版 MegaMoE**。本例原 MegaMoE 只作为数值参考（cap65536避免原热点截断）；没有一组可用于声称“相对原版 MegaMoE 加速多少”的本例完整性能数据。不要引用以前均衡/随机/热点集中/热点分散矩阵的倍数替代它。

## 2. 所有性能的共同分母

固定输入见 [README](README.md)：world8、两个EP4组、E8/Top2、M8192/H7168/F3072、home2/slots2/cap20480、canonical BF16 saved、clip10，目标κ=2.481662403792143。在线MoonEP每次F重新规划并刷新home权重，B使用该次F的placement。

每进程 B 和 FB 各10次预热、30次测量，逐样本取八rank最大wall，再计算中位/P90。五输出hash在计时前后、计时区外校验，Timer/profiler/诊断回读关闭；两次C0自重复后交替AB/BA。P90按`(n-1)q`线性插值；控制波动=所有有效C0轮对应统计量极差/中位数。

完整B包含恢复/计算/归主/finish/autograd及梯度转换；直接FB额外含当次F的在线MoonEP/home刷新。**不含真实router、shared MLP、整网FSDP collective或optimizer。** 单算子结果不能换算整网吞吐。

历史环境：Ascend950DT、CANN9.2、torch2.10.0+cpu、torch_npu2.10.0.post6；镜像SHA `2754601232edf86a4b7dc6e2aca87cf86648e14491e173458ebbf3d5cbd6d2f8`；SHMEM SO SHA `246ed665479f676b608d2f318824c02cd1b9eba935720b8efb373c2095bc7112`。实际库/环境/输入/准入收据以原项目证据为准，不只看分支名。

## 3. 第一轮：P1 acquire cache

节点 rack07os1，原任务队列2；C0=earlyDW1+DXcache，C1额外开P1cache。以下单位ms，表值显示四位小数；降幅使用原未舍入数据计算。

| 配对/顺序 | B 中位 C0→C1 | B 降幅 | B P90 C0→C1 | FB 中位 C0→C1 | FB 降幅 | FB P90 C0→C1 |
|---|---:|---:|---:|---:|---:|---:|
| 1 / AB | 18.7253→17.7906 | 4.9914% | 18.9088→17.8859 | 29.0959→27.8811 | 4.1753% | 29.1681→28.0862 |
| 2 / BA | 18.8215→17.4808 | 7.1229% | 18.9916→17.5699 | 29.2579→27.8139 | 4.9354% | 29.3081→27.9005 |
| 3 / AB | 18.8569→17.5250 | 7.0631% | 18.9460→17.7027 | 29.1398→27.8502 | 4.4256% | 29.2115→27.9569 |
| 4 / BA | 18.8493→17.4674 | 7.3313% | 19.0070→17.6375 | 29.1885→27.7468 | 4.9393% | 29.4762→27.8593 |

四组配对降幅中位：**B 7.0930%，FB 4.6805%**。六轮有效C0中位数控制波动B0.8395%、FB0.8218%；B门槛max(2%,2×0.8395%)=2%。四组B/FB及P90均改善，通过当时固定案例判据。

原因：P1的W2实际GM→L1 load对同一已完成panel反复执行所有producer的精确epoch检查和DSB。每AIC/专家调用内记录已完成acquire；第一次真实load仍完整WaitPanel+DSB，复用时省重复检查，cache不跨调用。没有改数值累加、贡献SUM次序、ready/epoch/退休协议、planner或owner分工。

原TimerV2：副本rank1/5的P1跨度约6.04/6.06→3.79/3.82ms；AIC尾部分别18.299→15.987、18.367→16.122ms。owner晚AIV0仍约17.25/17.59ms才结束，成为下一处限制。ON阶段量不等于OFF可节省预算。

## 4. 第二轮：owner early:late=2:1

节点 **rack07os5，双方统一TASK_QUEUE_ENABLE=1**。C0=P1cache1/owner1:1，C1仅改owner2:1。原队列2在节点71/73曾在MegaMoE调用前HCCL初始化报AICPU ret0x2a，根因未修复；因此本轮OFF不能冒充原队列2验证。

| 配对/顺序 | B 中位 C0→C1 ms | B 降幅 | B P90 C0→C1 ms | FB 中位 C0→C1 ms | FB 降幅 | FB P90 C0→C1 ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 / AB | 17.561855→16.661695 | 5.126% | 17.668273→16.758092 | 28.025620→27.146540 | 3.137% | 28.315931→27.386817 |
| 2 / BA | 17.570495→16.832755 | 4.199% | 17.853357→17.097224 | 27.990165→27.253075 | 2.633% | 28.123637→27.458447 |
| 3 / AB | 17.520915→16.766385 | 4.306% | 17.650030→16.944807 | 27.983920→27.189815 | 2.838% | 28.093422→27.329792 |

三组降幅中位：**B 4.306%，FB 2.838%**。五轮有效C0中位控制波动B0.506%、FB0.284%；三组B/FB及P90均改善。

**第四组多次因外部进程进入而作废，用户要求先交付三组；formal_retention_accepted=false。** 失败整组排除，包括其中本来eligible的单版，不拼成第四组。这个是三组有效观察结果，不是原四配对计划已完成。两轮使用不同节点/队列，7.093%与4.306%不能相加或连乘作为已测总收益。

本轮控制SO SHA `8385529b7b1f637ebad082338f734c8df692e1108491d703743ebbe81b355851`；候选OFF SHA `25deef9fb1687cd5ce2cbe2cce94f7e97122ccacd01c838d53d8390d4b9da44d`。其他配套库见原构建身份清单。

## 5. 反向流水为何改善，剩下什么瓶颈

原生产顺序保留：全部dHp/SwiGLU backward后，按专家遍历；有活动副本的专家执行dW2→dW1→dX，其他专家dW2→dX，末尾补其dW1。owner以既有worker读取partial并归并。

P1cache基线上，rank0/4全部32个早AIV1结束dW1后，最早晚AIV0才开始dW1，中间约3.0005/2.9693ms。晚worker仍等额承担42个dW1 tile，服务约2.16ms，等贡献约0.12ms：主要暴露的是已有贡献的归并服务，不只是副本尚未ready。

owner2:1复用原Task静态映射，将每worker早/晚dW1的42/42改56/28，dW2的21/21改28/14。tile唯一完整，单元素贡献者顺序不变，总算术和归约字节不减少。更多独立tile交给较早加入的AIV1，缩短晚AIV0尾部。

下表来自 **rack07os1 / 队列2 的原TimerV2**，不是上节rack07os5 / 队列1的同条件图。均为同rank前后比较、对应worker集合中位数：

| 指标 ms | rank0 1:1→2:1 | rank4 1:1→2:1 |
|---|---:|---:|
| 晚AIV0 dW1服务 | 2.156→1.435 | 2.161→1.443 |
| 晚AIV0 dW1等贡献 | 0.119→0.080 | 0.120→0.080 |
| 晚AIV0 dW1完成 | 17.254→16.157 | 17.587→16.412 |
| 早AIV1 dW1完成 | 11.471→12.030 | 11.822→12.262 |
| XSYNC_UNPERMUTE完成 | 17.339→16.218 | 17.631→16.459 |
| kernel跨度 | 17.568→16.450 | 17.863→16.694 |

早端工作增加，但仍远早于晚端完成；rank1/5 P1跨度基本未变。依赖链是“晚owner服务缩短→归主完成提前→最终同步提前”。最终同步等待也含到达时差，不能全算成网络延迟或直接删除barrier。

剩余限制：

- 非owner rank1/2/5/6 AIC尾部约15.91–16.19ms，已接近最终完成；继续削owner可能转为home专家dW1限制。
- 早worker仍会等当前tile贡献；等待与计算可交叠，不能累加wait计数当作可省时长。
- 总梯度读写/cast/FP32 SUM没有减少；owner晚加入、unpermute/dp和最终协议继续存在。
- 下一轮应先补同节点同队列完整OFF配对及ON证据，再单独选方案，不自动叠scan、双缓冲或ratio3。

## 6. 正确性已有证据与新分支要做的事

历史两轮均完成：大形状八rank连续三次Y/dX/dp/dW1/dW2逐位对照、EP2/4/8两层/热点切换/pool权重投毒/checkpoint/SGD、原独立参考及更新SHA对照。原无副本MegaMoE参考Y/dX/dp记录最大数值差0；dW沿用原BF16 partial门限，P1轮最坏relative L2约0.2756%。未放宽门限，不把数值误差0记录说成已保存参考存储SHA。

这些历史结果**不自动覆盖当前共享专家V2/V3、shared输入、legacy布局、UAM、真实FSDP或整网**。本次接入的CPU路由证明也不替代密集输入回读和NPU验证。比较当前分支时先固定构建/SO getter、输入SHA、placement及分母，分别记录新结果。

## 7. 原始证据索引（不随Git复制）

工作站技术项目根：

- `D:/megamoe_train/paper-python-v4pro-20261101`：源采样与P1轮。
- `D:/megamoe_train/paper-python-owner-tail-20261101`：owner轮与HTML。

远端归档根均为 `/home/z00956592/megaflux/<同名项目>/`；历史原件在节点71/73/75相应目录，设备/归档可用性需重新确认。本目录保存小型摘要和源文件SHA；clone代码仓不会得到远端PT/SO/模型。

| 原件 | 相对所属项目的路径 |
|---|---|
| 来源/映射 | P1项目 `SCENARIO.md`、`node71/evidence/sampling-search-a2/selected.json`、`inputs/e8-ep4-m8192-k2/workload.json` |
| P1完整样本及统计 | P1项目 `node71/P1-PERFORMANCE-RESULTS.json`、`REPORT.md`、`REPRODUCTION.md` |
| owner完整样本及门禁索引 | owner项目 `node75/OWNER-PERFORMANCE-THREE-PAIRS.json`、`node75/OWNER-PERFORMANCE-QUEUE-THREE-PAIRS.json` |
| owner流水前/后 | owner项目 `node71/analysis/control-no-waits/all_8_rank_backward_trace.json`、`node71/analysis/candidate-no-waits/all_8_rank_backward_trace.json` |
| owner流水分析 | owner项目 `PIPELINE-ANALYSIS.md`、`analysis/owner-effect-audit.json` |
| 可交互展示 | owner项目 `presentation/index.html`，单文件HTML，内嵌8rank/96轨道原始区间 |
| 输入/SO/交付指纹 | 两项目 `DELIVERY.json`、`LOCAL-DELIVERY-VERIFICATION.json`；大资产另见各项目远端索引 |

给同事分享时优先发送本目录和Git提交；只有分析原始流水时再单独获取大JSON/HTML。不要把个人机器的loopback预览URL当作同事可访问地址。
