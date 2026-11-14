# Python 高偏斜场景：owner dW1 尾部进一步优化

> 2026-11-02 合入状态：本报告对应实现及最终说明已合入 `megamoe-balanced`；文中“未push”描述2026-11-01原实验交付时状态。默认P1cache0、owner1:1不变，本次不新增设备验收。相对node/analysis等原始证据链接属于文中技术项目归档，大文件不随Git分发；当前可复用入口见[合入说明](20261102-balanced-python-integration.md)。

结论：已在 P1 cache 基线 b914741 上实现并验证 owner 早 AIV1 : 晚 AIV0 的 2:1 分工。固定大形状五输出逐位正确、EP2/4/8 生命周期及原 TimerV2 流水验证通过；**按用户要求交付三组有效 TimerOFF 配对：B 降幅中位 4.306%，FB 2.838%；原四组保留条件未完成，默认仍为 1:1**。代码位于独立分支 `codex/python-owner-tail-20261101`，没有 push 或修改四个共同分支。

技术项目：`D:/megamoe_train/paper-python-owner-tail-20261101`。原采样项目：`D:/megamoe_train/paper-python-v4pro-20261101`。大 PT、模型与完整环境仍在远端原项目/迁移副本中，当前项目回传原始 NPY、四份完整前反向 JSON、SO、日志、源码输入与指纹，具体见 DELIVERY.json。

## 问题定位

planner 将每个 EP 组初始 owner 行数 `[44819,4067,6672,9978]` 拉平到每 rank 16384，但归主工作仍集中在 rank0/4。每组 E1 复制到 rank1/2、E0 复制到 rank3；rank0/4 各自负责 E0/E1 的完整 owner 输出。行数均衡没有消除 owner 服务及早晚加入差异。

重新检查全部 32 个早/晚 worker：早 AIV1 全部结束 dW1 后，最早晚 AIV0 才开始 dW1，rank0/4 中间分别空出 3.0005/2.9693 ms。晚 worker 仍与早 worker 等额承担 42 个 dW1 tile，服务约 2.16 ms，等待贡献仅约 0.12 ms。因此该尾部主要暴露已有贡献的归并服务；继续只提前副本 ready 无法消除它。

最终同步大块等待还含到达时间差，不能全部解释成网络传输，更不能直接删 barrier。我们没有在 TimerOFF 添加分段打点；上述为同 rank TimerON 依赖证据，精确 OFF 收益必须由完整对照确定。

[全部 worker 审计](D:/megamoe_train/paper-python-owner-tail-20261101/analysis/input-pipeline-audit.json)，[详细流水分析](D:/megamoe_train/paper-python-owner-tail-20261101/PIPELINE-ANALYSIS.md)。

## 实现

复用原有 `MegaFluxOwnerSchedule::Task` 静态整数映射，使早 worker 分担更多独立 tile：

| 每 worker | 1:1 早 / 晚 | 2:1 早 / 晚 |
|---|---:|---:|
| dW1 tile | 42 / 42 | 56 / 28 |
| dW2 tile | 21 / 21 | 28 / 14 |

只新增三项构建/身份接线：CMake 的 `MEGAFLUX_OWNER_EARLY_RATIO`、构建脚本及 manifest、只读 SO getter `megaflux_worker_owner_early_ratio`。构建默认 1，仅允许 1/2；ratio2 要求现有 earlyDW1=1、DX panel acquire cache=1、owner16、compact0、row-ready0，仍受原 S9/flags0/panel2/operand-ready1 等限制。本实验额外固定 P1 cache=1；其生产默认依旧 0。

移植的是 Finance d338264 中三个控制改动，**没有借用该场景的性能结论**。实际实现差异 18 插入、1 删除；另加一个真实 Task CPU 测试及实验文档。Task/归约体、每元素贡献者 SUM 次序、完整 K/BF16 partial、ready/epoch/退休协议、UB/workspace、barrier、planner、专家生产顺序和 Timer schema 不变。五个关键头文件逐字节与 b914741 相同，见 [SOURCE-INVARIANTS.json](D:/megamoe_train/paper-python-owner-tail-20261101/SOURCE-INVARIANTS.json)。

## 流水实测

以下 C0=P1cache1/owner1:1，C1=P1cache1/owner2:1，均来自 rack07os1 / TASK_QUEUE_ENABLE=2 的原 TimerV2（与下节节点75 / 队列1的OFF分开描述），**各 rank 本地起点、仅同 rank 前后比较**；阶段量为对应 worker 集合中位数。

| 指标（ms） | rank0 C0→C1 | rank4 C0→C1 |
|---|---:|---:|
| 晚 AIV0 dW1 服务 | 2.156→1.435 | 2.161→1.443 |
| 晚 AIV0 dW1 等贡献 | 0.119→0.080 | 0.120→0.080 |
| 晚 AIV0 dW1 完成 | 17.254→16.157 | 17.587→16.412 |
| 早 AIV1 dW1 完成 | 11.471→12.030 | 11.822→12.262 |
| XSYNC_UNPERMUTE 完成 | 17.339→16.218 | 17.631→16.459 |
| kernel 跨度 | 17.568→16.450 | 17.863→16.694 |

晚端服务按任务量下降约三分之一，早端服务增加，但仍远早于晚端完成；其新增工作部分填入原来的贡献等待窗口。rank1/5 的 P1 跨度分别 3.792→3.793、3.822→3.828 ms，基本未变；AIC 尾部也大致不变。改善主要沿“晚 owner 服务变短→owner 完成提前→最终同步完成提前”传递，而非 P1 再次加速。总归约字节和算术量没有减少。

- [优化前反向 JSON](D:/megamoe_train/paper-python-owner-tail-20261101/node71/analysis/control-no-waits/all_8_rank_backward_trace.json)
- [优化后反向 JSON](D:/megamoe_train/paper-python-owner-tail-20261101/node71/analysis/candidate-no-waits/all_8_rank_backward_trace.json)
- [优化前前向 JSON](D:/megamoe_train/paper-python-owner-tail-20261101/node71/analysis/control-no-waits/all_8_rank_forward_trace.json)
- [优化后前向 JSON](D:/megamoe_train/paper-python-owner-tail-20261101/node71/analysis/candidate-no-waits/all_8_rank_forward_trace.json)
- [对比图](D:/megamoe_train/paper-python-owner-tail-20261101/analysis/pipeline-summary.png)、[逐 rank 统计](D:/megamoe_train/paper-python-owner-tail-20261101/analysis/pipeline-comparison.json)、[tile 审计](D:/megamoe_train/paper-python-owner-tail-20261101/analysis/owner-effect-audit.json)

每 rank 96 轨道，保留原始逐条区间；同轨 duration overlap=0，不绘制 wait 块。原 wait 数值仅用于文字诊断。不能跨 rank 相减估通信延迟，不能把 ON 缩短直接算成 OFF 加速。

## 正确性与构建

| 检查 | 结果 / 边界 |
|---|---|
| 真实 Task 分区 | 304 组覆盖、唯一性及份额检查通过，含零任务、非整除尾部与多 worker |
| CMake/default/getter/身份 | 10 组配置检查、三个新 SO 实际 getter 与各 155 项源码输入 SHA 通过 |
| 大形状 | ratio1-OFF、ratio2-OFF、ratio2-ON 各八 rank 连续三次，Y/dX/dp/dW1/dW2 全部逐位匹配冻结原 capture |
| 生命周期 | EP2/4/8、两版：两层、均衡/热点/切换、pool 权重投毒、checkpoint、SGD、原独立参考通过 |
| 两版生命周期比对 | 14 个 rank 对的五输出及 SGD 更新 SHA 完全一致 |
| TimerV2 导出 | 控制/候选前反向四份 JSON，八 rank、96 轨道、零重叠 audit 通过 |

[构建身份](D:/megamoe_train/paper-python-owner-tail-20261101/node71/BUILD-VERIFICATION.json)，[CMake](D:/megamoe_train/paper-python-owner-tail-20261101/node71/CMAKE-CHECKS.json)，[正确性矩阵](D:/megamoe_train/paper-python-owner-tail-20261101/node71/LIFECYCLE-COMPARISON.json)。既有独立参考门限未放宽；不扩称 legacy layout、其他 EP 或整网通过。

新 SO SHA256：

| 库 | SHA256 |
|---|---|
| owner1-off-a2 | `7f71bed9abf1f6bcdadff14cf4fc461a455d42a103db04641920ad670e630dc4` |
| owner2-off-a2 | `25deef9fb1687cd5ce2cbe2cce94f7e97122ccacd01c838d53d8390d4b9da44d` |
| owner2-on-a2 | `dc74a610777d0377514356e5815e650918a5bb0be12acb4a287a8ce97495c976` |

## TimerOFF：三组有效短验证

**用户选择先交付三组有效结果，本轮不再启动设备实验。** 原计划四组保留验收未完成，不能标记正式接受；以下仅为 rack07os5、双方 `TASK_QUEUE_ENABLE=1` 的固定capture三组短验证，不外推原队列2或整网，也不与上一轮P1收益相加。

C0=上轮P1cache1/owner1:1，C1仅启用owner2:1。两次C0自重复+三组AB/BA/AB，共8个有效进程，全部exit0、pre/post门禁及计时前后五hash通过。每进程B/FB各10预热+30测量，逐样本取八rank最大wall；Timer/profiler/诊断回读关闭，hash在计时外。B包括恢复/计算/归主/finish/autograd，FB每次含在线MoonEP/home刷新；不含真实整网FSDP/router/shared/optimizer。

| 配对 | 顺序 | B C0 / C1，ms | B降幅 | FB C0 / C1，ms | FB降幅 |
|---|---|---:|---:|---:|---:|
| 1 | C0→C1 | 17.561855 / 16.661695 | +5.126% | 28.025620 / 27.146540 | +3.137% |
| 2 | C1→C0 | 17.570495 / 16.832755 | +4.199% | 27.990165 / 27.253075 | +2.633% |
| 3 | C0→C1 | 17.520915 / 16.766385 | +4.306% | 27.983920 / 27.189815 | +2.838% |

三组配对降幅中位：完整B **4.306%**，直接FB **2.838%**。三组B/FB和P90均改善。有效C0共5轮，中位数控制波动：B 0.506%，FB 0.284%。波动=有效C0对应量极差/中位数；原B阈值计算为2.00%，但**缺第四组，完整保留条件仍未完成**。

| 配对 | B P90 C0 / C1，ms | B P90降幅 | FB P90 C0 / C1，ms | FB P90降幅 |
|---|---:|---:|---:|---:|
| 1 | 17.668273 / 16.758092 | +5.151% | 28.315931 / 27.386817 | +3.281% |
| 2 | 17.853357 / 17.097224 | +4.235% | 28.123637 / 27.458447 | +2.365% |
| 3 | 17.650030 / 16.944807 | +3.996% | 28.093422 / 27.329792 | +2.718% |

P90按(n-1)q线性插值；C0 P90控制波动：B 1.468%，FB 0.791%。

[完整30样本/逐rank/统计](D:/megamoe_train/paper-python-owner-tail-20261101/node75/OWNER-PERFORMANCE-THREE-PAIRS.json)，[八个有效进程](D:/megamoe_train/paper-python-owner-tail-20261101/node75/OWNER-PERFORMANCE-QUEUE-THREE-PAIRS.json)，[三组性能图](D:/megamoe_train/paper-python-owner-tail-20261101/analysis/paired-performance-three-pairs.png)。

原队列2在节点71/73发生MegaMoE调用前HCCL初始化AICPU ret0x2a，根因未确立，HCCL-first未稳定修复。队列1只作为双方统一的隔离条件，不表示原队列2修复；两节点旧轮和预检均未混入本表。

第四组多次被外部进程进入打断：a1的C1退出77；a2的C0退出77；a3的C0退出76。每次立即停止，后续新编号重新两次准入。**采用整组排除规则**：a2/a3中本来eligible的C1也不计入，三次末组尝试全部原件保留；前三组完整、逐次eligible的数据不删、不挑快轮。外部归属现场证据见node75/EXTERNAL-HOLDERS-node75-pair4-a1.json（8个宿主机sshd.service进程，不在自有容器）。未kill外部进程、未reset设备。

用户最终要求先交付三组，因此节点75新TimerON采集未启动；本报告流水量来自前述节点71已验证的原队列2图，**不冒充与队列1OFF完全相同的运行条件**。结构性改善由tile审计与正确性支持，OFF收益由上述三组独立完整wall支持，不能精确拆分为OFF各阶段预算。停止自有idle容器及真实设备门禁见DELIVERY.json。

## 还有哪些瓶颈

1. owner 晚加入仍在：rank4 晚 dW1 16.412 ms 完成、早端 12.262 ms，但其中时间差不是全部可省量；总梯度读写/转换/SUM 没减少。
2. 非 owner 的 AIC 尾部已接近最终完成：rank1/2/5/6 约 15.91–16.19 ms，继续削 owner 可能受 home 专家 dW1 限制，不能无条件加大早端份额。
3. 早 worker 仍存在贡献等待，顺序 tile 服务也会等待当前 tile；等待与计算有重叠，不能把等待加总估算收益。本轮没有叠 ready scan/双缓冲/ratio3。
4. owner 完成后的 unpermute/dp/最终协议仍必须保留。下一步先补齐同节点、同队列的完整 OFF 四组对照，再依据结果决定是否保留或改方向。

这轮交付的是已实现、数值和流水通过、三组OFF有改善的实验候选；默认 1:1、P1 默认0，未推广到生产或整网。复现见 [REPRODUCTION.md](D:/megamoe_train/paper-python-owner-tail-20261101/REPRODUCTION.md)。


配套采样/映射/运行/统计脚本交付于 `D:/megamoe_train/paper-python-owner-tail-20261101/code/reproduction-scripts.zip`；大模型、PT与原始trace不进Git，位置及范围见技术项目DELIVERY.json。
