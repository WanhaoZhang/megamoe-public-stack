# 持久专家副本缓存的实现边界

2026-11-15源码审计记录。此文是工程约束和后续验收清单，**没有实现Afore完整算法，也没有新增缓存性能数据**。18次v1筛查已完成，v3单独研究公开FSDP前向驻留；本文不改变运行代码。新增CPU规划参考见`AFORE-ADAPTATION.md`，它没有实际缓存命中。

## 现有数据流

`stack/tools/temporal_model.py`的owner provider从框架持有的FP32参数取得BF16计算权重，`grouped_experts.py`根据一层或两层域发布给Native。`source/native-balanced/ascend_deepep/fsdp2_bridge.py`在forward/recompute/backward路径执行`_publish`。现有计划重放缓存的是不可变执行计划；它不等于跨调用保留可复用的专家权重。

普通池使用32home+32replica，跨两层域池使用64home+0replica。物理slot编号只在对应池/域内有意义；不同层的同编号专家拥有不同参数。当前执行器串行复用这些池，并保留多份前向计划用于checkpoint重放及反向。

因此，直接增加一个Python字典并跳过`_publish`，会把“同slot编号”误认成“同一份有效权重”，不能作为副本缓存实现。

## 已核对的设备协议位置

- `kernels/mega_moe_bf16/include/mega_moe_replica_descriptor.h`：当前描述符为14个int64字段，只有地址、形状及rank信息，没有逻辑layer/optimizer版本、命中掩码或驻留提交记录。新缓存不能仅靠Python注释改变这份ABI。
- `mega_moe_replica_pipeline.h`中的`CopyW1Stripes`、`CopyW2StripeRange`/`CopyW2Stripes`及`CopyProjection`执行搬运并发布本调用epoch；`Publish`/`WaitStripe`/`Wait`检查该epoch。缓存命中即使不搬运，仍需在确认数据有效后满足同一次调用的就绪协议，不能直接把旧ready当新ready。
- 同文件的`ReadyCache`在地址/epoch/projection改变时失效；它缓存已通过的就绪检查，不代表持久专家权重缓存。
- `mega_moe_bf16.h`中复制函数由前向流水调用。反向恢复路径也必须逐一审计，不能只改前向后假设checkpoint重放和dx会读到正确权重。
- `fsdp2_bridge.py:295`的`_publish`仅向池写入当前home镜像，没有为跨层相同slot建立版本身份；`_enter`的串行池保护也仍然存在。

这些是当前源码边界，不是新ABI已经实现。最小设备改动仍需要独立工作区、明确缓存租约、版本提交与匹配构建，不能修改当前v3计时过程中冻结的SO或源码。

## 必须明确的身份与资源

1. 逻辑缓存键至少包括`(optimizer_version, layer, expert, compute_dtype)`。optimizer owner保持固定，Adam后旧计算镜像失效；不能跨更新复用旧版本。
2. 物理位置包括`(pool, destination_rank, slot)`。当前版本中的有效内容、正在复制的epoch、原始前向/重放/反向租约必须分别记录。
3. 所有rank共享确定性的副本部署决定；ready标记的epoch是通信协议状态，不能用权重版本号直接替换。
4. 只有没有活跃读取/搬运租约的槽可被回收。即使写入相同数值，也不能在无同步条件下并发覆盖设备正在读取的地址。
5. 多层缓存需要实际持久存储或支持层身份的槽位协议。只记录旧副本元数据而下一次发布已经覆盖其存储，不构成命中。
6. 未改变当前逐任务梯度协议前，每个实际处理任务的副本仍承担完整的梯度交付义务。权重命中不允许省掉dW贡献或owner等待。

主形状每层本地32个routed专家的BF16权重为`32×3×1024×4096×2 = 0.75GiB/rank`；四层共3GiB/rank。该值只是权重字节数，不是实现缓存后总峰值的预测，也不能忽略它与已有池的重叠。任何持久存储必须计入共同90GiB预算，向可使用它的全部基线开放。

## 首版可审查的分步路径

1. **只复用合法计算镜像。** 对固定home参数的BF16转换和镜像建立明确的版本身份；先在同一层、同一optimizer版本的重复调用上验证。若缓存改变框架图或BF16中间舍入，必须重新做逐参数梯度检查。
2. **再引入远端副本驻留。** 计划编译输出desired副本与actual-resident对照，权重通信根据“实际缺失”执行，梯度贡献者仍仅由实际非零quota导出。
3. **记录真实命中。** 包括版本/层/专家、slot、已避免的转换/权重字节、未避免的梯度字节，以及缓存服务/失效/在线决策时间。缓存计划或跳过日志不计作命中。
4. **保持公共预算。** 缓存容量固定并预登记，基线与新策略都获得相同可用槽位/内存上限。不能把闲置池的预分配字节当已释放HBM。
5. **冻结新ABI与构建。** 先审计Host/worker的边界、ready epoch及重放语义，重新构建匹配SO并检查实际加载能力。不能沿用旧SO指纹宣称新缓存已部署。

## 必须通过的正确性用例

- 同层重复调用命中；不同层相同expert编号拒绝误命中。
- 不同optimizer版本拒绝旧权重；至少三次Adam更新并与无缓存参考比较。
- 原始前向、checkpoint重放、反序反向及跨层保存态并存。
- 槽位冲突、热点切换、空专家、零任务副本、复制未完成及过期ready标记。
- 各rank缓存计划一致；非法缓存计划集体拒绝后能恢复。
- 输出相对L2≤0.004、dx/router/routed/shared梯度≤0.02，逐owner参数检查；全局梯度缩放一次。
- 对照实际SHMEM权重通信及原TimerV2依赖记录，证明命中真实省去了搬运，同时没有漏梯度。

完成这些之前，报告继续写“缓存未实现”。现有无探测固定/FIFO已经消除了多余观察工作，应作为缓存版本比较的起点，不能恢复较弱基线来获得正结果。
