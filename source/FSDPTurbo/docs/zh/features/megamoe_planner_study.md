# MegaMoE单microbatch规划实验

研究分支`research/single-mb-fbw-planner`保持参数和优化器owner固定，沿用公共规划校验及MegaMoE执行器。

`MegaMoeBalancedConfig.planner`新增实验值`eplb_budget_N`，其中N为0、1、2、4、8、16、32、64、128或256。N限制整个EP组额外物理实例预算，不改变每卡副本槽和接收容量；同卡实例合并后，实际非零副本数可以更少。原`eplb_fixed`行为保持不变。

需要配套研究分支的Host扩展和支持budget ABI1的planner动态库。旧动态库明确拒绝，不静默回退。配置构造仅验证名字；预算与实际EP及槽容量的关系由native规划入口验证。

这些名字用于训练代价的受控干预，不表示新策略已通过设备正确性或性能验收。CPU语义检查与完整训练步收益分别记录，原训练精度、逻辑路由、重计算及梯度归并契约保持一致。

## 参数读取对照

`MegaMoeBalancedConfig.parameter_read_mode`用于区分FP32主参数读取、形状检查和BF16计算镜像转换的开销，默认值为`legacy`。

|取值|行为|
|---|---|
|`legacy`|保留原有形状检查和实际参数回调各自执行的转换，作为匹配对照|
|`cast_once`|形状检查读取原参数，仅在实际参数回调中转换为BF16|
|`fp32_to_home`|回调保留FP32参数，由native的home镜像复制完成BF16转换|

三个方式都在每次调用时读取当前专家Parameter，不捕获FSDP包装前的参数。该对照仅允许`shared_expert_mode="framework"`；配置校验不会改动模型或参数dtype。模式加入池身份和跨rank签名检查，不能在同一池中混用。

`checkpoint_home_reuse`默认关闭，只允许使用配套native桥接实现的即时重计算权重复用对照。上述参数读取和复用探针的CPU控制检查不代表NPU数值、长期训练或性能资格。发布保留研究状态，不自动启用任何非默认方式。完整研究入口见独立源码仓库的`research/single-mb-fbw-planner`分支。
