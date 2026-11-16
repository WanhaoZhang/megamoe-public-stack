# Afore 风格基线：语义参考与训练接入边界

2026-11-15。原文：[Afore: Attention–FFN Disaggregation with Overlapped Reconfiguration of Experts](https://arxiv.org/html/2610.03203v1)，重点核对第5.1–5.2、6.1–6.2节；不是声称已经全文精读资料库。本次没有找到经核实的作者代码仓库，按论文算法写的参考不能冒称作者实现。

原文研究AFD推理。其规划先在既有副本间均衡tile，再搜索新副本，以计算收益扣除暴露迁移时间决定是否部署；槽位包含投影状态和保护集合。实际路由结果在Attention后产生，并用于FFN；缺失权重搬运与在途FFN重叠。训练适配需另做梯度和参数版本协议。

## 本次实际完成

`prototypes/afore_reference.py`是CPU/NumPy语义参考，不被训练栈导入。它复用当前`planner_contract.py`的`PlannerProposal(copies, quota)`与公共编译器，没有建立第二套生产执行接口。

- 输入为当前真实来源计数、调用方提供的驻留绑定、上一分配、保护槽、容量和成本参数。
- 按128行tile初始化、既有副本内搬移、受限新副本搜索；固定排序。以传入的聚合迁移成本函数和实际可用重叠窗口选择正净收益候选，未选候选完整回滚。
- 用固定source-major区间构造来源quota，真实尾块不填充成虚假token。公共编译器检查守恒、专家身份、容量和实际贡献者。
- 专门区分`projected_resident`与本次活跃`copies`：缓存了但没有任务的槽，不登记梯度贡献者。
- 训练适配的版本键为`(optimizer_step, layer, bf16)`；版本变更拒绝带活跃租约的旧状态。保护集合应覆盖全部训练前向、重放和反向租约，范围比推理的前一microbatch更严格。

`validate_afore_reference.py`通过10个CPU测试，其中包含120个固定seed的公共编译器用例。它们检查已有副本优先、正负迁移收益、真实尾块、容量拒绝、空专家、换槽保护、LRU、版本失效、配额与实际贡献者。原始输出及源码SHA在`diagnostics/afore-reference-tests.{json,log}`。这些用例没有检查NPU数学或真实权重就绪。

## 仍然没有完成

1. 没有把返回的投影槽表提交给执行器，没有发生实际缓存命中、权重搬运或通信省略；`weight_migration_executed`始终为false。
2. 没有测得训练形状的tile、迁移、竞争与可用重叠成本。测试中的成本参数只是触发边界用的数值，不是性能预测。
3. 没有AscendC在线实现、SHMEM就绪epoch/槽位租约改造或任何25更新Afore性能点。
4. 没有梯度交付成本项。保留非零配额副本的全部梯度义务，是后续训练接入的必需条件，不能根据权重命中省掉梯度。

公开FSDP的`reshard_after_forward=False`仅改变框架计算权重生命周期，也不能替代这里的远端副本缓存。具体设备接入清单在`CACHE-IMPLEMENTATION-BOUNDARY.md`。

## 下一步的最小可验收切片

先在一个固定层/optimizer版本上支持两个连续调用的真实驻留复用，测量原TimerV2中的实际权重流量与就绪等待；随后覆盖跨层同expert编号、热点切换、重放、反序反向和三次Adam更新。它通过之后再增加并发预取。基线与新策略共享相同缓存预算和通用执行优化，成本校准用独立数据，正式测试前冻结。

在此之前，报告应继续列“Afore训练设备基线未实现”，不能把这10个CPU测试或v3驻留pilot画成Afore性能。
