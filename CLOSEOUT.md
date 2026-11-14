# Planner EP8 实验收尾

2026-11-14：按用户“没效果可以收尾一下了”的要求停止后续实验，保留负结果。

本轮已完成统一 Planner 接口、五种已有策略接入、独立成本校准和 `step_time` 原型。
108 次主矩阵和 12 次第四轮均保留；共 120 次运行、2400 个测量步。
数据与代码身份另查 [收尾审计](CLOSEOUT-AUDIT.json)。这是已完成范围的审计，原完整验证仍未完成。

## 结论

- 主矩阵 `step_time` 的 900 次全局规划调用，搜索后计划采用数为 **0**，全部精确回退到 Native 起点。
- 完整训练步比 Native 起点慢约 **0.34%–0.85%**。当前没有证据证明新增搜索带来收益。
- MaxVio 0.2 相对 UltraEP 的四轮步时改善约 1.2%，达到原统计门槛；实际执行的仍是 Native 起点，不能归因于新的梯度服务模型。
- 成本模型保留 47.40ms 误差裕量，在线搜索本身有成本；单机拓扑差异不可分辨。没有针对主矩阵重新拟合或为正结果改变条件。

## 保留的限制

第四轮 MaxVio5 的一组训练在第22步出现全局梯度范数相对 MoonEP 相差2.415%的附加轨迹告警。
120次运行的初始逻辑参数、输入、路由及实际基础运行库一致，但该告警的具体数值根因没有解决。
不同训练轨迹的范数差不等于同权重梯度L2；同样，部分同权重检查也不能替代完整25步验收。

第一次同权重诊断通过前4步后出现HCCL错误；加主机对齐的重试也仅完成前4步，随后按用户要求中止。
原日志、线程栈、失败和中止状态全部保存。**不宣称整项25步数值检查已通过。**

两个独立路由银行72次、条件消融54次和隔离规划银行均未执行，已取消；它们不进入任何曲线。
`ACCEPTANCE.json` 保持 `complete=false`，同时明确 `closed_at_user_request=true`。

## 产物入口

- [完整结果和配置说明](reports/REPORT.md)
- [主矩阵逐点CSV](analysis/points.csv)、[逐次CSV](analysis/runs.csv)
- [新策略实际采用/回退记录](analysis/planner-decisions.csv)
- [四轮统计](analysis/stability-four-rounds.json)
- [源码补丁](patches/)、[实现说明](reports/STEP-TIME-IMPLEMENTATION.md)、[决策记录](reports/DECISIONS.md)
- [基线源代码与SO](evidence/BASELINE-FREEZE.json)、[新策略冻结清单](evidence/STEP-TIME-FREEZE.json)
- [释放本任务进程与八卡的记录](evidence/USER-CLOSEOUT-RELEASE.json)
- [讲解页](../balanced-kernel-explainer-20261106/overview.html#planner-comparison)

所有实验数据留在本项目；原三后端章节保留。自动设备接入已禁用，不再续跑。
