# TimerV2 默认 trace 导出修改（2026-10-28）

目标：kernel 打点默认只产前向、反向两个 all-rank Chrome Trace，使用用户提供的 forward/backward 参考格式。每 rank 固定 group0..31 的 AIC0/AIV0/AIV1 共 96 轨道；planner、dispatch、权重搬移、梯度发布及 owner 加载/转换/SUM/写回都归入原核轨道。阶段汇总包络明确标注含内部空隙，不冒充纯 busy 时间。原始记录保留，未知 schema 拒绝导出。

实施：新增独立离线导出器与 after-run wrapper，接入现有采集工具的生成入口；统一既有 CSV trace 轨道；文档说明两文件、重算归属、等待显示、事件覆盖。只修改导出/CPU工具，不启动或恢复已暂停的实验，不修改冻结 SO/训练数学。

验证：参考 trace 结构、CPU schema/不完整输入拒绝、现有主层与 MTP 全八 rank 的 forward/recompute/backward 原始 TimerV2 数据，核对每 rank 96 轨道、各采样事件类型的导出覆盖和周期边界。检查默认正好两个 all-rank trace，原始输入 SHA 不变；既有 backward parser 自检回归。提交 Native 和根仓 gitlink/使用说明，普通 fast-forward push 到现有 GitCode/GitHub 分支并读回 SHA。

限制：既有采样是 block/group/subblock 编号，不能补造物理 core ID 或跨 rank 时钟同步；本次属于离线工具验证，不构成候选数值或性能验收。

后续显示修正：首版跨不连续操作合成包络造成假并行，现改为原始子区间；外层范围用起止标记。增加交错 load/cast 反例、scope 处理及零重叠硬门禁；同一份新 rho6 原始 NPY 离线重导出并独立回读，不重新运行 NPU。
