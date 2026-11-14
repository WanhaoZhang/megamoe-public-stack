# 默认 kernel trace 验证（2026-10-28）

## 最新：修正不连续包络导致的假重叠

- 首版 `9356aab` 仅核对事件覆盖、轨道数和周期边界，遗漏了子操作包络相交检查；原来的“覆盖通过”不能当作串行显示正确的证据。
- 新增交错 load/cast、父 scope 与嵌套等待回归：共7项 CPU 契约测试通过。原始记录不平移、不裁剪；独立 load/cast/SUM/store 逐段绘制，粗范围仅作标记。正时长事件有重叠直接拒绝。
- 对本次新 rho6 单算子原始 NPY 离线重导出，前向/反向均8rank、每rank96轨道；再独立流式读取 JSON，检查所有正时长事件原始周期同轨道零重叠。前向22,472个 duration条与26,500个标记，反向656,128个 duration条与147,212个标记，原始子操作起止保留。
- 新图约22.98MB/413.38MB；归并微操作逐条展示使反向文件增大。最新文件与审计在项目 `rho6-kernel-trace-20261028-a1/serial-trace-v2/`、`TRACE-LAYOUT-V2-ACCEPTANCE.json`；旧版包络图放入 `superseded-envelope-view/`。
- 这次没有重新运行 NPU，之前的五输出/运行后检查证据保持原样。只修正可视化和后续导出默认。

## 首版覆盖验证（历史，显示包络已被上述修正替代）

本轮只做 CPU/离线验证，未启动或恢复 NPU 实验，未修改 kernel、训练数学或冻结 SO。

- `python integration/wuhan/test_timer_trace_cpu.py`：4 项通过。覆盖默认两个 phase 文件、96 轨道、重算只归入反向并保留原始间隔、原始输入不改写、不覆盖已有输出、缺 rank/未知 schema 拒绝，以及旧 CSV 中自定义权重/规约名称不被过滤。
- `python test/megamoe_bf16_test/selftest_backward_timer.py`：原有固定阶段、逐专家、等待分类回归通过。
- 使用既有主层/MTP 的原 rho6、EP4×EFS2、H7168/F3072、全八 rank 原始 `.npy` 离线导出，分别核齐原前向、重算和反向。每层 1,384,322 条 stage 记录，实际出现的 38 种事件 kind 全部由原生阶段、汇总事件或明确隐藏的 WAIT 覆盖；没有把未采样的 kind32 宣称为已采到。
- 每层只产生前向/反向两个 all-rank trace（另有小审计 JSON），每个 rank 96 条轨道。前向 10,126 个图事件，反向（含重算）34,420 个；原始周期与图中时间基准逐事件重新核对通过。
- 主层文件约 3.59 MB / 14.34 MB，MTP 约 3.59 MB / 14.33 MB。与参考 `all_32_rank_{forward,backward}_trace.json` 一样采用事件数组和 group/AIC/AIV 轨道；本次实际 world8，不伪造为 32 rank。
- 新旧采集工具准备模板已接 after-run hook，旧 `producer-timer-tools-a*` 和原始实验目录不改写。安装器保留模板源码 SHA 备份；只在将来准备新工具时生效。
- 独立 bundle 的 base-layout fallback、after-run shell `bash -n`、重复 hook 拒绝、实际模板安装幂等与语法检查均通过。

原始证据留项目 `D:/megamoe_train/moonflux接整网/balanced-8p/rho6-optimization-20261027/`：`default-trace-main-validation-a2/`、`default-trace-mtp-validation-a1/`、`default-timer-trace-validation-summary.json`。大型 JSON/原始 NPY 不提交仓库。

本报告不构成 P4/cache 数值问题修复、原 MegaMoE 精度门槛通过、整网收益或物理 core ID/跨 rank 时钟校准的证明。
