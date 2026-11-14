# train BF16 优化与默认融合校验的集成交付

本次以集成分支 `278d848` 为基础，合入已验证的 train BF16 适配提交 `eb2e053`，源码合并提交为 `9e5c12b`。保留集成分支的默认融合计划校验（`fused-v2` / validator ABI2）。W1 首四条 stripe 穿插实验 `a2b23f6` 不在本次交付中。

## 合入内容

- 前反向 Dispatch 细分、一次准备静态元数据和 S9 副本适配。
- BF16 SwiGLU、TopK/Gather 路径及反向 FP32 数值语义修正。
- 对应 Host/kernel 接口、精度检查、完整 F/B 计时入口、打点图和已有报告。
- 合并采集器的重复计时代码，保留 `--save-outputs`、预热与测量回执，以及实际 `fused-v2` 校验身份。

没有同步 W8A8 优化，没有提前 GMM2 调度，也没有引入新的路由或权重搬运策略。具体来源、实现和原有精度/性能记录见 [TRAIN-BF16-SYNC.md](TRAIN-BF16-SYNC.md)。默认融合校验说明见 [FUSED-PLAN-DEFAULT.md](FUSED-PLAN-DEFAULT.md)。

## 如何使用

这些 BF16 路径沿用形状门限自动选择，不需要新增启用环境变量。MoonEP 计划校验默认使用融合实现；旧框架传 `plan_validation="generic"` 仍兼容，并执行同一个 `fused-v2` 实现。

**必须使用同一源码版本构建并加载 `_C`、BF16/worker kernel、planner 和 ABI2 validator。** 本次同时包含反向 Dispatch 接口变化和 validator ABI2，不能只换 Python 文件或单独替换旧 SO。

在准备好 CANN、SHMEM 和子模块的自有隔离目录中：

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export SHMEM_HOME_PATH=/your/pinned/shmem/install
MAX_JOBS=4 bash test/megamoe_planner_test/build_benchmark.sh
MAX_JOBS=4 bash scripts/build_megaflux_timer_capture.sh integrated
```

运行环境设置和完整采集命令沿用 [TIMER-CAPTURE.md](TIMER-CAPTURE.md)。本次使用 `HCCL_CONNECT_TIMEOUT=180`。窗口容量、S9 panel restore 和既有设备空闲检查继续保留。

Windows 导出的源码包可能将 shell 转为 CRLF；可用 `git -c core.autocrlf=false archive` 导出，或仅在新建的隔离源码目录将 shell 换行规范化。Linux Git checkout 不需要为本次修改增加该步骤。

## 组合版本验收

本次验证使用原固定输入：八卡、两个 EP4 组，M8192/H7168/F3072/E8/TopK2、home2/slots2/capacity20480、rank_skew rho6/hot2/seed42、BF16 保存、limit10、MoonEP + S9/panel。

沿用 `eb2e053` 的五输出回执作为精度参考，不重跑旧基线。TimerOFF 三轮精度检查后进行 10 次预热 / 30 次完整 F/B 测量；TimerON 单独三轮与 TimerOFF 校验并导出两张 all-rank 图。各次设备运行前后均要求空闲门禁通过。

组合版本已完成验收：

- 在自有容器重建 `_C`、基础 BF16 依赖、worker、MoonEP planner 和 ABI2 validator，TimerOFF/ON 各有独立库；18 项 CPU 检查通过。
- `177.5.23.1` 上，八个 rank 的 TimerOFF 三轮五输出与固定 `eb2e053` 参考逐位一致，最后一次性能测量的输出也一致；TimerON 三轮五输出与这次 TimerOFF 逐位一致。
- 前反向 all-rank JSON 均导出成功，轨道审计零重叠。正式性能只使用 TimerOFF。接受的 OFF/ON 两次运行前后空闲门禁均通过。
- 28 个变更的 kernel/Host/CMake 文件与已验证 donor `eb2e053` 逐字节相同，交付不包含 W1 穿插提交及其宏。

| TimerOFF 完整 F/B，逐次取八个 rank 的最大值 | 中位数 | p95 |
| --- | ---: | ---: |
| 本次组合版本 `9e5c12b`，10 次预热 / 30 次正式测量 | 35.118160 ms | 35.587608 ms |
| 最新融合校验版 `05c1159`（文档提交 `278d848`）历史记录，同一机器 | 34.776100 ms | 35.175826 ms |

**本次组合版本相对最新融合校验版的历史记录，中位数慢约 0.98%，p95 慢约 1.17%；没有证据证明组合后有新增收益。** 两组不是本次交替配对实验，因此也不能把这约 1% 的差异直接归因于某一个优化。本次沿用已有基线，没有重新测旧版本。保留这两个记录，便于同事按其目标场景决定是否保留 train 适配。

精度校验是固定输入的版本回归和 TimerOFF/ON 一致性；不将其称为本次新增数学 golden 或收敛验证。设备配置为八卡、两个 EP4 组，此项验收不冒充真正 EP8。

可审查证据在 [reports/train-bf16-integration-20261029](reports/train-bf16-integration-20261029/)：`acceptance.json` 保存性能样本、五输出结论、动态库 SHA256 和门禁；`source-audit.json` 保存来源及排除项；`paired-ranks.json` 保存逐 rank 回执；`trace-manifest.json` 保存两张图的大小和 SHA256。

完整原始日志及打点图存于本地 `D:\gitcode\deepep\Profiling\moonep_train_integration\20261029\delivery-package`。图文件在其 `capture/kernel-trace/` 下，名称为 `all_8_rank_forward_trace.json` 和 `all_8_rank_backward_trace.json`，不将数百 MB 原始图放进 Git。对应远端目录为自有容器的 `/project/optimization/train-integrated-20261029-a1/integration-evidence/`；正式控制组为 `validation-b2`，Timer 配对为 `validation-b5`。

### 启动问题与复现注意事项

`177.8.24.1` 完成构建后未通过设备空闲门禁，未启动 NPU 验证；按约定转至 `177.5.23.1`，复用已构建并逐文件校验的同一套库，没有跨机器重跑旧性能基线。运行包 SHA256 为 `876f898aa1c1cd6d50293e27db9d2e526a70f1893daadf4e1f241a4953119a4e`，552 个文件哈希一致。

`177.5.23.1` 上的早期失败仅作为排错记录保留，没有计入验收：

- Python `shmem` 导入缺失：复制自有目录已有的依赖到本次独立目录，通过 `PYTHONPATH` 使用；未改系统 Python 或其他容器。
- SHMEM 初始化报告监听端口被占用：换用确认空闲且位于本机临时端口区间之外的端口；成功 Timer 运行使用 torchrun `22050`、两个 EP 组的 SHMEM `22060/22061`。实际使用时检查本机端口，不能假设这些端口永远空闲。
- Timer 单独补跑脚本曾把参考路径链接到宿主机绝对路径，容器内不可见：改为挂载内相对链接，并在设备启动前检查全部八份参考文件可读。该修正只涉及本次实验启动脚本，不涉及算子源码。

没有停止其他任务，也没有用失败尝试的时间替代正式性能。接受的 TimerOFF 只测一次，后续仅补齐 TimerON。构建与部署路径、源代码和二进制身份均有回执可查。

原报告约 1.79% 的改善来自 `91fa98d → eb2e053` 的跨机器对照。本次新增默认融合校验后，组合版本属于新的测量点；不能将两个独立优化的收益相加，也不能把专家算子 F/B 当作整网训练 step。此入口不包括真实 FSDP all-gather/reduce-scatter、Attention 或 optimizer。
