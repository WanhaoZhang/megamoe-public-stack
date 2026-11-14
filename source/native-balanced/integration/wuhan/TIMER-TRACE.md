# Kernel 打点默认输出

使用原算子的 AscendTimerV2。请求 kernel 打点图时，默认只导出两个 Chrome Trace / Perfetto 事件数组：

完整采集入口（含配对构建、同shape合成/真实输入回放、五输出校验与自动导图）见 [TIMER-CAPTURE.md](TIMER-CAPTURE.md)。该入口补齐历史实验目录中的串联逻辑，无需外部prepare脚本。

- `all_<N>_rank_forward_trace.json`
- `all_<N>_rank_backward_trace.json`

每个 rank 只有 96 条轨道，沿用参考图 `group0..31/AIC0、AIV0、AIV1` 的编号和数字排序。planner、dispatch、权重搬移、计算、partial 排空/发布、owner 本地/远端加载、BF16→FP32、FP32 SUM、结果转换和写回共用这些轨道，不按事件类型创建新轨道。原始 block/group/subblock 编号不是额外读取的物理 core ID；planner 的 pure-vector worker w 沿 TimerV2 约定显示在 group w/AIV0，不能用此图证明跨 kernel 的物理驻留。

## 默认导出

当前 `run_training_case.sh` 在成功训练结束后，发现 `$OUT/named-kernel-timer` 才离线自动导出至 `$OUT/kernel-trace`；普通训练不进入此分支。旧冻结 SO 和源码无需替换；独立采集工具可以携带 exporter 与 base layout 两个纯 CPU 文件，见 `timer_trace_tools.py`。

已有数据也可离线导出，不启动 NPU：

```bash
python integration/wuhan/export_timer_trace.py \
  --input /path/to/named-kernel-timer --out /path/to/kernel-trace --world-size 8
```

输入支持 `rankN/forward/forward_timer.npy`、`rankN/recompute/{forward,backward}_timer.npy`，以及单层回放 `rankN/step0/` 或 `rankN/`。同目录可带 `planner_pairs.npy` 和 `*_stage_{header,indices,pairs}.npy`。`--world-size` 会拒绝缺 rank；每个 rank 都必须收齐前向、反向。stage schema 1–6、magic、计数和重复写检查失败时拒绝导出，不静默漏事件。已存在的 trace 不覆盖。

历史 rho6 项目更新未来的采集工具准备脚本（不修改已生成的冻结工具或恢复任务）：

```bash
python integration/wuhan/install_timer_trace_defaults.py --project-tools /path/to/rho6-optimization-20261027
```

此安装器只更新 `prepare_named_diagnostics.py` 和 `prepare_producer_timer_queue.py` 的未来输出，并保留 SHA 命名的源码备份。成功的 torchrun 结束后才执行离线导出，不在训练逐层增加 CPU 回读。当前停止的整网实验不因此恢复。

## 展示与覆盖口径

- 前向文件是原前向；反向文件包含重算前向（名称前缀 `RECOMPUTE/`）及反向，两者保持本 rank 原始设备周期间隔，不额外输出第三个重算文件。
- 可以直接表示串行阶段的原有固定/每专家区间保留。若粗阶段与其细分工作相交，粗阶段改为起止标记，计算退回精确 tile 记录。权重搬移、dispatch、owner load/cast/SUM/store 等子操作逐条使用原始 `[start,end]`，**禁止按事件类型取 min(start)/max(end) 把中间其他操作跨过去**。
- `REDUCE_WORKER`、`REDUCE_PHASE`、`OWNER_TILE_SERVICE`、copy residency 等外层范围用 BEGIN/END milestone，完整范围保留在参数里，不和内部工作一起画成并行执行条。ready/publish 完成点逐条显示真实时刻；具有实际区间的 weight/partial publish 保留区间。发布请求不是全局可见证明。
- `trace-export-audit.json` 逐事件 kind 检查：采样记录必须由未冲突的原生阶段、原始子区间/标记或显式隐藏的 WAIT 覆盖。每条 lane 的正时长事件按原始周期检查不重叠，发现冲突即拒绝导出，不能平移/裁剪时间来凑串行。原始 `.npy` 完整保留。
- 默认隐藏 WAIT 条、保留真实空白；`--include-waits` 可显示等待。与其他区间嵌套的等待改为范围标记；累计等待只保留带时长参数的起点，不伪造连续等待的结束位置。每 rank 独立时间原点，不拼接跨设备时钟。
- 逐条保留归并子操作会增加文件大小；使用流式写出避免累计全部 rank 的数据。两个文件只有全部 rank/phase 验证通过后才发布，失败保留 `.partial` 取证。
- 本流程是诊断；Timer、抓取、hash、profiling 运行不能用于正式性能统计。本次导出工具验证不改变 P4/cache 数值失败或原 A 精度边界。

旧 CSV 入口 `test/megamoe_bf16_test/trace_parser.py` 也使用同样 96 轨道和两个按 phase 命名的 all-rank 文件；原有调用签名兼容，单 rank 文件改为显式 `--per-rank`。

2026-10-28 修正：首版 `9356aab` 对相同类型的不连续子操作合成包络，虽然事件覆盖和轨道数量检查通过，仍会在同一 AIV 上产生假重叠。该显示方式已撤销；不能用旧版包络解释并行性或各子操作连续耗时。数值回放通过与该显示错误是不同证据。
