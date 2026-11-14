# 共享专家前反向打点与单机近似校时

本入口保留原始 NPY，不改变算子计算。无 Timer 的性能基线不创建 probe、不读控制阶段时钟、不加入校时同步。Timer 版本是诊断采集；其耗时不能当成生产性能。当前17项独立算子验收及4组OFF/ON配对已通过，真实FSDP整网训练尚未验收；复现配方和结论分别见[使用指南](../../docs/内核X重派发实现与验证-20261101.md)、[最终结果](../../docs/内核X重派发八卡验证结果-20261101.md)。

## 独立构建 probe

只在已确认的自有环境中，使用现有 CANN 编译器和新的私有输出目录：

```bash
export CLOCK_PROBE_BUILD_DIR=/your/private/new/clock-probe-build
export ASCEND_DEEPEP_RUNTIME_INCLUDE_DIR=/existing/directory/containing/rt.h
bash integration/wuhan/clock_probe/build.sh
```

生成 `libdeepep_clock_probe.so`。该工程只支持本轮 Ascend950/A5，独立链接 runtime/ascendcl，不编译主算子，不构建或修改 SHMEM。当前实际probe已随独立算子采集验证；重新构建的SO仍须与所采数据一起固定哈希，不能用源码或CPU检查替代设备采集。

## 采集 API

```python
from ascend_deepep.control_timing import ControlTimingRecorder, record_control_timing
from device_clock_calibration import DeviceClockProbe, save_clock_capture

probe = DeviceClockProbe(clock_probe_so, device, samples=9)
recorder = ControlTimingRecorder(device_events=True)  # 只用于 Timer 诊断末轮
pre = probe.sample_batch()
with record_control_timing(recorder):
    result = complete_forward_and_actual_backward()
    torch.npu.synchronize()
recorder.resolve_device_timings()  # 只读已完成的 Event，不再同步
post = probe.sample_batch()
# 先将该次调用的 Timer NPY 保存到同一个 frame，再保存下面两个 JSON。
save_clock_capture(frame, pre, post, rank, world, recorder.records)
```

每次 probe 是 32 个 mixed block，记录 32 AIC 与 64 AIV 的 `GetSystemCycle`。每个样本的 host `monotonic_ns` 前后时间包住 launch 到 stream 完成，重复取 bracket 最窄的样本；全部样本保留。前后两个批次求 cycle→host monotonic 的 affine 映射，记录相对 1 ns/cycle 的 offset 漂移和 scale 漂移。绝不使用 planner 或主 kernel 的开始时刻强行对齐。

这是**rank 级近似校时**。逻辑 lane 不是物理核 ID，本入口没有记录物理核 ID，不能证明每个物理核完全同步。误差估计包含半个 host bracket 和半个观察到的 lane cycle spread；这是 affine/系统计数器可比较假设下的观察误差，不是所有未来物理核行为的硬保证。最大端点误差达到 `0.2 ms` 时，mapping 的 `resolution_warning=true`，这张图不适合归因 `0.2 ms` 量级的差异。

映射只允许落在 pre/post anchor 之间的事件，不允许外推。缺 rank、不同 host monotonic 域、重置/倒退的 cycle、无效端点、被修改的映射都拒绝校时导出。

## 离线导出

```bash
python integration/wuhan/export_shared_fb_trace.py \
  --input /your/private/capture/result \
  --out /your/private/new/clock-aligned-traces --world-size 8
```

输入为 `{framework,fused}/rankN/step0`，包含原始前后向 Timer、coordinate 数组、planner 数组，以及 `clock-calibration.json`、`host-control-timing.json`。融合链路额外包含 `[4,128,2]` 的共享前向摘要和 `[10,128,2]` 的共享反向Timer；前向schema2还有detail、逐tile数组及geometry元数据。ABI3采集还必须有 `[4,128,2]` 的 `x_redispatch_backward_timer.npy` 和明确的native/ABI3声明；导出器核对三相全64 AIV及pull相32个奇数AIV1，拒绝缺项或残留参与者。

默认推荐先看每个 mode 的 `all_8_rank_{mode}_full_fb_trace_clock_aligned.json`，同一张图包含该次真实前反向，并仅附一次完整 host scope。导出器在原始 cycle 上检查 F+B 同 lane 无执行条重叠，然后使用同一 pre/post 映射。原有前向、反向两张单向图继续保留，但它们只显示对应 device phase，不混入整个 F/B 的 host scope，避免前向横轴被反向 host 记录延长。

所有 rank、两个 mode、两个 phase 使用共同 host 原点。A/B发生在不同真实时间，图中保留这一差异。默认不显示wait，`--include-waits`才显示。共享前向标为`SHARED/GMM1_GATE_UP`、`SHARED/SWIGLU`、`SHARED/GMM2_DOWN`；共享反向显示重算GU/A、dA、VJP、五个梯度GEMM和`MERGE_SHARED_DX`。最后一项只表示shared的dXg+dXu合并，keep的两段重算为空合法。

V2的native backward INIT包住共享S0，导出前核验shared子区间包含关系并保留真实BEGIN/END。ABI3在S0前关闭纯native INIT，因为其S0中已有真实dY Dispatch及X/prob搬运；仅对完整声明ABI3并具备独立X Timer的采集采用这一规则，不套用到V2。X四相分别为source gather、source publish sync、X/prob pull、read-retire sync；前者包含X、dY和概率准备，不能全部归为X单项成本。默认隐藏两个sync scope，原始记录保留。

旧图没有 probe 时可显式添加 `--allow-uncalibrated` 导出 rank 内局部原点图；同时生成带 `uncalibrated` 后缀的完整 F/B 图，F/B 在同一 rank 内共用真实 cycle 原点，保留前后向间隔，但不附无法与 device 对齐的 host scope。这不会把旧数据伪装成校准结果。新图校时失败也不能自动退回并声称已同步。若显式使用该旧图选项，即使存在部分校准文件也全部按未校准处理。

已有仅前向入口可用 `--timer 1 --clock-probe-so ...` 采集，再离线 `--export-only --clock-aligned` 导出；旧的未校准导出方式仍保留。

## 控制侧时间的含义

`control_timing_scope`是可选contextvar recorder，关闭时不读时钟。开启后记录Python/host enqueue范围：权重发布、planner、validator、HCCL、执行准备、Timer清零、native launch、placement snapshot和finish。桥接层也设有backward live weight provider、replay和执行scope，但contextvar不保证跨autograd工作线程传播。本次fused结果实际只有13项前向Host记录，没有反向publish/replay/native_launch/finish分项；不能仅用该列表定位反向Host空泡，应结合实际kernel和external-phase记录。

这些 host scope 并不等于对应 device kernel 的耗时，异步调用返回后工作仍可执行。它们分别放在 host 轨道，嵌套 scope 不可相加；导出图保留完整 F/B host 上下文。

默认 `ControlTimingRecorder()` 保持只有 host 记录，不导入 Torch。Timer 末轮显式开启 `device_events=True` 时，利用本仓现有 `torch.npu.Event(enable_timing=True)` 用法，在每个 scope 前后向当前 stream 记录 Event；最终同步完成后 `resolve_device_timings()` 写入独立的 `device_elapsed_ms` 字段。host bar 的持续时间仍为原来的 host scope，不能替换成这个 device 值。Event 区间包含 stream 依赖等待、实际执行及 host 供给空隙，也包含诊断 Event 自身的影响，不是纯 kernel active time；嵌套 device 区间同样不可相加。跨 stream 的 scope 拒绝作为有效同 stream 区间。resolve 不新增同步，无 Timer 基线不创建 Event。要分析更细的执行占用和底层 kernel 归属，仍需结合标准 device profiler。

旧图 planner→MegaMoe 的空白没有上述 host 分段证据，不能根据新 hook 的名称直接给旧空白分摊耗时。`framework` 原生 Timer 只覆盖 routed kernel；Torch/Triton shared 的算子时间不在原生 Timer 中，不能用空缺的共享 Timer 宣称参考链路没有这些计算。

本次32份实际校时记录均通过；端点观测不确定度约7.7～10.2μs，跨rank两端观测不确定度和最大约19.6μs。毫秒级重叠可据此分析，约20μs以内的先后不作硬结论。导出audit的逻辑lane区间重叠计数为0；`physical_core_id_recorded=false`，不能由此推断物理核占用或PMU利用率。

## CPU 验证

```bash
cd integration/wuhan
python -m unittest test_clock_calibration_cpu test_shared_expert_forward_benchmark_cpu
```

测试包括最窄 bracket、共同原点、漂移/误差、缺失和篡改拒绝、禁止外推、关闭 recorder 不读时钟、AIC/AIV 映射、真实 scope 冲突拒绝，以及合成前反向完整离线导出。它们不启动 NPU。
