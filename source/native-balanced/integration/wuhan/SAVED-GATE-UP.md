# 路由专家保存格式与选择性重放

普通 S9 routed 接口提供 `save_gmm1_output=bf16|none|mxfp8|auto`，默认
`bf16`。格式由训练使用者选择，构造 pool 后固定；前反向与 Timer/NoTimer
使用同一配置，autograd context 记录本次前向实际解析的格式。

| 格式 | 反向 Gate/Up 来源 |
|---|---|
| `bf16` | 前向保存的 BF16 G/U，不重算 GMM1 |
| `none` | backward 内重算 GMM1；仍保存 packed X/prob 与路由状态 |
| `mxfp8` | 解码压缩 G/U、scale；限幅时使用对应导数 mask，不重算 GMM1 |
| `auto` | 沿用 native 的 FP32 limit/sentinel 规则，收据记录实际格式 |

显式 MXFP8 可配合 `swiglu_limit=10`，仍须满足 native 对齐与 UB 限制。
none 默认保留原反向调度；另有显式开启的 [none GM 流水](GM-PIPELINE.md)，
需配套重建 Host/worker，不能仅按保存量认定它更快。MXFP8 精度应按量化格式
评估；同格式 TimerOFF/ON 及框架链路比较不等于 BF16 跨格式逐位一致。

## 框架接入

```python
pool = NativeFSDP2Pool(..., save_gmm1_output="mxfp8")
bridge = FSDP2ExpertsBridge(pool, current_fsdp_weights)
```

FSDPTurbo 对应 `MegaMoeBalancedConfig.save_gmm1_output`；MindSpeed 分组参数为
`--optimization.megamoe_balanced_save_gmm1_output`。需要三个仓的配套版本。
shared fused 的 routed none/MXFP8 需要独立 native 适配；不要删除其能力校验。

普通整层 checkpoint 会重新执行专家前向。已有 FSDPTurbo UAM SelectiveReplay
可以让父层重算，同时将 experts 配置为 `keep/local` 或无损 `offload/cpu`。
它分别保存专家反向 tape 和供父层回放使用的输出 Y，回放时恢复 Y，原节点直接
调用 MegaMoe backward。不要让通用 MXFP8 后端再次量化混合 UInt8 tape。

UAM 与旧整层 `memory.recompute` 互斥。专家及其后代规则要优先于父层规则；
主层与 MTP 路径均须覆盖。具体配方见配套 FSDPTurbo、MindSpeed 使用说明。
KEEP/OFFLOAD 是存储策略，与 native 的 BF16/none/MXFP8 保存格式独立。

## 最小设备验证入口

在已确认空闲的单机八卡自有容器中，使用匹配的 TimerOFF worker/binding 和
冻结场景输入。按实际安装设置 CANN、SHMEM、PYTHONPATH、LD_LIBRARY_PATH；
`WORKER_NATIVE` 指向已核验的 `timer-capture-build.json` 及配套三个库。

```bash
python -m torch.distributed.run --nproc-per-node=8 --master-port=32700 \
  integration/wuhan/runtime_bootstrap.py --native-only \
  integration/wuhan/benchmark_saved_modes_uam.py \
  --scenario-inputs /absolute/case/inputs --out /absolute/run/bf16-checkpoint \
  --save-gmm1-output bf16 --variant checkpoint

python -m torch.distributed.run --nproc-per-node=8 --master-port=32701 \
  integration/wuhan/runtime_bootstrap.py --native-only \
  integration/wuhan/benchmark_saved_modes_uam.py \
  --scenario-inputs /absolute/case/inputs --out /absolute/run/bf16-uam \
  --save-gmm1-output bf16 --variant uam-keep \
  --reference /absolute/run/bf16-checkpoint
```

分别替换为 none、mxfp8，必须使用各自同模式基线。`uam-cpu` 对应无损 CPU
offload。入口包含真实 expert-FSDP2、五输出与梯度 shard、调用次数、完整 F/B
与峰值显存；它是单层集成探针，包含强制父回放的数值恒等边界，不是整网训练。
为匹配真实 EFSDP2 参数，它会统一两 EP 组的 home 权重，收据记录此输入变换。

## Timer 与 EP 组内对齐

原配对采集入口新增 `--save-gmm1-output`；不指定时继承 capture 格式或默认
BF16。新收据同时记录 requested/resolved；格式不同的历史 capture 输出不能
冒充同模式精度参考。原配对验证器也检查保存格式。

```bash
python integration/wuhan/export_timer_trace.py --input /absolute/run/timer \
  --out /absolute/run/ep-barrier-view --world-size 8 \
  --time-origin ep-barrier --ep-size 4
```

F 锚点为 worker0 planner Counts 起点；B 锚点为 group0/AIV0 的
XSYNC_GATHER 终点。对同一 rank 全部事件统一平移，保留锚点之前的负时间和
默认隐藏 wait。两个 EP4 组各有独立逻辑零点，这不是跨设备时钟校准。
存在 stage header/收据时核对 EP 与组内 epoch，缺证据时 audit 明确标注未验证。

## 2026-11-03 单层设备验证

普通 routed 的冻结验证版本为算子 `a935a07`、FSDPTurbo `8018b7b`、
MindSpeed `624999a5`。八张 950、两个 EP4 组、真实 expert-FSDP2；
M8192/H7168/F3072、E8/Top2、home2/slots2、capacity20480、limit10，
使用仓内固定 Python 高偏斜输入。探针统一两 EP 组的 home 权重，并用专家后
数值恒等边界强制父层回放；它不包含 attention、共享专家、优化器或整网其他层。

三种保存格式分别比较 checkpoint、UAM KEEP、无损 CPU offload，共9组通过。
同格式的五输出、autograd 输入梯度与真实 FSDP 梯度分片均逐位一致；UAM每步
为1次完整专家前向和1次反向，原checkpoint为2次前向和1次反向。
这是框架路径一致性与重复性检查，不是独立数学golden或MXFP8对BF16量化误差验收。

| 格式 | checkpoint完整F/B（ms） | UAM KEEP（ms） | UAM CPU（ms） |
|---|---:|---:|---:|
| bf16 | 40.002 | 30.084 | 85.875 |
| none | 52.313 | 42.571 | 82.583 |
| mxfp8 | 41.665 | 31.655 | 86.690 |

2次warmup、7次计时，取每次八rank最大值的中位数，包含桥接、UAM和真实FSDP规约。
BF16/none同机耗时分别下降24.79%/18.62%；MXFP8因机器占用切换了节点，
不计算跨机收益比例。CPU offload本单层更慢；整网跨层预取效果仍需验证。

三态TimerOFF/ON五输出逐位一致，六张图通过轨道零重叠、EP大小和组内epoch检查。
采图在计时外预先建立设备和通讯状态；图只覆盖pool内部F/B，并按两个EP4组
分别建立逻辑同步点，不能作为物理时钟校准或完整FSDP/UAM wall time。

上述真实EFSDP2/UAM结论覆盖balanced普通routed单层。共享构建后续完成的
算子兼容验收见 [GM流水说明](GM-PIPELINE.md)；它不替代真实主层/MTP整网、
多步训练、共享FSDP/checkpoint/offload及整网峰值验证。共享融合的routed
保存格式继续只允许显式BF16。

原始 trace、日志、SO、输入及中间定位过程留在独立实验目录；仓内仅保留必要
代码、回归测试、复现方法及最终验证结论。
