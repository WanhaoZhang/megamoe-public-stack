# 通用逐 tile 前向与 none 反向流水

本页适用于普通 routed S9，目标分支为 `megamoe-balanced` 和
`megamoe-balanced-shared-expert`。共享 V1/V2/V3 的 routed 保存格式仍为 BF16；
不要给共享融合入口启用 `none` GM 反向。

## 默认与修复

- 保留旧默认：S9 的 `auto` 有效值仍为 0，逐 tile 和 none GM 反向默认关闭。
- 前向返回的未使用物理槽 counts 清零；无副本时仍返回正确的零尾部。
- 无副本时的 home tiling 保留原保存格式，避免把 none/MXFP8 错当成 BF16。
  此修复需要重建 Host 扩展 `_C`。

## 构建与选择

先按仓库构建说明配置 CANN、SHMEM 和编译依赖。两种候选都必须使用新 Host
扩展及匹配 SO；不要借用其他目录的 `_C`。构建命令中的临时环境变量不会遗留
到后续训练进程。

```bash
python3 setup.py build_ext --inplace --force

# BF16 前向：GM 解耦并按 producer 的 Gate/Up tile 完成度消费。
MEGAFLUX_SWIGLU_DECOUPLE=on MEGAFLUX_GM_SWIGLU_TILE_READY=1 \
MEGAFLUX_BACKWARD_NONE_GM=1 \
MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 \
MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1 \
MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1 MEGAFLUX_OWNER_EARLY_RATIO=2 \
bash scripts/build_megaflux_timer_capture.sh routed-gm-tile

# 只比较 none 反向：保留原前向策略，单独开放新反向能力。
MEGAFLUX_SWIGLU_DECOUPLE=auto MEGAFLUX_GM_SWIGLU_TILE_READY=0 \
MEGAFLUX_BACKWARD_NONE_GM=1 \
MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 \
MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1 \
MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1 MEGAFLUX_OWNER_EARLY_RATIO=2 \
bash scripts/build_megaflux_timer_capture.sh routed-none-gm
```

配对构建入口使用 S9 flags0、panel restore2、operand-ready1。选择对应
`worker-variants/<prefix>-off` 作为 `WORKER_NATIVE`，并按
[Timer 使用说明](TIMER-CAPTURE.md)配置加载路径和 runtime bootstrap。
运行前读取实际 SO 的 getter 与 SHA；改变运行环境变量不能改变已编译的前向策略。

| 保存格式 | 前向逐 tile | 运行时 `MEGAFLUX_BACKWARD_NONE_GM` |
|---|---|---|
| BF16 | 仅 on＋tile1 构建启用 | 0，读取保存的 G/U |
| none | 不启用逐 tile，遵循编译的 UB/GM 策略 | 0 为原反向；1 为重算 G/U 后的 GM 流水 |
| MXFP8 | 不启用逐 tile，遵循编译的 UB/GM 策略 | 0，解码保存的 G/U |
| 共享融合 V1/V2/V3 | 保留原共享逐 tile 实现 | 必须为 0 |

`none` GM 同时要求 `save_gmm1_output="none"`、普通 S9 selector1/3 和实际
SO 的 `megaflux_backward_none_gm_version()==1`。明确请求不支持的组合会在 Host
报错。构建时同名变量开放能力，运行时同名变量选择路径，二者含义不同。
新反向仅将 W1 恢复排在 W2 前，复用 GM P1/P2、ready、P1/DX cache 和 early dW1；
不改变 W2/GMM2 数学。内核只重算 GMM1，不调用整个 MegaMoe forward；
消除框架 checkpoint 导致的重复前向仍需沿用配套 UAM 配置。

保存格式仍由整网使用者选择；框架接入及 UAM 说明见
[SAVED-GATE-UP.md](SAVED-GATE-UP.md)。

## 容量与回退

BF16 逐 tile 追加每物理专家、每预留 producer 一条 128 B 进度线；当前
home2＋slots2、128 producer 上界共 64 KiB。none 仍需两块反向 G/U 临时区，
capacity20480/F3072 时共 240 MiB；本次没有消除这部分重算显存。

前向 `on` 在 none/MXFP8 下仍会按原 GM 方案追加 raw Up 临时区；当前 shape 为
120 MiB。因此只评估 none 反向时使用上面的 auto/tile0 配方，避免混入前向改路。
BF16 保存格式本来已有 raw G/U，不增加这块 raw Up。

none 反向回退只需设 `MEGAFLUX_BACKWARD_NONE_GM=0`。前向回退需选择
auto/tile0 的匹配 SO。旧 Host 不携带 none 请求位，默认反向协议不变；
逐 tile SO 的工作区仍要求配套新 Host，不能据此混用旧 Host。

## 复现

输入生成和来源见[固定 Python 高偏斜场景](scenarios/python-olmoe-high-skew/README.md)。
每 rank M8192/H7168/F3072、E8/Top2、home2/slots2、capacity20480、limit10；
八卡分成两个 EP4 组。固定路由的初始最重 rank/均值为 2.7355，在线 MoonEP 后
每 rank 16384 行。它是源直方图映射后的受控输入，不是整网自然路由。

在通过设备门禁的自有环境中，沿用 Timer 文档中的 CANN、SHMEM、加载路径和
runtime bootstrap；`RUN` 为独立输出目录，`INPUTS` 为已核验的八 rank 输入。
两次使用同一 auto/tile0 OFF SO、同一匹配 `_C`，只改变运行时请求：

```bash
export WORKER_NATIVE="$PWD/worker-variants/routed-none-gm-off"
# PYTHONPATH、LD_LIBRARY_PATH 按 TIMER-CAPTURE.md 设置。
export TASK_QUEUE_ENABLE=2 WUHAN_PUBLISH_SYNC=device WUHAN_CONTROL_TAPE_DEVICE=npu
for policy in 0 1; do
  ref=()
  if [ "$policy" = 1 ]; then ref=(--reference "$RUN/none-0"); fi
  MEGAFLUX_BACKWARD_NONE_GM=$policy \
  python -m torch.distributed.run --nproc-per-node=8 --master-port=22500 \
    integration/wuhan/runtime_bootstrap.py --native-only \
    integration/wuhan/capture_kernel_timer.py --timer 0 \
    --scenario-inputs "$INPUTS" --out "$RUN/none-$policy" \
    --save-gmm1-output none --repeat 3 --warmup 2 --measure 7 --save-outputs \
    --shmem-port 22502 "${ref[@]}"
done
```

正式比较将两次再按反序执行一次，使用新输出编号。每组取逐次八 rank 最大
完整 F/B 耗时的中位数。真实 expert-FSDP2/UAM 单层入口见
[保存格式说明](SAVED-GATE-UP.md)：同为 `uam-keep`，各以同格式 checkpoint
收据作参考，分别运行 none-GM0/1；必须确认两者都是 1F/1B。

采图时使用相同 none-GM 请求调用 `run_timer_capture.sh`，详见
[采图说明](TIMER-CAPTURE.md)。新收据记录 `backward_none_gm_requested`，
TimerOFF/ON 请求必须一致；旧收据缺字段时标为 `unknown`。

无副本/冷专家回归入口为 `test_inactive_routed_modes_npu.py`，参数为
`--mode <format> --out <fresh-dir> --shmem-port <unused-port>`，同样经八进程
runtime bootstrap 启动。三种格式分别以同格式普通算子为参考，检查五输出、
counts 零尾部、冷专家梯度零、输入和参考不被修改。此入口不自带设备门禁。

## 2026-11-03 验证结论

保存格式保持可选；本次只证明以下固定场景，未执行整网训练。NoTimer none
对照同机、同 SO、同 `_C`，仅切换运行时请求；均保留 early-dW1、P1/DX cache、
owner ratio2、panel2/operand-ready1。

| 对照范围 | 原 none | none GM | 完整 F/B 耗时下降 |
|---|---:|---:|---:|
| 普通算子完整链路，ABBA 合并逐次 rankmax | 39.132 ms | 29.906 ms | 23.58% |
| 真实 expert-FSDP2/UAM KEEP 单层，均 1F/1B | 42.511 ms | 34.200 ms | 19.55% |

普通算子按 A/B/B/A 执行，每窗口 2 次 warmup、7 次测量，各策略合并 14 个
逐次八 rank 最大值后取中位数。五输出按相同 none 格式逐位一致；
EFSDP2 探针还核对 autograd 输入梯度和实际 FSDP 梯度分片。该探针不包含
attention、共享专家、优化器或整网其他层。checkpoint 的 2F/1B 不纳入
上述流水收益，避免重复计算减少框架重放的收益。

BF16 GM/tile 用 A/B/C/C/B/A 比较，原默认两个窗口为 25.666/25.645 ms，
GM 整专家为 25.776/25.802 ms，GM 逐 tile 为 25.773/25.738 ms；没有稳定收益，
因此只迁入显式能力，默认仍为 auto0/tile0。首 tile 更早不等于端到端更快。

none 的主要结构变化是副本权重梯度与 owner 归约重叠。两套图都有 704 条实际
AIC Gate/Up 重算区间；rank0 重算尾为 3.448/3.461 ms，几乎未变。rank2 的
副本 dW2 尾由 24.471 提前到 9.969 ms，dW1 尾由 21.276 提前到 12.245 ms；
rank0 的 W1 owner store 尾由 31.184 提前到 18.840 ms。部分本地 dX 后移，
不能据此声称各阶段都变快。上述 Timer 数字解释排程，正式收益以 NoTimer 为准。

三种保存格式均完成普通 routed 功能回归。无副本夹具覆盖每个 home 活跃及
一个 home 完全空载，包含 none-GM split=0/all。这是同格式链路一致性检查，
不是独立数学 golden，也不是 MXFP8 相对 BF16 的量化误差验收。

以上性能来自 balanced 普通 routed 构建。共享分支的匹配构建已通过 BF16、
none-GM、MXFP8 无副本/冷专家回归，并在 177.5.26.1 补齐以下兼容检查：

| 补充门禁 | 结果与范围 |
|---|---|
| 共享 V3、keep、内核 X redispatch，完整冻结 shape | 八 rank、两轮八输出精度通过；dP/dWr1/dWr2 逐位一致；计划及重复输出不变；native 调用每 rank 两次，prefix 审计通过，前向 tape 的 packed X 为空 |
| 共享构建的普通 routed none-GM，完整冻结 shape | 八 rank 与冻结 none 参考的输入、配置、保存格式、placement 和五输出 hash 全部一致；跨机仅作正确性对照 |
| 新版请求收据的 TimerOFF/ON | 使用 balanced A 构建；16 份收据的请求均为 1、逐 rank 和总体检查均 matched，五输出逐位一致、同轨道零重叠，前反向 EP 图已导出 |

本次共享融合仅补 keep 兼容验收，没有新增共享性能、recompute、双 live forward
逆序反向、checkpoint/UAM 或整网验收结论。普通 none-GM 与共享 V3 分开执行；
共享融合入口仍只允许 routed BF16，运行时 none-GM 必须为 0。共享 benchmark
显式开启计数以核对 native 调用与 packed X 释放；生产 pool 默认不变。

四组前反向 Timer 图已验证 OFF/ON 五输出一致、同轨道零重叠。Timer 中的
`AIV_SWIGLU` 在逐 tile 模式包含后续 tile 等待，不能当作纯向量计算时长。
EP barrier 导图仅做组内逻辑对齐，不是跨 NPU 物理时钟校准；Timer 不替代
NoTimer 性能。原始 trace、SO、张量、日志和中间定位资料留在实验目录。

供同事查看的四份关键 none 前后对照图已无损压缩入仓，附校验指纹、启动配置和
[优化交接说明](../../docs/experiments/20261103-none-gm-traces/README.md)。
