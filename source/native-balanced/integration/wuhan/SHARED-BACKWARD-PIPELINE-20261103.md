# 共享专家反向依赖与向量流水优化（2026-11-03）

本次在 `megamoe-balanced-shared-expert@cbccb9755af38d74fdd5015d091c9808270993f0` 上优化共享反向。此前保留 packed X 的融合链路比“外置共享专家＋MegaMoE”完整前反向慢约 6%～9%；本轮相同输入下已消除这项劣化，融合链路比外置串行链路快约 0.87%～1.26%。这是独立 MoE 验证，不能作为整网训练加速比例。

## 修改与使用

- **删除两处过强依赖。** `dW_down = dYᵀ @ A` 不依赖 SwiGLU 反向的 dG/dU，先计算 dW_down，再等待 dG/dU，覆盖向量梯度计算。共享 recompute 时，`dA = dY @ W_down` 不依赖重建的 A，把 A-ready 等待移到真正消费 A 的 dW_down 前。V2、V3 使用相同依赖关系。
- **扩大向量分块并双缓冲。** 共享 SwiGLU 梯度、重建激活、dX 合并从每次 256 元素改为至多 8192 元素/槽；本次 F3072 每槽两行，H7168 每槽一行。两个 UB 槽轮流预取、计算、回写，复用前等待向量读和 MTE3 写完成，退出时排空事件。
- **兼容布局与边界。** 使用真实 GM 行跨度，兼容旧 ABI1 交错 G/U 与 V2/V3 独立平面；支持奇数行尾块、宽于单槽的分列和无任务 worker。三输入 VJP 的双缓冲最多占用 96 KiB UB，不新增 GM 张量。FP32 导数计算、限幅导数及 BF16 舍入边界保持原定义。

重新构建匹配 worker 即生效；没有新增运行开关、Python 参数或 ABI。既有 `save_gmm1_output=bf16|none|mxfp8`、`shared_save_mode=keep|recompute`、`routed_input_mode=saved|redispatch` 保持独立可选。共享计算始终使用 BF16，路由保存格式不代表共享权重格式。旧外置共享模式保留。

V3 的原 X/prob 重派发及源读取退休协议保持原定义，本次不删除跨阶段同步、不修改路由 GMM、W1/W2 搬移或梯度归主算法。框架仓无需因本次内核内部调度修改而新增配置。

## 固定场景与计时口径

world8 是两个 EP4 组，不是真正 EP8。M8192/H7168/F3072/E8/Top2，home=slots=2，capacity20480，swiglu_limit10。使用已审计的 `python-olmoe-high-skew-e8-ep4-m8192` 固定输入，目标 κ≈2.4816624；不能用命令行 rho 值代替该路由。共享权重沿用 benchmark seed42。

NoTimer 对照均在 177.5.26.1 自有容器完成，源码版本运行次序交替；每个进程内外置/融合链路也按 AB/BA 交替。warmup2、measure10，以每轮八 rank 最大完整 F/B 耗时的中位数计，包含实际 autograd、共享与路由分支及本地合并；不包含 router、真实 FSDP AG/RS、optimizer。运行前两次空闲检查，运行中持续检查设备占用，仅操作自有容器。

## 保留 packed X：无 Timer 完整前反向

单位 ms；“旧融合”是本轮冻结源码基线，“外置串行”是候选源码同进程的共享框架链路。下降比例以耗时计算。

| 路由 G/U 保存 | 共享策略 | 旧融合 | 新融合 | 外置串行 | 新融合较旧融合下降 | 新融合较外置下降 |
|---|---|---:|---:|---:|---:|---:|
| bf16 | keep | 36.733 | 33.835 | 34.224 | 7.89% | 1.14% |
| bf16 | recompute | 39.366 | 35.707 | 36.080 | 9.30% | 1.03% |
| none | keep | 40.928 | 37.940 | 38.403 | 7.30% | 1.21% |
| none | recompute | 43.616 | 39.784 | 40.246 | 8.79% | 1.15% |
| mxfp8 | keep | 38.617 | 35.580 | 36.034 | 7.86% | 1.26% |
| mxfp8 | recompute | 41.315 | 37.540 | 37.869 | 9.14% | 0.87% |

六组各十轮配对中，新融合完整 F/B 均低于对应外置串行链路。本次 none 性能配置显式启用已有 `MEGAFLUX_BACKWARD_NONE_GM=1`；不把此收益外推到所有 shape、路由、默认 UB 路径或真实 FSDP 整网。相同模式新旧融合的 PyTorch allocated 峰值差为零；这不是板级总显存统计。

## 内核 X redispatch：兼容与完整链路补测

以下六组在 177.5.21.1 同机、同进程 AB/BA 测量。外置基线为“外置共享专家＋外置 X redispatch＋MegaMoE”；融合链路为共享与 X redispatch 均进入 MegaMoE。此表不是与旧融合的跨机对比，也不把其绝对耗时与上一表相减。

| 路由 G/U 保存 | 共享策略 | 外置完整链路 ms | 新融合完整链路 ms | 融合耗时下降 |
|---|---|---:|---:|---:|
| bf16 | keep | 38.203 | 35.856 | 6.14% |
| bf16 | recompute | 40.049 | 37.773 | 5.68% |
| none | keep | 42.522 | 39.304 | 7.57% |
| none | recompute | 44.274 | 41.007 | 7.38% |
| mxfp8 | keep | 39.998 | 37.088 | 7.28% |
| mxfp8 | recompute | 41.839 | 38.845 | 7.15% |

六组各八 rank、双次精度比较通过，与同格式 saved 融合链路八输出逐位一致；实际 X/prob 有效前缀核验通过，融合前向 autograd tape 的 packed X 均为零字节。反向仍需临时重建缓冲，不能将 tape 释放量直接当作总峰值显存下降。

## 精度与打点

六组均通过八 rank、双次精度比较：Y 与七项梯度符合既有外置链路判据，新旧融合八输出逐位一致，placement 与冻结输入一致。Timer OFF/ON 六组八输出逐位一致。补齐六组旧融合 saved 基线图，并导出六组候选图；每组均含外置和融合的前向、反向、完整 F/B 视图。

图使用每 rank 实测 pre/post probe 映射到同一 host monotonic 时域，保留校时不确定度，不强制各 rank 的 kernel 起点相同。十二组导出审计通过，同轨 duration 零重叠，默认隐藏 wait。采图耗时仅用于解释流水，性能结论使用无 Timer 实测。

| BF16 关键反向阶段 | 旧融合 | 新融合 |
|---|---:|---:|
| keep：共享反向前缀 | 8.342 ms | 5.490 ms |
| keep：SwiGLU VJP 核内中位时长 | 1.119 ms | 0.249 ms，已与 dW_down 重叠 |
| keep：dX 合并核内中位时长 | 1.871 ms | 0.107 ms |
| recompute：共享反向前缀 | 10.966 ms | 7.247 ms |

表中前缀取对应采样最慢反向 rank，向量阶段取该 rank 的核内中位值。BF16 keep 的旧 Timer 图来自 177.5.21.1，其余旧图及全部候选图来自 177.5.26.1；图用于解释依赖与重叠，同机 NoTimer 表用于性能结论。两类证据均显示尾部缩短，不只是首块提前。

CPU 检查：新增分块几何测试覆盖 360 个 shape/worker 组合，以及空尺寸、无任务 worker、ABI1 stride；既有共享/原 X 重派发/反向契约共 21 项通过。

补充兼容检查在 177.2.25.1 自有容器、相同源码及 SO 上通过：旧 ABI1 的 M257/H512/共享F256、均衡/热点/限幅边界各两次，共八 rank、48 次精度迭代；none/native/recompute 运行值 `MEGAFLUX_BACKWARD_NONE_GM=0` 的默认 UB 路径通过双次精度与同格式 saved 八输出逐位比较。旧 ABI1 门禁只修复设备状态头的 JSON 收据序列化，计算仍使用原设备头；这些补测不提供性能结论。

## 构建与复现

沿用[三种保存格式说明](SHARED-SAVED-FORMATS.md)的匹配构建与配置。以下为本轮实际构建开关，不依赖外层环境变量猜测库行为：

```bash
export MEGAFLUX_SHARED_EXPERT_FORWARD=1 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MEGAFLUX_SWIGLU_DECOUPLE=on MEGAFLUX_GM_SWIGLU_TILE_READY=1
export MEGAFLUX_BACKWARD_NONE_GM=1  # 构建能力；运行值另行选择
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1
export MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1 MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_OWNER_EARLY_RATIO=2
python3 setup.py build_ext --inplace --force
bash scripts/build_megaflux_timer_capture.sh UNIQUE_PREFIX
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  test/megamoe_bf16_test/test_shared_backward_vector_schedule.cpp -o /tmp/shared-vector-schedule
/tmp/shared-vector-schedule
```

固定输入按[场景说明](scenarios/python-olmoe-high-skew/README.md)生成并校验。设置 `WORKER_NATIVE`、PYTHONPATH、SHMEM/CANN 动态库路径到自己的配套环境后，使用新输出目录：

```bash
export MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE=1
export MEGAFLUX_BACKWARD_NONE_GM=0  # bf16/mxfp8 为0；本轮 none 主性能矩阵为1
python3 -m torch.distributed.run --nproc-per-node=8 --master-port=27100 \
  integration/wuhan/runtime_bootstrap.py --native-only \
  integration/wuhan/benchmark_shared_expert_fb.py \
  --out NEW_OFF_DIRECTORY --save-gmm1-output bf16 --shared-save-mode keep \
  --routed-input-mode saved --scenario-inputs INPUT_DIRECTORY \
  --triton-reference-file MINDSPEED_SOURCE/mindspeed_llm/fsdp2/ops/triton_swiglu_with_limit.py \
  --world-size 8 --ep 4 --tokens 8192 --hidden 7168 --ffn 3072 \
  --experts 8 --topk 2 --slots 2 --capacity 20480 --limit 10 \
  --repeat 2 --timer 0 --warmup 2 --measure 10 --shmem-port 27102
```

替换三个保存格式及两个共享策略形成六组。测原 X 重派发时，benchmark 使用 `--routed-input-mode native`，API/框架配置名为 `redispatch`；加 `--saved-dispatch-reference SAME_FORMAT_SAVED_DIRECTORY` 比较同格式输出与有效 X/prob 前缀。性能对照随输入策略变化：saved 比外置共享＋MegaMoE，native 比外置共享＋外置 X redispatch＋MegaMoE，两者不能混作同一基线。

采图切到匹配 ON worker，使用同配置 OFF 的 `--reference NEW_OFF_DIRECTORY`，增加 `--clock-probe-so ACTUAL_CLOCK_PROBE_SO`，改为 `--timer 1 --warmup 0 --measure 0`。使用 `export_shared_fb_trace.py --help` 所列参数导出；默认不显示 wait。框架 Triton 参考用于计时，纯 Torch 参考仅作无计时精度检查。

## 本轮库身份与待验边界

- Host `_C`（ABI 未变，沿用匹配库）：`22f595bf4866b300db7b6f1fd1f015bd4ddcf22d1de679a7ca6d2ee22508608a`。
- NoTimer worker：`a1469a8505af5ddcfbf59089b43702a3a01f063ac53e4e32046719871740fe25`。
- Timer worker：`871b8a6d3d9cb02a5853f935bcc3a738e4d83e1090eebf4349588785843d9951`。
- 实际 getter：共享保存格式能力均为7，V2/V3为2/3，GM decouple/tile-ready均为1，P1/DX cache均为1，owner ratio为2，Timer为0/1。

真实 FSDP 参数/梯度、checkpoint/UAM/offload、MTP、多 step 动态参数和恢复训练仍由整网接入验收。本次不增加反向前的 MegaMoE forward 重放，也不改变整网保存策略默认值。原始日志、调试过程及 NPY/JSON 采样保存在独立实验目录，不随本次代码提交。
