# 共享融合 none/MXFP8 八卡验证（2026-11-03）

本次扩展共享 V2/V3 的路由 G/U 保存格式，不改变共享专家的 BF16 计算及 keep/recompute 定义。使用方法见[保存格式与框架配置](SHARED-SAVED-FORMATS.md)。

## 代码范围

- 共享前向按本次实际格式分派模板，Host 分配、设备写入与反向解码保持一致。
- Python autograd 冻结实际保存格式；V2/V3 Host、设备检查和所加载 SO 的格式能力配套扩展。旧 ABI1 保持 BF16。
- none 复用内核 GMM1 重算；显式 none-GM 策略接入共享 V2/V3，V3 恢复路径优先 W1。MXFP8 复用已有 scale/限幅 mask 解码。
- FSDPTurbo、MindSpeed 接通三种路由格式与显式 routed_input_mode，旧默认保留。

## 验证边界

NoTimer性能数据来自177.5.21.1的自有隔离容器；补充Timer/默认UB功能门禁使用柜4的177.11.24.1，无跨机性能对比。每组双次空闲门禁及持续占用监测。world8 为两个 EP4 组，不是真正 EP8。固定 Python 高偏斜输入，M8192/H7168/F3072/E8/Top2，home=slots=2，capacity20480，limit10。共享权重沿用 benchmark seed42。

运行时为Python3.12.13、torch2.10.0+cpu、torch_npu2.10.0.post6及匹配的CANN9.2.0-dsv-20261030；补充机器使用相同版本的Python/Torch/NPU包和同一套Host/worker。

Python CPU 契约：算子24项、FSDPTurbo18项、MindSpeed7项通过；C++ 布局/调度与匹配 Host、Timer OFF/ON worker 构建通过。
none/MXFP8 各覆盖 keep/recompute × saved/native，八个 NoTimer 组合均为8 rank、2次精度比较；检查 Y 与七项梯度，同格式 routed dP/dWr1/dWr2 逐位一致。四个 native 组合与对应 saved 链路八输出逐位一致，X/prob有效前缀核验通过，前向 tape 的 packed X 为零字节。
Timer OFF/ON八输出逐位一致：4/4组；校时JSON导出与零重叠检查：4/4组；none默认UB路径native keep/recompute兼容：2/2组。最终库另补BF16 native keep回归通过。

共享融合仍未验收真实 FSDP整网、UAM/offload、checkpoint、多 step动态参数及断点恢复。本轮 native 前向不驻留 packed X，反向仍需要临时重建缓冲，不能据此声称单层总峰值显存按同额下降。

## 无 Timer 完整前反向耗时

同机 AB/BA 交替，warmup2、measure10，取每轮八卡最大值的中位数，单位 ms。saved 的对照是外置共享＋MegaMoE；native 的对照额外包含外置 X redispatch。后者与共享＋X redispatch 融合链路比较，两类不能混作同一基线。

| 路由保存 | 共享策略 | X策略 | 外置链路 | 融合链路 | 融合耗时变化 |
|---|---|---|---:|---:|---:|
| none | keep | saved | 38.347 | 40.758 | +6.29% |
| none | keep | native | 42.317 | 41.949 | -0.87% |
| none | recompute | saved | 40.255 | 43.598 | +8.30% |
| none | recompute | native | 44.326 | 44.636 | +0.70% |
| mxfp8 | keep | saved | 35.989 | 38.517 | +7.03% |
| mxfp8 | keep | native | 39.964 | 39.720 | -0.61% |
| mxfp8 | recompute | saved | 37.832 | 41.248 | +9.03% |
| mxfp8 | recompute | native | 41.847 | 42.498 | +1.56% |

负值表示融合耗时更低。native 的小幅差值有正有负，本轮未证明稳定性能收益；saved 融合链路慢约6～9%，保留外置选择。本次交付的是格式支持与可选内存策略。

## 复现

使用保存格式文档的匹配构建配方。固定输入按[场景使用说明](scenarios/python-olmoe-high-skew/README.md)生成并校验，不依赖仅设置rho替代这份路由。运行前确认八卡空闲，`WORKER_NATIVE`指向匹配的OFF/ON变体目录，Python路径指向本源码及同版框架Triton激活。

```bash
# none 的运行值可选0（原UB路径）或1（构建需提供能力）；MXFP8/BF16设0。
export MEGAFLUX_BACKWARD_NONE_GM=1 MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE=1
python3 -m torch.distributed.run --nproc-per-node=8 --master-port=25100 \
  integration/wuhan/runtime_bootstrap.py --native-only \
  integration/wuhan/benchmark_shared_expert_fb.py \
  --out NEW_DIRECTORY --save-gmm1-output none --shared-save-mode keep \
  --routed-input-mode native --scenario-inputs INPUT_DIRECTORY \
  --triton-reference-file MINDSPEED_SOURCE/mindspeed_llm/fsdp2/ops/triton_swiglu_with_limit.py \
  --world-size 8 --ep 4 --tokens 8192 --hidden 7168 --ffn 3072 \
  --experts 8 --topk 2 --slots 2 --capacity 20480 --limit 10 \
  --repeat 2 --warmup 2 --measure 10 --shmem-port 25102
```

先用 `--routed-input-mode saved` 产生同格式参考，再给 native 增加 `--saved-dispatch-reference SAVED_DIRECTORY` 严格比较。Timer ON 使用匹配 ON 库、同配置 OFF 的 `--reference`，以及 `--clock-probe-so`，warmup/measure 均为0；导出器 `export_shared_fb_trace.py` 默认隐藏 wait，保留校时不确定度。框架 Triton用于性能，纯Torch参考只用于无计时精度。

## 二进制指纹

- 匹配 Host `_C`：`22f595bf4866b300db7b6f1fd1f015bd4ddcf22d1de679a7ca6d2ee22508608a`。
- 最终 NoTimer worker：`53d8c14f8a853707b92873675d8118978d889e12122c99b9b4dffe8b85beb8ab`。
- 最终 Timer worker：`fc8ca81705f0c98614332e8e8109c06d8a70b7e089fa377b7c0844743de276ef`。
- 实际策略：shared F/B=1、GM SwiGLU=on、GM tile-ready=1、none-GM构建能力=1、early-dW1=1、P1/DX cache=1、owner ratio=2、S9 flags=0、panel restore2/operand-ready1。
