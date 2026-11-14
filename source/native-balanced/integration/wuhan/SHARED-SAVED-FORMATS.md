# 共享融合的三种路由保存格式

本次适配只扩展 `megamoe-balanced-shared-expert` 的共享 V2/V3 链路。共享专家计算仍使用 BF16；`save_gmm1_output` 描述的是路由专家 Gate/Up 的保存格式，三个选择相互独立：

| 配置 | 选择 | 行为 |
|---|---|---|
| `save_gmm1_output` | `bf16` | 反向直接读取保存的 BF16 Gate/Up |
| 同上 | `none` | 前向不保留路由 Gate/Up，反向内核重算 GMM1 |
| 同上 | `mxfp8` | 反向使用已有 scale 和限幅 mask 解码，不重算路由 GMM1 |
| `shared_save_mode` | `keep` / `recompute` | 保留共享 BF16 G/U/A，或仅在反向内核重算共享中间态 |
| `routed_input_mode` | `saved` / `redispatch` | 读取前向 packed X，或由 V3 反向按冻结 placement 重派发原 X/prob |

`auto` 保留原策略：有限正限幅选 BF16，禁限幅选 MXFP8。不替整网决定保存策略，也不在反向前重新调用 MegaMoE forward。

## 使用和二进制配套

`NativeFSDP2Pool(..., save_gmm1_output='none')` 或 `'mxfp8'`，再调用现有 `bridge.forward_shared(..., shared_save_mode='keep', routed_input_mode='redispatch')`。本次前向解析后的实际保存格式会冻结在 autograd context，反向使用同一格式。

需要重新构建匹配的 `_C` 与 worker。共享前向也必须按实际格式选择计算模板，不能仅移除 Host 的 BF16 限制，否则保存态分配与内核写入布局会不一致。新 worker 的 `megaflux_shared_expert_forward_saved_modes()`、`megaflux_shared_expert_backward_saved_modes()` 返回位集合：BF16=1、none=2、MXFP8=4。构建开启对应共享入口时应为 7；旧库缺少 getter 时只按 BF16 能力处理，禁止新 Host 静默调用不支持的旧库。ABI1 旧共享反向仍保留 BF16 限制，使用 V2/V3 才能选择新格式。

推荐配套构建开关：

```bash
export MEGAFLUX_SHARED_EXPERT_FORWARD=1 MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MEGAFLUX_SWIGLU_DECOUPLE=on MEGAFLUX_GM_SWIGLU_TILE_READY=1
export MEGAFLUX_BACKWARD_NONE_GM=1  # 构建能力
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1
export MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1 MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_OWNER_EARLY_RATIO=2
python3 setup.py build_ext --inplace --force
bash scripts/build_megaflux_timer_capture.sh UNIQUE_PREFIX
```

运行时共享 tile 流水设置 `MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE=1`。`MEGAFLUX_BACKWARD_NONE_GM` 的运行值仅在 routed `none` 时可选为 1，BF16/MXFP8 设置 0；默认仍为 0。none GM 模式复用已有重算与 dW 流水，并让 V3 的恢复路径优先 W1；X/prob 源读取退休与共享 S0 的同步不删除。none/MXFP8 的 routed 前向沿用已有整专家 GM SwiGLU，不把 BF16 的 raw G/U 别名和逐 tile ready 直接套到其他格式。

## 框架配置与待接入事项

FSDPTurbo 的 `MegaMoeBalancedConfig` 新增 `routed_input_mode='saved'|'redispatch'`。MindSpeed 新增 `--optimization.megamoe_balanced_routed_input_mode` 同名值。内核重派发要求 `shared_expert_mode='fused'`、`backward_restore='panel'`。其余旧默认值保留。

示例：

```text
--optimization.megamoe_balanced_shared_expert_mode fused
--optimization.megamoe_balanced_shared_expert_save_mode recompute
--optimization.megamoe_balanced_routed_input_mode redispatch
--optimization.megamoe_balanced_save_gmm1_output mxfp8
--optimization.megamoe_balanced_backward_restore panel
```

同事整网接入时仍需验证真实 FSDP unshard/reshard、共享与路由参数梯度、checkpoint/UAM 的 saved-tensor 生命周期、MTP，以及恢复训练。独立 MoE 验证不代表这些整网流程已经通过。experts 应保留已保存的自定义 autograd tape，避免外层 checkpoint 再执行整个 MegaMoE forward；路由 none 的 GMM1 重算与共享 recompute 均由 backward 内核完成。

## 独立验证入口

`integration/wuhan/benchmark_shared_expert_fb.py` 新增 `--save-gmm1-output` 与 `--scenario-inputs`。按三种格式 × 两种共享策略 × saved/native 两种输入链路执行；`native` 表示外置共享＋外置 X redispatch 基线，对比共享与 X redispatch 融合链路。每个组合检查 Y 和七项梯度，native 另核验 packed X 不驻留前向 tape，并可用 `--saved-dispatch-reference` 与同格式 saved 链路严格比较。

性能使用真实框架 Triton 激活；`--reference-activation torch-fp32` 仅作无计时精度检查。Timer ON 需要对应 OFF 收据和独立时钟 probe，导出默认隐藏 wait。[格式适配时的八卡结果](SHARED-SAVED-FORMATS-VALIDATION-20261103.md)保留历史矩阵和指纹；后续优化的最新性能、依赖关系、配套库与验证边界见[共享反向流水优化](SHARED-BACKWARD-PIPELINE-20261103.md)。
