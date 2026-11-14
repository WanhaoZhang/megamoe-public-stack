# 共享专家与 X 重派发接口契约

本文件说明算子接口和兼容边界。既有BF16链路的CPU/CANN构建及17项独立算子验收已通过，包含keep/recompute、原X/prob有效前缀、双live前向倒序反向、有限延迟与Timer OFF/ON；主形状为8卡、两个EP4组。none/MXFP8扩展的配套要求与独立验收见[三种保存格式](SHARED-SAVED-FORMATS.md)。真实FSDP整网训练和checkpoint/UAM生命周期尚未验收。既有构建与运行见[使用与复现](../../docs/内核X重派发实现与验证-20261101.md)，性能与内存口径见[最终结果](../../docs/内核X重派发八卡验证结果-20261101.md)。

## 模式与框架交接

| 选择位置 | 默认/兼容路径 | 显式融合选择 |
|---|---|---|
| 共享计算构建 | shared forward/backward关闭 | `MEGAFLUX_SHARED_EXPERT_FORWARD=1`、`MEGAFLUX_SHARED_EXPERT_BACKWARD=1` |
| FSDPTurbo共享模式 | `shared_expert_mode="framework"` | `shared_expert_mode="fused"`，默认接共享V2，显式redispatch接V3 |
| MindSpeed共享模式 | `--optimization.megamoe_balanced_shared_expert_mode framework` | 同参数设`fused`，默认接共享V2，显式redispatch接V3 |
| 算子bridge路由输入 | `routed_input_mode='saved'`，保留packed X | `routed_input_mode='redispatch'`，选择ABI3内核重建 |

共享模式与路由输入模式是两个独立开关。框架适配器现已透传`routed_input_mode`，缺省仍为saved，显式redispatch才启用ABI3；真实FSDP saved-tensor hooks与checkpoint/UAM验证仍需框架接入验收。原`mega_moe`后端和`megamoe_balanced`默认framework共享路径保留。

框架现有融合适配限制为DeepSeekV4、BF16、TP1、EP2/4/8、无bias、SiLU及正有限限幅，共享F/limit与routed相同。底层独立Fs字段及V2/V3的`limit=0`禁限幅能力不扩大框架已接通范围；本次大形状设备性能只验证EP4×2，另有EP2小形状功能检查。

## Python接口与保存态

模型仍调用原`experts.__call__`并传三组live共享权重，dispatcher使用算子bridge：

```python
y_total = bridge.forward_shared(
    x, ids, prob, shared_gate, shared_up, shared_down,
    limit=limit, shared_save_mode='recompute',
    routed_input_mode='saved',  # 显式改为redispatch才使用ABI3
)
```

bridge每次通过provider获取当前routed和共享权重；反向在unshard后重取live权重，不缓存前向unsharded指针。共享输入始终是本卡原X，不参加TopK/跨卡Dispatch，不占副本槽；返回`y_total`已包含共享和路由分支。

`pool.forward_shared(...)`返回`y, status, routed_saved, shared_saved`：

```text
routed_saved = (counts, routed_state, dispatch_output,
                logical_ids, physical_ids, copies, logical_counts)
shared_saved = (X, raw_gate, raw_up, activation, shared_meta)
```

- `keep`保存限幅前BF16 `raw_gate/raw_up`和BF16 `activation`，均为连续`[M,Fs]`；原X只保存引用。
- `recompute`把这三项替换为各自的empty tensor，反向在同一个native launch内仅重建共享GU/A；原X仍通过标准saved-tensor hooks管理。
- `saved`的`shared_meta`为CPU int32 `[1, mode, M, H, Fs]`，`mode=0/1`分别表示recompute/keep。
- `redispatch`的meta在末尾追加`1`，共6项；前向finish后释放packed X/prob，tape的`dispatch_output`槽为empty UInt8。反向按冻结排列和计数重建，不重新规划路由。

反向仍需要临时packed X/prob，释放前向tape不等于降低单层峰值；两种保存策略也不自动实现框架UAM/offload。

## Autograd与梯度

兼容saved调用有11个输入：

```text
bridge, X, IDs, prob, routed_W1, routed_W2,
shared_Wg, shared_Wu, shared_Wd, limit, shared_save_mode
```

新的redispatch调用在末尾追加`routed_input_mode`，共12个输入。保存的tensor tape仍为14项：

```text
prob, counts, routed_state, dispatch_output,
logical_ids, physical_ids, copies, logical_counts, control,
X, raw_gate, raw_up, activation, shared_meta
```

`control`随本次placement经过saved-tensor hooks，避免checkpoint重放后混用旧epoch；ctx只持有bridge/provider及shape、dtype、mode等元数据。七项native梯度为`dx_direct, dp, dWr1, dWr2, dWg, dWu, dWd`。`dx_direct`已合并路由直接dX与共享dX，Router继续消费dp，不在kernel重复累计Router dX。routed dW返回home前缀并保留Expert-FSDP/owner SUM，共享dW回原普通FSDP参数。

saved反向返回11项，redispatch反向追加一项`None`匹配12输入。仅支持一阶反向。版本、能力或保存态不匹配会明确拒绝，不静默回退到缺少共享梯度的旧入口。

## Native ABI与返回值

共享前向descriptor独立于共享反向版本：`MegaMoeSharedExpertDescriptor`仍为ABI1、14个uint64/112字节，字段依次为：

```text
abi, weight_gate, weight_up, weight_down,
raw_gate, raw_up, activation, ready,
tokens, hidden, ffn, timer, flags, reserved
```

权重为连续BF16 `[Fs,H]`、`[Fs,H]`、`[H,Fs]`，不在Host拼接。原FullDescriptor的25/26word入口保留；共享前向使用25word前缀、selector及14word扩展，共40word。`compute_train_replica_shared`在原参数后追加Wg/Wu/Wd，返回：

```text
y_total, counts, routed_state, dispatch_output, forward_scratch,
raw_gate, raw_up, activation, ready, shared_timer
```

Timer入口再追加native Timer；NoTimer的shared_timer为空。

| 共享反向版本 | 描述符大小 | 数据与行为 | native返回 |
|---|---:|---|---|
| ABI1 | 152 B | 旧连续rawGU契约，旧入口保留 | 七梯度 |
| ABI2 | 184 B | 独立raw G/U/A，keep或共享重算，读取前向packed X | 七梯度；Timer再附native/shared Timer |
| ABI3 | 200 B | ABI2前缀加显式redispatch标志与独立X Timer，内核重建X/prob | 七梯度＋临时重建buffer；Timer再附native/shared/X Timer |

能力getter分别为`megaflux_shared_expert_forward_abi()==1`、`megaflux_shared_expert_backward_v2_version()==2`及`megaflux_shared_expert_backward_v3_version()==3`。ABI1/2不解释ABI3的redispatch标志。ABI3初版仅支持BF16；当前扩展V2/V3支持BF16、none、MXFP8，并以实际SO格式能力位协商，旧ABI1仍限BF16。详见[三种保存格式](SHARED-SAVED-FORMATS.md)。

## 前向调度和Timer

双AIV完成InitRouting，AIC可并行计算共享Gate/Up。非tile路径按共享GU→SwiGLU→Down阶段执行；tile路径使用独立ready后缀，在不同tile间流水共享GU/SwiGLU/Down，AIV1继续Dispatch和权重搬运。每个Down任务仍须等对应128行的完整F维activation，使用完整K=Fs，不改变部分和或BF16舍入。

运行时`MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE=1`要求匹配worker，并要求实际routed GM SwiGLU已启用。前向descriptor的`flags=0`保留原整段布局；tile/detail Timer标志在原ready前缀后追加元数据，ready每调用清零。当前验收使用显式GM on、GM tile-ready构建和共享tile运行时组合，默认值及完整命令以使用指南为准。

共享前向legacy摘要为`[4,128,2]`；schema2还包含独立detail和逐tile数组，不能只根据固定摘要判定工作缺失。共享反向为`[10,128,2]`，ABI3另有`[4,128,2]` X Timer。导出保留计算阶段、默认隐藏wait；逻辑lane不是物理核ID。采集、校时误差和Host作用域边界见[校时说明](CLOCK-CALIBRATION.md)。
