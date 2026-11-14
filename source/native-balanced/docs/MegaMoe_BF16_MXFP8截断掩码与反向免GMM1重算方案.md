# MegaMoe BF16：MXFP8 保存、SwiGLU 截断与反向免 GMM1 重算方案

状态：**前向 MXFP8＋截断掩码已实现；2026-09-10 用户反馈 `8247167` 编译修复后
所列 smoke 均通过。反向读取和免 GMM1 重算仍为讨论方案，未实现。** 原方案依据 `train-bf16-forward-stable@f1638db` 整理，
前向实现从 `b7ba51e` 继续。默认保存方式仍为 BF16。

## 0. 本次前向实现与验证入口

普通 `compute_train(..., swiglu_limit, "mxfp8")` 现在允许开启截断。三个 mask 使用
本稿的 `gateHigh / upLow / upHigh` 定义，先将原始 BF16 精确转换为 FP32 与 limit
比较，再量化原始 gate/up。等于边界时 bit=0；量化后的值不参与打标记。
前向 SwiGLU 仍使用原始 UB，计算路径不变。

- `TrainingSavedStateLayout` 构造末尾新增 `saveClipMasks=false`；仅同时保存 MXFP8
  且开启截断才追加掩码。新增 `gateHighOffset/upLowOffset/upHighOffset`、
  `clipMaskPlaneBytes`、`clipMaskKind`、`layoutVersion`，旧前缀及 data/scale 偏移不动。
- 有掩码：`layoutVersion=2`、`clipMaskKind=kClipMaskThreeBitplanesV1 (1)`；
  其他格式：version=1、kind=0。每份 state 没有新增 GM header。
- Python autograd 的 `ctx.saved_gate_up_metadata` 保留解析后的格式、FP32 limit、
  version、kind；`diagnostic_holder["saved_gate_up_metadata"]` 可查看同份 CPU 元数据，
  无 D2H。直接 TorchBind 接入方必须自行保留相同参数/版本。
- 普通、UDMA、MGroup 的共享 epilogue 及 Workspace 指针均接入。
  UDMA/MGroup 当前公开 raw 入口没有新增 limit 参数，仍关闭截断；本次不扩展其公开 ABI。
- 前向三个 mask UB 各 4 KiB；Host 校验 218 KiB，预算取设备 UB 和 248 KiB 的较小值。
  关闭截断的 MXFP8 仍为 206 KiB，不访问 mask UB。mask 未新增双缓冲。
- mask 由 V 直接写 UB，数据、scale、mask 统一在 V→MTE3 后写出；复用缓冲前沿用
  epilogue 末尾 MTE3→V/S 等待。不新增全局 barrier，不做整块 mask 清零。
- 反向本轮仅随共享布局同步容量计算，仍使用原始 GMM1 重算及其真实 scratch。
  不读取/解压新 gate/up，不读取 mask，不改变梯度公式或反向阶段同步。
  因此允许前向先产出 v2 state 供验证；这不代表有损激活反向已适配。

MaskReg 的 FP32 lane 每四个 mask bit 中最低位有效，使用官方的
[MaskReg StoreUnAlign<uint32_t>](https://asc.gitcode.com/api/SIMD-API/basic_api/reg_vector_compute/reg_data_store/StoreUnAlign_MaskReg.html)
将 64 个 lane 打包成连续 8 字节。每行四次写出得到 256 个列标记；UB 行按 32B 存放，
GM 按 `F/8` 字节行跨度紧凑存储。`8247167` 已将 CastTrait 改为 static constexpr，
随后收到设备 smoke 通过反馈；不据此扩展其他 CANN 版本的兼容性结论。

示例 R65536/F4096：无掩码 MXFP8 扩展 **528 MiB**，新增三个 mask 共 **96 MiB**，
总扩展 **624 MiB**，state 共 **1594497024 字节**，比 BF16 持久化少 **400 MiB**。
新增 mask 不影响 SHMEM 或 forward transient 大小，也不会减少当前反向 scratch。

### 前向逐 bit 设备验证（同一个 smoke 入口）

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
set -o pipefail
SMOKE_EP=8 SMOKE_DEVICES=0,1,2,3,4,5,6,7 \
SMOKE_SAVE_GMM1_OUTPUT=mxfp8 SMOKE_CHECK_CLIP_MASKS=1 \
  python test/megamoe_bf16_test/smoke_compute_train.py \
  2>&1 | tee bf16_mxfp8_clip_masks.log
```

该选项使用 M129/K256/N512/TopK4 的前向用例，X 只有首列为 0/±1，W1 保留 BF16
边界值，使 GMM1 参考是精确的符号变化，不受 CPU/NPU GEMM 归约误差影响。
覆盖 limit=1/10/1.001/0、等于边界、相邻 BF16 值、空专家、128 行尾块、正负 up，
末次输入清零检查全部有效 mask word 被写成零。逐位检查**所有有效行和列**，
并检查 MXFP8 数据/scale、与 BF16 保存的 y/counts 对照。
不读取未使用容量和对齐尾部，也不把这种受控输入的通过当作真实训练梯度通过。

各 rank 均须看到 `[SAVED_CLIP_MASK] ... all_valid_values_scales_and_bits=PASS`、
`[CLIP_FORWARD] ... y_counts_vs_bf16=PASS`，最后看到 `[SMOKE] compute_train forward passed`。
不加 `SMOKE_CHECK_CLIP_MASKS` 时保持原大 shape，三种 limit=0/1/10，输出格式与分配大小。
原 `test_mega_moe.sh` 的 raw 入口仍无 limit 参数，不能拿其 `mxfp8` 测试代替截断验证。

本地已通过 16 项 Python 格式/边界/覆盖/模式检查、24 项 workspace 检查、共享 C++
布局及 UDMA 布局检查。Dispatch ready C++ 回归编译通过，但可执行文件被本机 Device Guard
策略拦截，未执行。最新设备 smoke/基础前向耗时及未覆盖范围见
[设备反馈记录](MegaMoe_BF16_MXFP8_saved_state第一版.md#2026-09-10当前设备反馈与合入边界)。
下文反向消费、移除重算和梯度验收仍待负责人实施。

### 2026-09-11：反向实现基线更新

已同步 train 的 PR13/15，反向采用 `BackwardSavedStateInfo savedState_`；其构造
复用支持全部保存模式的 `InitSavedState`。原 gate/up、scale、mask 地址和版本不变。
PR15 的截断跳过逻辑统一采用运行态判断，未引入第二套模板截断开关。

PR13 为对齐小算子，反向 `ComputeForwardAux/Recompute` 显式将 SiLU 结果舍入到 BF16，
再乘 up；梯度路径也显式舍入 SiLU 和 `dH*up` 中间值。后续 loader 只替换原始 gate/up
的数据来源，必须保留这些中间舍入，不能机械套用下文历史代数公式而省掉 Cast。
前向仍使用现有数学路径，不能声称新反向辅助 H/Hp 与前向逐位一致；完整梯度按训练
参考验收。压缩版本还需单独区分实现错误和量化误差。

## 1. 建议结论与讨论边界

建议前向继续保存**原始 BF16 gate/up 量化得到的 MXFP8 数据和 E8M0 scale**，在
`swiglu_limit>0` 时额外保存由量化前原始值生成的截断标记。反向按 tile 解压，
读取标记恢复截断分支，不再运行 GMM1 重算，也不在 GM 中展开完整 BF16 gate/up。

第一版建议使用三个独立 bit plane：gate 是否被上截断、up 是否被下截断、up 是否
被上截断。它们既能恢复梯度开关，也能明确恢复 up 的 `-L/+L` 边界值。保留现有
FP8 原始值语义，暂不引入“先截断再量化”的另一种格式。

这是一种**有损激活保存**方案：标记保持前向的截断判断，但不能消除 gate/up 的
量化误差。反向重建 H/Hp 后，误差还会进入 TopK 权重梯度和 W2 梯度；不能只验证
`gateInd/upInd` 或最终前向 y 就认为训练适配完成。

本次重点是普通 BF16 的训练反向。前向的共享量化/保存模块可同步覆盖普通、UDMA、
MGroup，但不能据此宣称 UDMA/MGroup 已有配套训练反向或设备验证。

## 2. 方案制定时的代码基线（前向最新状态见第 0 节）

| 环节 | 当前实现 | 本方案要改变的部分 |
|---|---|---|
| 保存模式 | `bf16` 默认；另有 none/mxfp8/auto | 默认不变，只扩展显式 MXFP8＋截断 |
| Host 限制 | 显式 MXFP8＋正 limit 会报错 | 接口、mask、反向适配完成后才允许开启 |
| 前向 | SaveGateUp 在 ComputeSwiglu 前执行，原始 BF16 UB 供实际计算 | 在同一原始 UB 上生成截断标记 |
| 反向 P0-C | `GMM1RecomputePhase()` 重算 gate/up，写两块 BF16 GM scratch | 保存值消费路径跳过该阶段计算 |
| 反向 P1 | 从 scratch 加载 gate/up，计算 H/Hp、dp partials，重新加载后算 dZ | 两次加载改为读取保存值并解压/恢复截断值 |
| 反向掩码 | 用 raw 与 clamped 的差生成 `gateInd/upInd` | 压缩＋截断路径读取前向标记 |

源码位置（路径相对仓库根目录，行号按基线）：

- `csrc/ops/mega_moe.cpp:92`：MXFP8＋截断的 Host 拒绝条件。
- `kernels/mega_moe_bf16/include/block_epilogue_swiglu_bf16.h:116`：保存先于 SwiGLU。
- `kernels/mega_moe_bf16/include/block_saved_gate_up_mxfp8.h:35`：现有量化、scale 和 DMA 写出。
- `kernels/mega_moe_bf16/include/block_epilogue_swiglu_backward_bf16.h:518`：截断和 indicator。
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h:339`：epilogue 绑定重算 scratch。
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h:1099`：调用 GMM1 重算。
- `kernels/mega_moe_bf16/include/mega_moe_workspace_info.h:232`：反向两块重算区。

## 3. 数值契约

设原始 GMM1 两半输出为 BF16 的 `g/u`，转换到 FP32 后进行截断和 SwiGLU；
`L` 为同一次 forward 的、实际传入 tiling 的 FP32 正 limit。

```text
gc = min(g, L)                     # gate 只有上界
uc = clamp(u, -L, L)               # up 有上下界
H  = BF16(SiLU(gc) * uc)
Hp = BF16(H * packed_topk_weight)
```

当前反向的 `ind = 1 - min(1, abs(raw-clamped)*1e30)` 意图是：未截断为 1，截断为 0。
本方案明确采用边界包含语义：`g==L`、`u==-L`、`u==L` 时均不截断，indicator 为 1。
比较发生在原始 BF16 精确转换的 FP32 值与 FP32 L 之间，不能用舍入后的 BF16 L 比较。

注意现有 `1e30` 差值放大写法在极小正 limit、极小差值下不一定得到严格 0/1。
若要支持这类 limit，需先与反向负责人确认采用上述比较语义；不能声称与旧算式在
所有浮点边界上逐位相同。首轮重点覆盖实际使用的 L=1、10 及附近 BF16 可表示值。
异常 NaN/Inf 的梯度行为另列验证，不以有限值的指示函数推导覆盖异常输入。

### 3.1 为什么不从解压值重算 indicator

已用仓库 W8A8 CPU MXFP8 参考构造以下例子：同组包含 16，scale 为 0.0625。

| 数值 | 前向原始值 | MXFP8 解压值 |
|---|---:|---:|
| gate | 10.0625 | 10 |
| limit | 10 | 10 |
| gateInd | 0 | 从解压值判断得到 1 |

这种错误会改变梯度是否归零。掩码必须来自前向原始值，并在反向直接使用；
解压值再次越过边界时也不能推翻前向标记。

### 3.2 三个 bit plane 的定义

每个 packed 元素位置 `(r,c)` 保存以下三个 bit，1 均表示发生对应截断：

| 标记 | 定义 | 反向用途 |
|---|---|---|
| `gateHigh` | `g > L` | gateInd=0；恢复 gc=L |
| `upLow` | `u < -L` | upInd=0；恢复 uc=-L |
| `upHigh` | `u > L` | upInd=0；恢复 uc=+L |

`upLow` 和 `upHigh` 不允许同时为 1。全部为 0 是合法的未截断状态。
关闭截断时不分配、不生成、不读取这三个 plane，逻辑 indicator 恒为 1。

先前讨论的“gate/up 各 1 bit，合计 64 MiB”只覆盖两个梯度开关；本稿多保存 up
的截断方向，合计变成 96 MiB。两位方案也能实现，但须额外保证原始 up 的符号在
FP8 量化下始终可恢复，包括量化到有符号零的情况。当前没有这项设备证据，因此
第一版建议多花 32 MiB 明确记录方向，避免靠解压浮点比较猜测 `+L/-L`。

若团队优先追求两位方案，应单独冻结“FP8 sign bit 保留原始符号”的契约，并验证
正负零、下溢、极端组动态范围及 Cast 行为，再决定省去第三个 bit plane。

### 3.3 反向恢复值与梯度

先解码原始保存值，再在 FP32 寄存器中应用标记和边界：

```text
gq = decode_mxfp8(saved_gate, gate_scale)
uq = decode_mxfp8(saved_up,   up_scale)

gc_hat = L   if gateHigh else min(gq, L)
uc_hat = -L  if upLow else (+L if upHigh else clamp(uq, -L, L))

gateInd = 1 - gateHigh
upInd   = 1 - (upLow | upHigh)

s = sigmoid(gc_hat)
dUp   = dH * SiLU(gc_hat) * upInd
dGate = dH * uc_hat * s * (1 + gc_hat*(1-s)) * gateInd
```

对被截断元素直接恢复 FP32 的 ±L，避免边界被 FP8/BF16 再次舍入。未截断元素
即使解压后越界，也只限制参与函数计算的近似值，indicator 仍来自保存标记。
在有限值验证中，截断支路的梯度必须为零；如需定义异常输入的零梯度，不能只靠
`NaN*0`，须明确 select/异常策略并与现有训练语义对齐。

`ComputeForwardAux`、SwiGLU gradient 以及任何仍会调用的 forward-recompute helper
必须使用同一套恢复逻辑。H/Hp 只做非线性重建，没有执行 GMM1，这部分计算仍需保留。

## 4. 保存格式、偏移与版本

定义 `R=maxRecvTokens`、`F=N/2`、`G=ceil(F/32)`。首版沿用现有 F 对齐约束，不
放开非完整 32 元素组。行序保持 `[local expert][source rank][row]` 的 packed 行序。

现有 gate/up 数据、scale 和旧 saved-state 前缀偏移不变，在当前 MXFP8 末尾追加：

```text
旧前缀 | gate FP8 | up FP8 | gate E8M0 | up E8M0 | gateHigh | upLow | upHigh

maskRowBytes   = G * sizeof(uint32_t)
maskPlaneBytes = AlignUp(R * maskRowBytes, 512)
gateHighOffset = AlignUp(existingMxFp8TotalBytes, 512)
upLowOffset    = gateHighOffset + maskPlaneBytes
upHighOffset   = upLowOffset + maskPlaneBytes
totalBytes     = upHighOffset + maskPlaneBytes
```

每个 mask plane 的逻辑类型为 `uint32[R,G]`；第 `j` 个 word 的 bit `b` 对应
`c=32*j+b`，最低位对应组内第一列。mask 和 scale 的 group 索引一致，但各自存储类型
不同，不能混用字节偏移。元素的 mask word 地址为 `base + (r*G+j)*4`。
128×256 tile 每行拥有 8 个完整 word，可独立写出，无需跨核 OR、原子操作或 read-modify-write。

模式处理建议：

| 保存模式 | limit | 格式/标记 | 反向目标行为 |
|---|---|---|---|
| none | 0 或正值 | 无 gate/up 扩展 | 保留现有 GMM1 重算 |
| bf16（默认） | 0 或正值 | 现有原始 BF16，无新增 mask | 可作为先接入保存值的无压缩验证路径；未适配时仍重算 |
| mxfp8 | 0 | 现有 FP8+scale，无 mask | 接入保存值后跳过 GMM1 |
| mxfp8 | 正值 | FP8+scale+三个 mask | 本方案目标，接入后跳过 GMM1 |
| auto | 任意 | 暂保持现有选择规则：关闭截断用 MXFP8，正 limit 用 BF16 | 不自动把原来 BF16 的截断训练切成有损压缩 |

所有前向分支保持现有输出 Tensor 个数。建议给新格式定义明确的布局版本及
`clipMaskKind=none/three_bitplanes_v1`，在共享 Host/Device 布局代码中定义，不能依赖
私有偏移硬编码。Python 包装层根据同次 forward 的已解析模式、normalized limit 和
版本保存 ctx 元数据；反向显式接收并验证这些信息。直接使用 TorchBind/Host 的接入方
同样必须保留版本及参数，不能仅用“buffer 足够大”猜格式。

本稿不要求往 GM 的每份 state 增加头部或进行 D2H 查询；ctx 的 CPU 元数据由同版
Host/包装层的格式契约生成。新增版本参数的具体 ABI 名称可评审后定，但必须做到旧
consumer 对新 clip-mask 格式报错，不能静默当成普通 MXFP8。部署时 Host、Device、
Python 扩展须一起更新。反向保存值 loader 未接入前，不允许将新 state 静默当作
普通 MXFP8 读取。第 0 节允许前向先产出新格式；配套反向仍重算，不消费扩展区。

## 5. 前向实现与初始化

建议分配三个小的 mask UB 缓冲，直接从原始 gate/up 生成完整 word。起步版本采用
清楚的顺序，确认精度后再考虑融合量化遍历：

1. WaitForCube 后，原始两个 BF16 tile 已可见。
2. 在原始值上比较 g>L、u<-L、u>L，并按约定列顺序打包到独立 mask UB。
3. 量化原始 gate/up；保持现有 E4M3FN、32 元素组、E8M0、舍入与饱和规则。
4. V→MTE3 后写出 FP8、scale、mask。缓冲独立，可与后续 SwiGLU 计算重叠。
5. 实际前向 SwiGLU 继续读取原始 BF16 tile，使用原有 FP32 limit，不消费解压值。
6. 复用 mask/quant UB 和通知下一轮 Cube 之前，确认所有相关 MTE3 写出已完成。

不得为打标记先把完整 gate/up 写到临时 GM 再读取；也不应为了存 mask 新增全局
barrier。现有量化前 CTRL[60] 设置及量化后恢复必须保留，不能再次遗漏 Cast 的
饱和模式前提。

初始化规则：Host 仍 `empty(totalBytes)`，只清零旧前缀中原本要求清零的范围。
三个 mask plane 不做整块 memset/zero_；每个有效 word 必须完整赋值，即使本组
没有截断也要写 0，不能只写发生截断的 bit。未使用容量、空专家和对齐尾部不读不写；
末尾非有效 bit 写 0。不同 tile/核必须拥有不重叠的完整 word。

以 128×256 tile 为例，每个 mask plane 为 128×32=4096 B。现有量化 UB 上界
206 KiB，加三个 plane 的 12 KiB，前向预算约 **218 KiB**。该值是建议布局预算，
需实现时把所有实际 live buffer、编译器需求一起核对，并使用平台查询值进行 Host
容量检查；不能只按纸面总量宣称编译一定成功。首版不要叠加 mask 双缓冲。

## 6. 反向加载、UB 和 GMM1 重算移除

### 6.1 两次加载路径都要改

当前 `SwiGLUGrad()` 的第一轮 gate/up 加载用于生成 H、Hp、dp partials；这些操作
覆盖 gate/up 所在 UB 后，会重新加载 gate/up 再计算 dGate/dUp。首版保留这个顺序，
两轮都走统一的 saved-tile loader。不要只修改第二轮求导而遗漏第一轮辅助输出。

对 MXFP8，loader 按 tile 执行 MTE2 数据/scale 搬入、解码和恢复；对 BF16 验证路径
沿用原 BF16 tile 搬入；对 none 仍读取重算 scratch。禁止先把整份保存值解压到两块
完整 GM BF16 plane，否则显存和额外访存优势会明显缩水。

建议以现有 3×64 KiB BF16 工作 tile 为基础，gate/up 分别加载，复用同一个压缩
暂存区：

| 反向 AIV 每个 tile 的 UB | 预算 |
|---|---:|
| dHp/dH、gate、up 三个 BF16 工作 tile | 192 KiB |
| 单个 gate 或 up FP8 暂存 tile，串行复用 | 32 KiB |
| 单个 scale 暂存，128 行各按 32 B 存放 | 4 KiB |
| 三个 mask plane，可在两次加载间保留 | 12 KiB |
| 合计 | **240 KiB** |

按当前 Host 248 KiB 上限估算还剩 8 KiB；实际仍取平台容量与支持上限的较小者。
不能同时额外驻留 gate/up 两份完整 FP8 tile，那会超出本预算。高位暂存区复用前
须等待对应 V 读取结束，MTE2 完成后 V 才能解码。mask 的加载和解包不能占用仍在
使用的 dHp 或 gate/up 数据区。FP8/E8M0 解码可分寄存器块完成，不另建完整 FP32 tile。

恢复边界值应在后续 FP32 计算寄存器中应用，不能把非 BF16 可表示的 L 先舍入写回
BF16 UB。Host 的 backward UB 检查必须由 `max(旧各阶段, 新P1预算)` 计算，原来的
仅 192 KiB P1 预算不足以覆盖这个 loader。

### 6.2 可以删除与必须保留的内容

保存值消费路径可以跳过 `GMM1RecomputePhase()`，取消该阶段 gate/up GM 写出，并
把 backward workspace 的 `ptrRecomputeGate/ptrRecomputeUp` 两个 plane 大小设为 0。
Host 分配公式、Device 偏移和 epilogue 参数绑定必须一同更新；none 路径仍要原来的
重算区，不可全局删除定义。建议用独立 `GateUpSource` 表达“重算 scratch/BF16 saved/
MXFP8 saved”，不要继续把参数名都叫 recompute 然后偷偷改地址。

必须保留：

- P0 的计数/路由恢复、dY gather 和 Dispatch。它们还服务于 GMM2 输入梯度。
- AIC 的 `firstPipe.Destroy()` 及 AIV0/AIV1 的 drain、Destroy、ready 发布。
- P1 开始前对所有 AIV0/AIV1 UB 释放的等待。重算消失后这段等待可能更显眼，
  不能因它原来位于 P0-C 附近而一同删掉。
- P1 的 MTE2→V、V→MTE2 复用依赖以及 MTE3→MTE2 等待；当前两次加载之间的
  Hp/dp 写出缓冲还在被 DMA 使用。
- P1 pair-local dZ 完成事件、全体 AIV 的 dZ/Hp/dp 可见性与计数。
- packed X、packed TopK 权重、元数据、progress；后续 dW1、dX、路由仍然依赖它们。
- `ptrHp`、`ptrGradZ`、`ptrDpProducts` 等其他 backward workspace。

当前 `bwFlagAiv1Ready` 同时累计 P0 的 AIV1 释放与 P1 两组 AIV 完成，目标为
`3*blockNum`。首版不要改这套计数契约。移除重算不等于移除 P0，也不等于 P1 可以
越过这些 ready 条件。

`WaitForCube()` 原来的注释涉及同一 AIC 的 gate/up Fixpipe 写入；切换 saved 读取后
要更新数据来源说明，但该等待仍保护 dHp Fixpipe。前向 state 的可见性依靠正确的
同流顺序或跨流 event，不依靠重算阶段偶然提供的延迟。

## 7. 内存与性能估算

示例固定 M=4096、K=7168、N=8192、EP=8、每卡4专家、R=65536、F=4096；
所有数字为每个 rank、每份 state，MiB=2^20 B。

| 保存内容 | 扩展大小（不含旧前缀） | saved-state 总字节数 |
|---|---:|---:|
| 不保存 gate/up | 0 | 940185600 |
| BF16 gate/up | 1024 MiB | 2013927424 |
| 现有 MXFP8 gate/up＋scale | 528 MiB | 1493833728 |
| 本方案 MXFP8＋scale＋三个标记 | **624 MiB** | **1594497024** |

三个标记各 32 MiB，起始偏移分别为 1493833728、1527388160、1560942592。
相比保存 BF16，每份持久化 state 仍节省 **400 MiB**；相比无截断 MXFP8 增加 96 MiB。
反向保存值路径另可取消两块重算 BF16 scratch，共 **1024 MiB**。

这两项不能无条件相加称为“峰值下降 1424 MiB”：多前向后反向时有多少 state 仍然
存活、反向 scratch 何时分配/释放、分配器缓存和其他算子重叠，都决定真实峰值。
若有 B 份相同 shape 的 state 同时存活，其持久化部分相对 BF16 约少 B×400 MiB；
反向 scratch 的减少按实际同时执行的 backward 数量另算。

性能模型应看整段前后向：

```text
净变化 ≈ 前向量化/打标记/额外写出
       + 反向压缩加载/解码/恢复/掩码开销
       - 重算 GMM1 在关键路径上的贡献
       - 被取消的重算 GM 写出等开销
```

P0-C 原来与 AIV gather/Dispatch 重叠，不能把 profiler 中 GMM1 重算的整段时间都当作
可省端到端时间。反向仍有两轮 gate/up 加载；先把精度、内存和同步做对，再评估是否
值得合并加载或复用寄存器。当前 MXFP8 的历史前向耗时约 31993/31957 μs，仅供旧版
背景参考，不是该“截断＋标记”方案的性能预测，也没有证明新反向收益。

## 8. 训练数值风险与验收

### 必须接受的近似范围

标记使截断分支选择来自真实前向，但未截断 gate/up 仍为量化近似，因此：

- dGate/dUp、进而 dX、dW1 会受影响。
- 由解压值重建的 H/Hp 不保证等于前向实际使用的 BF16 H/Hp。
- TopK 权重梯度的 `dp=reduce(dHp*H)` 和 W2 梯度的 `dW2=dO^T*Hp` 也会受影响。

应保持原有 BF16 舍入顺序：先形成 BF16 H，再乘 packed weight 成为 BF16 Hp；dp
也不能悄悄改成用未舍入的 FP32 H。即便保持这个顺序，输入近似仍会造成数值差异。
若要求所有梯度与原训练逐位等价，仅保存有损 MXFP8 不足以满足，需要保留更多精确
中间结果或继续重算，并重新评估内存目标。

### 验证分层

1. **CPU 格式参考**：以 BF16 位模式模拟原始 g/u，使用同一 FP32 L，验证三个标记、
   word 位序、解码、±L 恢复及上述梯度表达式。覆盖 L=1/10、等于边界、边界前后一个
   BF16 ULP、零、负 gate 大值、量化跨边界、up 量化为零、极端组动态范围。
2. **前向设备验证**：标记逐 bit 对照；保留原始数学路径，比较 y/counts 与同配置
   BF16 保存基准。保存值检查单列，不能以 y 正确替代。覆盖空专家、M 尾块、连续
   调用时截断模式相反、全部未截断，检出未完整覆盖旧 mask 的情况。
3. **分离梯度参考**：先比较 Device 新反向与“CPU 同样解压＋同样标记”的参考，检出
   实现错误；再比较该压缩参考与精确 BF16 重算基准，量化误差与实现误差分开报告。
4. **完整反向设备验证**：dX、dTopkWeights、dW1、dW2 分别报告 max abs、绝对误差分位数、
   有分母下限的相对误差及整体范数误差。对截断区的 dGate/dUp 单独检查归零；混合
   不同模式/limit 的多个 forward 后倒序 backward，检查 ctx 模式和生命周期。
5. **稳定性/收益**：先单机，现有双机再回归；分别记录前向、反向、总时间、实际峰值
   allocated/reserved 内存、P1 解码及 ready 等待。EP512 没有设备证据，继续待验证。
   最后由训练负责人做多步训练的 loss/梯度统计，决定可接受阈值。

现有 saved MXFP8 抽查允许量化步长差异，它不能直接作为四种训练梯度的验收阈值。
阈值需由训练方按模型和目标制定，本稿不预先声称可达到原 BF16 的容差。

## 9. 分阶段适配与分工建议

| 阶段 | 工作 | 验证边界 |
|---|---|---|
| A：冻结格式 | 确认三标记/两标记取舍、边界语义、布局版本、消费开关 | CPU 格式参考通过，接口评审 |
| B：前向产出 | MXFP8＋三个标记、Host 分配、UB 和全覆盖写出 | 前向 y 与 mask 校验；反向仍可保留重算作对照 |
| C：反向读取 | 接入 saved loader、所有 H/Hp/dZ 使用位置；先做 BF16 saved 无压缩对照 | 验证去掉重算后的数据来源、行序与同步 |
| D：MXFP8 消费 | 接入解压/标记恢复，停用 GMM1 重算并按模式缩减 scratch | 四种梯度、连续调用、生命周期和峰值内存 |
| E：性能优化 | 根据实际瓶颈考虑融合 mask/quant 遍历及加载优化 | 端到端收益，保留可回退的基准 |

前向负责人处理共享保存模块、workspace/tiling、Host 分配与产出校验；反向负责人
处理 P0-C 分支、P1 loader/恢复、Host/Device scratch、梯度与训练回归。双方先冻结
共享布局和版本参数，避免并行修改不同的偏移公式。阶段 B 若允许诊断运行正 limit
MXFP8，必须有明确的“仍重算”消费策略；不得把生成标记完成当作取消重算已经完成。

代码范围清单：

- 共享布局：`mega_moe_training_workspace.h`、普通/UDMA `mega_moe_workspace_info.h`。
- Host/接口：`csrc/ops/mega_moe.cpp`、三个 BF16 Host、`csrc/torch_classes.cpp`、
  `ascend_deepep/__init__.py`，同步检查 backward/debug/timer 参数链。
- 前向：`block_saved_gate_up_mxfp8.h`、`block_epilogue_swiglu_bf16.h`、三种 kernel 绑定。
- 反向：`mega_moe_bf16_backward.h`、`block_epilogue_swiglu_backward_bf16.h`、
  必要的 `mega_moe_bf16_backward_catlass_impl.h`，更新 producer/consumer 注释及 timer。
- 验证/文档：保存值参考、模式测试、训练梯度参考和前后向接入说明；既有 sh 入口保留。

## 10. 需要团队确认的决策

1. 是否接受第一版三个 bit plane（示例 96 MiB），以明确恢复 up 截断方向？如坚持
   两个 bit，是否愿意把 FP8 符号保留作为必须通过的独立数值契约？
2. 是否认可保存**原始值**的 MXFP8，而非新增截断后量化格式？后者可能改善某些组
   的动态范围，但还涉及 FP32 L→BF16 再舍入及保存语义变更，应独立评估。
3. 是否接受 H/Hp、dp 和全部梯度都可能有压缩误差，并由训练方提供验收阈值？
4. 反向如何校验同份 state 的版本/模式并选择 loader，谁负责共享文件及消费路径验收？
   前向显式 MXFP8＋截断已开放；当前反向仍重算，不是压缩消费路径已开放。

前向已采用三个 plane 和原始值量化；后续若更改这两项需显式升级格式。
反向仍需确认误差阈值和消费策略，再实施 C→E。本稿不修改当前默认 BF16 保存方式，不执行
train 合入，也不把尚未适配的反向标记为可用。
