# MegaMoe BF16：MXFP8 gate/up 保存第一版

本版在前向 kernel 内压缩保存 gate/up。前向 GMM1、SwiGLU、GMM2 的数学计算仍保持
原来的 BF16/FP32 路径；FP8 只用于跨前后向保留的激活。2026-09-10 用户反馈
`8247167` 修复后普通 BF16 设备构建、所列 smoke 均通过；验证范围见本文末尾。
反向算法仍由反向负责人适配；当前反向继续执行 GMM1 重计算，不读取新增的压缩内容。

2026-09-11 已同步 train 的 PR13/15，统一采用 PR15 的运行态截断判断；保存格式、
掩码定义及 L2 bypass 不变。反向持久视图改为 `BackwardSavedStateInfo savedState_`，
后续消费保存值时须保留 PR13 的 BF16 中间舍入。合并前 `3a9d230` 已收到精度通过和
保存 bypass 收益反馈；合并版尚待设备回归，详见
[集成记录与回归方法](MegaMoe_BF16_稳定前向优化迁入说明.md)。

## 运行态保存模式（默认保存 BF16）

`save_gmm1_output` 是每次调用的参数，不是 build 宏或可变的进程全局开关。

| 参数值 | gate/up 的行为 | 限制 |
|---|---|---|
| `auto` | limit=0 保存 MXFP8；开启截断保存 BF16 | 保持此前默认行为 |
| `none` | 不保存 gate/up，不分配 gate/up 和 scale | 仍保留 packed X、TopK 权重、路由元数据和 progress |
| `bf16`（默认） | 保存原始 BF16 gate/up | limit=0 或开启截断均可 |
| `mxfp8` | 保存 E4M3FN 数据及 E8M0 scale；开启截断追加三个 bit plane | 前向 smoke 已反馈通过；反向仍重算 |

普通、UDMA、MGroup 的 raw `compute` / `compute_with_timer` 均增加末尾字符串参数，
默认 `bf16`，省略该参数的调用保存 BF16。普通 `compute_train` 及三种 backward
入口则在原 `swiglu_limit` 后追加该参数。返回值个数不变。例如：

```python
y, counts, state = op.compute(shmem, x, ids, weights, w1, w2, max_recv, "none")
y, counts, state, transient = op.compute_train(
    shmem, x, ids, weights, w1, w2, max_recv, 0.0, "bf16")
# 整网 autograd 入口：省略 save_gmm1_output 即默认保存 BF16。
y, counts = train_op(x, ids, weights, w1, w2, max_recv, save_gmm1_output="bf16")
```

Host 将模式解析为 `tiling.savedGateUpMode`（0=auto、1=none、2=BF16、3=MXFP8），
auto 在 Host 解析为实际格式。Device 的 `Process()` 在所有 expert/tile 循环之前选择
`ProcessWithSaveMode<...>()`，三种实现都编译进同一个 SO。模式作为模板参数传到
GMM1 epilogue；`none` 编译路径不执行保存、量化、溢出模式设置及保存专用等待。
正常 SwiGLU、GMM2 输入写出和流水同步继续执行。此方式会增加编译代码量；
运行时分派及代码布局是否影响性能需设备测试，不能保证与旧 SO 的耗时完全相同。

共享布局构造末尾为 `(N, saveMxFp8, saveGateUp=true, saveClipMasks=false)`。none 的 `totalBytes=legacyBytes`，
gate/up 和 scale 容量为 0，Device 四个保存指针为空。原有字段偏移、初始化范围、
SHMEM 和 transient 分配不变；MXFP8 无掩码为 206 KiB UB，有掩码为 218 KiB UB。
前文 shape（M4096/K7168/N8192/EP8/R65536/每卡4专家）的 saved-state 为：

| 模式 | 字节数 |
|---|---:|
| none | 940185600 |
| BF16 | 2013927424 |
| MXFP8 | 1493833728 |
| MXFP8＋截断掩码 | 1594497024 |

`MegaMoeBF16Train` 在每个 autograd ctx 中保存本次模式和 limit，普通/timer/debug
反向都传回该模式。不要从下一次 forward、模块的可变属性或环境变量推断上一份 state
的布局。直接调用 C++/TorchBind backward 的接入方必须传入对应 forward 的相同参数。
state 没有自描述格式头。当前 backward 只调整容量检查和布局，仍重算 GMM1；
未来适配时，none 必须保留重计算，其余模式才能切换为读取保存值。

### 同一 SO 对比三种模式

首次更新需重新构建普通 BF16（脚本也会重建其 backward 和 Python 扩展）；要测试
UDMA/MGroup 时也须重新构建对应 SO，不能把新 Host 的 none 模式配给旧的保存 kernel。
构建参数沿用本机原配置，测试 sh 入口不变：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950

# 三次保持相同 shape/seed；先检查精度。之后沿用原性能参数，只切换保存模式。
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 --mode precision --warmup 0 --loops 1 --repeat 1 --save-gmm1-output none
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 --mode precision --warmup 0 --loops 1 --repeat 1 --save-gmm1-output bf16
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 --mode precision --warmup 0 --loops 1 --repeat 1 --save-gmm1-output mxfp8
```

none 显示 `[SAVED_GATE_UP] ... SKIP`，仍检查最终 y、counts 和旧 saved-state 数据；
BF16/MXFP8 还须各 rank 保存值抽查 PASS。多机的 manifest 会检查模式一致。
training smoke 可用 `SMOKE_SAVE_GMM1_OUTPUT=none/bf16/mxfp8/auto` 选择模式，
省略则为 bf16；各模式测 limit=0/1/10。默认 smoke 检查前向容量和输出有限值。
`SMOKE_CHECK_CLIP_MASKS=1` 配合显式 mxfp8，使用小 shape 的精确 GMM1 用例验证全部
有效掩码 bit、数据/scale 和 BF16 保存的 y/counts 对照，详见
[截断掩码方案第 0 节](MegaMoe_BF16_MXFP8截断掩码与反向免GMM1重算方案.md)。不替代反向回归。

本地 CPU 验证覆盖三种分配的旧偏移不变，以及多次不同模式前向后倒序调用普通、
timer、debug 反向的参数保留。它不验证 CANN 指令、Device 编译或真实性能。

## 显式 auto 下 swiglu_limit 决定实际分配

| public swiglu_limit | 语义 | gate/up 格式 | scale 分配 |
|---|---|---|---|
| 0 | 关闭截断 | 两个 FP8 E4M3FN plane | 两个 E8M0 scale plane |
| 1、10 等有效正值 | 开启截断 | 两个原始 BF16 plane | 不分配，Device 指针为空 |

Host 先把 public limit=0 转换为已有 tiling 关闭哨兵 FLT_MAX，Host 和 Device 都通过
`SaveGateUpAsMxFp8(tiling.swigluLimit, tiling.savedGateUpMode)` 选择格式，同时检查
mode 是否为 none 决定是否分配。不要把 public 0 直接传给这个内部选择函数。
**只分配实际使用的格式，不先分配完整 BF16 再附加 FP8。** auto 开启截断仍保留 BF16。
显式 mxfp8 开启截断则追加 gateHigh/upLow/upHigh 三个 bit plane，按原始值生成，
仅覆盖有效 word，不全量清零。新增格式及后续梯度误差边界见截断掩码方案。

普通 `compute_train` 已有 swiglu_limit 参数，可以覆盖两种分配。
普通 raw compute、UDMA、MGroup 当前公开前向入口仍默认关闭截断；显式 auto 时保存 MXFP8，省略模式则保存 BF16；
三版共用格式选择与量化代码，本版不为 UDMA/MGroup 新增公开的截断参数。

## MXFP8 数值规则

参考 W8A8 的 `kernels/mega_moe/include/block_epilogue_swiglu_mx_quant.h`：

- FP8 E4M3FN 数据、E8M0 scale；沿每行 F=N/2 方向每 32 个元素一组。
- gate、up 分别量化；保存的是截断、SiLU、TopK 乘权之前的原始 GMM1 BF16 值的量化结果。
- 原样沿用 ComputeMaxExp、ComputeScale、ComputeDataForQuantTargetFp8 的向量计算。
  包括 BF16 指数提取、零/特殊值处理、缩放、饱和转换和 CAST_RINT 舍入，不改成另一套 amax/448 规则。
- 对正常有限数据，可按 E8M0 指数恢复 scale；byte=0 表示 2^-127，byte=255 为特殊值。
  反量化为 decode_e4m3(data) * decode_e8m0(scale)，随后按当前反向的 BF16/FP32 精度约定计算。
- scale 随每一份 saved-state 保留，不能被后续 forward 的 scale 覆盖。

此压缩会引入新的反向数值误差。前向最终 y 不使用压缩后的值，因此前向 PASS 不能证明
反向梯度或训练收敛通过。外部量化与内部量化对比时，应统一格式、分组、舍入和特殊值规则。

## Host / Device 共用布局

构造函数：
`TrainingSavedStateLayout(M,K,TopK,EP,expertPerRank,R,N,saveMxFp8,saveGateUp,saveClipMasks)`。
R=maxRecvTokens、F=N/2。先用同次调用的 mode 和规范化 FP32 limit 解析实际模式：
`saveGateUp=(mode!=none)`，`saveMxFp8=(resolvedMode==mxfp8)`，
`saveClipMasks=saveMxFp8 && (tilingLimit!=FLT_MAX)`。不能仅凭 limit 判断格式。
原 packed X、cumsum、路由索引、完整计数、packed TopK 权重和 backward progress 偏移不变。

令 `A(x)=AlignUp(x,512)`，D 为 1（MXFP8）或 2（BF16）：

| 字段 | 字节布局 |
|---|---|
| legacyBytes | 原字段区域末尾；初始化范围仍为 [0,legacyBytes) |
| gateOffset | A(legacyBytes) |
| gateUpPlaneBytes | 保存时 A(R*F*D)，none 为 0 |
| upOffset | gateOffset+gateUpPlaneBytes |
| gateUpScaleBytes | 保存 MXFP8 时 A(R*ceil(F/32))；其他为 0 |
| gateScaleOffset | upOffset+gateUpPlaneBytes |
| upScaleOffset | gateScaleOffset+gateUpScaleBytes |
| gateHighOffset | upScaleOffset+gateUpScaleBytes |
| clipMaskPlaneBytes | MXFP8＋截断为 A(R*ceil(F/32)*4)，其他为 0 |
| upLowOffset | gateHighOffset+clipMaskPlaneBytes |
| upHighOffset | upLowOffset+clipMaskPlaneBytes |
| totalBytes | 保存时 upHighOffset+clipMaskPlaneBytes；none 为 legacyBytes |

数据 plane 为 `[R,F]` 行主序；scale 为 `[R,F/32]`，Host 要求 F 整除 32。
第 r 行第 c 列地址为 `dataBase+(r*F+c)*D`；MXFP8 对应 scale 地址为
`scaleBase+r*(F/32)+c/32`。这里只保存便于向量解压的逻辑顺序，不套用权重/GEMM 专用 scale 重排。
有效行与 packed X 一致；未使用容量和 padding 未初始化，反向必须按 counts/cumsum 读取有效范围。

掩码按三个独立 `uint32[R,ceil(F/32)]` plane 保存，列 c 对应 word `c/32` 的 bit `c%32`。
`gateHigh=(g>L)`、`upLow=(u<-L)`、`upHigh=(u>L)`，等于边界时为 0。

Host 保持 `empty(totalBytes)` 后仅清零旧前缀。新增数据、scale 和掩码的有效部分由 kernel 覆盖，
不增加整块初始化。C++ stub 同步分配新容量、保留旧 memset 范围。
raw 三输出、compute_train 四输出及 Python autograd 保存接口不变；仍是一个 uint8 saved-state。
它不含独立格式头，解读者必须保留同一次 forward 的 shape、R、save_gmm1_output、
swiglu_limit 及布局版本。现有 ctx 保存模式/limit 和 saved_gate_up_metadata；
version=2/kind=1 表示三掩码格式，其他 version=1/kind=0。不能只通过总字节数猜格式。
新旧 Host/SO/测试不能混用；不要把 FP8 plane 直接 reinterpret 成 BF16。

M4096/K7168/N8192、EP8、每卡 4 专家、R65536：

| 格式 | gate/up 数据 | scale | 完整 saved-state |
|---|---:|---:|---:|
| MXFP8，关闭截断 | 512 MiB | 16 MiB | 1,493,833,728 B |
| BF16，任意支持的 limit | 1 GiB | 0 | 2,013,927,424 B |
| MXFP8，开启截断 | 512 MiB | 16 MiB，另有 96 MiB mask | 1,594,497,024 B |

MXFP8 相对 BF16 每份状态无掩码时节省 496 MiB、有掩码时节省 400 MiB。
其他 saved 字段、SHMEM 和 forward transient 容量不变。
极小 R 时，单独的 scale plane 对齐开销可能超过数据压缩节省，不能假定所有小 shape 都减半。

## UB、同步与性能

当前 tile=128x256。AIV0 原始 BF16 gate/up 共占 128 KiB；新增两块 FP8 tile、scale
写出缓冲和复用的量化临时区后，`SavedGateUpQuantUbLayout::totalBytes=210944`，即 206 KiB。
开启截断追加三个各 4 KiB 的 mask UB，总计 218 KiB。Host 按模式查询并检查实际设备
UB 与 248 KiB 上限的较小值，Device 有静态检查；改变 tile 大小时必须重新布局。

MXFP8 路径先量化两路，保留原 BF16 输入；FP8 数据和 scale 使用独立 UB 缓冲，MTE3
写出可以与随后读取原 BF16 的 SwiGLU 重叠。输出 Hp 后的 MTE3 完成等待保留，
全部保存和输出完成后才发布 ready/NotifyCube，下一 tile 才能复用量化缓冲。
BF16 路径继续使用基准：保存 gate → 保存 up → 等两块完成 → SwiGLU。
`8247167` 基准的保存目标使用默认 L2 策略；后续保存写出 bypass 实验见文末。

量化会增加 AIV0 的归约、转换等工作，可能延长 Cube 等待。判断收益应比较
“融合前向”与“串行 BF16 保存前向＋外部量化”的完整耗时；不能只与未压缩前向 kernel 比。

## 反向负责人接入

1. 根据同一次 forward 的模式、规范化 limit、布局版本选择数据源。none 保留重算；
   BF16 可按原 gate/up 方案读取；显式 MXFP8 在关闭或开启截断时均为压缩格式。
2. MXFP8 使用 `ptrSavedGate/Up` 和 `ptrSavedGateScale/UpScale`，分 tile 解压；
   开启截断还须读 `ptrSavedGateHigh/UpLow/UpHigh`，恢复 FP32 边界值和梯度开关。
   不可从解压近似值重新判断 indicator；具体公式和两轮 loader 见截断掩码方案第 3、6 节。
   避免先在 GM 中展开完整两块 BF16，否则抵消显存收益。
3. 验证后停止 GMM1RecomputePhase，删除其两块完整 scratch，并同步修改 Host/Device
   backward workspace 偏移和分配。当前版本仍保留这些重计算区，不提前删除。
4. 保留 P0 dY gather、计数恢复/Dispatch、UB 释放及 P1 必需的同步；不能直接删除整个 P0。
5. 对比原重计算与解压路径的 gradX、gradTopkWeights、gradWeight1、gradWeight2，覆盖
   各模式、limit=0/正值、多次前向后反向、尾块、空专家。
   mask 保留截断分支，但 H/Hp、TopK/W2 梯度仍可能受压缩误差影响，需训练方验收。

## 验证入口与当前证据

- 原 test_mega_moe.sh 启动方式不变。precision/both 在计时后检查 sampled gate/up 完整
  32 元素组及 scale，屏显 `[SAVED_GATE_UP] ... format=MXFP8_E4M3_E8M0 ... PASS`。
  参考来自 CPU GMM1 和 W8A8 golden；允许一个 FP8 步长及 BF16 GEMM 差异，检查是存储
  抽查而非完整梯度验证。performance 跳过抽查。
- smoke_compute_train.py 覆盖 limit=0/1/10，并验证对应分配字节数、输出有限值。
  默认 smoke 不校验全部保存值或梯度；严格掩码 smoke 的范围见运行态保存模式一节。
- 本地 C++ 布局检查覆盖 EP2/8/512、两种格式、尾容量、scale 空间和旧偏移。
  Python 检查覆盖复用 W8A8 计算体、分配、舍入/饱和、分组、正常样本及破坏检出。
- 实际 Save/Store 方法的延迟 DMA 模拟通过 54 次 tile，覆盖 scale 的 UB/GM 不同
  行步长、边界哨兵、连续复用及 SwiGLU 提前改写原始 gate。模拟量化计算使用替身，
  只证明被模拟的地址和读写依赖，不证明 CANN 向量计算或硬件同步。
  实际 Python 抽查函数也用 CPU Tensor 替身覆盖两种格式、空专家和未初始化尾容量。
- 本地没有 CANN/NPU 环境；设备反馈按下方日期记录，不以 CPU 测试替代设备验证。
  压缩保存值的反向消费和完整训练精度仍待适配、验证。

设备 CANN 9.2 编译反馈：`ShiftRights` 的 scalarValue 必须为 `int16_t`。
提取量化函数时曾把 `SHR_NUM_FOR_BF16` 误声明成 `uint16_t`，现已恢复 W8A8 原来的
`int16_t`；同时将数据打包步长常量恢复为原来的 `int64_t`。补充了常量类型/数值与
W8A8 源码的对照检查。该修正不改变量化规则或内存布局，该历史编译问题已修复；最新设备反馈见文末。

串行 BF16 基准为 `d2977cf`，设备反馈精度通过，31,782 / 31,705 μs。
`62f34a5` 的 up 保存重叠实验未见性能收益；本版 BF16 fallback 已恢复串行基准。

### 2026-09-10：保存值检查报 NaN

设备运行进入 `check_saved_gate_up` 后报 `max_abs_error=nan`，说明此次失败在保存值
校验阶段，不能据此认定最终 y 或保存值精度通过。旧错误信息没有区分原始 FP8 NaN、
解码非有限值及 CPU GMM1 参考异常。

源码对照发现：W8A8 的 `GroupMatmulSwiglu` 在调用量化 epilogue 前设置
`AscendC::SetCtrlSpr<60, 60>(0)`，新保存路径提取量化函数时遗漏了这个调用前提。
根据 [CANN SetCtrlSpr 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_00091.html)，
CTRL[60]=0 时 Cast 的 `SatMode::SAT` 才决定单指令饱和行为；CTRL[60]=1 时由全局
模式控制。比如组最大值 1.875 的 scale 为 1/256，归一化后为 480，超过 E4M3FN
最大有限值 448，需要饱和为 448；不能依赖调用者留下的全局模式。

修正仅作用于 MXFP8 保存路径：记录原 CTRL[60]，切到单指令模式，完成 gate/up
量化，经过 V→S 等待后恢复原模式，再执行写出及后续 BF16 SwiGLU。未改变内存布局、
初始化范围、量化规则或比较阈值；正 limit 的 BF16 保存分支不经过这个设置。
该遗漏已修复，后续设备运行已报告通过；这不代表异常输入全部覆盖。
新增 V→S 等待的性能应以修复后的版本为准。

校验失败现在直接屏显 rank、expert、gate/up、抽样 packed 行和 group 起始列，
以及第一个异常的 `(sample, group, lane)`、FP8 字节、scale、参考值和该组 32 字节。
会分别报告 CPU 参考非有限、E8M0=0xff、FP8=0x7f/0xff、解码溢出及有限值不匹配，
无需上传 bin/json。CPU 检查覆盖饱和边界和这些异常分类；不替代设备验证。

更新代码并重新编译普通 BF16 后，先跑一次保存值和最终输出精度：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 &&
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode precision --warmup 0 --loops 1 --repeat 1 --save-gmm1-output mxfp8
```

通过标准为所有 rank 的 `[SAVED_GATE_UP] ... PASS` 和最终精度成功信息；通过后再
使用原来的性能命令比较耗时。若设备构建此前需要显式 runtime include 等参数，
继续沿用原构建参数。

## 2026-09-10：当前设备反馈与合入边界

代码核对点 `8247167`（前向三掩码实现 `169ceff`，CastTrait 静态存储修复 `8247167`）。
用户反馈“smoke 都过了”，对应此前提供的 BF16/MXFP8 大 shape smoke 和
`SMOKE_CHECK_CLIP_MASKS=1` 的小 shape 掩码校验命令；本次没有重新收集逐 rank 日志。
该反馈支持普通 BF16 前向验证，不扩展为反向消费压缩值、UDMA/MGroup 或 EP512 通过。

普通 8 卡基础 test（raw、关闭截断）耗时反馈如下，单位 μs：

| 保存模式 | 本轮耗时 | 相对同轮 none=31452 |
|---|---:|---:|
| none | 31452；随后复测 31368 | — |
| bf16 | 31949 | +497，约 1.58% |
| mxfp8 | 31858 | +406，约 1.29% |

保留单次结果，不把未配对运行平均后当作稳定基线。none 两次相差 84 μs；当前不足以
认定整体性能回退，也没有证明 MXFP8 稳定快于 BF16。用户决定保持本版、不继续调优。
这些数值不是 MXFP8＋截断掩码的性能；基础 test 未开放正 limit 的精度/性能入口。

仍待后续：反向读取 BF16/MXFP8 保存值、掩码恢复、按模式取消 GMM1 重算与 scratch、
四类梯度和训练验收；正 limit 的大 shape 性能；新版本双机回归；EP512 设备验证。
UDMA 长循环和 MGroup 设备验证保持原待办。默认 BF16 保存不变。


## 保存写出 L2 bypass 实验（待设备验证）

在 `8247167` 已测基准上，仅调整保存目标 GlobalTensor 的缓存提示：

- BF16：`savedGateGlobal_`、`savedUpGlobal_` 使用 `CACHE_MODE_DISABLE`。
- MXFP8：共用 `Store()` 的目标使用相同策略，覆盖 gate/up 数据、两份 scale、
  开启截断时的 gateHigh/upLow/upHigh；所有 rank、expert、tile 一致。
- 普通、UDMA、MGroup 共用保存模块；none 不执行保存 DMA。

不新增 build/run 开关；不改量化、掩码、布局、初始化和等待顺序。Hp、路由输入及
其他非保存目标继续原有策略。当前反向仍重算，不改变其读取路径。
目的是评估较晚才消费的保存数据是否应绕过 L2；尚未确认前向收益，也没有反向消费
阶段的端到端数据。CPU 测试不能证明缓存提示的设备行为。

更新代码后沿用原 build 入口重新构建，按相同 shape/seed/warmup/loops 对比三种模式：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950

set -o pipefail
for save_mode in none bf16 mxfp8; do
  bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
    --mode both --warmup 20 --loops 100 --repeat 1 \
    --save-gmm1-output "$save_mode" \
    2>&1 | tee "bf16_saved_l2_bypass_${save_mode}.log" || break
done
```

基础 test 固定关闭截断。三掩码写出需另外复用第 0 节的严格 smoke 命令
（见截断掩码方案），确认全部有效 mask/data/scale 和 y/counts 通过。
比较 `max_of_rank_medians_us`，同时观察相对同轮 none 的增量；基准为
none 31452/31368、BF16 31949、MXFP8 31858 μs。未实测前不将本实验列为稳定正收益。
