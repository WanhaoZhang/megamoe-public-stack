# MegaMoe BF16 稳定前向优化迁入说明

> 当前保存契约：[MXFP8 saved-state 第一版](MegaMoe_BF16_MXFP8_saved_state第一版.md)。
> 默认 bf16；none 不保存 gate/up；显式 mxfp8＋截断追加三个 bit plane。
> 仅 auto 按 limit 选择格式。用户已反馈 `8247167` 的前向 smoke 通过，反向仍重算。
> 下文历史 BF16 容量/性能结论不能直接用于量化版本，当前分配与适配按上述文档。

日期：2026-09-10。基线：`origin/train@64049c0`。
整理分支：`train-bf16-forward-stable`；来源分支核对点：`deepep_moe_xx@6a9a4a6`。

## 2026-09-11：同步 train 的 PR13 / PR15

本轮将 `origin/train@cc57ed2` 合入 `train-bf16-forward-stable`，保留提交历史。
合并前稳定基准为 `3a9d230`。PR15 使用 `limitEnabled` 的运行态分支；本轮采用该实现，
删除重复的 `Impl<true/false>` 截断模板，不再做两种分支写法的性能 A/B。
保存格式选择用的 `ProcessWithSaveMode<...>` 模板仍保留，它是不同的功能。

保留内容及冲突适配：

- PR13 的 BF16 反向舍入、Unpermute 的 MTE3→V 复用等待、现有训练/测试修正。
- PR13 的 `BackwardSavedStateInfo` 改为复用 `InitSavedState`，按本次模式和 limit
  解析完整 gate/up、scale、mask 布局；保持所有 forward transient 别名为空。
- 保留 `save_gmm1_output` 的 Python/绑定/Host 参数链，包含普通、timer、debug backward；
  默认 bf16，none/mxfp8/auto 按对应 forward 的参数回传，布局元数据仍在 ctx 内。
- 保留 EP512/2048 专家、直接来源偏移、UB 容量检查、Cumsum 修正和分块清零。
- 保留 BF16/MXFP8 保存与截断掩码、有效区覆盖、保存目标 L2 bypass。
- Combine 保留 slice-aware 共享语义：普通前向及反向传 0，MGroup 传 `slice.rowBegin`。
  PR13 的反向修正意图仍成立：packed 的跨专家 `cumMRows_` 不能加到专家内通信坐标。
  不能直接将共享 helper 改为忽略全部偏移，否则 MGroup 的非首片会路由错误。
- CATLASS=`48f7461`、SHMEM=`f4a4dfd`，本轮未改依赖指针。

当前反向仍执行 GMM1 重算，不读取新增 gate/up/mask；没有缩减其实际重算 scratch。
PR13 的反向 SiLU/梯度 BF16 中间舍入必须保留，后续保存值 loader 应在此基线上接入。
前向保存格式和掩码定义不因 PR15 改变。

本地检查：16 项保存模式/掩码检查、24 项 workspace 检查通过；实际共享 C++ 布局
30 组及 UDMA 布局通过；提取当前 `InitSavedState`/`BackwardSavedStateInfo` 的实际代码，
在 EP2/8/512、四种模式和四种 limit 下验证 48 组地址、容量、空 transient、无写入。
这些是 CPU 编译/检查，不是 CANN 编译或设备性能结论。

### 设备回归（统一版本，无需分支写法 A/B）

重新运行原构建入口 `bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950`。
然后复用保存文档中的 none/bf16/mxfp8 基础 `--mode both` 和 training smoke 命令，
严格掩码 smoke 仍用 `SMOKE_SAVE_GMM1_OUTPUT=mxfp8 SMOKE_CHECK_CLIP_MASKS=1`。
本轮没有改基础测试启动脚本；BigOp 测试脚本的变化来自上游 PR13。

反向完整回归可使用 PR13 的单机 8 卡入口：

```bash
set -o pipefail
env -u NODE_RANK -u NNODES -u MASTER_IP \
  ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  ASCEND_DEEPEP_SHMEM_IP_PORT=tcp://127.0.0.1:8899 \
  PRECISION_NPROC=8 \
  bash test/bigop_fused/run_precision.sh both train \
  2>&1 | tee bf16_pr13_pr15_train_precision.log
```

注意 PR13 测试将精度 `Failed`/`WARNING` 视为诊断结果，可能返回退出码 0；不能只看
shell 是否退出成功。必须检查 `[fast summary]` 和各梯度结果，`Failed` 不能记录为通过。
该入口默认 bf16 保存且 limit=0；不代替正 limit smoke 或将来的压缩消费反向验收。

用户已报告合并前 `3a9d230` 精度通过，bf16=31217/31591/31546、mxfp8=31480/31604、
none=31385 μs，观察到保存 bypass 收益。这个结果不能直接沿用为本次合并版结果。
合并版待设备回归后再进入 train。EP512、UDMA 长循环、MGroup 和新版本双机仍待验证。

## 当前集成状态：已合入 train 的 PR12

在 `train-bf16-forward-stable@13d3736` 上 merge `origin/train@cc50455`，保留双方历史。
CATLASS 固定为 PR12 的 `48f7461afcff7c9282c6372c19d8768d9c6e0d9d`（v1.6.1），
接受对方 CANN 9.1 的兼容选择；SHMEM 保持 `f4a4dfdb6bc47a9ba4aded2b20c38638a3ddb44c`。
PR12 集成之后又完成了下文反向 EP512 适配和前向 gate/up 持久化。
当前 raw/compute_train 返回项不变；测试启动命令不变，py 的布局校验和保存值抽查已同步更新。
前向 gate/up 保存已收到设备反馈，当前 bf16 31949、mxfp8 31858、none 31452/31368 us；
均为普通 8 卡关闭截断的基础 test，不能作为带掩码性能或反向消费验证。
详见 [gate/up 持久化与反向接入](MegaMoe_BF16_gate_up持久化与反向接入.md)。

公共函数末尾参数统一为：

```cpp
bool bf16, double swiglu_limit, bool bf16_direct_dispatch,
const std::string &save_gmm1_output
```

前三个参数均无默认值，所有 6 个调用点显式传入，避免把 bool 转为截断值。
保存模式是独立的第四个尾部参数：BF16 路径传入本次调用模式，原量化入口使用 auto。
下表列出前三个参数：

| 调用入口 | bf16 | swiglu_limit | bf16_direct_dispatch |
|---|---|---|---|
| 普通 raw compute / compute_with_timer | true | 0.0 | true |
| 普通 compute_train 的独立 Host 前向 | true | swigluLimit | true |
| backward Host | true | swigluLimit | true |
| MGroup | true | 0.0 | false |
| UDMA | true | 0.0 | false |
| 原 MegaMoe 量化入口 | false | 0.0 | false |

普通 raw 和 training 前向都接受 EP512/4 专家，并校验总专家数 <=2048、TopK <=总专家数、
Dispatch 同期 UB <=查询到的实际 UB 容量。两者都调用相同前向 Device，保留 L2 bypass、
直接来源偏移、Cumsum 紧凑 UB、16 KiB 分块清零和原有同步。
raw 返回 `(y, counts, saved_state)`；当前 compute_train 返回
`(y, counts, saved_state, dispatch_output, transient)`，需显式传入
`swiglu_limit`，末尾保存模式默认 bf16。Dispatch 输出独立拆分说明见
[可选持久化设计](MegaMoe_BF16_Dispatch输出可选持久化.md)。
后续反向也已接入 2048 专家及直接来源偏移，见文末补充；EP512 前后向仍未设备验证。

本地检查：24 项现有 workspace 检查通过；MSVC C++17 编译实际共享 UB 布局及从
6 个调用点提取的参数表达式，结合实际 Host 的资源检查和 swigluLimit 赋值逻辑，
验证关闭截断得到 FLT_MAX（不是 1.0）、非零 limit=7 得到 7、EP512 前向启用直接偏移，
旧路径仍拒绝 EP512 核数超限，超 UB 配置仍拒绝。该 CPU 探测不替代 CANN/NPU 编译。

后续反向接口修正：

- 删除 Python backward 的 dummy transient 分配，并从绑定、Host、launcher 和 Device 入口删除该参数。反向只初始化 saved-state 地址，不计算前向临时区地址；反向算法、scratch 和同步保持原样。
- 补齐 backward timer 的 C++ 绑定参数 `swiglu_limit` 并显式传给 Host。Python 原先已经传入此参数，旧绑定签名不匹配；普通、debug、timer 现在使用一致的参数链。
- 关闭截断时前后向的 Min/Max 已在后续优化中跳过，反向同时省去截断梯度掩码，见下节。

三个反向绑定的输入统一为：

```python
op.backward(shmem, grad_y, topk_weights, w1, w2, counts,
            training_saved_state, dispatch_output,
            max_recv_tokens, swiglu_limit, save_gmm1_output)
# backward_debug / backward_with_timer 使用相同的输入顺序。
```

普通 autograd 不持有前向 transient；根据 `retain_dispatch_output` 决定是否持有
独立 Dispatch 输出。显式使用 `diagnostic_holder` 时仍会保留两者，供诊断读取。
**反向 kernel ABI 已变化，必须一起重新编译安装 Host 扩展和 BF16 SO，不可混用旧 SO。**

本地执行实际 Python autograd 方法，验证三个分支在 limit=0/7 下不分配 dummy、参数顺序正确；
MSVC C++17 探测实际三个绑定方法，验证 limit 传到 Host，且 timer Tensor 保留。
新旧前向 workspace 地址/大小对照一致，反向 saved-state 视图的临时区指针为空；
24 项现有 workspace 检查通过。这些检查不替代设备编译和反向梯度回归。

**31309 us、8 卡精度通过是 `13d3736` 与旧 CATLASS 的历史结果。**
不能将这份旧依赖结果沿用到 PR12 集成版。EP512 仍标记待设备验证。

### 关闭截断时跳过前向 Min/Max

用户在 `b438d61` 推送后反馈：第一项普通 BF16 精度通过，耗时 **31310 us**；
第二项 compute_train smoke 均通过。该 smoke 仅检查前向输出有限值，不代表反向梯度验证。
这轮结果作为下面优化之前的对照，后续测试应保持相同 shape、循环、timer 和软件环境。

普通 BF16、UDMA、MGroup 前向共用 `block_epilogue_swiglu_bf16.h`。
`ComputeSwiglu()` 在每个 tile 的向量循环外判断现有 `kMegaMoeSwigluLimitDisabled`
标记（FLT_MAX），当时选择 `ComputeSwigluImpl<false/true>`；9 月 11 日已统一采用 PR15
运行态条件分支。Host 对 `swiglu_limit <= 0`
的映射保持原样；关闭路径通过 `if constexpr` 去掉 gate/up 的 Min/Max，
以及只为截断准备的 Duplicate 和两个 Muls，不在逐元素循环中增加运行时判断。
开启截断时保留 `gate=min(gate,L)`、`up=clamp(up,-L,L)` 的原计算顺序。
SwiGLU、BF16 舍入、TopK 乘权、写出与同步不变，无新增接口或编译选项。

本地核对两种展开后的运算顺序：开启路径与原实现一致，关闭路径仅删除截断及其准备操作；
枚举 65280 个有限 BF16 值，确认 FLT_MAX 截断对这些值均为恒等变换。
NaN/Inf 不在等价检查范围，关闭路径现在直接使用原值，不保证与原先夹到有限阈值的行为一致。
设备编译、精度和性能收益仍待验证，不能预先认定更快。
按原有命令重新编译后，普通 BF16 精度及耗时与 **31310 us** 对照，
并重复 compute_train 的 limit=0/10 smoke；开启截断的完整精度仍需独立参考校验。

用户随后要求反向采用相同判断。`block_epilogue_swiglu_backward_bf16.h` 的
`ComputeForwardAux`、`ComputeSwigluRecompute`、`ComputeSwigluGrad` 均在向量循环外
当时使用同一关闭标记选择模板路径，9 月 11 日改为 PR15 运行态条件分支。前两者跳过 H 重算的截断及阈值准备；梯度路径直接使用
原始 gate/up，跳过截断、gateInd/upInd 的生成，以及 dGate/dUp 对指示值的乘法。
sigmoid 本身仍需常数 1，不能把所有 Duplicate 一并删除。
开启截断时保留全部原操作及其顺序，包括阈值边界的原有梯度行为。
反向 GMM、通信、TopK 权重处理、梯度写出、UB 布局和同步保持原样。

本地检查三个开启路径的运算顺序与原代码一致，关闭路径无截断/掩码操作；
从实际梯度运算链提取的 4147 组标量 FP32 探测通过，覆盖关闭标记、limit=1/7/10、
BF16 输入及阈值内/等于/外的取值，并在非边界点对照解析梯度。
该 CPU 检查不覆盖 CANN 编译、向量 mask 或 NPU 完整梯度。
设备需在 limit=0 和开启截断的相同输入下验证四类梯度；前向 smoke 不替代这项检查。

### CANN 9.2 编译时规避未使用的 W4A8 头文件

2026-09-10 用户设备使用 CANN 9.2.0 编译普通 BF16 时，CATLASS `48f7461` 的
`block_mmad.hpp` 连带包含 `block_mx_a8w4_prologue.hpp`，其中 `RegCompute` 和
`RegComputeNkNz` 报错：`simd_vf function must be a free function or static member function`。
这是未使用的 W4A8 声明在头文件解析阶段触发的错误，并不表示 BF16 执行了 W4A8 运算。

沿用 PR12 反向已有方式，在 BF16 包含 `block_mmad.hpp` 前临时定义以下两个头文件保护宏，
跳过 W4A8 prologue 及依赖它的 MMAD specialization，包含完成后立即 undef：

```cpp
#define CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
#define CATLASS_GEMM_BLOCK_MMAD_A8W4_MX_HPP
#include "catlass/gemm/block/block_mmad.hpp"
#undef CATLASS_GEMM_BLOCK_MMAD_A8W4_MX_HPP
#undef CATLASS_GEMM_BLOCK_MX_A8W4_PROLOGUE_HPP
```

本次补齐普通 BF16 前向和 UDMA 前向；MGroup 复用普通 BF16 实现，反向原有规避保留。
不新增编译选项，不改 W4A8 目标或第三方源文件；CATLASS 保持 `48f7461`，SHMEM 保持 `f4a4dfd`。
这是 BF16 的头文件包含规避，不是 W4A8 的兼容修复。后续更换 CATLASS 时需重新核对保护宏，
如要在同一编译单元使用 W4A8，需要重新处理该规避。
本地已核对指定 CATLASS 提交的包含关系及保护宏；用户随后反馈普通 BF16 精度通过、31310 us，
compute_train smoke 通过。UDMA/MGroup 不能据此认定已经设备验证。
仍使用下面原有 BF16 构建命令，无需额外开关。

### 设备重新编译和回归

在稳定分支更新代码并同步指定的子模块，使用独立 BF16 build 目录：

```bash
git pull --ff-only &&
git submodule update --init --recursive 3rdparty/catlass 3rdparty/shmem &&
git submodule status 3rdparty/catlass 3rdparty/shmem
```

确认 CATLASS 为 `48f7461`、SHMEM 为 `f4a4dfd`，保持当前使用的 CANN/PyTorch 环境。
下面命令同时重建普通 BF16 SO，并通过构建脚本重建安装 Python Host 扩展：

```bash
set -o pipefail
MEGA_MOE_BF16_BUILD_DIR="$PWD/build_mega_moe_bf16_pr12" \
  bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  2>&1 | tee bf16_pr12_build.log
```

仅在构建成功后测试。现有 py 保持单机 SERVER_NUM=1、SERVER_INDEX=0，
CSV 使用 M4096/N8192/K7168；测试入口不作修改：

```bash
set -o pipefail
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode both --warmup 20 --loops 100 --repeat 1 \
  2>&1 | tee bf16_pr12_raw.log
```

检查所有 rank 的精度和最终 SUCCESS，并保存新的 max_of_rank_medians_us。
该命令测 raw 前向，不能替代 training 独立 Host 路径的设备检查。
PR12 已有 forward-only smoke，可原样运行 8 卡，覆盖 compute_train 的 limit=0 和 10：

```bash
set -o pipefail
SMOKE_EP=8 SMOKE_DEVICES=0,1,2,3,4,5,6,7 \
  python test/megamoe_bf16_test/smoke_compute_train.py \
  2>&1 | tee bf16_pr12_train_smoke.log
```

应看到每个 rank 两种 limit 的输出及最终 `[SMOKE] compute_train forward passed`。
这份原有 smoke 只检查输出有限值和入口执行，不做 golden 精度或性能比较；不能用它
宣称非零截断精度已通过。整网接入后应进一步用同一输入对照带截断的 PyTorch 前向参考。
上面的前向回归不覆盖反向接口修正。反向需另外在相同输入及 limit 下比较普通/debug/timer
的四类梯度，并覆盖关闭截断与实际触发截断的输入；timer 分支需启用 timer 构建。
EP512 仍未设备验证。
收到用户对当前环境回归的确认后，才将稳定分支合回 train。

## 本批新增迁入

只迁入普通 BF16 **Dispatch 写 packed X 时绕过 L2**，提取自 `2ec55aa` 的生产代码，
采用 `813bb32` 中已经使用的 `bypass`。完成本分支设备对照后固定采用该策略，
不再提供缓存策略编译选项、宏或条件分支。
没有整体 cherry-pick 上述提交：它们同时含有隔离测试和调测工具。

实现只给 Dispatch 的目标 `GlobalTensor` 设置 `CACHE_MODE_DISABLE`；
源数据读取、GMM1 自己创建的输入 Tensor 及其读取策略保持 train 原样。
普通 BF16 调用处创建目标 Tensor 并固定设置提示，再传入共享 Dispatch 函数。
MGroup 调用处传入采用原默认策略的 Tensor；UDMA 保持原实现。
普通 BF16 的 expert/256 两种 Dispatch 分工均使用同一个目标 Tensor 提示。
保存布局、返回接口、通信内容、同步和 GMM 调度保持 train 基线。

来源记录（xx 分支性能跟进文档第 21 节）为同一大 shape：
`M=4096, N=8192, K=7168, EP=8, TopK=16`。

| Dispatch source / destination | 完整 BF16 耗时（us） |
|---|---:|
| normal / normal | 31906 |
| normal / bypass | 31614、31508 |
| bypass / bypass | 31624、31518 |

目标单独 bypass 的两次耗时较 normal 低约 0.92% 和 1.25%。其中第一份完整日志
包含输出精度和保存权重通过；后续反馈只有整体耗时，且不是随机交错配对实验。
这些记录支持提取该优化，但不表示本次干净分支、所有 shape 或双机均已完成回归。
源 bypass 没有额外稳定收益证据，因此不迁入；该优化也没有消除已记录的慢组现象。

## train 已包含，不重复迁入

- InitRouting 的 packed TopK 权重批量重排及容量不足时的回退。
- Unpermute 索引批量读取、输入/输出搬运流水。
- 默认整 expert 就绪策略，以及 expert 模式不发布逐行 ready 计数。
- 三输出 saved-state、packed TopK 权重持久化和普通 BF16 双机测试。

## 后续单独处理

- 当时迁入批次没有保存 gate/up；后续新增保存区见本文末尾及专项说明。
- 反向算法适配由反向同事负责；本次集成保留 PR12 已恢复的入口和现有约束。
- 初次稳定优化迁入未包含 PR12；当前已 merge PR12，并按文首记录固定依赖。
- 不引入 GMM1 B-cache 实验、SwiGLU 批量权重加载、task swap、人工延迟、
  expert0 屏障、隔离 kernel、指令采集和慢组诊断工具。

## 本分支设备验证结果

2026-09-10 用户按前述两组命令反馈，shape 为 `M=4096, N=8192, K=7168, EP=8`。
对照提交为 `10d0883`；两组均使用 `--mode both --warmup 20 --loops 100 --repeat 1`。

| Dispatch 目标策略 | max_of_rank_medians_us | median_of_rank_medians_us |
|---|---:|---:|
| normal | 31807.529 | 31800.592 |
| bypass | 31226.272 | 31221.825 |

最慢 rank 中位耗时降低 581.257 us，约 1.83%。两组 8 个 rank 的 y 和
expert_token_nums 均 PASS，所贴日志的逐 rank 误差统计一致，最终均出现 SUCCESS。
这是当前 shape 的一组设备对照，不代表其他 shape 或双机均已回归。
据此固定使用 bypass，移除本批引入的缓存策略构建选项和宏。

用户随后反馈选项清理提交 `6dd9fd9` 的耗时为 31338 us，认为可以接受。
按上一轮相同指标比较，比 bypass 对照高约 0.36%，比 normal 对照低约 1.48%。
该次反馈未附完整精度日志，不能据此补记新的精度检查结果。

## EP512、每卡 4 专家的普通 BF16 前向适配

本批目标是均衡路由下 EP512、每卡 4 专家（总计 2048 专家），经现有 PyTorch
`MegaMoeBF16.compute()` 接入 Host/Device。测试脚本保持原样；UDMA、MGroup 的 Host
支持范围和反向实现保持原样。MGroup 复用普通 BF16 的三个 Device 函数，获得相同修正。

已完成的修改：

1. Cumsum UB 按 `EP * AlignUp(expertPerRank * 4, 32)` 分配。
   GM 计数行仍按 `AlignUp(EP * expertPerRank, 128)` 个 int32 存放；这是 512 字节对齐，
   与计数发送、512 字节分段等待匹配。DataCopyPad 只取本卡专家列，UB 每来源行对齐 32 字节。
2. Dispatch 直接读 `preSumBeforeRankWorkSpace_[source * expertPerRank + expert]`，
   删除 `prevSum_[10]` 及跨专家累加，避免固定来源槽位限制。
3. ResetTokenPerExpert 复用最多 16 KiB 零缓冲分块写回整个 GM 计数矩阵，
   完成 MTE3 后再进入原有全核同步，随后 Unpermute 才复用 UB。
4. 普通前向总专家数检查提高到 2048，并校验 TopK 不超过总专家数。
   普通前向调用 Host tiling 时启用直接偏移路径，取消其 `EP <= AIC数 * 10` 限制；
   后续 backward 已同步取消这一固定缓存限制；UDMA 和其他旧调用仍执行原限制。
   没有新增外部构建开关。
5. Host 和 Device 共用 `ForwardDispatchUbLayout` 计算同期 buffer 大小，Host
   在生成 tiling 前检查总量是否超过实际查询到的 UB 容量。布局不改变 saved-state 接口。

| EP | 每卡专家数 | 原 Cumsum UB | 修正后 |
|---|---:|---:|---:|
| 8 | 4 | 4 KiB | 256 B |
| 256 | 4 | 1 MiB | 8 KiB |
| 512 | 2 | 2 MiB | 16 KiB |
| 512 | 4 | 4 MiB | 16 KiB |

以 `M4096, N8192, K7168, TopK16, EP512, expertPerRank4`、严格均衡为例：

- 每卡接收 65536 行，每个专家 16384 行，每个来源到目标 rank 128 行。
- `max_recv_tokens=65536` 是每 rank 的接收容量，不再乘 EP。
- FirstBuffInit 的同期 UB 总量 118816 字节（约 116.03 KiB），小于 Ascend950 的 248 KiB。
- 每 rank 完整 GM 计数矩阵为 4 MiB；Cumsum UB 为 16 KiB，清零 UB 最多 16 KiB。
- 此例旧 saved-state 为 944391168 字节；追加 gate/up 后为 2018132992 字节。SHMEM 布局仍需 1883435008 字节，2 GiB 分配满足此例。
  改变 M/K/TopK 时应重新计算，并以 Host 的容量检查为准。

PyTorch/Host 接口和返回值不变：

```python
y, expert_token_nums, training_saved_state = op.compute(
    shmem, x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens)
```

`shmem` 的 WorldSize 是 512、Rank 是全局 rank 0..511；每卡 weight1 为 `[4,N,K]`，
weight2 为 `[4,K,N/2]`，topk_ids 为 int32 且有效值在 `[0,2048)`。
所有 rank 使用同一组 shape、专家划分及足够的 SHMEM 容量，生命周期覆盖异步前向执行。
`compute_with_timer()` 走 raw Host 路径；PR12 已恢复 `compute_train()` 和训练 wrapper，
本次已将同样的 EP512 检查接入其独立 Host 前向；反向后续适配见下节。

本地验证：

- MSVC C++17 编译实际共享布局头，并用从当前 Device 源码提取的清零函数及来源偏移语句
  进行 CPU 模拟，60 组配置通过（包含 EP512、首尾 rank、空专家、清零边界和非清零核）。
- EP512/4 专家的 UB 和 saved-state 大小由共享 C++ 布局计算验证；从 Host 源码提取的实际检查
  在 28/32 AIC 下允许该普通前向配置，保留旧路径拒绝行为，并拒绝超 UB 的 K 配置。
- 前一轮 160 组 GM 跨距搬运/UB 对齐/前缀和 CPU 模拟通过；本轮 24 项现有 workspace 检查通过。
- 已核对 TorchBind、launcher 的 1024 rank 地址表、2048 专家计数的动态 buffer，
  以及 GM/saved-state 的地址公式。InitRouting 的每专家计数 buffer 按实际专家数计算；
  公共 tiling 头里未使用的旧 MAX_MOE_EXPERT_NUM 声明不作为普通 BF16 的入口限制。

结论：已移除此次确认的 EP512/4 专家软件阻塞，可以进入 CANN 编译和设备验证；
本地没有 CANN/NPU，尚无 Device 编译、512 卡精度、通信可达性、长循环或性能通过证据。
小 EP 原性能也需回归，因为 Dispatch 现在多读每个专家的来源偏移。

## EP512／每卡 4 专家的普通 BF16 反向适配

本轮扩展已有反向入口，不新增 gate/up 保存或改变反向梯度公式：

- `DispatchTokensBackward` 删除 `prevSum_[10]`，直接读取前向保存在
  `preSumBeforeRankWorkSpace_[source * expertPerRank + expert]` 的来源行偏移。
  EP512 在 32/28 个 AIC 上分别会分配最多 16/19 个来源槽，不再受固定数组长度约束。
- `ResetTokenPerExpert` 不再将整个 GM 计数矩阵放入 UB。复用前向的 16 KiB 清零块大小，
  分块写回，并等待 MTE3 完成后进入原有全核同步；P6 的 pipe 已在此之前释放。
- `RestoreTokenPerExpert` 仍然逐来源 rank 搬运计数行，但由单 AIV 串行恢复改为所有 AIV
  按 `sourceRank = aivCoreIdx + n * blockAivNum` 分工。各 AIV 写入的物理行互不重叠，
  恢复后的原有 `SyncAll` 保证反向 Dispatch 只能读取完整表。EP512/4 专家时整表为
  4 MiB，但每行只有 8 KiB，每个 AIV 只需要一行 UB。
- Host 总专家数上限提高到 2048，增加 TopK 不超过总专家数检查。
  普通 backward 传入 `bf16_direct_dispatch=true`，取消 `EP <= AIC数 * 10` 的缓存限制；
  `swigluLimit` 仍在独立参数位置显式传入，UDMA/其他旧路径不随之放开。
- Host 继续校验匹配前向的 UB 条件，并单独校验反向阶段峰值：P0 的恢复/搬运/gather、
  SwiGLU 三块 UB tile、Unpermute、P6 TopK partial 与路由缓冲、退出时清零块。
  预算取实际查询到的 UB 与既有 248 KiB 限额的较小者。

M4096/K7168/N8192、EP512/4 专家、dispatchRows=2 时，各阶段为：

| 阶段 | UB 大小 |
|---|---:|
| P0 恢复计数行 + ping-pong 搬运 + gather | 94,208 字节（92 KiB） |
| SwiGLU 三块 tile | 196,608 字节（192 KiB） |
| Unpermute | 129,024 字节（126 KiB） |
| P6 TopK partial + route | 133,120 字节（130 KiB） |
| 退出前计数表清零块 | 最多 16,384 字节（16 KiB） |

这些阶段不同时持有上述缓冲，峰值为 192 KiB。saved-state、反向 GM scratch 布局、
GMM1 重计算、通信拓扑和各阶段同步协议保持原样。

本地 MSVC C++17 探测使用实际 Host 检查、调用表达式和 Device 清零/来源偏移代码：
EP8/256/512、每卡 4 专家、28/32 AIC、limit=0/7 均按预期通过入口检查；
超专家数、超 TopK、超 UB 和低于所需 UB 的设备配置被拒绝。
60 组 shape 的计数清零边界及来源地址检查通过，包含 EP512、空计数、不均衡计数和非末核不清零。
地址参考来自独立生成的来源专家排序路由，不代表实际跨机通信已验证。
**EP512 前后向均仍标记待设备编译/验证，不声明 512 卡训练已经跑通。**
本轮不修改测试启动脚本。先回归已有 8 卡完整 forward→backward 的四类梯度和连续调用；
后续有大 EP 环境时，由整网入口验证 EP256/512 的精度、通信和连续训练。

### 普通 BF16 扩展到 EP1024、每卡 32 专家

普通 raw 前向、training 前向和反向统一采用以下组合约束：EP rank 数最多 1024、
每卡专家数最多 32、总专家数最多 2048。因此 `EP64 × 32 experts/rank` 和
`EP1024 × 2 experts/rank` 均可进入 Host；`EP1024 × 3`、`EP65 × 32` 和
`EP1025 × 2` 会在 Host 拒绝。UDMA、MGroup 继续保留原支持范围。

这项适配不只是放宽 Host 检查，还修正了两个实际容量边界：

- `PrevSumBuf_` 只保存本卡每个专家的一个 int32 前缀值。其 UB 从原来的
  `syncBytes × expertPerRank` 改为 `AlignUp(expertPerRank × 4, 32)`；64×32 时
  从 256 KiB 降到 128 B。写入循环和 `DataCopyPad` 的有效字节数保持不变。
- 普通 BF16 的 SHMEM 同步头部统一使用 64 KiB rank 槽和 64..72 KiB AIV
  generation counter，payload 固定从 72 KiB 开始。Host 分配、精确清零、前向和
  反向共用同一组偏移函数；MGroup、UDMA 继续使用各自的旧布局。

源码布局和 Host 边界检查已覆盖上述两个端点；1024 rank 的 ACLSHMEM/HCCL 建链、
通信可达性、连续训练和性能仍须在大 EP 设备环境验证，不能由本地布局测试代替。

### gate/up 保存已实现，反向读取切换待负责人接入

普通 BF16/raw/compute_train、MGroup、UDMA 已直接从 GMM1 UB 保存原始 BF16 gate/up，
追加在 saved-state 末尾，旧字段偏移和 progress 清零位置保持原样。
Host 分配扩大但只清零 legacyBytes，新 plane 由有效 tile 覆盖；典型 R65536/N8192 每卡增加 1 GiB。
当前反向仍使用原 GMM1 重计算和 scratch，不消费新 plane；不提前删整个 P0 或重计算缓冲。
布局、初始化范围、MTE3→V 等待、内存增量、负责人切换步骤和验证边界见
[gate/up 持久化与反向接入](MegaMoe_BF16_gate_up持久化与反向接入.md)。

## 构建与后续验证

使用原构建和测试入口，无需指定缓存策略：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 &&
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode both --warmup 20 --loops 100 --repeat 1
```

该命令不编译额外阶段 Timer，`both` 保留精度和 profiler 性能采集。
双机仍按现有 py 顶部配置，每机执行同一组命令，两台均出现全局成功才算通过。

选项清理后的本地检查包括 Bash 语法、残留选项扫描和普通 BF16/MGroup 调用链核对。
本地无 CANN/NPU；普通 8 卡 gate/up 前向的设备反馈见最新保存文档，不能扩大为
EP512、UDMA/MGroup、新版双机或压缩消费反向的验证。EP512 仍待设备验证。
精度模式在计时结束后抽查保存值；正 limit 大 shape 性能与反向读取适配仍为后续。
