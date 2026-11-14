# MegaMoe BF16 前向性能：实验结论与遗留问题

更新日期：2026-09-05。供下周继续讨论和实施。

范围：普通 `MegaMoeBF16` 前向及共享权重重排在 MGroup/UDMA 的接入。UDMA 的其他独立改动仅记录状态，不把它的收益、故障或 buffer 生命周期套用到普通 BF16。

本次修改前的普通 BF16 提前启动提交为 `db6c761`；整理时分支 HEAD 已到 `268af98`（另一个任务的 UDMA transient workspace 压缩）。本文记录用户实测、源码分析和本地验证，三者分别说明。

## 1. 当前决定与使用方法

默认恢复为：**当前 expert 的 Dispatch 全部就绪后，GMM1 才开始计算该 expert。** 这仍是 expert 间流水，不要求所有 expert 都到齐。

保留 256 行就绪实验，通过普通 BF16 的构建参数选择：

```bash
# 默认：等整个 expert 就绪。也用于从实验模式切回默认。
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950

# 实验：一个 256 行 ready block 就绪，即可计算其中的 128 行 GMM1 tile。
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --gmm1-ready-mode 256

# 显式指定默认模式；如需正常分段计时，两种模式都加相同的 --enable-timer。
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --gmm1-ready-mode expert --enable-timer
```

构建输出会显示 `[INFO] GMM1 readiness mode: expert/256 (build-time selection)`。

- 脚本每次都向 CMake 传入模式，默认命令会覆盖上次构建目录中的 `256` 缓存。
- 直接使用 CMake 时，设置 `-DASCEND_DEEPEP_BF16_GMM1_READY_MODE=expert` 或 `256`；CMake 本身会保留已有 cache，切换时应显式传值。
- 这是构建选择，运行测试时不需要新参数；已经构建的 SO 不会因运行命令改变模式。重新构建后启动新的测试进程加载 SO。
- 初版对照保留两种模式下的 256 行计数与发布，只切换 GMM1 等待策略。第 11 节继续清理默认 expert 模式：不再发布行计数，只在各 peer/core 分工的数据和权重可见后发布 expert 完成；显式 256 模式保留分段协议。128 行 GMM1 tile 与 `<3,1>` swizzle 沿用原配置。
- `256` 是就绪/发布粒度，不是一次底层搬运 256 行。当前 `CopyGMToGMPerTokenV2` 的双缓冲内部批次来自 `dispatchRows_`，默认 host tiling 传入 2 行。
- 临时 GMM1 串行等待诊断已删除，原诊断构建参数不再接受；普通 `--enable-timer` 保留。历史诊断结果不能代表普通计时性能。
- 整 expert 默认值与诊断清理完成后，继续实现了 InitRouting 权重重排与 Unpermute 搬运流水，见第 9、10 节。Swizzle 沿用原配置，运行期 ABI 不变。

源码入口：[构建脚本](../scripts/build_mega_moe_bf16.sh)、[CMake](../CMakeLists.txt)、[Process / Dispatch / UnpermuteV2](../kernels/mega_moe_bf16/include/mega_moe_bf16.h)、[GMM1 等待与 scheduler](../kernels/mega_moe_bf16/include/mega_moe_bf16_catlass_impl.h)。

## 2. 已有实测：不再重复无依据的参数试探

shape：每 rank M=4096，hidden K=7168，CSV N=8192（SwiGLU 后宽度 4096），TopK=16，EP=8，每 rank 4 个 expert。本节 swizzle 分析按 32 AIC / 64 AIV、均衡路由计算。

| 环境 | 提交/配置 | 用户报告端到端 us | 结论 |
| --- | --- | ---: | --- |
| 旧环境，已不可用 | `9627e93`，细粒度 Dispatch 之前 | 31842 | 历史基线，不与新机器直接归因比较 |
| 旧环境 | `52cd5bf`，128 行就绪 | 32908 | 提前启动回退 |
| 旧环境 | `4441802`，expert 完成后绕过逐 tile 查询 | 32923 | 单独减少查询未解决回退 |
| 旧环境 | `3da4809`，256 行就绪 | 33012 | 单独扩大 ready 粒度未解决回退 |
| 旧环境 | `418e084`，保留分段但等整个 expert | 31943 | 主要回退随等待策略恢复而消失 |
| 新环境 | 整 expert 等待、普通 Timer | 31940.519 | `max_of_rank_medians_us`，当前比较基准 |
| 新环境 | 整 expert 等待、串行 wait diagnostics | 34818.241 | 诊断改变流水，不能当正常基线 |
| 新环境 | `db6c761` 恢复 256 行提前启动 | 32954 | 比新环境基准慢 1013.481 us，约 3.17% |

新环境那组完整结果：min/median/max of rank medians 为 31907.575 / 31932.809 / 31940.519 us；min of rank mins 为 30932.457 us。不要拿最小单次值代替稳定中位数。

目前足以停用这版提前启动默认值，但不足以把原因定为“某个固定 group 坏了”“L2 冲突”或“HBM 争用”。128/256、expert-ready latch 都没有产生收益，不继续围绕阈值盲试。

新环境正常计时下，rank5/group24 的 GMM1 总和 19.054 ms，参考 group 为 18.973 ms，仅差 0.081 ms。用户确认当前样本没有异常慢 rank/group，不再专门追查 group24。

理论文档的 GMM1/GMM2 为约 18.754/9.377 ms，计算和约 28.131 ms；正常 GMM1 约 18.9 ms 已接近该模型。28.131 ms 不含完整 InitRouting、SwiGLU、状态保存、同步、Unpermute，不能作为本实现可承诺的端到端时间。

理论来源：[训练 Shape 多机 EP 分析](../../../moe/learning_materials/megamoe_a5/costmodel/website_upload/megamoe多机EP性能分析/megamoe训练新Shape多机EP通信计算掩盖分析.md)。其中多机拓扑、通信模型与当前 EP8 实机也需分别对待。

## 3. Swizzle `<3,1>` / `<3,0>` 会不会影响提前启动？

**会，而且是供数顺序和访存复用的联合问题；不能仅按更早启动判断总耗时。**

使用的是 [CATLASS GemmIdentityBlockSwizzle](../3rdparty/catlass/include/catlass/gemm/block/block_swizzle.hpp)。`3` 是 tile 条带宽度，不是 3 个 group；方向参数改变 M/N tile 的遍历顺序。

均衡 shape 下，每 expert 有 16384 行；GMM1 调度网格为 128 个 M tile × 16 个 N tile，共 2048 个任务。32 AIC 下每核 64 个任务，完整 expert 后 startBlockIdx 不偏移。坐标 `(m,n)` 是 tile 编号：

| 配置 | 前六个任务 | 前 32 个任务覆盖的 M tile | 前 32 个任务覆盖的 N tile |
| --- | --- | --- | --- |
| `<3,1>`，当前值 | (0,0), (0,1), (0,2), (1,0), (1,1), (1,2) | 0…10，即输入行 0…1407 | 0…2 |
| `<3,0>` | (0,0), (1,0), (2,0), (0,1), (1,1), (2,1) | 0…2，即输入行 0…383 | 0…10 |

这两个方向还在交替条带上反向遍历，不能只用首批任务推断完整 expert 行为。

Dispatch 按 source rank 划连续范围：每 source 对本 expert 提供 2048 行，4 个搬运 owner 各负责 512 行。这些 owner 并行工作，不是 rank0 搬完才轮到 rank1。若取“每个 owner 恰好发布了自己前 256 行”的静态快照：

- 已就绪的 M tile 是 0、1、4、5、8、9……，中间有尚未就绪的区间。
- `<3,1>` 首 32 个任务中 18 个可开始；`<3,0>` 为 22 个。
- `<3,0>` 把输入 A 的需求集中到少数行，但同时把 B 的需求从 3 个 N tile 扩到 11 个 N tile，改变权重复用与缓存压力。

这些坐标和计数已用仓内真实 swizzle 头文件进行 host 枚举验证，两个方向均覆盖完整 2048 个任务且无重复。上述 ready 快照是假设，不是硬件上实测的同时到达状态，也不代表利用率或收益预测。

当前的 `Gmm2BlockScheduler = BlockScheduler`，GMM2 内还构造 GMM1 scheduler 来计算 SwiGLU ready 目标。未来如验证 GMM1 swizzle，必须把 GMM1/GMM2 的选择分开，并保持用于 ready 目标的 GMM1 scheduler 一致；直接替换公共别名会同时改变 GMM2，无法判断回退来自哪里。

遗留项 D-01：先完成生产者发布顺序、GMM1 全任务遍历、A/B 复用的联合分析，再决定是否做一个限定 GMM1 的候选版本。本次不添加 swizzle 开关，不要求用户立刻测试 `<3,0>`。

## 4. InitRouting：明确候选与未证明的部分

[GatherRoutedTopkWeights](../kernels/mega_moe_bf16/include/mega_moe_bf16.h) 在路由算子返回后、count 交换和 Dispatch 开始前执行。`AIV_INITROUTING_SORT` 的计时还包含此函数、buffer 初始化和后续 SyncAll，不是纯排序时间。

当前按 sorted route 逐元素读取 GM 索引、读取 GM BF16 权重、标量写入 UB，然后用 MTE3 连续写入 SHMEM。当前 shape 为每 rank 65536 条、64 AIV 平均每核 1024 条。

I-01 已实现容量受限快速路径：把源权重和本核索引批量搬入 UB，完成局部重排后继续用 MTE3 发布。源权重表最大 128 KiB；索引/输出分批处理，独立临时 UB 总量最多 140 KiB。超过 65536 条 route 的 shape 保留原路径。用户已反馈修复版编译、精度通过，端到端 31856 us，InitRouting 662 us；稳定收益幅度尚未确定，见第 9 节。

不能恢复 SIMT 标量写 SHMEM 的老办法：源码注释记录了 count 通知先可见、权重数据尚未对远端 MTE2 可见的风险。输出发布和输入重排是两个独立问题。

已确认的是逐元素操作存在；尚未测得它占 InitRouting 约 0.8 ms 的比例，不能声称整个 0.8 ms 都能消除。

## 5. ops-transformer 的 Unpermute 可借鉴点

参考仓说明：[MEGAMOE_REFERENCE_REPOSITORIES_AND_OPTIMIZATIONS](../../learn_design/MEGAMOE_REFERENCE_REPOSITORIES_AND_OPTIMIZATIONS.md)。本次读取的是本地 refs，不宣称这些提交是联网查询到的最新版本：官方工作树 `35f0aafb0`；海思参考仓通过本地 commit 对象查看下述历史改动，无需切分支。

### 当前 BF16 路径的实际语义

- [SwiGLU epilogue](../kernels/mega_moe_bf16/include/block_epilogue_swiglu_bf16.h) 已应用 routed TopK 权重，GMM2/Combine 返回的结果已经包含权重因素。
- `UnpermuteV2()` 使用 `expandedRowIdxGather[token*TopK+expert]` 从 `ptrD` 找回各 route，沿 TopK 顺序相加。
- 当前使用 BF16 输入与 BF16 累加结果：第一条 route 是 `Adds(...,0)`，后续为 BF16 `Add`；没有每 token 先 Duplicate 清零，也没有再次乘 TopK 权重。
- 优化前配置是两个 BF16 输入 buffer、一个输出/累加 buffer。9 月 7 日实现改为两个输出 buffer，见第 10 节；仍不保留完整 TopK FP32 累加器。

### 参考实现与可移植性

| 参考点 | 核实到的内容 | 对当前 BF16 的判断 |
| --- | --- | --- |
| `ops-transformer_haisi:0108cd9ae` | 新增 `UnpermuteV3` / `UnpermuteVectorProcess`，MicroAPI 转换、乘权重、加法；两个输出 buffer 和明确 MTE3→V 复用依赖 | 双输出缓冲与依赖组织可借鉴；不能重复乘权重。该提交的 Process 仍调用 V2，仅有新增函数，不能凭提交标题认定已启用或已有性能收益 |
| `ops-transformer_haisi:997efb34e`，`hs/jsw_zero_redundant` | `UnpermuteCombineDedup` 的第一条有效 rank 结果直接覆盖累加 buffer，删除清零，并调整事件 | 当前普通 BF16 已无该清零；去冗余布局不同，不能宣称直接移植可省一轮清零 |
| 官方 `arch35/stage/mega_moe_unpermute.h` | 按批读权重、2～6 输入 buffer；BF16 输入转 FP32，FP32 累加后统一转 BF16；支持已预乘权重模板 | 可借鉴容量规划和 buffer 复用；FP32 累加会改变当前 BF16 每步舍入行为，需单独作为数值方案评估 |
| 官方 host `CalcUnpermuteBufferConfig` | 按可用 UB、hidden、TopK、每核 token 数决定 weight batch 和 ring 深度 | 借鉴按容量推导的思路；当前无需再读权重，应把预算用于本核 route 索引、输入预取和输出缓冲 |

官方参考源码：[Unpermute stage](../../ref/ops-transformer/mc2/mega_moe/op_kernel/arch35/stage/mega_moe_unpermute.h)、[host buffer 配置](../../ref/ops-transformer/mc2/mega_moe/op_host/op_tiling/arch35/mega_moe_tiling.cpp)。

U-01 推荐先设计**保持 route 顺序和 BF16 舍入语义的搬运流水**：批量读取本核路由索引，明确下一输入的预取和两个输出 buffer 的复用依赖，减少逐 token 输出与后续计算之间的等待。不能仅把 buffer 从 2 改成 4/6 就认定更快。

具体依据：当前每个 token 开头用 MTE3→MTE2 事件，把下一 token 的输入搬运放在前一 token 输出完成之后。两个输出 buffer 可以把依赖限定到“同一输出槽位再次使用”时，让前一输出搬运与后一 token 的处理有机会重叠。当前 K=7168 时，一行 BF16 为 14 KiB；两个输入加两个输出为 56 KiB，加本核 1024 个路由索引约 4 KiB，基本数据区约 60 KiB。容量不是这个 shape 的主要障碍，但仍需证明 MTE3→V、V→MTE2 的配对和最终 drain，不能直接删除现有事件。

之后再判断 MicroAPI 能否减少 UB 中间读写。历史 V3 每个 expert 后仍写回 BF16 结果，并不是把全部 TopK 一直留在寄存器中累加；当前又没有逐 route 乘权重，所以它的收益来源不能照搬。

这是独立于 Dispatch/GMM1 启动策略的候选；本轮仅分析，不改数值和尾部同步。当前缺少普通 Timer 下 Unpermute 独立耗时，无法给出毫秒收益承诺。

## 6. 合并遗留清单

下列是本轮会话和已检查源码中确认的待办，不代表对整个仓库完成了全面缺陷审计。

| 编号 | 问题/候选 | 当前状态 | 下周处理条件 |
| --- | --- | --- | --- |
| D-01 | 256 行提前启动回退约 1 ms；swizzle 与供数节奏不匹配的可能性 | 默认停用，保留构建实验入口；默认模式冗余分段发布已清理，见第 11 节；提前启动回退根因未定论 | 本轮只验收默认模式清理；后续调度实验须有更强依据 |
| I-01 | InitRouting TopK 权重逐元素 GM gather | 已实现；用户报告当前 shape 编译、精度通过，E2E 31856 us、InitRouting 662 us | 保留候选；小幅收益仍需与测量波动区分，其他 shape 和 MGroup/UDMA 未完成设备验收 |
| U-01 | Unpermute 索引整理、输入预取和输出复用 | 9 月 7 日已实现共享 helper，普通 BF16/MGroup/UDMA 接入；host 流水测试通过 | CANN 编译、设备精度与普通 Timer 下的 Unpermute/E2E 收益待验收，见第 10 节 |
| D-02 | 实际 Dispatch 内部 copy batch 为 2 行 | 是源码配置，不等于已证明瓶颈 | 若继续优化搬运，按 UB 和事件资源推导；与 ready 粒度分别处理，不同时改 GMM |
| P-01 | 串行 wait diagnostics 引入额外 FIX 同步，使约 31.94 ms 变成约 34.82 ms | 临时诊断代码、构建开关及专用汇总工具已删除，保留历史结论 | 后续使用普通 Timer；GMM1 Timer 本身也包含握手/后续 tile wait |
| P-02 | Timer 样本与 profiler 统计口径不同 | trace 通常看最后采样迭代，性能是 tail=50 的 rank 中位数 | 记录 shape、SO/提交、ready 模式、timer/diagnostics、loops；不以某张图或单次最小值归因 |
| W-01 | UDMA 持久路由元数据与 transient workspace 去冗余 | `8323352` 直接产出 metadata；`268af98` 压缩临时布局，默认 EP8 省 513 KiB/rank | 独立跟踪 UDMA 设备精度、反复调用和性能；不把节省容量当成 E2E 收益。保留 ptrA1 在 Combine 前的 packed X 快照 |
| DOC-01 | 部分 kernel README 仍描述两返回值/旧 workspace 公式 | 测试 README 已描述 `(y, expert_token_nums, training_saved_state)` | 统一普通/UDMA kernel 文档与当前 API、持久/临时/SHMEM 生命周期；本次仅补充构建模式说明 |

rank5/group24 不作为继续追查的缺陷。早先旧环境的慢 group 不能用新环境样本证明已存在。

工作顺序：保留默认整 expert 模式及 I-01 已通过当前 shape 精度的版本 → U-01 的搬运流水方案 → 有更强证据后再恢复 D-01 的调度实验。避免同时修改启动、swizzle、数值累加和通信布局。

## 7. 本次验证与边界

- `bash -n`：构建脚本语法通过。
- 构建开关初版通过了包含诊断兼容性的 9 个 mock 场景。清理临时诊断后重新检查默认、256、256 后重建默认、显式 expert、256+普通 Timer、缺少参数、非法 128、帮助、拒绝旧诊断参数共 9 个场景。使用临时目录、mock CMake/Python/SHMEM 运行真实脚本，验证模式/Timer 参数和屏显；未编译 NPU 代码。
- Clang C++17 编译并运行已有 `test_dispatch_tile_ready.cpp`：256 行 ready / 128 行计算、覆盖、乱序到达、尾块、reset、完成后绕过查询、整 expert gate 的 host 协议测试通过。
- 真实 CATLASS swizzle 头文件的 host 枚举通过；仅对设备修饰符和无关设备 API 声明做 host 适配，未修改仓库中的 CATLASS。
- 本机没有可用 CANN/NPU 验证环境；上述结果不代表 Ascend 编译、8 卡精度或端到端性能 PASS。恢复默认后的实际耗时仍以新构建 SO 的设备结果为准。

本次不要求用户补发 JSON 或马上跑新实验。后续确实需要设备验收时，给一组可复制命令，将模式和必要统计输出到终端。

## 8. 合入 train 前的调测代码清理

按用户要求删除当天的临时诊断实现：GMM1 等待前后的 FIX→Scalar 强制同步、累计计数和 Timer slot 别名、公共 Timer 的累计写入接口、诊断 CMake/脚本开关、专用终端汇总脚本及其两份测试。CSV parser 恢复普通事件名称和时间区间格式，移除测试 README 中已失效的诊断命令。

UDMA 临时 stage/hang 诊断已在 `97c6c87` 删除；本轮核对没有恢复这些逻辑。普通阶段 Timer、精度校验、Dispatch ready 协议测试和 workspace 布局测试保留。

清理后验证：上述 9 个脚本场景通过；合成普通 Timer CSV 的事件名、时间戳、duration 和 139776 元素布局检查通过。公共 Timer、BF16 Timer 包装和 CSV parser 已逐文件核对，与 `12bb58f` 之前的源码一致（忽略 CRLF/LF 差异）。

整 expert 默认模式、显式 256 行构建入口和本文遗留记录保留。本次只清理工作分支，不执行向 train 的合并或推送；硬件回归仍是合入前待完成的验证。

## 9. InitRouting 权重批量重排实现

初版接入普通 BF16 的 `Process()`，随后按用户要求接入 MGroup 和 UDMA：路由算子结束后，先调用
[`TryGatherTopkWeights`](../kernels/mega_moe_bf16/include/mega_moe_topk_weight_gather.h)，随后才创建 Dispatch/transport 的 `firstPipe`。三者共用同一个 helper，容量不足时分别调用原来的标量 gather。

### 路径选择与数据

- `M * TopK <= 65536` 且非零时启用；否则返回原有 `GatherRoutedTopkWeights()`。
- 使用独立 TPipe，先分配源权重 UB，再分配索引和输出 UB；最后等待 MTE3 发布完成并 Destroy，再进入 FirstBuffInit。
- 源权重最多 128 KiB；一批最多 2048 条 route，索引 8 KiB，输出 4 KiB。总量最多 140 KiB，不与 Dispatch/GMM 的 UB 占用叠加。
- 当前 shape 为每核 1024 条 route，一批完成。每个活跃 AIV 读取完整源权重表一次；64 AIV 共发起 8 MiB 源权重读取请求，实际外部存储流量受缓存影响。收益依赖批量读取相对标量 gather 的实际效率，不能只按删除的标量指令数估算。
- `DataCopyPad` 只读取有效 GM 字节，UB 分配向 32 字节对齐；输出也只写本核有效 route，支持尾批和无工作核。
- 将 route 索引乘 2 得到 byte offset，使用 uint16 Gather 重排 BF16 原始位模式；不做 Cast 或浮点计算。最大偏移 131070 Byte，小于 16-bit Gather 的 2^17 Byte 限制。

接口依据：[官方 Gather 文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/API/ascendcopapi/atlasascendc_api_07_0092.html) 的 count 重载、uint16 类型、字节偏移和对齐约束；另参考本地 ops-transformer 中 Muls(int32 索引)→Gather 的用法。文档与 host 检查不替代目标环境 CANN 编译。

### 同步边界

| 依赖 | 实现 |
| --- | --- |
| 上批 Gather 读完索引后，MTE2 才能覆盖索引 UB | V→MTE2 |
| 源权重及本批索引到达后，Vector 才能使用 | MTE2→V |
| 索引乘 2 完成后执行 Gather | PIPE_V barrier |
| 上批 MTE3 读完输出槽后，Gather 才能覆盖它 | MTE3→V |
| Gather 结果完成后才能写 SHMEM | V→MTE3 |
| 输出发布完成后才能释放 UB、进入 count 通知阶段 | MTE3→Scalar，再 Destroy；保留原有跨核 SyncAll |

### 本地验证与设备边界

[`test_topk_weight_gather.cpp`](../test/megamoe_bf16_test/test_topk_weight_gather.cpp) 使用生产计划结构验证了容量上限、随机 route 排列、每核覆盖、尾批、无工作核、65536/65537 边界和全部 65536 种 BF16 位模式，Clang C++17 编译运行通过。

还使用临时同步 CPU API 替身执行了实际 Gather helper，检查位模式、输出前后哨兵、分批、回退与 UB 释放；替身不是 Ascend 编译器或异步流水模拟器，不能证明设备指令支持、同步可见性或性能。

构建入口不增加新参数，仍使用 `bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer`，默认整 expert 等待。与 `2cefcc2` 对照时应保持相同 shape、普通 Timer 和 profiler 统计口径；最新用户反馈见下文。

### 设备函数标注修复

用户对 `66d81a1` 的 CANN 构建发现 `TopkWeightGatherPlan` 构造函数被视为 host 函数，无法从 aicore 调用。原因是初版只有 constexpr，没有设备函数标注；原来的 CPU API 替身测试也没有检查调用目标限制。

现按仓内 `mega_moe_training_workspace.h` 的方式，在 `__CCE__` 下同时给构造函数和 `BatchCount()` 添加 `__host__ __aicore__`，普通 host 编译时宏为空。未改变容量、索引或流水算法。

修复后 host 计划/位模式测试通过。临时使用 Clang CUDA 的 host/device 属性代替 CCE 属性，并关闭 constexpr 自动双端推断：原版和仅修构造函数版本均按预期拒绝设备调用，两处均修复的版本通过。这只验证调用目标标注；随后用户另行反馈了目标环境编译与精度通过。

### 用户设备反馈：31856 us / InitRouting 662 us

在设备标注修复提交 `cc0a446` 推送后，用户反馈编译无问题、精度无问题，端到端 31856 us、InitRouting 662 us。沿用本任务当前默认 shape 的上下文记录；未独立读取远端 HEAD、完整 profiler 输出或重复实验统计。

- 相对新环境此前整 expert 等待的 31940.519 us，数值下降 84.519 us，约 0.265%。这是小幅改善迹象，不能从这次反馈确定已超过测量波动。
- 相对先前约 0.8 ms 的 InitRouting，662 us 在数值上少约 138 us（约 17%）。旧分项是近似值，且不能确认是同环境同口径的配对基线，因此只作量级参考。
- 662 us 是当前 InitRouting 分段耗时，不是新 Gather helper 的独立耗时；不能将上述差值精确归因或直接等同于 E2E 节省。
- 当前普通 BF16 shape 的编译/精度验收已有用户报告；MGroup、UDMA、其他 shape 和长期稳定性不随之自动通过。本次保留实现，不要求用户马上追加参数扫描或诊断采集。

### MGroup / UDMA 接入

用户随后要求两条路径也使用该优化。MGroup 的 `Process()` 通过现有普通 BF16 头文件获得 helper；UDMA 显式 include 共享 helper。调用参数均为原始 BF16 权重、`expandedRowIdx`、`ptrRoutedTopkWeights`、`M*TopK` 及现有 AIV 分工。

两条路径都在 FirstBuffInit 之前运行快速路径；返回 false 时仍在 FirstBuffInit 之后运行各自的旧 Gather。原有 SyncAll、flag reset、通信提交、GMM 调度及持久状态保存顺序保留，新增临时 UB 不与这些阶段叠加。

UDMA 保留原 Timer 口径：Gather 位于 SORT 计时结束之后、`AIV_INITROUTING_EXPERT_TOKEN_COUNT` 区间内。因此不能只看 UDMA 的 SORT 分项判断本次优化效果，也不能直接与普通 BF16 的 InitRouting 分项对比。

沿用各自构建入口，无新增参数：

```bash
bash scripts/build_mega_moe_bf16_mgroup.sh -soc_type Ascend950 --enable-timer
bash scripts/build_mega_moe_bf16_udma.sh -soc_type Ascend950 --enable-timer
```

共享 helper 的 host 测试通过；新增两处调用完成源码边界检查，MGroup/UDMA 的目标编译、设备精度和端到端性能仍待验收。普通 BF16 的 31856 us / 662 us 不作为这两条路径的验收结果。

## 10. Unpermute 搬运流水实现（2026-09-07）

本次基于 `0d3f3fa`（已合并 train 的 saved-state 变更）实现 U-01。
[`TryUnpermute`](../kernels/mega_moe_bf16/include/mega_moe_unpermute.h)
接入普通 BF16 与 UDMA 的 `UnpermuteV2()`；MGroup 调用继承的普通 BF16 实现。
三者均在原有 Combine 完成、跨核和跨卡同步之后读取本地 `ptrD`，不改变通信提交、GMM 调度或持久状态布局。

### 数据与容量

- 每批最多读取 1024 个 int32 路由索引，批大小向下取完整 token；尾批 `DataCopyPad` 仅读有效字节。仍逐 route 从 UB 读取索引，减少的是标量 GM 访问，并未消除标量索引工作。
- 当前 M=4096、TopK=16、64 AIV 时，每核 64 token；1024 次标量 GM 索引读取变成一笔 4 KiB 索引 DMA。
- 使用两个 BF16 输入行和两个 BF16 输出/累加行。每个 token 先加载前两条 route，消费输入槽后预取同槽下一条 route；TopK=1 和奇数 TopK 也按实际 route 数配对事件。
- 输出在两个槽间轮换；只在同一输出槽再次被 Vector 写入时等待上次 MTE3 读取完成，下一 token 输入不再等待上一 token 输出。
- 第一条仍为 `Adds(...,0)`，后续仍逐次 BF16 `Add`；保持 TopK 顺序与每步舍入，不再次乘权重，也不换成整段 FP32 累加。
- UB 为 `4 * K * sizeof(BF16) + 4096`，K=7168 时为 60 KiB。容量来自现有 `Catlass::Arch::Ascend950::UB_SIZE`；行长不满足 32 Byte 对齐、TopK 超过 1024 或 UB 不足时，调用原三行实现。空闲核不分配 UB。
- 新 UB 由独立的尾部 TPipe 管理，原阶段的 TPipe 已释放；所有输入、输出事件及 pipe 排空后再 Destroy。GM 行偏移使用 64 位乘法。

### 同步依赖

| 依赖 | 实现 |
| --- | --- |
| Scalar 读完旧索引批后才能覆盖索引 UB | S→MTE2 |
| 新索引到达后才能 GetValue | MTE2→S |
| Vector 读完输入槽后才能复用该输入槽 | 每槽 V→MTE2，两个事件 |
| 输入到达后才能执行 BF16 Add | 每槽 MTE2→V |
| MTE3 读完输出槽后才能覆写该输出槽 | 每槽 MTE3→V，两个事件 |
| TopK 累加完成后才能写出 | 每槽 V→MTE3 |
| 退出及 UB 释放 | 两个输入空闲事件、两个输出空闲事件全部消费，PIPE_ALL 排空 |

### 已验证与待验收

[`test_unpermute_pipeline.cpp`](../test/megamoe_bf16_test/test_unpermute_pipeline.cpp)
直接调用生产 helper，使用独立 MTE2/Vector/MTE3 FIFO 和随机执行顺序的 host shim。
Clang C++17 `-O2 -Wall -Wextra -Werror` 编译运行通过，共 94 个用例，覆盖：
当前 shape 的单 AIV 工作量、奇偶 TopK、跨批和尾批、空闲核、精确 UB 容量边界、拒绝路径、GM/UB 越界检查、输出保护区、逐步 BF16 舍入结果和事件全部回收。
临时删除输入或输出复用等待的负对照均被测试检出。该模型不模拟硬件时延、指令内部并发或 CANN 的真实 BF16 算术实现。

新布局构造函数及 BatchCount 同时添加 host/device 标注。临时 Clang CUDA 属性检查关闭 constexpr 自动双端推断后通过；删除 device 标注的负对照被拒绝。这是调用目标检查，不代表 CANN 编译。

本机无 CANN/NPU。三个入口的设备编译、精度、Unpermute 分项和 E2E 均待验收；不预设收益。输入激活总读取量没有减少，因此不能把索引指令减少比例当成总体加速比。默认整 expert 等待与普通 Timer 保持原配置，没有新增调测开关或打点。

### 设备验收入口

在设备仓库根目录，用现有默认 shape 先做一组普通 BF16 验收：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1,2,3,4,5,6,7 \
  --mode both --warmup 10 --loops 100 --enable-timer
```

结果沿用终端 `[SUCCESS]`、`[PERF]` 和普通 Timer。对比需保持设备环境、shape、权重/路由模式、warmup/loops 与 Timer 一致，并与本次修改前的 `0d3f3fa` 比较；历史 31856 us 只作参考。MGroup 与 UDMA 仍分别使用原构建、测试入口验收，普通 BF16 通过不代替它们通过。

## 11. 默认 expert 模式去除分段发布

统一基线为 `928f688`：本地与远端 `train`、`deepep_moe_xx` 已在该点对齐。本次修改在工作分支开展。

此前为了对照，默认路径是 `DispatchTokens<true>` 与 `GroupMatmulSwiglu<true,true>`：生产者仍按 256 行边界切分，每段搬运后执行 MTE3→Scalar 和行计数 AtomicAdd；消费者只等待单独的 expert 完成计数。默认消费者不使用这些行通知。

现在给 Dispatch 增加编译期 `WaitForFullExpert` 参数，生产者与消费者共用同一个构建模式：

| 构建模式 | Dispatch | GMM1 |
| --- | --- | --- |
| 默认 `expert` | 每个 peer/core 分工连续提交 payload，再提交对应权重；完成可见性同步后贡献一次 expert 完成计数 | 仍等待 `EP * dispatchBlockNumPerEP`，每个 expert 只等一次 |
| 显式 `256` | 沿用 256 行边界切片、payload/权重可见性同步和行计数发布 | 沿用各计算块的行就绪等待，以及整个 expert 完成后的查询绕过 |

默认模式删除的是分段级额外等待和原子更新。底层 `dispatchRows_` 两行双缓冲、peer/core 分工、当前 expert 的实际行数裁剪、空分工完成通知、末尾事件回收、GMM tile/swizzle 和权重数值均沿用原逻辑。就绪标志的 workspace 分配和初始化保留，便于两种构建模式继续使用同一布局。

MGroup 的 `DispatchTokens()` 默认模板实参仍为旧协议，UDMA 使用独立 Dispatch。本次调整普通 BF16 的默认生产者，不重新开启提前计算实验。

验证：扩展原 `test_dispatch_tile_ready.cpp`，覆盖完整分工发布、被刻意污染但不应读取的行计数、payload 已到而权重未到、容量裁剪、空核、乱序到达和重复 reset；Clang C++17 编译运行通过。另在临时 host shim 中直接编译执行修改前后的生产 Dispatch 函数体，逐个验证 24 组分工与容量场景；默认模式只发布 expert 完成，256 模式的搬运/发布统计与修改前一致。Host shim 将模拟写入推迟到可见性同步，不替代 Ascend 指令与硬件内存序验证。

本机无 CANN/NPU；本次默认模式清理的目标编译、精度和性能待验收。对照使用 `928f688`、相同普通 Timer 与 warmup/loops。此前 Unpermute 用户反馈为 480→460 us；缓存接入后单次 E2E 32009 us 只作记录，未据此认定回退。可以继续使用数据缓存，每次仍重新计算 NPU 输出并做精度检查。

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer --gmm1-ready-mode expert
```

随后沿用基线测试命令和原 warmup/loops，观察 Dispatch 首个 expert 与 E2E；本次不增加诊断打点，也不要求先扫描更多参数。
