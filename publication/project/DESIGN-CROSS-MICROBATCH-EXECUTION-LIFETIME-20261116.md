# 跨microbatch共享专家执行资源与梯度累积：机制、代码与最小证伪计划

状态：设计草案，未实现、未运行。2026-11-16按用户要求只读分析；长期goal保持paused。本次只新增本文档与归档摘要，不修改验收源码、构建、SO、配置或历史结果。以下实验数量、接口和分支名均为建议，不代表已启动或已验收。

## 1. 先分开两项产出

第一项是公共执行器能力：同一参数版本的专家权重可以跨native调用直接作为计算操作数使用；受管远端副本的Wgrad直接累积到可跨调用持有的FP32缓冲，再明确交付给owner。先检验这项能力是否减少完整更新的暴露工作。

第二项才是planner候选：在当前已就绪任务、权重驻留与未交付梯度状态下，决定使用哪个副本、是否启用新副本、保留哪些资源、何时交付哪一段梯度。只有第一项有效，而且第二项超过强简单组合，才值得讨论新的planner论文贡献。

首版保留串行native调用，保留已有分组、checkpoint和kernel内合法屏障；跨调用保留状态不要求一个kernel常驻整个optimizer step，也不要求多个MoE kernel同时运行。跨任务tile交错属于后续可能的执行扩展，不能当作首版已具备的能力。

## 2. 当前已经共享了什么，新增的是什么

`tools/grouped_experts.py:41`已将多个任务的pointwise FFN行拼接；受单次16384行限制，两条完整16K序列通过两波执行。Attention上下文仍完整。这个分组已经能让同次调用共享专家权重与一次调用的梯度输出，不能把它重命名为新方向。

当前不同调用之间的路径大致为：

1. `_publish()`将当前层owner计算权重拷进公用home镜像。
2. 命中版本缓存时，仍将cache bank内容拷进该调用的replica计算槽，再发布fresh epoch。
3. backward分配本调用的dW输出并写出。
4. `LifetimeDevice.gradients()`观察计数/贡献者，将输出转换并加入公共FP32 bank。
5. 立即、窗口或更新末优先归并；容量不足会提前flush。

新增能力针对第2—4步：计算直接读有效驻留存储；受管副本的梯度tile直接加入跨调用累积区。不会减少Wgrad的数学计算量，也不会自动减少FP32 bank读改写；这些成本必须保留。

## 3. 数学等价与资源身份

对固定参数版本v、层l、逻辑专家e、执行rank r，定义专家执行身份：

`ExpertExecutionKey = (parameter_version, layer, expert, execution_rank)`

其本地梯度满足：

`G[v,l,e,r] = sum(delta_W[a] for a in actual_assignments[v,l,e,r])`

每份assignment及其路由权重不变，dx与路由权重梯度仍随相应反向正常完成。改变的是权重和梯度存储的复用方式，以及远端梯度的交付分段。

同版本同层同专家的不同microbatch可共享权重；不同层同编号专家、不同参数版本不能共享权重内容。不同microbatch的激活、路由元数据、输出和dx保存态仍独立。RNG映射、loss有效标签归一化、GA4缩放和EP缩放均保持。

单个执行身份不等于一个不可拆的物理缓冲。权重lease与梯度segment分别管理：权重可在无读者后回收，未交付梯度必须继续保留；未来反向需要相同版本权重时可以重新装载。不能为每个保存的前向无限期钉住所有权重槽。

主形状H4096/F1024下，一个专家共3HF个参数；BF16权重约24MiB，FP32梯度约48MiB。继续以每rank八条目权重192MiB、八条目梯度384MiB的共同预算设计；优先复用现有分配，不能同时多留一套而不记账。通信接收面、临时tile及其他工作区另计，整卡仍不超过90GiB。

## 4. 为什么它可能改变planner决策

假设专家e的owner是rank0，rank1已有其当前版本权重，并有一个尚未提交的梯度累积区；rank2更空，但尚未执行该专家。当前已就绪microbatch有e的任务：

| 选择 | 当前调用 | 更新内新增义务 |
|---|---|---|
| 继续交给rank1 | 可能少一些即时负载均衡 | 可以复用权重与同一未提交梯度segment |
| 在rank2启用副本 | 可能让本次调用更快 | 可能增加24MiB权重准备和一份48MiB梯度交付 |

这里是逻辑payload示意，不是实测耗时。新副本的搬运可能被完全隐藏；rank1也可能严重拥塞，所以复用不一定更好。

重要差异是边际梯度发送量与状态有关：

- 当前身份已有可追加的OPEN segment：新增任务可能不增加一次完整梯度发送，但仍增加计算和累积成本。
- 身份首次出现，或旧segment已经封口：新增任务会开启一个新segment，最终需要交付完整参数梯度。
- 提前交付可以释放容量、提早服务owner，但后续再次使用会产生新segment和额外发送。

因此“使用哪个副本”与“何时归并”相互影响。固定每调用副本罚项不能完整表达这个状态；仅做更新末归并也未必适合有限容量。但如果热点稳定、容量足够、尾部能隐藏，固定布局＋更新末归并可能已经很好，新算法应允许没有优势。

## 5. 与REM和MegaFlux的准确区别

### REM

[REM §3.2/B.3](https://arxiv.org/html/2609.32398v1)描述两个时间尺度：根据历史负载在global-batch边界配置副本，microbatch内根据当前负载派发；副本独立计算梯度，在global step后累积归主，optimizer状态固定在原owner。附录还描述每次forward前同步权重、逐层预取。

因此，跨microbatch梯度累积、固定owner和更新末归主均不是本方案的新贡献。REM论文并未给出我们这套MegaMoE epilogue与计算槽的实现，不能据此断言其代码没有直接累积优化。

待研究的差异仅是：在有限共享资源和动态就绪条件下，是否需要联合调整副本参与、实际驻留/回收与梯度分段交付，而不是沿用本次更新中的固定布局与统一交付边界。REM式控制也必须获得相同的驻留、直接累积、部分flush等公共能力。

若只实现“副本保留＋GA累积＋步末发送”，成果应归类为REM式训练适配/执行器优化。不能宣称比REM更少归并：一个被使用的远端身份若全更新只发一次，已经达到该身份的最低完整发送次数。

### MegaFlux

[MegaFlux §3/A.2](https://arxiv.org/html/2610.00671v1)为当前调用选择副本和tile配额，在核内按权重/梯度就绪流水执行；反向考虑owner侧完整梯度代价，使用`max(load_r + lambda * remote_contributors_r)`代理，lambda固定12，并协调早期梯度生产与归并。

它已经处理contributor readiness和副本专家优先生产，不能把普通ready扫描或“考虑梯度”称为新增机制。

本候选增加的是跨调用、同参数版本的权重/梯度状态与分段交付决策。某副本的梯度义务是否已经存在，取决于前面实际执行与交付的历史，不能仅由本次q是否非零确定。首版仍使用其合法调用内流水，不重新设计所有GEMM。

“区别”只是问题范围，不自动构成创新。所有公共执行器能力需同时提供给MegaFlux式单调用策略和REM式策略。还要排除[Afore](https://arxiv.org/html/2610.03203v1)已有的副本复用/保护槽/暴露迁移成本判定，以及PlaceMoE已有的梯度重叠机制。

## 6. 需要实现的最小状态与协议

以下为计划接口，不是已有API：

| 对象 | 关键字段/约束 |
|---|---|
| ResidentWeightLease | key、bank/slot、generation、W1/W2或panel-ready、在途读者；参数内容版本与调用epoch分开 |
| GradientSegment | key、segment_generation、FP32 sink、准确贡献IDs、OPEN/SEALED/INFLIGHT/COMPLETE、最后写入ready |
| CallBinding | 不可变计划与物理slot到resident/sink的映射、参数版本、call/replay身份、实际贡献者 |
| UpdateCompletion | 所有计划反向恰好一次；所有segment交付完成后才允许norm/clip/Adam |

动作限定为：acquire/fill权重、绑定已就绪调用、append实际梯度贡献、seal特定segment、submit、观察complete、回收。不能在发送仍读取的同一缓冲上继续add；需要另一空闲条目/双buffer，或明确等待。所有占用计入八条目预算。

checkpoint只绑定原始计划，不重复规划或登记贡献。反向可重新获得物理lease，但必须恢复相同逻辑专家与参数版本。零任务副本不登记梯度。

首版保持确定的全rank提交序列；不能让各rank独立按本地deadline调用不匹配的P2P。后续选择性seal也需生成全rank一致的交付manifest。权重和梯度复用均以设备完成事件/核内ready确认，不能仅以CPU函数返回认定资源空闲。

## 7. 现有代码具体修改位置

基线为`gradient-ready-20261116/stack`的`research/step-lifetime-grad-ready@39c5794`，本轮复核其工作区干净。若后续实施，建议从该交付建立独立`research/step-lifetime-exec-session`；本轮没有创建分支。

| 文件/入口 | 当前行为 | 最小改造 |
|---|---|---|
| `tools/step_lifetime.py:31` | 已有UpdateContext/ReplicaLease/GradientTicket | 增加计算存储绑定与梯度segment代次；seal与complete分开；权重和梯度租约独立 |
| `tools/lifetime_device.py:85,122,171,229` | 缓存bank→计算槽；接收本调用raw dW后add；确定P2P flush | 绑定resident计算地址和远端sink；受管副本不再走外部全平面add；沿用数值owner发布路径；先保留共同drain/元数据观察以隔离变量 |
| `tools/gradient_bank_ready.py` | 已验证最后写入ready的强引用/代次 | 扩展到segment generation和seal快照；inflight期间拒绝追加/复用 |
| `ascend_deepep/megaflux_worker_pipeline.py:47` | sidecar携带cache指针/raw模式 | 新版本可选resident/sink绑定；旧模式兼容；计算排布、输出目标和交付模式分开，不复用ready=0切换所有行为 |
| `ascend_deepep/fsdp2_bridge.py:297,452` | 每调用publish；backward取独立dW并交给lifetime | 保存逻辑身份与绑定，托管sink时返回dx/dp及未托管梯度；owner Parameter对象和布局不替换 |
| `csrc/ops/mega_moe_bf16.cpp:899`、torch binding/launcher | 分配本调用gradWeight1/2，输出类型继承BF16权重 | 添加可选外部sink描述；受管远端不要求全量独立dW平面；home/未受管副本保留旧输出与明确fallback |
| `include/mega_moe_replica_pipeline.h:146`、`mega_moe_backward_restore.h:97`及前后向权重寻址 | 将远端/缓存内容拷到replica槽 | 命中时计算operand直接指向有效bank；miss时填充该bank并保留panel流水；为本调用发布合法ready；所有前向、重计算、DGrad权重读取一致 |
| `include/mega_moe_bf16_backward.h:1453` | DW输出地址按连续BF16专家平面计算 | DW阶段解析受管sink；其他GEMM/任务排布保持；空专家只清理本次输出，不清空已累计bank |
| `include/mega_moe_bf16_backward_catlass_impl.h:907,987` | GMM1DW/GMM2DW写BF16 C并发布tile-ready | 新tile级累积epilogue；唯一写者、正确写入完成再publish；不能只把BF16指针改成float指针 |
| `include/mega_moe_gradient_pipe.h` | 当前调用partial-ready及owner归并 | 区分tile生产完成与跨调用segment封口；首版继续确定交付，不引入任意Wgrad延迟 |
| `tools/temporal_model.py:183`、grouped/prefix runtime | 单次分组/原计划重放；跨层域无replica；lifetime只接受单层域 | 首版固定同层分组，仅增加调用与lease绑定；不先扩大64槽或重写跨层虚拟专家域 |
| `tools/train.py:387,427` | lifetime.finish后norm/clip/Adam | 验证全部segment交付后发布owner梯度；GA4/有效标签归一化不变，EP缩放每份贡献一次 |

上述目录前缀为`stack/source/native-balanced/`，tools文件除外。实现需要新ABI与真实二进制能力校验，不能仅Python侧伪造开关。控制与候选优先使用同一个新SO的不同能力模式，并与旧锚点回归。

两个容易混淆的细节：

1. `_publish()`只有一份公用home镜像，跨层时内容会被替换。不能因为Parameter版本未变就全局跳过publish；必须校验镜像当前绑定的layer/version，或另外设计可驻留home镜像。首版保持此行为共同不变。
2. 当前GMM输出BF16，外部转换后加到FP32。首版sink应保留每调用BF16舍入边界，再FP32累积，避免把精度变化混入机制对照。若改成FP32直接输出，需要独立数值族，并给基线相同能力。tile级累积仍需要FP32读改写，也可能占用额外AIV/UB资源，不能预先认定更快。

首版只管理拿到lease的远端副本，home和资源不足的调用显式走匹配旧路径；记录实际覆盖率与fallback，不能静默换planner。若分配上暂保留了未使用的旧输出平面，也不能宣称已节省该部分峰值HBM。

## 8. 最小验证：先花小成本决定是否值得做planner

### G0：数值与协议资格，不计性能分数

复用现有小模型/独立参考入口，覆盖：同专家多调用累积、空专家、全远端且owner无本地任务、热点切换、条目1强制中途seal、inflight写入拒绝、版本切换、checkpoint重放与反序反向。

沿用输出相对L2≤0.004，dx/router/routed/shared各参数梯度≤0.02；L2/H512/F512/S256/EP8小模型至少三次Adam更新，检查未clip梯度、缩放一次、参数变化和无NaN/Inf。新SO/绑定/实际贡献精确一次必须核对；任何失败先停性能，不删屏障绕过。

### G1：四个机制控制，回答“执行器能力有价值吗”

首先使用EP8、主H/F形状下的一条短合法调用序列；同一参数版本、同层四次native调用，固定输入、dy、IDs、copies/quota、任务/重计算次序、B3交付边界及预算。选取不超过八个活跃远端身份的高复用情况，作为有利诊断条件，明确不是自然训练泛化证据。

| 模式 | 计算直接使用resident权重 | Wgrad直接累积到sink |
|---|---|---|
| K00 | 否：现有bank→计算槽 | 否：现有raw dW→外部add |
| K10 | 是 | 否 |
| K01 | 否 | 是 |
| K11 | 是 | 是 |

先各一个新进程，共四个性能进程，只用于筛查。K00使用bank-ready控制；四项同FP32交付格式、同在线元数据/校验开销、同交付次数，不能K11才少做公共检查。计划冻结/重放仅用于机制隔离，不能把它当成在线planner成绩。

记录全调用序列到全部梯度完成的rank最大wall，不能只计GEMM；记录实际copy字节、dW平面写出、bank读改写、覆盖率、发送次数/字节和峰值。已有AscendTimerV2扩展用于另行诊断resident就绪、sink写入、seal/submit/complete；先通过所用路径打点覆盖资格。正式计时关闭Timer/profiler。

若有利条件下都没有净收益，先确认新epilogue/间接寻址的成本；不直接开发复杂planner。这里的负结果只限定当前实现和形状。

### G2：四个完整更新控制，回答“不是只胜过慢原型吗”

保持当前L4/EP8/E256/K8/H4096/F1024/S16384、MB1/GA4/W2、FSDP/数值/Adam配方和全部数据不变。先用冻结MaxVio10/same，只跑以下四项各一个新进程：

1. 固定同层分组＋fixed home，公共优化齐全。
2. 固定同层分组＋Native原生核内归并，公共优化齐全。
3. 当前B3＋bank-ready，作为外部服务控制。
4. 新执行器K11＋同一Native策略和固定分组，使用相同交付边界。

各25更新，5预热/20测量，每更新取八rank最大完整wall，覆盖全部在线统计、规划、编译/校验、GA4、重计算、交付、clip、Adam及设备完成；无离线计划代替在线规划。输入/参数/路由/重放身份一致，整卡峰值≤90GiB。

这样首轮是四个机制进程＋四个完整更新进程，共八个性能进程，另有必要数值资格；不是直接启动大矩阵。本轮一个也不执行。

筛选规则：如果第4项只胜过B3，仍未胜过第1/2项中的较强者，不进入新planner开发；它至多是在偿还公共服务路径成本。如果没有约3%的完整更新信号，也不扩矩阵，保留工程结果。该门槛是控制投入的筛选条件，不是普适不可能性结论。

如果有≥3%信号，冻结最强控制与候选，另做三个独立配对、轮换顺序，六个新进程。前三个或20个更新都不能代替独立运行。只有重复同向且实际完成时间改善，才进入下一层研究。

### G3：最小决策证据，回答“有没有planner问题”

此阶段仅在G2成立后考虑，仍不是执行授权。先固定同层任务分组，避免重做尚未有效的MaxUtil方向。采用新公共执行器，比较：

- REM-style：更新内固定副本布局、当前microbatch配额、梯度尽量更新末交付。系统适配不引入DO-loss，不冒称完整REM复现。
- 强简单组合：Native逐调用配额＋普通驻留缓存＋按专家选择性部分flush；不能只拿“满了全flush”的旧B3当强基线。
- 候选：读取OPEN/SEALED梯度状态与当前ready任务，联合选择配额、驻留和交付。只看当前路由和历史，不看未来银行。

只找两类可证伪的决策反例：

1. 已有dirty副本与新空闲rank之间的选择，是否出现“新副本使当前调用更快，但让完整更新更慢”；强简单组合是否已能消除？
2. 有限预算下，保留dirty bank复用与提前交付换槽之间，是否存在统一步末/普通LRU不能处理的重复尾部；候选是否真的改变它？

容量使用真实八条目；若仅条目1或人为构造路由有效，作为诊断，不当主结果。先用现有same与已冻结的变化路由检查；若根本没有这些冲突，或简单组合已获得全部收益，就停止新planner题面。

只有反例成立，才扩展任务选择与驻留的联合：在同一新执行器上比较固定分组/FIFO、独立任务选择＋独立复制、真正联合决策。否则“联合”只增加复杂性。自然训练路由与独立银行仍是论文阶段必要证据，不能靠上述人工诊断替代。

## 9. 三种可能的交付与停止边界

| 结果 | 应如何归类 |
|---|---|
| K11比K00快，但完整更新没有胜过强控制 | 当前原型优化/负结果，不继续叠加planner |
| K11使完整更新更快，REM式/简单策略同样获得收益 | 公共执行器或REM式训练适配成果，有工程价值，尚无新planner贡献 |
| 同执行器上联合状态决策稳定胜过REM式与强简单组合，且可解释并泛化 | 新planner论文候选，再开展独立路由、自然训练与消融 |

真正需要证明的是：参数版本内的副本使用历史和梯度交付历史改变了后续复制的边际代价，联合决定这些边界可以改善完整更新。不能用命中率、字节下降或一个0.71%单轮结果代替这一证据。

本设计仅为下一次决策提供可审阅方案，当前研究目标与设备队列继续暂停。
