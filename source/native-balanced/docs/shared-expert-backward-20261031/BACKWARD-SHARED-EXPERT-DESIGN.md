# MegaMoE 共享专家反向融合设计

日期：2026-10-31。作者侧任务：共享专家 **backward 融入 native MegaMoE 并安排流水**；前向由同事开发。状态：完成源码静态分析后的设计稿，未修改技术实现、未构建或进行设备实验，所有新增接口/事件名称均为提案。

## 1. 结论与首版边界

采用“**本地 shared backward 前缀＋原 routed backward＋本地 dX 合并**”的联合 kernel 设计。AIC 的共享纯 GM GEMM 与现有 P0 搬运交叠；AIV1 在原 panel restore 全部完成后处理共享激活反向；共享 AIC/AIV 资源全部退休后，切回原 P1/P2/C1/owner/P5/P6。首版不改 routed 专家遍历顺序、tile/K 累加、owner 分工或通信屏障。

这是可实施的粗粒度流水候选，不是“共享计算可以免费藏进空洞”的收益承诺。为定位问题另保留完全串行 native 联合路径。完整 MoE block 的 B/直接 FB、峰值 HBM 与正确性决定是否采用。

首版支持范围：Ascend950、BF16 输入/权重、DeepSeek-V4、TP=1、无 bias、正有限 limit、canonical routed 布局、saved BF16、EP2/4/8、panel restore2/operand-ready1、worker flags0、owner16、compact0。共享宽度 Fs 独立于 routed F，从实际三个权重推导，不要求二者永远相同。首要固定场景是 M8192/H7168/F=Fs3072、E8/EP4组、Top2、home2/slots2、capacity20480、limit10，后接普通 FSDP8＋expert-FSDP2。

先支持 KEEP 及全层 non-reentrant checkpoint；UAM shared-recompute 在独立保存适配完成后进入支持矩阵。OFFLOAD、TP>1、bias、不同激活路径、其他 saved 精度和细粒度/native FSDP 不在尚未验证时放行。配置不支持时，所有 EP rank 在启动任何 collective/native kernel 前一致回退原框架路径或报错，不能各 rank 自行分叉。

## 2. 固定代码与同事前向设计的关系

本地代码根：`D:/megamoe_train/shared-expert-branch-20261031/source/wuhan`，四仓分支均为 `megamoe-balanced-shared-expert`。开始分析时远端与本地一致，表中固定的是实现核对版本：

|仓库|本次核对 SHA|
|---|---|
|wuhan|b10827b1e9a4bcc7a23d5a2efd6c3f3d5856f81d|
|ascend_deepep_hkq_wuhan|7c41ebf938363995838056677e73e5ea6a289d02|
|MindSpeed-LLM_wuhan|111aa86a2deb2f5697ae7ed1eed8e9532ce6ce8d|
|FSDPTurbo_wuhan|da8df1f7c48297a7ab1cf5fdbf493ecead1dce69|

收尾复核时native远端推进到 `6db276532a34e18b9c5c92720467e3a2cb6e7fa9`，仅新增同事的共享专家/反向重派发设计文档，没有改动框架或kernel；本设计列出的实现文件与该远端提交逐一核验Git blob相同。没有替同事切换工作区或覆盖新提交，源码行号仍对应上述实现版本。

native 当前包含 C1 与 panel-acquire cache 实现；cache 默认0，原已验收性能组合显式1。所有新旧对照锁定同一实际 cache 开关和 SO getter，不能借切换它归因共享融合。owner32、N4、其他 worker 候选不叠加。

同事文件：[shared-expert-redispatch-fusion-design-20261031.md](D:/shared-expert-redispatch-fusion-design-20261031.md)。它是前向与联合方案的讨论稿，不是已经落地的 ABI。本设计承接其“shared 使用本地 X、三权重独立传址、前向 Y 相加、两套 FSDP 不合并、X redispatch 独立、尊重 checkpoint/UAM”方向，并补全其尚未确定的反向 lane/ready/保存与窗口边界。

需修正/澄清三处：

1. 同事文档的 rho6 行数 `[45880,6552,6552,6552]` 与我们当前冻结 capture 的 `[45876,6552,6552,6556]` 有4行差。两者均约70/10/10/10，但必须按实际 PT/route/plan SHA 定版，不能按 rho 参数重新生成后混用。
2. 当前参数默认 `use_triton_swiglu_limit=False`，但原 rho6 launch 第111–112行显式为 **True**。本首场景冻结 Triton FP32 中间激活路径；默认 torch 逐算子路径有额外 BF16 中间舍入，不能混成同一个精度基线。
3. 当前正常 balanced tape **不保存原始本地 X**，只保存 routed 数值状态、dispatch 与冻结 plan/control；capture 模式才额外保留 X/Y。融合接口必须新增对原始 X 的 hooks-visible 引用。

## 3. 当前调用边界与原反向流水

当前模型：`DeepseekV4SparseMoeBlock.forward` 先 `experts(flat,ids,prob)`，再 `shared_experts(residual)`，最后相加；flat 与 residual 是原始 X 的 view，没有额外 Norm/scale。只有 `.mlp.experts` 进入 EP dispatcher，shared 模块保持普通 FSDP。`dispatcher_megamoe_balanced.py` 只建立 NativeFSDP2Pool/bridge，没有旧 redispatch cell，旧 launcher 在这里未接通。

当前 native 的实际 GM saved 路径如下，不按文件头历史 P3/P4 注释猜测：

|阶段|AIC|AIV0|AIV1|
|---|---|---|---|
|入口|初始化；跨 rank helper 在 AIC 上直接 return|原进入同步|原进入同步|
|P0-A|saved BF16 通常无 GEMM|与 AIV1 一起 GatherRoutes、同步、恢复计数|与 AIV0 相同|
|P0-B|可做独立纯 GM 工作|逐专家拉取 peer 的 dY，发布 dispatch ready|恢复 replica W1/W2 panel|
|P1|dHp GEMM|SwiGLU backward，写 dZ/Hp/dp partial|同左；所有写退休后发布|
|P2/C1|每专家 dW2→被复制专家 dW1→dX|dX combine/send|P1退休后进入 owner Reduce|
|P3|补其余 dW1|P5 清空/owner remainder|owner remainder|
|P5|原 AIC 可能已结束|dX发送退休、本地/跨rank完成、dX unpermute|同左|
|P6|—|ptrD复用、dp归约/发送/unpermute、reset与退出|同左|

重要事实：

- `GatherRoutes()` **复制原始 dY**；当前 early-topk 路径在 P1 对相应梯度乘概率。不能按旧注释写成 dY×prob 后再次乘一次。
- 原始 dY 输入只读。被复用覆盖的是内部 packed `ptrDO`，saved 模式 `ptrDX=ptrDO`；shared 不得把自身 dX 写进原始 dY。
- P1 使用双 AIV；P1 后 AIV1 做 owner，P2 AIV0 做 combine，不能把其中一条当成天然空闲 lane。
- `CrossRankSyncInWorldSize()` 只执行 AIV 同步，**不能证明新增 AIC 共享结果已写完**。新增 AIC→AIV 完成协议不可省略。
- routed P5 是 TopK BF16 行在 FP32 中按原顺序求和，再 BF16 落盘；与前向通用 `TryUnpermute` 的 BF16逐项累加不是同一数值边界。

## 4. 数学、dtype 与八项参考输出

记 R 为目标 BF16 舍入；X `[M,H]`、Wg/Wu `[Fs,H]`、Wd `[H,Fs]`。首版无 bias。三次共享 Linear 的逻辑参数身份保持原样。

```text
G = R(X @ Wg.T)                    U = R(X @ Wu.T)
g = min(fp32(G), L)                u = clip(fp32(U), -L, L)
A = R(silu_fp32(g) * u)            Ys = R(A @ Wd.T)
Y = R(Yr_bf16 + Ys_bf16)
```

反向输入为同一份未乘 routing probability 的总 dY `[M,H]`：

|输出/状态|计算|完整 K|
|---|---|---|
|dA `[M,Fs]`|R(dY @ Wd)|H|
|dWd `[H,Fs]`|R(dY.T @ A)|M|
|dG `[M,Fs]`|R(fp32(dA)·u·silu′(g)·1[G≤L])|逐元素|
|dU `[M,Fs]`|R(fp32(dA)·silu(g)·1[−L≤U≤L])|逐元素|
|dWg `[Fs,H]`|R(dG.T @ X)|M|
|dWu `[Fs,H]`|R(dU.T @ X)|M|
|dXg、dXu `[M,H]`|R(dG @ Wg)、R(dU @ Wu)|各自 Fs|
|dXs `[M,H]`|R(fp32(dXg)+fp32(dXu))|逐元素|
|dXdirect `[M,H]`|R(fp32(dXr_bf16)+fp32(dXs))|逐元素|

Triton 参考把 raw G/U 和 dA 升 FP32，使用包含等号的限幅导数，最后 dG/dU 降 BF16。必须保存限幅前 raw 值，单有已 clamp 的值无法区分“原值等于边界”与“原值越界”。不擅自替换 exp/SiLU 近似、舍入模式或限幅表达式。

首版 dXg/dXu 采用**两次独立 GEMM、分别 BF16 落盘后相加**，不改成一次 K=2Fs 的拼接 GEMM；Wg/Wu 不在 Host 拼成新永久 Parameter。即使形式相同，新 GEMM 与框架 Linear 的实现仍可能有差异，需实测定位。

总验收八项：`Y, dXdirect, dP, dWr1, dWr2, dWg, dWu, dWd`；所以 backward 新接口返回 **7项**，不是8项。`dP→Router→dXrouter` 继续由框架执行，完整 MoE 输入梯度由 autograd 汇合。kernel 不加第二次 router dX。

原框架的多个 BF16 分支贡献可能以不同括号相加。新的 `dXdirect=R(dXr+dXs)` 再加 router 梯度，**不能先承诺完整框架 dX 逐位相同**。先分开检查 shared、routed、native direct merge，再检查包括 Router 的完整 dX；差异必须归因并经约定判据验收，不能临时放宽。

## 5. 与前向协作的保存 ABI：先冻结，再分别实现

提案名称 `SharedExpertsTapeV1`，不表示源码已有该类型。首版拟固定 raw 布局为 BF16 `GU[M,2Fs]`，每行 gate 在前、up 在后，ld=2Fs；A 为 `[M,Fs]` 行主序。Wg/Wu 仍独立传址，前向两个 GEMM直接写 GU 的不同列区域，不经 Host cat 权重。若前向最终采用 planar G/U，两边必须共同调整 layout_id/leading-dimension 契约并单独验收，不能让反向根据字节数猜布局。

|tape字段|KEEP|SHARED_RECOMPUTE|用途/约束|
|---|---|---|---|
|schema_version、math_id、raw_layout_id|保存|保存|严格匹配可查询 SO capability/ABI|
|M/H/Fs、dtype、strides、limit|保存|保存|实际参数形状；正有限 L；不是从 routed F 猜|
|原始本地 X `[M,H]`|保存引用|保存seed引用/对应policy句柄|原始token顺序；无额外 clone；不可用 packed X 替代|
|raw GU `[M,2Fs]`|保存真实BF16前向结果|不跨F/B保存|通过 save_for_backward，可被hooks拦截|
|A `[M,Fs]`|保存真实BF16前向结果|不跨F/B保存|dWd使用实际前向A；重算必须相同math_id|
|routed counts/state/四plan张量/control|沿原契约|沿原契约|shared保存策略不暗改 routed输入策略|
|routed dispatch tensor|仅KEEP_PACKED持有|仅KEEP_PACKED持有|REDISPATCH_X明确不跨F/B保存，由B新建目标tensor|
|前向调用身份/逻辑权重版本/plan身份|保存元数据|保存元数据|层号、microbatch/调用、optimizer数学版本；不是裸GM地址|
|shared Y、dA/dG/dU/dX|不保存|不保存|Y为输出，其余为backward临时|

数值 Tensor 全部经 `save_for_backward`，不藏进 Python ctx 普通字段、全局字典或共享池。KEEP 的 GU/A 只读，禁止覆盖成 dG/dU。允许存储 view，但要明确 offset/stride、共享 storage 与 hooks 去重；不能把 GU/A 塞进 routed opaque byte state，使 UAM 看不到归属。

原始 X 的 seed 在 recompute 模式仍可能占112MiB；不能称“recompute免X保存”。不保存共享最终 Ys，重算共享状态也不需要再计算 Ys。

版本契约分三类：前向 tape身份、前向冻结 plan/status、当前 backward 执行 generation。保存的前向 epoch只证明配对，不可当本次通信 ready；当前 bridge 已用 frozen placement 创建新 executor epoch，这个原则保留。M0需定下token在联合框架入口产生、反向live参数获取后比对的位置；同一optimizer数学版本内的梯度累积microbatch与checkpoint重放必须有效配对，FSDP存储重建不能被误判为参数更新。

共享权重在反向从 live 参数重新获取，普通和 expert 两组 FSDP AG 均须完成。禁止保存跨 reshard 的 unsharded裸地址；只检查shape也不足以证明权重版本正确。首版限定标准 F→B→optimizer-step，不支持跨step异步更新/旧图复用；框架需提供逻辑参数更新 token/事务校验，FSDP的存储重建不算数学更新，不能直接把临时tensor `_version` 当等价token。debug可在计时外用指纹核对，不能通过常驻全权重解决版本问题。

### 前向必须交给反向的交付清单

1. 共同确定 math_id 与 GU 布局、leading dimensions，保证两GEMM与激活输出数据已退休，state不在下一层被覆盖的pool内。
2. KEEP 返回 X引用、raw GU、A；RECOMPUTE 返回 X seed＋精确激活元数据及模式，不强制存 GU/A。
3. 明确 tensor 持有者、Host返回/ready边界和 checkpoint重放时的同一 schema；capture开关不得改变正常/重放保存数目。
4. 输出相加保留先 routed BF16、shared BF16 再加的边界；不要重用 GU/A 存 Y。
5. X redispatch 若启用，保留冻结 physical IDs、copies、counts、prefix/逆索引、原概率与plan控制；不能只留 logical IDs。

本设计在前向尚未实现时，可用独立参考 forward adapter 产生此 tape 测反向，adapter和正式融合前向分列，不宣称前向已完成或有提速。

## 6. 首版流水 S0/S1：分工与就绪依赖

S0 是同一 native 调用内完全串接的定位对照；S1 是本设计推荐的粗粒度前缀。两者数学、输出、保存模式完全相同，仅排程不同。第一轮 routed_input_mode=KEEP_PACKED。

### 6.1 S1 时序

```text
AIC   : [重算GU，仅RECOMPUTE] → dA → [dWd，KEEP可提前]
          ║ 与原P0部分搬运交叠       ↓等 A_READY / DZ_READY
        [dWd若未算] → dWg → dWu → dXg → dXu → 等共享向量退休
        ─────────→ 原routed P1 → 原P2/C1 → 原剩余dW1

AIV0  : 原P0-A全员Gather/同步 → 原P0-B dY dispatch
        ─────────→ 原routed P1 → 原dX combine/owner/P5/P6

AIV1  : 原P0-A全员Gather/同步 → 全部panel restore返回且退休
        → [A=activation(GU)，仅RECOMPUTE] → shared VJP(dG/dU)
        → 等两shared dX GEMM退休 → dXs合并/发布 → 原routed P1
        ─────────→ 原early owner Reduce/P5/P6

P5    : 原route dX TopK归并并BF16舍入 → 加本地dXs → 写最终dXdirect
```

具体规则：

1. AIC 的 shared helper 只做本地纯 GM 输出，不走 Fixpipe→AIV UB 握手、不访问 routed groupList/owner/stripe ready。每个 GEMM 独立 CATLASS 对象作用域、局部shape与cursor，退出前退休；不修改 `problemShape_`、cumMRows、C1 cursor或done bitmap。
2. AIC 可在原P0执行本地 dA；RECOMPUTE先做原X的共享gate/up。KEEP下 A已就绪，dA发布后可算dWd，与AIV1的VJP交叠。RECOMPUTE下等AIV1发布A_READY再算dWd，不能读未初始化A。
3. AIV1 必须完成**整个** CopyExpertsPanel，包括W1和W2；其copy对象/drain完成后才进入共享短scope。不能只因Wd本地可用就截断 routed restore。AIV0保持原逐专家dY dispatch，没有减半其既有团队或改变P0-A屏障。
4. shared向量工作只使用32条AIV1（一般写blockNum），沿现CopyExpertsPanel的映射，在GetSubBlockIdx()==1集合中采用`worker=GetBlockIdx()/GetTaskRation(), workers=GetBlockNum()`的连续逻辑号分片M×Fs；不把全64lane编号直接当半团队索引，不硬编码task ratio。AIV0此时可继续高区P0 dispatch，或进入原P1等待AIC，但没有shared任务。
5. AIV1可在RECOMPUTE先发布A_READY，让AIC开始dWd，再做VJP。VJP完成后，所有lane的dG/dU写退休，发布DZ_READY。AIC等待相应全体发布后计算三个dW和两个dX；每输出tile一个写者、完整K，无原子浮点累加。
6. 首版把dXg/dXu放两个私有GM平面，AIV1等全部AIC的DX_GEMMS_DONE，再in-place把dXg变为dXs。AIV1 drain、销毁共享TPipe/临时event后发布SHARED_VECTOR_RETIRED。
7. AIC在全部共享输出已退休、全部AIV1低UB scope已退出前，不允许进入原routed P1。AIV1随后才进入原双AIV P1。原AIV0即使已在P1等待，也不得迫使尚未就绪的AIC提前发射Fixpipe。
8. P1后AIV1的owner、P2的C1及dX、P3尾部和P5/P6保持原样。这里的共享计算在owner之前完成，不存在owner阻塞等待某个被自己占用的shared vector资源的环。

S1的代价是共享前缀推迟 routed P1、C1与owner开始时间。只有被P0覆盖的部分和省去的框架开销可能形成净收益；若共享前缀太长，它会恶化现有关键链。S0/S1必须用相同完整block B/FB比，不以共享首tile提前或AIC更忙判成功。

### 6.2 UB/event预算

现有P1/P2低区为192KiB，高区供P0 StageUb。shared AIV1 helper首版按16×256元素chunk处理，显式布局所有BF16输入/输出及FP32临时，设计上限≤128KiB低区；这是待编译/核算的约束，不是已证实占用量。若超预算，先缩chunk，不侵入P0高区。

共享与原TPipe不得同时控制同一AIV的UB；每次scope drain/Destroy后再移交。AIC不使用shared向量UB；shared纯GM GEMM不能偷偷依赖AIV FIX消费。跨阶段hard event必须归还、按新scope配对初始化，不能把旧P1 event状态当完成信号。

### 6.3 P5 dX合并的精确位置

首版 dXs留在共享私有GM平面，不预写 `gradX`。在现有 `UnpermuteBackward()` 每行先完成原TopK的FP32累加，**先Cast成原BF16 dXr**，再加载同一原始token行dXs，做BF16结果相加后写最终gradX。不能直接把dXs加到未舍入的route FP32 accumulator中只Cast一次。

复用Unpermute原输入槽前先退休其最后MTE2/Vector读者；新增加载也要受MTE2/V和输出MTE3/WAR控制。`DataResBuf_`的结果行此时保存已舍入dXr，绝不能把dXs加载到这一结果槽而覆盖它；实现需从已退休的输入槽选择另一bounded行区，或在明确预算下使用DataResFp32Buf的视图并重新建立类型/读写依赖。先保留route BF16值，再相加。当前反向不是通用forward TryUnpermute快路径，不能只修改那个helper就宣称已接入反向。新增shape快路径若出现，必须与fallback使用同一merge语义。空/零tile场景仍覆盖所有合法M行，无漏写/双加。

在接口稳定后可另测“dXu写gradX、形成dXs预填gradX”的省内存变体，但需证明所有初始化/覆盖/读写退休；不作为首版隐含别名。

## 7. 新完成协议：保持原协议，不借槽

提案 `SharedBackwardControlV1` 独立于 routed DZReady/DOReleased/DXReady、panel ready、owner ready和前向shared ready。控制区每call初始化并使用新run generation，阶段名至少为：

|事件|生产者集合|消费者|发布前必须完成|
|---|---|---|---|
|RAW_GU_READY，重算模式|全部AIC|AIV1|所有raw GU GM写出|
|DA_READY|全部AIC|AIV1|全部dA写出|
|A_READY，重算模式|全部AIV1|AIC dWd|共享A写出|
|DZ_READY|全部AIV1|AIC dWg/dWu/dX|dG/dU全部写出|
|DX_GEMMS_DONE|全部AIC|AIV1|dXg/dXu全部GM写出|
|SHARED_DW_DONE|全部AIC|联合完成检查|三份dW写出、所有GEMM对象退休|
|SHARED_VECTOR_RETIRED|全部AIV1|AIC routedP1、P5本地merge|dXs写出、共享UB/event/TPipe退休|

首版为**每phase独立arrival平面**，每个lane写该平面内自身cacheline的generation，例如`arrival[phase][producer_id]`；所有phase记录保持到本次lease完整退出。不得用单个“最新(generation,phase)”槽依次覆盖各阶段：快生产者可能覆盖RAW_GU_READY，导致慢消费者永远等不到精确phase。若后续改为单调bitmap/计数，必须另证实不会丢阶段及epoch混淆。不能把旧值`>=目标`当本次完成。零tile核心也发布本阶段arrival。consumer先校验完整生产者集合且epoch精确匹配，再执行适当acquire/fence读取payload；GM写退休→发布不能仅凭C++语句顺序。

first implementation可粗粒度all-core-ready，不引入tile stealing；fused off必须完全保留旧控制布局。对齐、Host/Device字节大小、generation wrap拒绝/安全重置、超时进度与错误上报要写CPU/构建测试。新控制区只随本次lease使用，pool释放前包括所有AIC/AIV写者/读者均已结束。

必须保留原P0本地/跨rank发布、原dY专家ready、全AIV的dZ/Hp退休、全部AIC的dW2读dO退休、P5 dX写出与ptrD读者退休、P6 dp远端写完成、reset/退出同步。shared本地完成不替代任何一个原有跨rank happens-before。

## 8. 可选 X redispatch：第二阶段独立实现

KEEP_PACKED仍是默认对照。当前balanced没有框架X重派发成本，不能为了宣称“融合redispatch”先强制增加一次通信。新模式 `REDISPATCH_X` 的收益目标首先是降低跨F/B存活量，不保证backward峰值或时间下降。

新模式前向不跨F/B持有dispatch输出，但保留原始X、原prob、冻结physical IDs/copies/counts和路由prefix/逆索引/控制。Python释放引用不够：当前Host强校验dispatch tensor存在和bytes，native地址也从它派生；新ABI必须同时调整Host校验、分配、layout和指针。

### 8.1 whole-X-ready首版，分阶段复用原SHMEM源bank

```text
RX0 当前lease/新generation；读取本层冻结计数和路由元数据
RX1 所有AIV按冻结expandedRowIdx gather X和BF16 prob到可远端访问的源bank
    → local drain/会合 → 跨rank发布完成
RX2 按冻结physical专家/source-prefix pull peer数据
    → 写本次backward新分配的packed X/prob
RX3 所有peer源读取及目的GM写出退休 → local会合 → 跨rank确认
    → 允许原P0-A用同一源bank覆盖成dY
原P0/P1/P2/C1/P5/P6继续；所有routed dW1读恢复的packed X
```

当前source SHMEM展开区是 `M*TopK*H*2`（224MiB），接源prob；不是直接远端访问普通X指针。目标dispatch是按capacity的280MiB＋40KiB。本模式复用前者但仍须分配后者作为backward临时。

复制可以复用forward的offset/GMcopy原语，不能调用带InitRouting副作用的完整forward来重新排序/规划。恢复前缀/计数操作若涉及现有window表，要先审计具体写集合并与原RestoreTokenPerExpert逐项一致，不把“调用同名函数”当证明。

不重新运行planner；不只依赖logical IDs；BF16 prob转换位置保持原bridge；有效行packed X/prob须逐位相等，capacity尾部未定义字节不作判据。所有rank对模式、阶段序列、容量/shape/参与者集合一致，空专家/空slot不能跳过RX发布/退休同步。

AIC的独立shared dA/重算GU可与RX的AIV搬运交叠；shared VJP仍等AIV1完成RX与原restore后执行。新增两处通信边界的成本计入B/FB，不能把whole-X-ready版本冒充逐专家流水。tile-ready X恢复、独立bank双缓冲只在此版本正确后另做候选。

## 9. 框架、FSDP与保存策略接线

### 9.1 联合入口与参数身份

入口提升至MoE block，由融合分支显式传shared参数，保留 `experts.__call__`，避免直接绕过expert模块hook；外部旧shared MLP、Add、旧redispatch launcher在联合模式关闭，防止两次计算/相加/重派发。旧路径和ABI保留。

提案联合Function的Tensor输入顺序：`X, ids, prob, Wr1, Wr2, Wg, Wu, Wd`，另有bridge/typed metadata。返回梯度分别对应原输入，ids和metadata返回None。新requires_grad判断包含三shared参数，覆盖只训练shared、只训练routed、只训练router及部分冻结，不能照搬旧四Tensor判断。首版可固定计算全套梯度再按needs_input_grad返回，避免不同rank局部跳通信；后续按配置一致的需求掩码优化。

Wg/Wu/Wd仍注册在 `shared_experts.{gate,up,down}_proj.weight`，保留state_dict与optimizer state；不放入experts参数树、EP shard或replica池。native shared梯度只是本rank贡献，交ordinary FSDP归约，不手工EP allreduce/除EP。routed owner SUM→expert-FSDP原顺序不变。返回dtype按原live Parameter元数据转换，不能因为reduce_dtype=FP32就任意改变native计算/落盘精度。

AG就绪后的权重若为DTensor，沿当前routed接线取得可求导的`to_local()`视图后传native，不能detach或用无梯度拷贝冒充live权重；联合Function对该输入返回local梯度，再由DTensor/FSDP原有autograd连接回Parameter。这个连接须在真实FSDP检查，不能仅看state_dict名字未改就认为正确。

### 9.2 已核实的 FSDP hook 范围

当前A3 custom配置普通和expert hook都挂在父decoder layer，EP与expert-FSDP只匹配`.mlp.experts`。首版在这个限定配置内，证明父层pre-forward/pre-backward hook均使两套live权重ready；反向跨读shared权重不能只验shape。

native FSDP会忽略hook_module，shared单独wrapped或更细的重算/预取布局也可能改变调用顺序；这些配置未经验证不能绕开shared模块后默认继续。提供显式capability/配置检查及全rank一致回退。计算流和FSDP AG流之间的等待沿原hook契约建立，不新增一个“看起来是BF16”的缓存裸指针。

### 9.3 KEEP、checkpoint、UAM的明确状态机

|模式|前向存活|反向动作|放行条件|
|---|---|---|---|
|KEEP|X引用、raw GU、实际A，经hooks|直接读取|首版基线|
|全层non-reentrant checkpoint|保存交checkpoint接管|重放层恢复同schema/tape，再按本次tape模式执行|同一数学/保存枚举，epoch重配测试通过|
|UAM shared-recompute|X seed及保存域元数据；不常驻GU/A|只本地重建GU/A，不算Ys，不重跑路由通信取激活|显式policy适配与恢复stream/生命周期验证通过|
|UAM OFFLOAD|按原policy卸载独立shared域|unpack/prefetch后等待就绪、消费、回收|首版未完成适配时回退/拒绝|

checkpoint的重放tape模式与原调用一致，不在checkpoint外额外强引用GU/A，不因capture只采一次而改变save序列。已有shared-recompute场景不能在反向既由UAM重放shared又在kernel再重算一遍；mode/token明确唯一恢复者。

**旧 `shared_experts=recompute` 规则不会自动匹配融合Function**；仅增加同名module_scope也只改变归属栈，没有创建recompute region。需要显式把policy映射到SharedTape保存/重算模式，seed与shared状态独立于routed opaque域；不允许把tuned.yaml静默变成KEEP。适配未完成时整块回退旧路径，不以未兼容配置宣称完成整网融合。普通checkpoint与UAM遵守框架现有互斥/组合规则。

## 10. Workspace与峰值预算

所有大小用实际M/H/Fs/R推导，Host/Device同一个版本化layout，512B对齐，控制cacheline分离。以下为BF16、M8192/H7168/Fs3072/R20480的数值区体积：

|区域|体积|生命周期/别名|
|---|---:|---|
|原始X|112MiB|KEEP引用或recompute seed；不clone|
|raw GU|96MiB|KEEP只读saved；RECOMPUTE私有scratch|
|A|48MiB|同上，dWd读退休前存活|
|shared dA|48MiB|VJP读取，首版独立|
|shared dG/dU|96MiB|dWg/dWu与两个dX均读完前存活|
|dXg、dXu|各112MiB|两个独立GEMM；dXg在merge后变dXs|
|共享三权重梯度输出|共126MiB|真实输出，直到框架接管，不是pool partial|
|routed packed X/prob|280MiB＋40KiB|KEEP跨F/B；redispatch变B临时，峰值未自动消失|
|现有SHMEM源展开bank|224MiB|RX/P0分阶段复用；不是新增bank|
|现有routed backward数值scratch|约640.94MiB＋flags|dX已alias dO，不可当免费shared空间|

首版不借用原routed数值区。shared新增backward私有scratch保守为 `dA48+dG/U96+dXg112+dXu112=368MiB`；RECOMPUTE另需GU/A144MiB，共512MiB（对齐/control另算）。KEEP的GU/A是saved144MiB，不在这里再次算成新增clone。dXs不再新增第三个112MiB平面。三份dW输出和两套FSDP权重同时存活也必须纳入峰值。

|缓冲|最后读者/写者|之后才能做什么|
|---|---|---|
|原始dY|shared dA/dWd及原Gather读者|只读；首版从不覆盖|
|原始X|shared dWg/dWu及RX源gather|本调用所有消费者完成后交框架回收|
|raw GU/A KEEP|VJP/dWd；可能有hooks或重放消费者|禁止写回scratch|
|dA|所有AIV1 VJP|以后可单独评估arena复用|
|dG/dU|shared dWg/dWu及dXg/dXu全部AIC|所有读退休后可复用|
|dXg/dXu|AIV1 merge|dXg转为只读dXs，dXu可待后续arena优化|
|dXs|P5逐行merge|P5读完及本调用finish后回收|
|packed dO/ptrDX|原dW2→原dX→combine协议|不授予shared任何额外alias|
|SHMEM ptrD|原P5 dX读者|原跨rank退休后才P6 dp复用|
|SHMEM RX source bank|所有peer RX读及目的写退休|才可P0覆盖为dY|

后续可按live interval做arena coloring减少scratch，但每个alias单独证明读者退休；不能根据“不跨F/B常驻”把backward临时成本说成零。还需检查连续跨层pool、prefetch depth、两套AG、SHMEM外分配和UAM CPU恢复字节，分别报告allocated/reserved/SHMEM及forward末尾、B峰值、step峰值。

## 11. 代码分工、ABI和开关

不直接扩展旧positional tuple造成Python/SO错配；新增版本化shared entry/descriptor及只读capability getter，旧无shared入口原样保留。schema/布局/math_id/mode/shape/bytes不匹配在设备启动前统一拒绝。以下是拟修改清单：

|仓库/文件|反向侧职责|与前向协作|
|---|---|---|
|native `ascend_deepep/fsdp2_bridge.py`|联合Autograd、live参数、shared tape/needs_grad/梯度返回|双方先冻结tape与输入顺序|
|native `csrc/ops/mega_moe_bf16.cpp`|7项B输出、shape/dtype/bytes、allocation/layout与全rank门禁前的检查|shared forward返回状态与layout一致|
|native `csrc/torch_classes.cpp`、`csrc/launcher.cpp`|新版本化方法、launch参数、getter|不互相改旧签名|
|native `mega_moe_bf16_backward.h`|S0/S1接线、AIV1向量任务、P5merge、ready|不修改前向A/B初始化方案|
|native 新shared backward/CATLASS helper|无owner副作用的GM GEMM/VJP/局部shape/事件|可共享基础类型，避免共同编辑大文件冲突|
|native `mega_moe_training_workspace.h`/workspace_info及Host layout|SharedWorkspaceV1、可选RX布局、生命周期|保持旧routed布局/offset不变|
|FSDPTurbo balanced dispatcher/config|shared参数透传、pool签名/模式、一致性检查|同事forward融合入口共同接口|
|FSDPTurbo UAM adapter|独立shared保存域、shared-only重算/offload适配|禁止恢复shared时重跑全MoE|
|MindSpeed DeepseekV4 block/arguments|联合调用与旧路径切换，关闭重复shared/Add/launcher|只由共同接口补丁集中修改|
|wuhan|固定子仓gitlink、配置/复现/验收报告|先子仓后根仓，不强推|

逻辑开关彼此独立：`shared_fusion=off/on`、`shared_backward_schedule=serial/prefix_p0`、`shared_state=keep/recompute`、`routed_input=keep_packed/redispatch_x`、`activation_math_id`。这些名称是提案，不能当作当前可用命令。首版off严格旧行为；共享forward是否启用、tape是否可消费与B模式必须共同验证，不能允许前向未保存状态而B偷偷猜测。pool签名增加Fs、shared布局/math_id、保存/输入/排程模式和ABI版本，不能只沿用旧routed的M/H/F；同shape跨层可以串行复用scratch，但每call的live参数、tape和generation不可缓存为上一层的引用。

双方实现前只需确认接口/文件分工，不应彼此向共同大文件独立追加另一套schema。前向尚未提交时，反向可先用参考adapter独立正确性开发；禁止擅自改同事前向方案或替对方宣称性能。

## 12. 实现顺序与验收门槛

|里程碑|范围|进入下一阶段的条件|
|---|---|---|
|M0|ABI/tape/math_id与两套FSDP hook固定|双方前后向保存/读取样例一致；Host拒绝矩阵通过|
|M1|纯本地shared backward helpers＋参考adapter|三dW、两dX支路、VJP和边界值；完整K与布局覆盖|
|M2|native S0联合串接，KEEP_PACKED|原routed精确回归＋八项输出/Router dX；全调用生命期正确|
|M3|S1前缀与P0交叠，KEEP_PACKED|相对S0新输出无差异；ready/UB/零tile与流水打点通过|
|M4|RX whole-X-ready独立版本|有效packed X/prob逐位等前向，generation/空slot/生命周期通过|
|M5|S1+RX；UAM adapter分别接入|每一新增维度独立对照，不混保存策略或遗留redispatch|
|M6|真实FSDP8/Expert-FSDP2、checkpoint/UAM与整block性能|逐步参数/梯度、生命周期与资源隔离通过，再谈晋级|

### 12.1 数值与功能矩阵

- 共享off：旧接口、默认构建、原五输出与路由生命周期逐位不变；不多分配shared缓冲、不创建新对象、不插新barrier。
- shared-only：独立reference检查dA/dG/dU、dWd/dWg/dWu、dXg/dXu/dXs；采用真实输入/原权重/dY，正确处理Fs与routed F不同。
- 激活边界：±L、等L、BF16邻近值、零/正负/极端有限数；Triton与torch路径分开；未支持的非有限输入契约明确，不拿随机值代替边界测试。
- 总Y、native dXdirect、dP、路由两dW、共享三dW共八项；同时经真实Router计算完整X梯度。固定同一X/ids/prob/权重及**外部固定dY**的算子对照中，原routed三项dP/dWr1/dWr2应逐位不变；debug单独观察dXr确认merge没有污染原分支。整网若前向Y存在已解释的数值差异，下游dY也可能变化，不能把该逐位要求不加区分地套到SGD/整网比较。
- 新GEMM与BF16加法括号差异先定位，使用项目已有独立BF16判据并在跑前固定；不能只因融合而放宽，不能用一个很宽误差门限掩盖错位/限幅mask错误。若要逐位承诺，须匹配实际算子累加/舍入后专门证明。
- 共享宽度、M尾块、零tile核心、空专家/空owner/空slot、多个热点、TopK零概率、不同groupList、容量边界；布局/stride不支持时前置拒绝。
- 多层共享pool、重复调用、热点切换、多generation、三步SGD、故意污染pool、梯度累积、仅训练shared或部分参数冻结、checkpoint重放和UAM恢复；不支持retain_graph/double backward要显式界定，不能无声破坏。
- RX有效区X/prob逐位，前向冻结plan输入SHA守恒；输入、保存或逻辑权重版本错配必须报错，不能只核shape。
- 真实普通FSDP与expert-FSDP验证参数归属/grad缩放/AG-RS顺序，确认shared梯度没有走EP owner或重复归约。

### 12.2 性能与显存

正式性能的主分母是**同输入、同保存模式的框架原shared＋当前最佳routed**，测完整MoE block F、B、直接FB；S0/S1再比较排程成本，必要时测完整step。原96lane routed JSON与历史32.952285ms都不包含shared/FSDP，不当作联合MoE分母。

每个保存模式分别设A=框架shared+KEEP_PACKED、B=融合+KEEP_PACKED、C=融合+RX；如支持D=框架shared+RX，可用于分离输入恢复贡献。整层checkpoint/UAM另分组，不能跨组比收益。重算A/GU、packing、RX、dXmerge、finish、原要求的FSDP通信都计入相应层级；hash/诊断回读/导图在外。

算子/独立block TimerOFF可沿每进程warm10、measure30、逐样本max8rank、两轮C0自重复＋四组AB/BA。保留原B门槛：四组均改善、中位收益>max(2%,2×C0波动)；FB中位正且FB/P90无超max(1%,对应波动)回归。真实FSDP/step/HBM另列口径与样本，不借单kernel成绩外推。

TimerON配对库独立进程核八输出后采联合图；新增命名shared recompute-G/U、dA、A/VJP、三dW、两dX、dXs merge、RX publish/pull/retire、P5 direct merge、各ready wait与epoch。保留原96lane/每rank及原始区间，同轨不重叠，不跨rank相减算通信；通过开关校准插桩扰动。框架trace另列两套FSDP AG/RS、Router、checkpoint/UAM恢复和完整block。

设备阶段沿既有健康/进程/root-fuser双准入、独立环境、逐进程前后检查；失败编号保留，占用/门禁失败停止，不干预外部任务。本设计本轮未启动这些实验。

## 13. 实现前必须与前向方确认的六件事

1. GU是本设计拟定的 `[M,2Fs]` gate-first row layout，还是独立planar？确定stride与ABI，不在实现时各自猜。
2. KEEP是否返回实际BF16 A及raw GU？RECOMPUTE由谁保留X seed、谁唯一负责重建？
3. 前向/反向共同math_id冻结Triton参考；默认torch路径何时另验收？
4. 融合入口如何保留experts调用、两套父层custom hook和live参数连接？逻辑权重版本token由哪层维护？
5. UAM shared保存域适配由哪一侧集中实现？未兼容配置如何一致回退，而不是静默改KEEP？
6. 新Host binding/tiling/tape/header共享文件由谁负责合并，如何避免前向与反向互相覆盖？

以上是接口协作事项，不是把已知正确性约束推迟到性能测试再决定。代码仍在共同实验分支，未来先合对应子仓，再更新根gitlink；已验收megamoe-balanced不随本设计改默认。

## 14. 源码证据索引

路径均在第2节代码根下，行号固定于上述SHA，完整哈希见随附 `SOURCE-EVIDENCE.json`。

|证据|相对路径:行|
|---|---|
|shared三Linear、总Y相加|MindSpeed-LLM/.../models/deepseek_v4/modeling_deepseek_v4.py:1826,1999–2034|
|Triton raw保存、FP32 VJP、边界等号|MindSpeed-LLM/.../ops/triton_swiglu_with_limit.py:86–171|
|当前dispatcher仅路由bridge|FSDPTurbo/.../expert_parallel/dispatcher_megamoe_balanced.py:23–107|
|父层custom hook、EP与shared区别|MindSpeed-LLM/examples/fsdp2/deepseek4/pretrain_deepseek4_flash_284b_4k_fsdp2_A3.yaml:15–56|
|FSDP与expert-FSDP hook选择|FSDPTurbo/.../fully_shard_parallel/fully_shard_parallel.py:125–152；.../expert_parallel/expert_fully_shard_parallel.py:57–100|
|旧正常tape没有X、前向clone plan、backward新epoch|native/ascend_deepep/fsdp2_bridge.py:49–109,207–257|
|packed X/prob layout、saved state|native/kernels/mega_moe_bf16/include/mega_moe_training_workspace.h:264–390|
|routed workspace别名|native/kernels/mega_moe_bf16/include/mega_moe_workspace_info.h:224–329；native/csrc/ops/mega_moe_bf16.cpp:173–213|
|原始dY只读Gather、P0 peer pull|native/kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h:608–693,810–897|
|跨rank同步只含AIV|同上:1135–1156|
|全AIV dZ/Hp退休、dO读退休、C1顺序|同上:1189–1393|
|当前P5的FP32 TopK→BF16|同上:1500–1561|
|P0/P1/early owner/P2/P3/P5/P6实际Process|同上:1847–2215|
|权重panel写退休与发布|native/kernels/mega_moe_bf16/include/mega_moe_backward_restore.h:94–154|
|checkpoint非重入入口|FSDPTurbo/fsdp_turbo/memory/recompute/recompute.py:36–45|
|UAM active-region捕获/重放与seed|FSDPTurbo/.../core/capture/saved_tensor.py:550–648；.../core/recompute/region.py:128–143；.../core/recompute/executor.py:396–479|

历史capture launch、精确routing-facts和同事稿的路径/哈希一并留证；不改写同事稿或冻结原数据。
