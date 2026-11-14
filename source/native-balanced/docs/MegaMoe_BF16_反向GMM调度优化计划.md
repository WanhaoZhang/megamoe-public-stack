# MegaMoe BF16 反向 GMM 调度优化计划

## 1. 目的与当前结论

本文说明前向已经验证的 GMM1 调度优化如何用于反向，并给出从定位、单阶段改造、设备验证到合入的执行顺序。

当前结论：

1. 前向的“自适应 N-stripe 分核”可以作为反向 GMM 的候选调度方法，但必须按反向阶段分别验证，不能直接替换反向统一的 `3_1` 调度。
2. 前向的“256 行就绪”只可能用于反向 P0-B Dispatch 到 P1 GMM2Grad 的生产者/消费者流水。P0-C、P2、P3、P4 没有相同的输入就绪关系，不应套用该协议。
3. 第一轮只优化 P1 的分核，不改变 Dispatch 就绪粒度。确认 P1 局部和反向端到端都有收益后，再评估 P2。
4. 只有 Timer 证明 P1 仍明显等待 backward Dispatch 时，才设计反向 256 行就绪。不能为了复用前向方案而提前增加计数器和同步。
5. 调度选择在构建期完成，候选集合固定且很小；GMM 任务循环中不做运行态遍历、排序或自动试跑。

## 2. 前向修改提供了什么证据

当前前向在相同默认 Shape 下的实测结果为：

| 方案 | GMM1 分核 | Dispatch 就绪 | 端到端耗时 |
| --- | --- | --- | ---: |
| `legacy` | `GemmIdentityBlockSwizzle<3, 1>` | 整 expert | 31858 us |
| `adaptive-expert` | 自适应 N-stripe | 整 expert | 31337 us |
| `adaptive-ready-first` | 自适应 N-stripe，并优先计算已就绪的 256 行块 | 256 行 | 30967 us |

由此可以拆出两段收益：

- 只改任务分核：约 521 us，约 1.64%。
- 在自适应分核上增加 256 行就绪：再降低约 370 us，约 1.18%。
- 默认高性能方案相对旧基线：约 891 us，约 2.80%。

这组数据只证明两个机制在前向默认 Shape 上有效。反向应沿用“先分离验证调度，再验证通信计算重叠”的方法，不能把前向收益直接当作反向预期收益。

前向自适应选择器的可参考部分包括：

- 以 128×256 的 L1 输出 tile 为基本任务。
- 根据 `M tile` 数、`N tile` 数和 AIC 数选择有限的 N-stripe 宽度。
- 只枚举针对 28、32、36 AIC 预先确定的少量候选，不遍历所有分法。
- 用首波需要覆盖的 A/B 地址单元估算访存局部性，并用每条 lane 可连续复用 B 的 M-tile 轮数处理同分候选。
- 选择结果只参与坐标计算，不在每个 tile 上执行复杂决策。

## 3. 反向当前数据流与适配边界

反向五个 GMM 阶段目前都使用：

```cpp
using BlockScheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<3, 1>;
```

定义位置：`kernels/mega_moe_bf16/include/mega_moe_bf16_backward_catlass_impl.h`。

| 阶段 | 计算 | 当前输入就绪条件 | 自适应分核 | 256 行就绪 |
| --- | --- | --- | --- | --- |
| P0-C Recompute | `Z = Xpacked @ W1.T` | 仅 `save=none` 执行；X/W 已可读 | 可研究，但必须和 P1 的任务归属协调 | 不适用 |
| P1 GMM2Grad | `dH/dHp = dO @ W2` | 每个 expert 等待 backward Dispatch 的完整计数 | 第一优先候选 | 后续可研究 |
| P2 GMM1Grad | `dX = dZ @ W1` | P1 的 dZ 全部完成后进入 | 第二优先候选 | 不适用 |
| P3 GMM1DW | `dW1 = dZ.T @ Xpacked` | P2 尾部完成；纯 AIC 输出 GM | 需独立模型 | 不适用 |
| P4 GMM2DW | `dW2 = dO.T @ H/Hp` | P2 尾部完成；纯 AIC 输出 GM | 需独立模型 | 不适用 |

### 3.1 P1 为什么先做

P1 是最接近前向 GMM1 的第一候选：它仍按 expert 处理路由后的行，输出 tile 经过 AIC→AIV 流水进入 SwiGLUGrad。若当前 `3_1` 造成某些核 B 地址跨度大、任务尾部不均或活跃核利用不足，自适应 N-stripe 有机会改善它。

P1 也有前向没有的约束：

- 每个 AIC 完成的 tile 在配对的 AIV0/AIV1 之间交替发布。
- `nextTargetAiv` 的奇偶跨 expert 延续，不能因坐标重排而重置。
- AIC/AIV 的 tile 顺序、事件号和 UB 槽位必须保持一一对应。
- 第一块计算前仍等待 `bwFlagDispatch_[expert]` 的整 expert 完成值。

因此，P1 的第一版只替换任务坐标映射，不修改双 AIV 交替、事件协议、UB 布局和 Dispatch 等待。

### 3.2 P0-C 为什么不能单独改

P0-C 仅在 `save_gmm1_output=none` 时重算 gate/up。源码要求 P0-C 和 P1 的 scheduler 及 `startBlockIdx` 演进保持一致：写 gate/up tile 的 AIC 随后还要产生对应 dHp tile，并发布给 AIV1 使用的就绪事件。

因此：

- 默认 `bf16` 保存模式的第一轮实验不涉及 P0-C。
- 验证 `none` 模式时，P0-C 和 P1 必须使用兼容的坐标映射和核归属。
- 如果不能证明两阶段的 tile 生产者关系不变，就保留 P0-C 的 `3_1`，同时禁止 P1 在 `none` 模式使用不兼容的新映射。

### 3.3 P2 为什么单独验证

P2 把 GMM 产生的 dX tile 通过 ping-pong UB 交给 AIV0 Combine。它可以参考自适应 N-stripe 的坐标生成方式，但必须保持：

- AIC 与 AIV0 看到相同的 tile 顺序。
- `WaitForVector`、`NotifyVector`、`WaitForCube`、`NotifyCube` 的 ping-pong 槽位不变。
- Combine 使用的 `mLoc/nLoc/actualShape` 和 GMM 输出 tile 完全一致。

P2 在进入阶段时 dZ 已经完成，不存在 backward Dispatch 的 256 行就绪问题。

### 3.4 P3/P4 为什么需要新的判断模型

P3/P4 是转置 A、L0C 直接写 GM 的权重梯度：

- P3：`{m=N, n=K, k=Me}`。
- P4：`{m=K, n=F, k=Me}`，其中 `F=N/2`。

它们的 M/N 含义、A/B 布局、输出流量都不同于前向 GMM1。前向选择器偏好的 stripe 宽度未必适合 P3/P4。只有阶段 Timer 表明 P3/P4 是主要瓶颈时，才为这两个阶段建立独立候选和评分，不直接复用前向选择结果。

## 4. 实施步骤

### 阶段 A：固定反向基线

目标是得到无 Timer 端到端基线和带 Timer 的阶段分解。记录默认 `bf16` 保存模式下：

- `AIC_GMM2GRAD`：P1 GMM2Grad。
- `AIV_SWIGLU_GRAD`、`AIV_P1_WAIT`：P1 向量计算与等待。
- `AIC_GMM1GRAD`、`AIC_P2_TAIL`：P2 GMM1Grad 与尾部等待。
- `AIC_DW1`、`AIC_DW2`：P3/P4 权重梯度。
- 完整 backward 和训练 forward+backward 端到端时间。

基线未稳定前不改 scheduler。Timer 只用于定位；最终性能结论以不带 Timer 的构建为准。

### 阶段 B：只改 P1 为 adaptive-expert

在验证分支增加反向独立的构建期配置，例如：

```text
--backward-gmm-schedule legacy
--backward-gmm-schedule adaptive-p1
```

第一版 `adaptive-p1` 只做以下改动：

1. P1 根据 `Me`、`K` 和 AIC 数，从固定候选中选择 N-stripe 宽度。
2. 生成新的 `(mTile, nTile)` 顺序。
3. 保留整 expert 的 `bwFlagDispatch_` 等待。
4. 保留 `startBlockIdx`、双 AIV 交替、事件号和 ping-pong 协议。
5. 选择器每个 expert 调用一次，不能在 tile 循环中遍历候选。

进入下一阶段的条件：

- 四项梯度精度通过。
- P1 Timer 有稳定收益，且 `AIV_P1_WAIT`、P1 尾部和慢核离群没有恶化。
- 不带 Timer 的 backward 或训练端到端有可重复收益。

若只有 P1 局部变快而端到端无收益，停止扩展该方案，先确认后续阶段或跨 rank 尾部是否吞掉收益。

### 阶段 C：单独评估 P2 adaptive-expert

在 P1 已确认后增加 `adaptive-p1-p2` 对照，或者先在独立实验配置中只改 P2。P2 仍按完整阶段就绪，不增加行级计数器。

重点检查：

- `AIC_GMM1GRAD` 是否下降。
- `AIC_P2_TAIL` 和 AIV0 Combine 等待是否增加。
- tile 重排后是否让 Combine 的远端写回更集中，造成尾部变差。

P2 无端到端收益时，生产方案只保留 P1 的有效改动。

### 阶段 D：按 Timer 决定是否研究 P3/P4 和 P0-C

- P3/P4 占比高：分别做 task/address 枚举，建立适配转置 A 和 GM 输出的候选评分。
- `save=none` 是必须支持的高频场景：联合设计 P0-C/P1 的同核 tile 映射。
- 默认训练使用 `bf16` 保存且 P0-C 不执行：不要让 P0-C 的复杂性阻塞 P1/P2 的收益合入。

### 阶段 E：最后评估 backward Dispatch 256 行就绪

只有以下证据同时成立时才进入：

1. `AIC_BW_DO_WAIT` 明显占据 P1 首块或前几个 tile 的时间。
2. backward Dispatch 和 P1 在时间线上存在可利用的重叠窗口。
3. adaptive-expert 已经稳定，没有慢 group 和长尾回归。

需要新增的协议至少包括：

1. backward Dispatch 在对应 256 行 payload 完成 MTE3 可见性后发布行计数。
2. P1 根据当前 `mTile` 等待覆盖该 tile 的 ready block，而非等待整个 expert。
3. task mapping 优先顺序必须匹配 backward Dispatch 的实际发布顺序，不能直接假设前向的偶数 ready block 先到。
4. workspace 中行计数的布局、对齐、初始化、跨迭代复用和最终整 expert 完成值必须明确。
5. 保留 expert-ready 回退，供不均衡或新 Shape 出现退化时使用。

如果增加行级同步后 P1 出现稳定慢 group、端到端变差，或必须加入全核同步才能稳定，则停止该方案，生产默认保持 adaptive-expert。

## 5. 选择器设计约束

反向选择器沿用前向已经证明可控的原则：

- AIC 数只覆盖 28、32、36 三种固定集合。
- 每种核数只评估 2～4 个预定义 stripe 候选。
- 评分使用整数运算，输入仅为阶段的 `mTiles/nTiles/blockNum`。
- 不读取运行时性能计数，不试跑候选，不根据本轮耗时重新选择。
- 无合适候选或 Shape 太小时回退 `3_1`。
- P1、P2、P3、P4 分别选择，不能用一个阶段的结果代表所有阶段。

建议评分依次考虑：

1. 一轮能激活多少 AIC，避免尾波只剩少量核。
2. 每个 AIC 的 tile 数最大值与最小值之差。
3. A tile 和 B tile 的重复读取范围。
4. 连续 B 列块长度和 GM 地址跳变次数。
5. 与配对 AIV 的消费顺序是否保持连续。

## 6. 验证矩阵与合入条件

每个候选先过轻量验证，再扩展泛化范围。

### 6.1 第一轮

- AIC：32。
- Shape：`M=4096, N=8192, K=7168, TopK=16, expert/rank=4, EP=8`。
- 路由：balanced，固定 seed。
- 对照：legacy 与单一候选，交错顺序各跑 3 次。
- 指标：四项梯度精度、阶段 Timer、无 Timer 端到端。

### 6.2 轻量泛化

- `M=2048, N=8192, K=7168`。
- `M=4096, N=4096, K=7168`。
- `M=4096, N=8192, K=4096`。
- 一组 imbalanced 路由。
- 一组 28 AIC 和一组 36 AIC。

### 6.3 保存模式

- `bf16`：默认路径，必须通过。
- `mxfp8`：若当前产品配置支持，必须验证精度与 P1 行为。
- `none`：会执行 P0-C，必须单列验证，不能用默认模式结果代替。
- 非零 `swiglu-limit`：验证保存格式和反向裁剪梯度路径。

### 6.4 合入门槛

- `grad_x`、`grad_topk_weights`、`grad_weight1`、`grad_weight2` 全部通过现有阈值。
- 默认 Shape 三次交错测试均不劣于 legacy，并有稳定端到端收益。
- 没有新增固定慢 group、跨 rank 长尾或 watchdog 问题。
- 轻量泛化允许小幅波动，但不能出现明显退化；退化 Shape 必须有明确回退条件。
- 生产分支只保留验证有效的构建期选项、必要测试和说明，不带定位模式、额外 Timer 或临时日志。

## 7. 当前基线采集命令

以下命令只采集当前反向 `3_1` 基线，不代表新调度已经实现。

### 7.1 带 Timer 的阶段分解

```bash
cd /home/x00610564/ascend_deepep_hkq
git switch train-bf16-forward-stable
git pull --ff-only

bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer

BF16_BWD_DIR=test/megamoe_bf16_test/output/backward_schedule_baseline_$(date +%Y%m%d_%H%M%S)
python3 test/megamoe_bf16_test/profile_mega_moe_bf16.py \
  0,1,2,3,4,5,6,7 \
  --phase backward \
  --m 4096 --k 7168 --n 8192 \
  --topk 16 --expert-per-rank 4 \
  --routing-mode balanced \
  --warmup 10 --loops 50 \
  --capture-backward-timer \
  --profile-ranks 0,1,2,3,4,5,6,7 \
  --profile-dir "$BF16_BWD_DIR"

python3 test/megamoe_bf16_test/parse_backward_timer.py \
  --dir "$BF16_BWD_DIR" \
  --rank-start 0 --rank-end 7 \
  --summary --per-expert \
  --r 65536 --k 7168 --f 4096
```

这里 `r=65536` 对应 balanced 默认 Shape 的 `M×TopK=4096×16` 路由行数；`f=N/2=4096`。

### 7.2 无 Timer 的最终性能构建

```bash
cd /home/x00610564/ascend_deepep_hkq
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
```

无 Timer 性能测试必须重新执行这条构建命令，不能只在运行命令里去掉 Timer 参数。

### 7.3 反向精度

先用较小 Shape 做完整 CPU golden：

```bash
bash test/megamoe_bf16_test/test_mega_moe_backward.sh \
  0,1,2,3,4,5,6,7 \
  --mode precision \
  --m 64 --k 512 --n 1024 \
  --topk 16 --expert-per-rank 4 \
  --routing-mode balanced \
  --save-gmm1-output bf16 \
  --check-intermediates
```

主 Shape CPU golden 很慢时，先保留小 Shape 完整精度，再用主 Shape 的阶段 Timer 和无 Timer 性能验证；准备正式合入时仍需按项目现有大算子精度流程完成主 Shape 覆盖。

## 8. 分支与提交边界

建议先在 `deepep_moe_xx` 或新的反向验证分支实现和调测，每次提交只包含一个阶段的一个变量。满足对应阶段的合入门槛后，再把最小代码迁入 `train-bf16-forward-stable`。

稳定分支最终建议保留：

- 反向 `legacy` 回归方案。
- 经验证的反向自适应方案。
- 若行级 ready 确实改善反向端到端，再保留一个明确命名的高性能组合。

稳定分支不保留：

- 只用于定位的隔离模式和人为 delay。
- 大量逐 tile 打点。
- 运行态遍历候选。
- 需要全核同步才能掩盖慢 group 的调度方案。

## 9. 执行时的停止规则

1. 一次只改变一个阶段或一个就绪协议。
2. Timer 变好而无 Timer 端到端不变时，不继续叠加优化。
3. 出现新的固定慢 group 时，先恢复 expert-ready 和旧映射做对照，不用全核同步当正式修复。
4. 任何 ready-first 方案都必须证明发布顺序与消费顺序一致。
5. 未覆盖 `save=none` 前，不宣称 P0-C 已适配。
6. P3/P4 未单独建模前，不把前向 selector 作为其默认值。
