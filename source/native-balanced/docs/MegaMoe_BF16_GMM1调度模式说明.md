# MegaMoe BF16 GMM1 调度模式说明

## 1. 目标

普通 BF16 forward 的 GMM1 调度同时涉及两个选择：

1. GMM1 tile 如何分配给 AIC；
2. GMM1 等待整个 expert，还是在 Dispatch 每完成 256 行后提前启动。

当前保留四种构建模式用于生产默认和设备 A/B，但运行时 ready 协议只有整个 expert 与 Ready256 两种。生产默认使用 `adaptive-auto`。

## 2. 模式总览

| 模式 | GMM1 任务映射 | Dispatch ready 粒度 | 定位 |
|---|---|---|---|
| `adaptive-auto` | 自动选择 adaptive ready-first 或 legacy `3_1` | 自动选择 256 行或整个 expert | 生产默认 |
| `adaptive-ready-first` | adaptive N-stripe，并优先访问先 ready 的 M 块 | 256 行 | 强制高性能流水方案 |
| `adaptive-expert` | adaptive N-stripe | 整个 expert | 隔离 adaptive 映射收益的诊断模式 |
| `legacy` | `GemmIdentityBlockSwizzle<3, 1>` | 整个 expert | 稳健回退和旧基线 |

默认构建不需要额外参数：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
```

等价于：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule adaptive-auto
```

模式是构建时配置。测试命令不会改变已加载动态库的调度模式。

## 3. `adaptive-auto`

`adaptive-auto` 先使用调用级 Shape 选择 Ready256 候选：

```cpp
expertPerRank < 8 &&
m * topk >= expertPerRank * 2 * 256
```

左侧限制完整 expert ready 的启动暴露不超过 `Tdispatch/8` 时直接使用 ExpertReady；第二项要求严格均衡下每个本地 expert 预期至少有 512 行。

满足调用级条件后，每个 rank 只检查自己的第一个非空 expert：实际行数不少于 512 时，该 expert 使用 Ready256；实际行数不足 512 时，该 rank 整次调用使用 ExpertReady。后续 expert 一律使用 ExpertReady。AIV Dispatch 生产者和 AIC GMM1 消费者调用同一个判断函数，不会产生 rank 内协议不匹配。

Ready256 使用以下组合：

```text
adaptive-ready-first + 256-row ready
```

调用级或 rank 级条件不成立时选择完整的稳健组合：

```text
legacy 3_1 + whole-expert ready
```

调用级判断只使用 invocation-wide Shape 参数；rank 级判断复用 `UpdateGroupParams()` 已经得到的实际 expert 行数，不增加 GM 读取或路由表遍历。不同 rank 可以选择不同协议，因为 row-ready counter 只在本 rank 的 Dispatch AIV 与 GMM1 AIC 之间使用。

默认 Shape `M=4096, TopK=16, EP=8, AIC=32` 时，`dispatchBlockNumPerEP=4`：

| 每卡专家数 | 预期每 expert 行数 | 自动选择 |
|---:|---:|---|
| 4 | 16384 | 首个非空 expert 满 512 行时 `adaptive-ready-first + 256` |
| 8 | 8192 | `legacy 3_1 + expert` |
| 16 | 4096 | `legacy 3_1 + expert` |

专家较少时，启动 expert 可以在后续数据仍在 Dispatch 时提前启动 GMM1。专家数达到 8 后，完整 expert ready 的理论启动暴露已经不超过 `Tdispatch/8`，默认避免额外的分段发布和 counter 轮询。

## 4. `adaptive-ready-first`

强制构建命令：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule adaptive-ready-first
```

该模式包含两项必须配合的设计：

- adaptive N-stripe 根据当前 expert 的 M tile 数、N tile 数和 AIC 数选择 N 方向条带宽度；
- ready-first 将 M tile 按预计的 Dispatch 发布轮次排列。对齐且至少包含两个 256 行块的 producer partition 使用 phase-major 顺序，例如 512 行 partition 对应 `0,2,4,...,1,3,5,...`；小于 512 行或不规则 partition 使用连续块顺序。

GMM1 的计算 tile 仍为 128 行。一个 256 行 ready 块对应两个 GMM1 M tile。Dispatch 完成该块的输入和 routed TopK weight 后更新 row-ready counter，GMM1 随后可以开始计算，不必等待整个 expert。

该模式仍只允许第一个非空且实际行数不少于 512 的 expert 使用 Ready256；后续 expert 等待完整 expert。强制模式用于绕过调用级自动条件，不绕过这个 rank 级安全范围。

## 5. `adaptive-expert`

强制构建命令：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule adaptive-expert
```

该模式使用 adaptive N-stripe，但等待整个 expert 完成。它主要用于把收益拆成两部分：

```text
legacy
  -> adaptive-expert              观察任务映射收益
  -> adaptive-ready-first + 256   观察细粒度 ready 和流水收益
```

它不是专家较多场景的默认回退。实测 random routing、每卡 16 expert 时，adaptive 映射可能比原始 `3_1` 更慢，因此自动模式在不满足流水条件时直接回退到完整 legacy 组合。

## 6. `legacy`

强制构建命令：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 \
  --gmm1-schedule legacy
```

该模式保持原有行为：

- 使用 `GemmIdentityBlockSwizzle<3, 1>`；
- Dispatch 完成整个 expert 后发布完成计数；
- GMM1 对每个 expert 只等待一次，不查询 256 行 row-ready counter。

它适合每个 expert 行数较少、细粒度发布无法形成有效流水的场景，也是性能回归和问题定位的稳定基线。

## 7. 为什么删除 `legacy-ready`

已删除的实验模式 `legacy-ready` 使用：

```text
legacy 3_1 mapping + 256-row ready
```

这两个部分不匹配。`3_1` 的 M tile 访问顺序没有按照 256 行块的发布顺序组织，AIC 可能先取得一个尚未 ready 的 tile 并停在该任务上，而其他已经 ready 的 tile 无法立即执行，形成队头阻塞。

一组 random routing、每卡 4 expert 的结果为：

| 模式 | 端到端时间 |
|---|---:|
| `legacy` | 31657 us |
| `legacy-ready` | 32474 us |

因此 256 行 ready 只与 ready-first 映射配套使用，不再提供 `legacy-ready` 构建选项。Ready256 只为整次调用保留并清零一套 row counter，所选启动 expert 不依赖本地 expert 编号。

## 8. 已有性能依据

random routing、每卡 4 expert 的一组结果：

| 模式 | 端到端时间 | 相对上一项 |
|---|---:|---:|
| `legacy` | 31657 us | 基线 |
| `adaptive-expert` | 31519 us | -138 us |
| 旧版全 expert `adaptive-ready-first + 256` | 31407 us | -112 us |

这组历史结果表明 adaptive 映射和细粒度 ready 都有独立收益，但不能直接替代新版“仅启动 expert Ready256”的设备回归。

random routing、每卡 16 expert 的一组结果：

| 模式 | 端到端时间 |
|---|---:|
| adaptive 路径 | 33981 us |
| `legacy` | 33299 us |

这组结果表明 adaptive 映射并非对所有 expert 数都有收益，也是 `adaptive-auto` 在专家较多时回退到 legacy 的直接依据。以上数据只用于说明当前选择原因，最终性能仍需在目标 Shape、routing 分布和设备环境中复测。

## 9. 代码位置

- 构建模式及宏定义：`CMakeLists.txt`
- 构建脚本参数：`scripts/build_mega_moe_bf16.sh`
- 自动选择条件：`kernels/mega_moe_bf16/include/mega_moe_training_workspace.h`
- Dispatch 和 GMM1 共用的协议选择：`kernels/mega_moe_bf16/include/mega_moe_bf16.h`
- adaptive、ready-first 和 legacy tile 坐标选择：`kernels/mega_moe_bf16/include/mega_moe_bf16_catlass_impl.h`
