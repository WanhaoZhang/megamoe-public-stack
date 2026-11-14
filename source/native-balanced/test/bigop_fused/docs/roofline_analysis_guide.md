# Roofline 达成率分析指南

MegaMoE BF16 反向的 roofline 达成率（实测 TFLOPS ÷ 峰值 410.4 TFLOPS）是
衡量"GEMM 有没有吃满 cube"的核心指标。本文档记录用什么命令测、怎么读结果、
哪些 shape 参数影响达成率以及如何验证。

---

## 1. 快速查看单组 shape 的 roofline

```bash
# 生产 shape（单机 8 卡）——预期 roofline ~95%+
NPROC=8 BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
  bash test/bigop_fused/run_speed.sh 4096 3 10
```

输出中的 roofline 行：

```
[seq=4096] v2 bwd roofline: event=76.80ms, 16RFK floor=75.03ms
           -> ~401 TFLOPS (98% of roofline); event-floor=1.77ms overhead
```

| 字段 | 含义 |
|------|------|
| `event` | 实测反向耗时（Event 计时窗口） |
| `floor` | 理论最短时间 = 16RFK ÷ 410.4 TFLOPS |
| `TFLOPS` | 实际达到的算力 = 16RFK ÷ event |
| `% of roofline` | **达成率 = floor / event × 100** |
| `overhead` | event − floor（非 GEMM 开销：等待/launch/fill） |

---

## 2. 批量扫描：看哪些参数影响 roofline

### 2.1 K（hidden size）轴——最关键的维度

K 决定 **tile 数据复用率**（K-reuse = K/256，每片 B 矩阵被 MMAD 复用的拍数）。
K < 2048 时复用不足，cube 大部分时间在等数据搬运。

```bash
SPEED_CASES="4096:1024:3072:16:8 4096:2048:3072:16:8 4096:4096:3072:16:8 4096:7168:3072:16:8" \
NPROC=2 BIGOP_TIMING=1 \
bash test/bigop_fused/run_speed_matrix.sh 3 10
```

预期（tile 分析预测，实际以跑出来的为准）：

| K | K-reuse | 预期 roofline% | 原因 |
|---:|:---:|:---:|---|
| 1024 | 4 拍 | **~40-50%** | recompute/dHp/dW2 三个 GEMM 复用不足 |
| 2048 | 8 拍 | ~70-80% | 刚过门槛 |
| 4096 | 16 拍 | ~90% | 复用充足 |
| 7168 | 28 拍 | **~95%+** | 生产 shape 水平 |

### 2.2 F（ffn size）轴——次要维度

F 决定 GEMM 的 N 维（tile 网格宽度/负载均衡），**不影响计算强度**。
K 足够大时 F 的影响很小。

```bash
SPEED_CASES="4096:7168:1024:16:8 4096:7168:2048:16:8 4096:7168:3072:16:8 4096:7168:4096:16:8" \
NPROC=2 BIGOP_TIMING=1 \
bash test/bigop_fused/run_speed_matrix.sh 3 10
```

预期：F 从 1024 → 4096 变化时 roofline% 差异 < 5pp（对比 K 轴的 55pp）。

### 2.3 seq（M）轴——影响 tile 数量

M 决定行带数（R = M×topK），影响 tile 总数 / AIC 负载均衡。
M < 512 时 tile 太少（每 AIC 分不到 4 个），roofline 急剧下降。

```bash
SPEED_CASES="128:7168:3072:16:8 512:7168:3072:16:8 2048:7168:3072:16:8 4096:7168:3072:16:8 8192:7168:3072:16:8" \
NPROC=2 BIGOP_TIMING=1 \
bash test/bigop_fused/run_speed_matrix.sh 3 10
```

### 2.4 默认矩阵（seq/F/topK 三轴，K 固定）

```bash
# EP2（K=1024 小号矩阵）
NPROC=2 BIGOP_TIMING=1 bash test/bigop_fused/run_speed_matrix.sh 3 10

# EP8（K=7168 生产矩阵）——需要 8 卡
NPROC=8 BIGOP_TIMING=1 bash test/bigop_fused/run_speed_matrix.sh 3 10
```

---

## 3. 结果在哪看

```
speed_matrix/<时间戳>/
├── summary.csv           全部配置 × 指标（Excel 打开，看 v2_roof_pct_roofline 列）
├── speed_matrix.png      6 联图（面板③ = roofline% 对比柱状图）
└── case_XX_*.json        每组的结构化数据
```

面板③读法：每根柱 = 一组 shape 的 roofline%，95% 虚线是参考线。
低于 95% 的柱 = 该 shape 的 GEMM 没吃满 cube。

---

## 4. 影响因素速查

| 参数 | 影响 | 门槛 | 主/次 |
|------|------|------|:---:|
| **K (hidden)** | tile 数据复用（计算强度） | K ≥ 2048（8 拍） | **主** |
| **F (ffn)** | N 维 tile 数（负载均衡） | F ≥ 1024 | 次 |
| **M (seq)** | 行带 → tile 总数/AIC 均衡 | M ≥ 512 | 次 |
| **topK** | 行带放大（R=M×topK） | 不直接（约掉） | 弱 |
| **EPR** | 每专家行数 → tile 尾差 | M×topK/EPR ≥ 128 | 弱 |
| **EP** | maxRecv → workspace（不影响 GEMM 效率） | — | 无 |

核心判据（两个同时满足 → roofline ≥ 90%）：

```
① K-reuse = K/256 ≥ 8          （计算强度够，cube 不饿）
② tiles/AIC = (R/128 × N/256)/32 ≥ 4 （tile 够分，负载均衡）
```

---

## 5. kernel 内部逐 GEMM 看（需要 --enable-timer 构建）

整体 roofline 低时，用分阶段表定位是哪个 GEMM 慢：

```bash
# 重编（有额外开销，只做归因用，不做 speedup 报告）
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer

MEGAMOE_TRACE=1 \
NPROC=8 BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
  bash test/bigop_fused/run_speed.sh 4096 3 10
```

输出分阶段表：

```
stage               max_us   %KERNEL   TFLOPS   % of roofline
AIC_RECOMPUTE*8     20500    26.7%    ~375     91%
AIC_GMM2GRAD*8      10300    13.4%    ???      ← 含等待（等 AIV1 epilogue）
AIC_GMM1GRAD*8      20500    26.7%    ~400     97%
AIC_DW1*8           20500    26.7%    ~400     97%
AIC_DW2*8           10300    13.4%    ~400     97%   ← 纯 GEMM 对照
```

读法：
- **DW2 97%** = 纯 GEMM（无握手）能打满 → GEMM 本体没问题
- **GMM2GRAD 低** = 不是 MMAD 慢，是 AIC 在等 AIV1 释放 UB（单缓冲握手）
- 各 stage 的 `% of roofline` 乘以其 RFK 权重 → 整体 roofline

---

## 6. 常见低 roofline 的原因

| 现象 | 原因 | 验证方法 |
|------|------|----------|
| roofline ~40% | K < 2048（tile 复用不足） | 换 K=7168 对比 |
| roofline ~60% | fill fix 未合入（train 上还有 12GB zeros） | 检查 csrc 是 zeros 还是 empty；合入 fill-fix-v2 |
| 单个 GEMM 低 | P1 单缓冲握手（AIC 等 AIV1） | MEGAMOE_TRACE 看 GMM2GRAD vs DW2 |
| 整体低 + overhead 大 | workspace fill / launch / 同步 | 看 event-floor 差值 |
| 小 M 时急剧下降 | tile/AIC < 1（行带不够分） | 增大 M 或 topK |
