# bigop_fused — MegaMoE BF16 融合算子 vs 小算子链 测试套件

本目录包含 MegaMoE BF16 融合算子（BigOpV2）与纯 torch 小算子链（BigOp golden）
的精度门禁与性能对比全套工具。所有工具共享同一组 `BENCH_*` 环境变量注入 shape，
支持单机和多机。

---

## 快速开始

```bash
# 精度门禁（最常用，跑一组 shape 矩阵，PASS/FAIL）
bash test/bigop_fused/run_precision.sh

# 性能对比（v2 vs bigop，fwd/bwd/fwd+bwd 三组计时 + speedup）
NPROC=8 bash test/bigop_fused/run_speed.sh 4096 3 10

# 生产 shape 示例（单机 8 卡）
NPROC=8 BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
  bash test/bigop_fused/run_speed.sh 4096 3 10
```

---

## 工具导航：什么场景用什么

### 精度验证

| 命令 | 场景 | 耗时 |
|------|------|------|
| `bash run_precision.sh` | 日常门禁（6 组 shape 矩阵，矩阵模式复用进程） | ~5 min |
| `bash run_precision.sh both train` | 显式指定：前后向都测 + train 模式 | ~5 min |
| `PRECISION_ISOLATE_CASES=1 bash run_precision.sh` | 排查跨 case 状态污染（逐 case 独立 torchrun） | ~20 min |
| `PRECISION_CASES="4096:7168:3072:64:8" bash run_precision.sh` | 只跑指定 shape | ~2 min |

### 性能对比

| 命令 | 场景 |
|------|------|
| `bash run_speed.sh [seq] [warmup] [iters]` | 单组 shape 性能对比 |
| `bash run_speed_matrix.sh [warmup] [iters]` | 矩阵扫描（seq / F / topK 各轴），自动出图 |
| `BIGOP_TIMING=1 bash run_speed.sh ...` | 小算子侧分步表 + 反向分解行 |
| `MEGAMOE_TRACE=1 bash run_speed.sh ...` | 融合算子前向 kernel 总时间、反向分阶段表和 Chrome Trace（需 `--enable-timer` 构建） |
| `BIGOP_TIMING=1 MEGAMOE_TRACE=1 bash run_speed.sh ...` | 双侧全开（完整归因） |
| `BENCH_JSON_OUT=result.json bash run_speed.sh ...` | 结构化 JSON 输出（矩阵扫描用） |

### 离线分析（不需要 NPU）

| 命令 | 场景 |
|------|------|
| `python ep_probe.py` | EP 扩展推算：通信时间、speedup 上限随 EP 变化 |
| `python roofline_probe.py` | 计算 vs 通信赛跑图：crossover 点、regime 地图 |
| `python roofline_probe.py --bw-cage 50 --f 2048` | 模拟慢网络看通信何时成为瓶颈 |
| `python plot_speed_matrix.py --dir <dir>` | 矩阵扫描结果聚合出图 + CSV |

### 多机（双机/多机）

| 命令 | 场景 |
|------|------|
| 机器 A: `NNODES=2 NODE_RANK=0 MASTER_IP=<IP> NPROC=8 bash run_speed.sh ...` | 主节点 |
| 机器 B: `NNODES=2 NODE_RANK=1 MASTER_IP=<IP> NPROC=8 bash run_speed.sh ...` | 从节点 |
| 同理 `run_precision.sh` | 双机精度门禁（设 `PRECISION_CASES` 指定单组） |

多机检测：设了 `NODE_RANK` 就自动切换到多机模式，不设就是单机。位置参数两种模式一致。

### 一键多机（scripts/batch_launch.sh）

上面的方式需要逐台 SSH 手动执行。`scripts/batch_launch.sh` 在一台控制机上统一拉起
`HOST_ORDER` 的前 N 台（第一台固定为 node rank 0 / MASTER_IP），自动完成
预检 → 共享盘探测 → 构建 → 校验 → 运行，替代手动多机流程：

```bash
export HOST_ORDER=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4   # 第一台即 rank 0
export CODE_DIR=/root/ascend_deepep_hkq                  # 每台机器上的仓库路径
export REMOTE_USER=root SSH_AUTH=password                # 免密则 key
export CONDA_BASE=/home/anaconda3 CONDA_ENV=megamoe      # 可选：节点上激活的环境
export ASCEND_ENV=/opt/cann/set_env.sh                   # 可选：CANN 环境
export BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8

bash scripts/batch_launch.sh --plan 4           # 只看计划，不 SSH
bash scripts/batch_launch.sh --prepare-only 4   # 预检+构建+校验，不跑测试
bash scripts/batch_launch.sh 4                  # 4 机 32 卡正式跑 run_speed.sh
```

- CODE_DIR 在共享盘上时仅 rank 0 构建一次，其余节点通过 `PYTHONPATH` 直接加载
  共享产物（优先于环境里残留的其他 checkout 的 ascend_deepep 安装）；
  每节点独立 checkout 时各节点并行构建（自动探测，`BUILD_LAYOUT` 可强制）。
- `SYNC_GIT=1` 让节点 `git fetch` 并 checkout 控制端当前 commit（需先 push）；
  默认校验各节点 commit 与控制端一致，不一致则停止（`ALLOW_SHA_MISMATCH=1` 放行）。
- 控制端日志在 `./launch_logs_<run_id>/node_<r>_<ip>_<stage>.log`；
  测试结果留在各节点 CODE_DIR 下（多机时 `run_speed.node<r>.log` 按节点区分）。
- 自定义命令模板示例：
  `bash scripts/batch_launch.sh 2 'NODE_RANK={NODE_RANK} bash test/megamoe_bf16_test/run_backward_dual_node.sh {MASTER_ADDR}'`
- 完整参数说明见 `bash scripts/batch_launch.sh --help`。

---

## 环境变量

### Shape 注入（所有 runner 共用）

| 变量 | 含义 | 默认 | 约束 |
|------|------|------|------|
| `BENCH_HIDDEN` | K（hidden size） | 1024 | %256==0 |
| `BENCH_FFN` | F（SwiGLU 输出宽，N=2F） | 2048 | N%512==0 |
| `BENCH_EXPERTS` | E 总专家数 | 16 | E%EP==0 |
| `BENCH_TOPK` | topK | 2 | <=E |
| `BENCH_SEQ` / `BENCH_SEQS` | token 数（单组/逗号多组） | 128 | <=16384 |
| `NPROC` | 每节点卡数 = EP 并行度 | 2 | |

### 内存

| 变量 | 含义 | 默认 |
|------|------|------|
| `BENCH_LOCAL_MEM_BYTES` | SHMEM 对称窗口大小 | 1 GiB |

对称窗口需求 ≈ `2 × M × topK × K × 2B`（dispatch + combine 两块）。
启动时会打印估算；超限时提示建议值。大 topK 或大 M 需要调大。
速查（M=4096, K=7168）：topK=8 → 1GiB 够；topK=16 → 2GiB。

### 观测开关

| 变量 | 含义 | 前提 |
|------|------|------|
| `BIGOP_TIMING=1` | 小算子分步表 + 分解行 | 无（默认构建即可） |
| `MEGAMOE_TRACE=1` | 前向 kernel 总时间、反向分阶段表和 Chrome Trace 时间线 | `.so` 需 `--enable-timer` 构建 |
| `BENCH_JSON_OUT=<path>` | 结构化 JSON 结果 | 无 |
| `BIGOP_BWD_RECOMPUTE=0` | 小算子反向用旧口径（12 RFK，仅对比） | 无（默认 1 = 16 RFK 重计算） |

---

## 文件说明

### 入口层

| 文件 | 作用 |
|------|------|
| `run_speed.sh` | 性能对比入口（单机/多机统一） |
| `run_speed_matrix.sh` | 矩阵扫描入口（多组 shape 批量跑） |
| `run_precision.sh` | 精度门禁入口（矩阵/隔离两种模式自动选择） |

### 核心层

| 文件 | 作用 |
|------|------|
| `test_speed.py` | 性能 bench 主体（v2 vs bigop + event/wall 双口径 + roofline + 分解 + JSON） |
| `test_precision.py` | 精度主体（四组梯度对比：dx/dw1/dw2/dtopk；含矩阵模式） |
| `golden.py` | BigOp 小算子标杆（纯 torch 手写 fwd+bwd，BIGOP_BWD_RECOMPUTE 开关） |
| `fused.py` | BigOpV2 封装（调融合算子，diagnostic_holder 透传） |
| `make_fake_moe_layer.py` | 共享的 MoELayer 构造（v2 和 bigop 用同一份路由） |

### 观测层

| 文件 | 作用 |
|------|------|
| `timing.py` | StepTimer + step_stats()（小算子侧逐步计时） |
| `MEGAMOE_TRACE.py` | kernel AscendTimerV2 解码（MEGAMOE_TRACE=1 时输出分阶段表） |
| `plot_speed_matrix.py` | 矩阵扫描结果聚合 → 6 联图 + summary.csv |

### 离线分析

| 文件 | 作用 | 输入 |
|------|------|------|
| `ep_probe.py` | EP 扩展推算（四级网络拓扑，通信时间 + speedup 上限） | 带宽参数（CLI） |
| `roofline_probe.py` | 计算 vs 通信赛跑图（crossover / ceiling / regime / 绝对时间） | 带宽参数（CLI） |

---

## 输出解读

### run_speed 的三行关键输出

```
[seq=4096] v2 bwd:        event=76.80ms  wall=82.10ms  (iters=10)
[seq=4096] bigop bwd:     event=75.20ms  wall=80.50ms
[seq=4096] v2/bigop bwd speedup: event=0.98x  wall=0.98x
```

### v2 roofline 行（融合算子的 GEMM 效率）

```
[seq=4096] v2 bwd roofline: event=76.80ms, 16RFK floor=75.03ms
           -> ~401 TFLOPS (98% of roofline); event-floor=1.77ms overhead
```

- `floor` = 纸面 FLOPs ÷ 410.4 TFLOPS（理论最短时间）
- `98% of roofline` = GEMM 贴着 cube 跑，几乎无空转
- `overhead` = event − floor（非 GEMM 开销：等待、launch、残余 fill）

### bigop 分解行（小算子的成分拆解）

```
[seq=4096] bigop bwd 分解: GEMM=46.70ms(75.2%) ≈398 TFLOPS(97% of roofline)
           | alltoall=9.42ms(15.2%) | other=6.01ms(9.7%) | 分步合计=62.13ms
    cube : 非cube = 75.2% : 24.8% -> 理论加速上限 ≈ 1.33x
```

### kernel 分阶段表（MEGAMOE_TRACE=1）

```
stage                    AIC_max   AIV_max   %KERNEL  note
AIC_RECOMPUTE*8          20500          -    26.7%  ≈375 TFLOPS（91%）
AIC_GMM2GRAD*8           10300          -    13.4%  ← 注意：含等待
AIV_SWIGLU_GRAD*8            -      18000    23.4%
```

**注意**：`AIC_GMM2GRAD` 和 `AIV_COMBINE` 桶里含有等待时间（等对端核释放
UB），不全是纯 GEMM/搬运——看 TFLOPS 列时要记住这一点。

### Chrome Trace 时间线（MEGAMOE_TRACE=1，自动生成）

```
seq4096_rank_0_trace.json    ← Chrome Trace 格式
```

拖到 ui.perfetto.dev 或 chrome://tracing 看逐核泳道时间线（AIC/AIV0/AIV1
逐组，per-expert 块，wait 阶段单独标色）。与 profile_mega_moe_bf16.py
独立产出的 trace 格式完全一致（同一份解码管线）。

---

## 与其他目录的关系

### 深入阅读

- [roofline 达成率分析指南](docs/roofline_analysis_guide.md) — 什么 shape 能打满、K/F/M 的影响、kernel 逐 GEMM 定位

| 目录 | 作用 | 和本目录的关系 |
|------|------|----------------|
| `test/megamoe_bf16_test/` | kernel 团队的独立测试（kernel vs CPU golden） | 本目录是 v2 vs 小算子 对比；那边是 kernel 自身验证 |
| `test/megamoe_bf16_test/profile_mega_moe_bf16.py` | profiler trace 采集 | 生成 Chrome Trace 时间线（含 torch_npu.profiler 系统级 trace） |
| `test/megamoe_bf16_test/parse_backward_timer.py` | timer CSV 离线解析 | 本目录的 MEGAMOE_TRACE.py 是 in-process 版本 |
| `test/bigop_fused/ep_probe.py` + `roofline_probe.py` | 部署规模推算 | 参数需从 bench 实测校准（见下） |

### 校准离线工具

`ep_probe.py` / `roofline_probe.py` 的带宽参数需要从实测校准：

| 参数 | 校准方法 |
|------|----------|
| `--bw-hccs` | 单机 alltoall 分解行的流量 ÷ 实测 ms |
| `--bw-rack` | 双机（同柜）bench 的 alltoall 分解行反推 |
| `--bw-cage` | 跨柜 bench（有环境时）；无则用 NIC 理论 × 0.5-0.7 |
| `--cube-tflops` | v2 roofline 行的 achieved TFLOPS |

---

## 常见问题

**Q: `local_mem_size is too small`**
A: 对称窗口不够。设 `BENCH_LOCAL_MEM_BYTES=<更大的值>`。启动时会打印需求估算和建议值。

**Q: `MEGAMOE_TRACE=1` 没输出**
A: `.so` 没用 `--enable-timer` 构建。重编：`bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950 --enable-timer`

**Q: 多机启动卡住不动**
A: 检查两台机器是否几乎同时启动（torchrun 会互相等待）；检查端口 29518/8899 防火墙放行。

**Q: 大 topK 内存不够**
A: 每 rank 内存 ≈ 对称窗口 + workspace（~M×topK×EP×(2K+8F)×1.5B）+ 权重×2 + 激活。topK 和 EP 是放大器，M 减半或 topK 减半可显著缓解。

**Q: 精度 FAIL 但性能正常**
A: 先用 `PRECISION_ISOLATE_CASES=1` 排除跨 case 污染；再看是哪个梯度超差（dx/dw1/dw2/dtopk）。
