# Python 高负载不均：固定八卡算子场景与性能说明

> 本目录现也随优化合入 `megamoe-balanced-shared-expert`；下文balanced历史构建/性能用于复现来源，不代表shared组合已验收。请先看[共享分支合入说明](../../../../docs/experiments/20261102-shared-python-integration.md)。

本目录让同事在同一份路由、shape 和数值输入上验证 routed MegaMoE。日常使用**不需要下载 OLMoE、Python 语料或传输约 4 GB capture**：从冻结的源直方图生成四份路由，再按固定 CPU RNG 构造八 rank 张量，逐 tensor SHA 校验。

来源标签始终为 `NEW_PYTHON_TRAINING_HISTOGRAM_DERIVED`：**官方预训练 OLMoE + 真实 Python 文本的新采样，再映射为八卡算子输入**。它不是论文作者的 Python P95 原始 trace，也不是 DeepSeek 自然 router 的逐 token 路由。这个区别不能在复用时省略。

2026-11-02 已将场景工具与 P1-cache/owner2:1 两项优化一起合入 `megamoe-balanced`，保留原提交 `b914741`、`847a1f9` 和最终三组报告 `ea98005`；共同基线为 `7c41ebf938363995838056677e73e5ea6a289d02`。P1 cache 默认0、owner ratio默认1，合入不改变生产默认。下文性能来自已完成的历史算子实验；**本次合入没有新增设备或整网验收**。源码与历史已构建输入的比对、构建与回退见 [合入说明](../../../../docs/experiments/20261102-balanced-python-integration.md)。

## 1. 文件与体积

| 文件 | 用途 |
|---|---|
| [source-profile.json](source-profile.json) | 冻结的 64 专家源计数、checkpoint revision、采样方式、IDs/prob SHA |
| [case.json](case.json) | 目标 shape、计数、κ、seed、四份路由预期 SHA、历史 planner 结果 |
| [expected-inputs.json](expected-inputs.json) | 八 rank 六输入的 shape/dtype/tensor SHA，以及有条件使用的历史五输出 SHA |
| [generate.py](generate.py) | CPU 路由重建；可选生成并核验密集张量 |
| [PERFORMANCE.md](PERFORMANCE.md) | 历史完整 B/FB、流水依据、当前分支适用范围 |
| [historical-performance.json](historical-performance.json) | 小型性能摘要与原始证据文件 SHA 索引 |
| [build-reference.json](build-reference.json) | 历史SO指纹、155项原始编译输入SHA及本次源码匹配范围 |

只携带文本配置、生成器和说明。运行生成的四份 NPY 总计 **262,656 B**；八份历史输入 PT 总计 **3,994,584,408 B**。官方三片权重约 13.84 GB、两份反向 trace 约 820 MB、展示 HTML 约 11.24 MB，均不随场景进入 Git。生成产物请放到独立技术项目输出目录。

## 2. 原始场景怎样采到

模型为 `allenai/OLMoE-1B-7B-0924`：

- HF revision：`6d84c48581ece794365f2b8e9cfb043c68ade9c5`。
- ModelScope revision：`d799cc18394603b9bd1abc813be923515bbb5bee`；历史采样从 ModelScope 下载，三片 SHA 已与固定 HF LFS 匹配。
- 官方训练 mix revision：`1e44595eaffc7491dfab23947ea4d5a62b33aff3`，`data/starcoder/python-0000.json.gz` 前 2048 文档，保持原顺序，追加 EOS 后 pack 成 4096 上下文。
- 得到 4,061,029 tokens、991 个完整上下文；每事件取连续八个上下文，即 32,768 tokens。
- BF16、`train()`、梯度开启、gradient checkpointing；固定预训练权重，**零 optimizer 更新**。选中输入另完成真实 causal loss backward，179 个梯度张量均有限。
- 预先固定选择规则：最多 32 事件，选择首个含 κ∈[2.7,3.3] 层的事件，在该事件中取 κ 最接近 3 的层，再以层号较小者打破平局。

实际选中 **event0 / layer9（零起算，第十个 MoE 层）**。源模型 E64 / Top8，262,144 条有效路由，无 padding。

定义 `κ = E × Σ(count[e]²) / (Σcount[e])²`，均衡时为 1；**它不等于最热专家/平均专家，也不等于最重 rank/平均 rank**。

源 κ = **3.0070735886693**。平均专家 4096 条；E46 最热 28,054 条，为平均 6.8491 倍，覆盖 85.6140% tokens。全量 64 维计数见 source-profile.json。源 IDs/prob SHA 只标识原采样，不要求映射后的 IDs/prob 与其相等。

同事若只想复用本例，不需要重新跑模型；如果要研究自然共选关系、其他层或训练演变，必须另行采样并使用新场景编号。本目录没有分发模型、原文本或完整采样环境。

## 3. 映射到八卡算子的 shape

| 参数 | 固定值 |
|---|---|
| world / EP 分组 | 8；`[0,1,2,3]`、`[4,5,6,7]` 两个 EP4 组 |
| 每 rank M / H / F | 8192 / 7168 / 3072 |
| 逻辑 E / TopK | 8 / 2 |
| home / replica slots / capacity | 2 / 2 / 20480 |
| 保存态 / 权重布局 / clip limit | BF16 saved / canonical / 10 |
| p | FP32 常量 0.5 |
| 每 EP 组 tokens / assignments | 32768 / 65536 |

两个 EP 组复用相同的四份路由；全局 rank `r` 使用 `rank(r % 4)-ids.npy`，密集输入使用独立的 `42+r` seed。mesh 另一维度大小为 2，但 benchmark **不执行真实 expert-FSDP all-gather/reduce-scatter**，不能称作整网 EP4×expert-FSDP2 性能。

| 每 rank 输入 | shape / dtype |
|---|---|
| X、dY | `[8192,7168]` / BF16 |
| W1 | `[2,6144,7168]` / BF16，即 `[home,2F,H]` |
| W2 | `[2,7168,3072]` / BF16，即 `[home,H,F]` |
| prob | `[8192,2]` / FP32 |
| logical_ids | `[8192,2]` / int32、每 token 两个专家互异 |

H/F 沿用当时目标八卡算子设置；本例 E8/Top2 和路由重构不代表完整 DeepSeek 模型的自然训练配置。X/dY/权重/p 也是受控合成输入。

### 64 专家怎样变成 8 专家

1. 将源 64 维计数降序排列并归一化，构造累计分布。
2. 在 `[0,1]` 上对 8 个等宽区间做 CDF 积分，得到 8 维权重，乘以目标总条目 65536。
3. 每专家最多 32768 条，因为同一 token 不允许重复选该专家；超额按剩余专家权重比例再分配。
4. 向下取整后按余数从大到小补齐，余数相同按专家号小者优先。
5. `np.random.default_rng(17)` 先置换专家号。
6. 逐专家按精确度数循环分配给全组 32768 个 token，再用同一 RNG 打散 token 顺序，切成四个 rank 的 8192×2 数组。

这会改变源 Top8 的共选关系。重构后 E1 恰好被所有目标 token 选中，是映射约束的结果；不能将它描述为原始 Python 模型的自然 E1 行为。

| 专家 | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 目标条目 | 12051 | 32768 | 3495 | 572 | 5497 | 1175 | 8107 | 1871 |

目标 κ = **2.481662403792143**，平均 8192 条，最热专家/平均为 4。源 κ≈3 在压缩/限额后变为此值，不强行修改回 3。

### planner 前后的详细负载

每 EP 组初始每 rank 连续拥有两个逻辑专家：

| EP rank（全局） | 初始 home | 初始条目 | 历史 MoonEP 后物理专家及条目 | 合计 |
|---|---|---:|---|---:|
| 0（0/4） | E0、E1 | 44819 | E0=5645，E1=10739 | 16384 |
| 1（1/5） | E2、E3 | 4067 | E2=3495，E3=572，副本 E1=12317 | 16384 |
| 2（2/6） | E4、E5 | 6672 | E4=5497，E5=1175，副本 E1=9712 | 16384 |
| 3（3/7） | E6、E7 | 9978 | E6=8107，E7=1871，副本 E0=6406 | 16384 |

初始平均 16384，最重 owner/平均 **2.73553466796875**。每组活动副本 3 个，执行行数最大/平均降至 1；逻辑专家直方图和 κ 未改变。rank0/4 分属两个独立组，是同一 owner 位置，不是两个独立自然热点样本。

执行行数均衡后，E0/E1 的完整 dW1/dW2 仍需归主至 rank0/4，owner 服务量和晚 AIV0 加入会形成尾部。新版本 planner 若改变 placement，应保存自己的计划及 SHA；不能只看总行数一致便声称与历史逐位等价。

## 4. CPU 生成与核验

下列示例在 Linux checkout 根目录运行，`CASE` 是脚本目录，`RUN` 是新的实验输出目录。Windows 也可直接用相同 Python 脚本和绝对路径。

```bash
CASE="$PWD/integration/wuhan/scenarios/python-olmoe-high-skew"
RUN=/absolute/experiment-project/python-case-a1

# 只构造路由：需要 NumPy；不导入 torch_npu，不使用设备。
python3 "$CASE/generate.py" --out "$RUN/routes-only"
python3 "$CASE/test_scenario_cpu.py" -v

# 路由 + 八 rank 密集输入：新目录，约 4 GB 磁盘空间。
# 使用历史运行时 torch==2.10.0+cpu；该版本和 torch_npu 在原 NPU 环境配套。
python3 "$CASE/generate.py" --out "$RUN/full" --with-tensors

# 也可先只生成某几个全局 rank，用于分批准备；不能覆盖已有目录。
python3 "$CASE/generate.py" --out "$RUN/rank0-check" --with-tensors --ranks 0
```

源路由 CPU 证明使用 NumPy 2.5.4。生成器会核对四份 NPY 文件 SHA 和原始 int32 数组 SHA；总计数守恒、Top2 互异、目标直方图和 κ 均需精确通过。不同 NumPy 版本若造成差异，会失败退出，不能更新预期值来掩盖差异。

密集张量使用 **CPU** `torch.Generator().manual_seed(42+global_rank)`，保持下列顺序：

1. FP32 `randn([M,H]) × 0.2` → BF16 X。
2. FP32 `randn([M,H]) × 0.1` → BF16 dY。
3. FP32 `randn([home,2F,H]) × 0.01` → BF16 W1。
4. FP32 `randn([home,H,F]) × 0.01` → BF16 W2。

prob 为 FP32 常量 0.5，不消耗 generator；IDs 由路由生成器产生。不要换成设备 RNG、先转 BF16 再乘 scale、改调用顺序或用全局 seed 混入别的随机调用。

输出 `full/inputs/rankN-inputs.pt` 和 JSON 收据。生成器逐个核六输入 tensor SHA，并用采集入口同一个 CPU loader 回读序列化结果。`.pt` 文件 SHA 用于本次传输完整性，历史相等以**逐 tensor SHA**为准，因为不同序列化元数据可能改变文件字节。

**本次落库验证范围**：本地只有 NumPy，没有 Torch，已经实际重建路由并匹配历史四文件/八 rank logical_ids；密集张量与新增 NPU 入口本轮仅静态/语法检查，未声称重新跑过设备。脚本的密集输入强校验用于同事在配套运行时执行时发现差异。

## 5. 用当前代码采集正确性收据和 TimerV2 图

现有 `capture_kernel_timer.py` 增加可选 `--scenario-inputs`。不传时原有合成负载/capture 行为不变；该参数与 `--capture` 互斥。固定场景的 shape 由输入 config 决定，**不要用 `--rho` 代替本例 κ**。

前提：已有匹配 CANN/SHMEM、binding、CATLASS 子模块及 Python 环境。只在独立目录构建配套库；下面的 `case-python-a1` 必须是新编号，不覆盖现有运行 SO。请先清理无关继承实验环境变量，再显式选择所测开关；本场景本身不要求 owner2:1 或 P1-cache。

```bash
# checkout 根目录；该脚本生成 worker-variants/case-python-a1-{off,on}。
# 复现历史owner2:1候选时，显式设置下列三项；默认构建不会自动开启。
export MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_OWNER_EARLY_RATIO=2
bash scripts/build_megaflux_timer_capture.sh case-python-a1

# 已通过设备准入后，在准备好的容器环境运行：
# OFF 三次重复 + ON 对 OFF 五输出逐位核验，随后离线导出。
export TASK_QUEUE_ENABLE=1  # 与下面新B/FB示例一致；两版统一选择并保存
bash integration/wuhan/run_timer_capture.sh \
  --variants "$PWD/worker-variants/case-python-a1" \
  --out "$RUN/trace-a1" --world-size 8 --master-port 32250 \
  --scenario-inputs "$RUN/full/inputs" --repeat 3 --shmem-port 32260
```

`--scenario-inputs` **仍在每次 F 执行在线 MoonEP 与 home 刷新**，没有将历史 placement 悄悄放到计时外。CPU loader 会校验固定 shape/world8/EP4、window、文件及六输入历史 SHA。

生成图：

- `trace-a1/kernel-trace/all_8_rank_forward_trace.json`
- `trace-a1/kernel-trace/all_8_rank_backward_trace.json`

默认不传 `--include-waits`，不画 wait 块；每 rank 32 AIC + 64 AIV 轨道，原始逐条区间、同轨零重叠。时间是 rank 本地起点，禁止跨 rank 相减估计通信延迟。TimerON 只解释排程，不能替代 TimerOFF wall 性能。

OFF 首次运行的五输出只建立该版本重复性基线，**不是独立参考正确性验收**。合入/内核改动仍需原独立参考、同 plan 五输出对照、EP2/4/8 生命周期/checkpoint/SGD 检查。expected-inputs.json 中的历史输出 SHA 只有在输入、placement、数值路径都一致时才可当逐位目标；不能仅因 planner 变化就判定 kernel 数值错误，也不能悄悄放宽门限。

### 设备准入与失败编号

在宿主机 root 使用 `integration/wuhan/gate_and_run.py` 包裹实际容器命令；阶段前须额外完成两次健康/进程/root-fuser 全八卡准入。示例中的 `gate()` 来自同一现有工具：

```bash
# HOST_REPO 为宿主机上的此 checkout；RUN_HOST 为宿主项目输出路径。
PYTHONPATH="$HOST_REPO/integration/wuhan" python3 - "$RUN_HOST/admission-a1" <<'PY'
from pathlib import Path
import sys, time
from gate_and_run import gate
p = Path(sys.argv[1]); p.mkdir(parents=True, exist_ok=False)
if not gate(p, 'first'): raise SystemExit(75)
time.sleep(2)
if not gate(p, 'second'): raise SystemExit(75)
PY

# 执行上面的采集/后面的benchmark命令时，用它包裹 docker exec：
# python3 "$HOST_REPO/integration/wuhan/gate_and_run.py" \
#   --project "$RUN_HOST" --name trace-a1 -- docker exec <own-container> ...
```

不要只运行注释示例便宣称已门禁；填写并记录真实自有容器、命令、路径、NIC 和端口。每个设备进程 pre/post 必须通过，失败/占用立即停止本阶段，保留编号，修正后换新编号重新准入；不 kill 外部任务，不 reset 设备。网卡/HCCL/SHMEM 和原日志归档规则继续沿用项目约定。

## 6. 完整 B / 直接 FB 对照入口

`replay_captured_layer.py` 新增同一个 `--scenario-inputs` 模式，复用原 autograd bridge 计时，**在线 MoonEP**，要求 `--reference` 指向同版 TimerOFF 三次检查生成的 receipts。原 `--capture` 模式仍是固定 captured placement，两者口径必须区分。

```bash
# 在通过门禁的单节点容器内，使用当前构建 OFF 库：
export WORKER_NATIVE="$PWD/worker-variants/case-python-a1-off"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$WORKER_NATIVE:$PWD/ascend_deepep/lib:${LD_LIBRARY_PATH:-}"
export WUHAN_PUBLISH_SYNC=device WUHAN_CONTROL_TAPE_DEVICE=npu OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1  # 若选其他值，C0/C1双方必须一致，另立结果组

torchrun --nnodes=1 --nproc-per-node=8 --master-port=32350 \
  integration/wuhan/runtime_bootstrap.py --native-only \
  integration/wuhan/replay_captured_layer.py \
  --scenario-inputs "$RUN/full/inputs" --reference "$RUN/trace-a1/control" \
  --restore panel --phases backward fb --warmup 10 --repeat 30 \
  --shmem-port 32360 --out "$RUN/perf-c0-a1"
```

每个版本使用**自己的匹配 OFF 收据/库**，loader/benchmark 核输入、库 SHA、plan SHA、计时前后五输出；跨版本另核同输入、同 plan、同数值路径的五输出。上例只展示单进程命令，不能代替外层 root 门禁或配对实验。

完整 B：本轮 forward 在计时外准备 saved state，计时内运行 autograd backward，包括对应恢复、计算、归主、finish 和梯度转换。直接 FB：同一计时区内 autograd F+B，含每次 F 的在线规划与 home 权重刷新。不要用 capture 工具里一次 pool F/B 的分段 wall 冒充这里的独立完整 B/FB。

每进程各 10 预热、30 测量；每个样本取八 rank 最大 wall，再算中位数/P90（`(n-1)q` 线性插值）。关闭 Timer、profiler 和诊断回读，hash 在计时外。配对先两次 C0 自重复，再 AB、BA、AB、BA；不要合并不同节点、任务队列或有外部占用的单版样本。当前工具保存逐 rank samples 和 rank-max samples；实验报告还应保存 build/getter/环境/门禁及统计过程。

历史 owner2:1 第四组未完成，这个事实也不能因同事重用脚本而变成“已通过四组”。本次新增入口尚无设备运行收据，正式接受仍需完成对应验证。

## 7. 本例与共享专家融合的关系

这是 **routed experts 负载/反向排程**场景，未包含模型里的 shared MLP。可以作为同事共享融合实验的 routed 一侧输入，但还需补充 shared FFN 宽度、权重、保存态、keep/recompute、saved/redispatch 和独立 shared reference；统计完整 block 的 Y、dX、route/shared 梯度与生命周期。

本目录接入不改同事共享前向，不自动开启任何共享融合开关；既有共享融合报告不能直接套到本场景。详细历史性能及证据位置见 [PERFORMANCE.md](PERFORMANCE.md)。
