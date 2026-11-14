# EP8 轻载保留原路由与计划校验融合

本轮起点 `9eddbbe`，实验代码提交 `e78d204`，分支 `work/planner-megamoe-bench`。测试形状固定为真正 WORLD=EP=8、每卡 4 个 home 专家和 4 个 slot，M4096/H7168/F3072/TopK6。

## 改动和验证范围

1. 在 MoonEP 已完成全局 histogram 校验之后，计算最大原始卡负载。`max_load*1000 <= M*TopK*1050` 时保留原 owner assignment，copy 表保持空；否则执行原有严格均衡 MoonEP。整数乘法使用 int64，不增加 D2H、`.item()` 或逐步主机决策。
2. 新开关 `MEGAFLUX_MOONEP_DIRECT_PERMILLE` 默认为 0；本轮独立构建 0 和 1050 两个 planner SO。Python `direct_permille` 显式选择并核验实际 SO 导出值，初始化时把策略/校验配置纳入已有跨卡一致性检查。实验分流仅允许真正 EP8 的小规模 planner。
3. 计划校验融合复用分支已有 `FusedDevicePlanValidator`，用两个 kernel 和原有一次集合通信替代 tensor 小算子校验；保留计划快照、错误传播、容量和一致性检查。此次没有新改校验 kernel。
4. MegaMoe/S9 计算、权重搬运和梯度规约实现不变；MTE 双缓冲实验、worker schedule、owner compact、row-ready 实验均关闭。反向 panel restore=2、operand-ready=1、delay=0。
5. 本轮没有改变接收缓冲分配方式，也没有删除 publish/barrier/finish 或 FSDP2 生命周期保障。

这里“保留原路由”表示不迁移专家、S9 active=0 消费原 logical IDs；仍经过 histogram、路由编码及完整校验，不等于已经消除了整个 planner 调用。

| 文件 | 作用 |
|---|---|
| `CMakeLists.txt` | 默认关闭的阈值构建选项，仅作用于 MoonEP planner |
| `include/mega_moe_moonep_direct.h` | 共享 CPU/device 整数判定，使用 int64 中间乘积 |
| `kernels/mega_moe_bf16/mega_moe_replica_planner_control.cpp` | MoonEP 条件编译分支内保留 owner assignment，其他 planner 分支不变 |
| `kernels/mega_moe_bf16/mega_moe_moonep_planner.cpp` | 导出真实阈值，供运行时核验 |
| `ascend_deepep/moonep_planner.py` | 显式策略参数、实际 SO 身份检查、既有 generic/fused 选择 |
| `ascend_deepep/execution_plan_device.py` | 初始化时把策略与校验选择加入已有跨卡配置一致性检查 |
| `ascend_deepep/fsdp2_bridge.py` | 传递显式策略选项，拒绝不支持的组合 |
| `test/megamoe_planner_test/` | 独立构建、CPU/设备门禁、三组精度、正反序性能、加载库和输入回执 |

## 独立对照

| 标记 | variant | 路由策略 | 计划校验 |
|---|---|---|---|
| A | strict-generic | 原严格均衡 MoonEP | generic |
| B | direct-generic | 最大/均值不超过 1.05 保留原路由 | generic |
| C | direct-fused | 同 B | fused |

每个场景先 original、A、B、C 精度，再 A/B/C/C/B/A 性能。B/A 评估分流，C/B 单独评估校验融合，C/A 评估组合。各场景同输入、权重、容量、依赖和实际加载库身份；容量沿用覆盖 original/MoonEP 的共同校准值。

分流的时间差包含停止专家迁移后整条副本路径的变化，不能全部解释成孤立权重 copy 时间。校验融合的时间差是在相同计划、相同 worker 下的完整 F/B 差值，包含调度影响。本轮 original 用作数值参考，性能百分比的基准 A 是原严格均衡 S9。

- balanced_random：随机排列但严格计数均衡。
- random：各源 rank 独立 seed，每 token 无放回选取 TopK，允许轻微自然不均衡。
- controlled 1.5 / 2：rank0 所属专家设定精确热卡配额，参数为最大卡负载/均值。
- 五项精度：y、dx、路由概率梯度 dp、owner dw1/dw2。每次精度进程执行三个相同输入 epoch，前后向之间覆盖权重池。相同计划的候选要求五输出 bitwise；计划变化按原算子数值门槛及额外 relative L2≤0.05 验证。
- 性能关闭 Timer，每进程 warmup=10、repeat=30；每个样本取八 rank 完整 F/B 墙钟最大值，再统计中位、p95 和标准差。范围含 planner、home 发布、权重恢复、owner SUM、输出所有权和完成等待，不含整网训练或 FSDP reduce-scatter。

## 环境与复现入口

使用 177.2.25.1 自有容器 `moonep-megamoe-ep16-20261027`，容器 ID `0ba47b08a7b324b33362331bafb1400d59253e32fead4c223b3eaf5ef64faa81`。名称含 ep16 不表示本轮 EP16。

新目录 `/data/x00610564/moonep_megamoe_large_ep_20261027/optimization/routing-20261029-a1`；容器内 `/project/optimization/routing-20261029-a1`。旧 mte 和 planner-bench 目录仅作只读依赖来源，未覆盖。

新 worker 与两个 planner 在该目录使用 CANN 9.2.0、Timer OFF、4 CPU 线程构建。未变扩展和依赖 SO 逐文件比对源代码并记录 SHA256。共享 control 文件的差异仅处于 MoonEP 条件编译分支；使用该分支的 planner 已重新构建。

实际运行版本为 PyTorch `2.10.0+cpu` 配套 `torch_npu 2.10.0.post6`；八个 rank 均使用 NPU，PyTorch 版本字符串中的 `+cpu` 不代表本轮在 CPU 执行。实际运行代码指纹 `dff25b014b09b5898ee6af0a78f1c33e18dcb3e7adcaa5881db4f476e226a77b`，外部依赖指纹 `6b866fad3e683571f4d01aec79890c2f92ecac460329d7a13bb72bc9689e8ace`。

每次启动都校验容器 ID/标签/唯一可写挂载，检查两次八卡进程、利用率、HBM 和 fuser，结束后再次检查。允许用户明确授权的空闲 Warning/Alarm 卡。任何门禁失败即停止后续实验；不停止其他进程。

构建和部署入口：`test/megamoe_planner_test/remote_routing.py`、`build_routing_variants.sh`。测试入口：`routing_campaign.py`、`routing_compare.py`、`routing_contract.py`。原始证据位于实验目录的 `bench-build-evidence`、`campaign-evidence` 和 `test/megamoe_planner_test/output`。

## 已完成的门禁

- 16 项 CPU 合约测试通过；实际编入设备 TU 的整数阈值 helper 经 C++ 边界测试，覆盖等号、首个拒绝值、超过 int32 的中间乘积和禁用/非 EP8 情况。
- 真正 EP8 的六次连续规划及 generic/fused 差分通过；17 类错误注入全部保留原错误语义，非法计划没有进入 MegaMoe 执行。

| 原始最大/均值 | 选择 | 副本数 | 128 对齐 required |
|---:|---|---:|---:|
| 1.0 | 原路由 | 0 | 24576 |
| 2.0 | MoonEP | 7 | 24576 |
| 1.0498046875 | 原路由 | 0 | 25856 |
| 1.0501302083333333 | MoonEP | 7 | 24576 |
| 1.0067138671875（random） | 原路由 | 0 | 24832 |
| 1.0（再次切回） | 原路由 | 0 | 24576 |

门禁输出目录名为 `small-hot-a1`，实际执行入口是 `--mode routing-contract`。规划测试固定 M4096/E32/TopK6，执行器初始化使用 H512/F1024，只验证计划协议；完整 F/B 使用下文的 H7168/F3072。

构建产物 SHA256：

- `worker-variants/route-bench/libascend_deepep_megaflux_early_bf16.so`：`0b4e56ba30a717cd21ac2c3cca2995e85b009dfd10832f4c602260965a30f090`，与前轮 mte-base 完全一致。
- strict planner：`7f17167f0ec055b37e95f4a4f4d161a93047010a6f99851e1f017e9609afec62`。
- direct1050 planner：`e63aa02d65ef1d90e5f07ef7d6fee68539fe3d7f99f5c590e6599f2f2f28774e`。

## 大形状完整 F/B 结果

41 次八卡设备运行全部通过前后占用门禁：1 次协议/校验门禁、16 次精度运行、24 次正式性能运行。性能共 720 个 max-rank 样本。以下均为单层完整 F/B，不是整网 step。

表中“正 / 反”分别是 A/B/C 与 C/B/A 两个运行方向，时间单位 ms；不把两轮混成一个平均结果。

| 场景 | A：旧严格均衡 S9 | B：分流＋普通校验 | C：分流＋融合校验 |
|---|---:|---:|---:|
| 严格均衡（balanced_random） | 29.326265 / 30.217785 | 29.682210 / 29.348055 | 28.426510 / 28.472085 |
| 自然 random | 34.744730 / 34.815320 | 29.661420 / 29.701870 | 28.960855 / 28.751765 |
| 受控 1.5 倍 | 36.028375 / 35.641080 | 35.662640 / 35.592780 | 34.725695 / 34.819825 |
| 受控 2 倍 | 38.047085 / 37.868305 | 37.646295 / 38.180220 | 36.777350 / 36.588050 |

耗时变化，负数表示更快；各单项按自己的对照计算，百分比不能直接相加。

| 场景 | B/A：仅分流，正 / 反 | C/B：仅校验融合，正 / 反 | C/A：组合，正 / 反 | C/A p95，正 / 反 |
|---|---:|---:|---:|---:|
| 严格均衡（balanced_random） | +1.214% / -2.878% | -4.230% / -2.985% | -3.068% / -5.777% | -5.302% / -7.073% |
| 自然 random | -14.630% / -14.687% | -2.362% / -3.199% | -16.647% / -17.416% | -17.304% / -13.752% |
| 受控 1.5 倍 | -1.015% / -0.136% | -2.627% / -2.172% | -3.616% / -2.304% | -3.432% / -2.877% |
| 受控 2 倍 | -1.053% / +0.824% | -2.308% / -4.170% | -3.337% / -3.381% | -9.176% / -3.067% |

### p95 与波动

| 场景 | 方向 | A p95 / 标准差 | B p95 / 标准差 | C p95 / 标准差 |
|---|---|---:|---:|---:|
| 严格均衡（balanced_random） | 正序 | 30.240540 / 0.465811 | 30.114380 / 0.174217 | 28.637110 / 0.084419 |
| 严格均衡（balanced_random） | 反序 | 30.758820 / 0.183943 | 30.238050 / 0.302813 | 28.583390 / 0.079328 |
| 自然 random | 正序 | 35.472080 / 0.341807 | 30.565520 / 0.270387 | 29.334150 / 0.217954 |
| 自然 random | 反序 | 35.208860 / 0.233269 | 30.931550 / 0.678018 | 30.366890 / 0.424526 |
| 受控 1.5 倍 | 正序 | 36.346110 / 0.972534 | 38.684330 / 1.041648 | 35.098540 / 0.468515 |
| 受控 1.5 倍 | 反序 | 36.252030 / 0.223557 | 36.825460 / 0.944937 | 35.208950 / 0.244633 |
| 受控 2 倍 | 正序 | 41.103560 / 0.974278 | 39.657320 / 0.666831 | 37.331750 / 1.622124 |
| 受控 2 倍 | 反序 | 38.170770 / 0.157277 | 39.194420 / 0.579332 | 37.000220 / 0.833540 |

### 精度和计划

各场景每个配置均通过五输出数值门槛，且三个重复 epoch 的输出 hash 一致。下表相对 L2 是五项输出、八个 rank 中的最大值；更详细的逐输出指标保存在同名 JSON。

| 场景 | A 对 original 最大相对 L2 | B 对 original 最大相对 L2 | C 对 original 最大相对 L2 | B/C 对照 |
|---|---:|---:|---:|---|
| 严格均衡（balanced_random） | 0 | 0 | 0 | B、C 与 original 逐位一致 |
| 自然 random | 0.00205078382 | 0 | 0 | B、C 与 original 逐位一致 |
| 受控 1.5 倍 | 0.00276246882 | 0.00276246882 | 0.00276246882 | B 与 A、C 与 B 逐位一致 |
| 受控 2 倍 | 0.00275349176 | 0.00275349176 | 0.00275349176 | B 与 A、C 与 B 逐位一致 |

| 场景 | 原始最大/均值 | A 副本数 / required | B、C 副本数 / required | 固定容量 R |
|---|---:|---:|---:|---:|
| 严格均衡（balanced_random） | 1 | 0 / 24576 | 0 / 24576 | 27776 |
| 自然 random | 1.00671386719 | 7 / 24576 | 0 / 24832 | 28160 |
| 受控 1.5 倍 | 1.5 | 7 / 24576 | 7 / 24576 | 41600 |
| 受控 2 倍 | 2 | 7 / 24576 | 7 / 24576 | 55424 |

### 判断

- 自然 random：原始负载只偏离均值约 0.67%，旧 MoonEP 仍生成 7 个副本。分流后副本为 0，输出与 original 逐位一致，两个方向都出现明确完整 F/B 收益。
- 严格均衡：旧 MoonEP 本来就没有副本，单独分流的两次中位数变化方向不同，不支持声称稳定收益。
- 1.5 倍负载：分流未改变计划；单独分流的中位数虽小幅下降，但 p95 两次都变差，因此不据此宣称稳定收益。
- 2 倍负载：单独分流的中位数变化方向不同，不能声称稳定收益。C 的中位数和 p95 两轮都下降，但每轮各有一次长尾，最大耗时分别为 45.739250 / 41.191940 ms，高于 A 的 42.535820 / 38.199100 ms，标准差也更大。没有删除这些样本，不能宣称最坏耗时或所有波动指标都改善。
- 校验融合的收益以 C/B 单独列出；组合收益以 C/A 列出，不能把它们误称为 MegaMoe 主 kernel 的加速。

采用结论：当前固定 shape 的基准验证可以显式使用 C（direct1050＋fused）；自然 random 的主要收益来自不再为了 0.67% 的不均衡创建副本，其余场景主要受益于校验融合。通用默认仍为 threshold=0＋generic，不把这组结果外推为其他 shape、真实训练路由、整网或 EP16～256 的默认策略。

### 证据与复算

所有 24 组性能都已从八 rank 原始样本重新计算 max-rank 序列、中位和 p95，并与远端报告逐项相等。

北京时间 2026-10-29 11:02:18 最终检查：八卡无占用，自有容器只剩原保活 `tail` 进程。全过程未启动整网训练，未停止或修改其他任务。独立环境和原始张量证据保留，便于后续复测。

归档 SHA256：`506b1ccf12e54774c4b555ef14d7ceced9fe2e48988d5e669ecbab97e52ba0cf`。

本地原始回执：`test/megamoe_planner_test/output/routing-20261029/evidence/`；汇总：`docs/planner-megamoe-routing-20261029.json`。

远端总控：新实验目录下 `campaign-evidence/suite-a1`；四个完整案例位于 `test/megamoe_planner_test/output/m4096-{balanced,random,load15,load2}-a1`。


## 接入方式与采用边界

本轮两个优化都通过显式实验配置选择，未更改通用默认值：

- 构建 planner 时指定 `-DMEGAFLUX_MOONEP_DIRECT_PERMILLE=1050`，并把该独立 SO 目录放在新进程的 `LD_LIBRARY_PATH` 最前。
- `NativeFSDP2Pool(..., planner='moonep', plan_validation='fused', direct_permille=1050)` 对应 C；将 `plan_validation` 改为 `generic` 对应 B。
- A 必须选择 threshold=0 的 planner SO，并传 `direct_permille=0`。同一进程内不切换库。
- 测试入口 `gate_and_run.py --worker-variant route-bench --routing-variant direct-fused` 自动选择相应独立库，核验实际加载路径和 SHA256。
- 1.05 是这组固定 shape 的初始阈值；本轮没有扫描阈值寻找全局最优，也没有据此改变 EP16～256 或其他 shape 的默认路由。
- 各场景容量 R 仍固定一致；`required` 是当前计划的需求，不代表已经回收了预分配内存。
