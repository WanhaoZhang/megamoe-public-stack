> 当前校验已统一、保留兼容参数，使用和框架迁移见 [融合校验默认实现](../integration/wuhan/FUSED-PLAN-DEFAULT.md)。下文 generic/fused 对照为历史实验记录，不是当前可选运行路径。

# EP8 Planner + MegaMoe 增量交付：使用、证据与保留建议

本说明供合入 `codex/moonflux-8p-integration` 时审阅。交付基线为
`0618eebc2403bb5adb0aa1c615e870223ec78472`，保留该分支最新的全 rank Timer trace 导出与串行区间修正。

本次交付两部分：**可单独保留的通用 benchmark**，以及**默认关闭、显式选择的 EP8 原路由策略**。
融合计划校验内核已经存在于目标分支；这里补充它的独立对照和使用配置，不重复实现。
没有带入无稳定收益的 MTE 双缓冲、owner 索引候选，也没有修改 GMM 算术或调度。

## 1. 修改清单

| 类别 | 交付内容 | 是否改变设备执行 |
|---|---|---|
| benchmark | 四种链路、四类路由、精度门禁、完整 F/B 正反顺序比较 | 仅独立测试入口 |
| 复现与诊断 | 精度回执绑定输入/源码/依赖/SO；逐样本 max-rank；独立阶段诊断 | 不混入正式计时 |
| CPU 检查效率 | 可选 1～4 进程离线张量比较，阈值不变 | 不使用 NPU |
| 原路由策略 | 已获得全局计数后，轻度不均衡不创建副本 | 仅显式启用时改变 MoonEP 计划 |
| 融合计划校验 | 复用已有 `FusedDevicePlanValidator`，提供 A/B/C 对照和一致性门禁 | 仅显式选择 fused 时启用已有路径 |
| 接入检查 | 校验实际加载 planner SO 的阈值；各 EP rank 的策略配置必须一致 | 初始化时检查 |

生产文件只有以下七个；其余为测试、说明和历史证据：

- `CMakeLists.txt`：`MEGAFLUX_MOONEP_DIRECT_PERMILLE`，默认 0。
- `include/mega_moe_moonep_direct.h`：设备/CPU 共用整数阈值判断。
- `kernels/mega_moe_bf16/mega_moe_moonep_planner.cpp`：编译阈值及导出实际阈值。
- `kernels/mega_moe_bf16/mega_moe_replica_planner_control.cpp`：在已验证计数后选择原 owner assignment 或原严格均衡算法。
- `ascend_deepep/moonep_planner.py`：显式参数、SO 检查、校验模式选择。
- `ascend_deepep/execution_plan_device.py`：已有初始化一致性检查增加策略签名。
- `ascend_deepep/fsdp2_bridge.py`：透传参数并限制为小规模 planner 的真正 EP8。

S9 计算、权重搬运、panel restore、owner 梯度 SUM 内核及同步协议保持目标分支版本。
先前大 EP 接入、operand-ready 修复及融合校验实现已经在基线内，不属于本次新增。

## 2. 优化做了什么

设每 rank 输入 token 数为 M，每 token 选择 K 个不同专家；各 rank 的 M、K 相同。
原 owner 卡负载由全局逻辑专家计数求和，平均卡负载是 `M*K`。

```text
max_owner_load * 1000 <= M * K * direct_permille
```

`direct_permille=1050` 表示 max/mean 不超过 1.05 时保持原 owner 路由，copy 表为空；
超出阈值则执行原严格均衡 MoonEP。乘法使用 int64，不新增逐步 D2H、`.item()` 或主机调度。
依然保留全局 histogram、路由编码、容量/一致性检查、计划快照及完成检查；并未跳过整个 planner。

该路径仍由 S9 执行，不是把调用直接替换成原始 MegaMoe API。
不搬专家时减少的是额外副本权重搬移和相应 owner 归约工作，home 发布和正常反向仍在。

**阈值 1.05 是当前场景的已验证选项，不是所有 shape 的最优阈值。** 本次没有实现成本模型、
新 Native 算法、规划后动态分配大缓冲或 EP16～256 的自适应分流。

## 3. 开启方式与关闭方式

适用边界：Ascend950、单机 WORLD=EP=8、小规模 MoonEP planner、`slots=home<=16`。
禁止与强制 GM planner 混用。已验证的大形状为 M4096/H7168/F3072/TopK6、E32，即每卡 4 专家，slots=4。
大 EP 路径和通用默认行为不变。

先在自己的新源码目录、独立容器中准备固定依赖。具体宿主机参数见
[`test/megamoe_planner_test/README.md`](../test/megamoe_planner_test/README.md)。

```bash
source /path/to/cann/set_env.sh
export SHMEM_HOME_PATH=/path/to/pinned/shmem/install
bash test/megamoe_planner_test/build_benchmark.sh
bash test/megamoe_planner_test/build_routing_variants.sh
```

第一个脚本构建默认 baseline 和唯一的 `planner-bench` worker；第二个只新增两个隔离 planner SO：

| 目录 | 编译阈值 | 用途 |
|---|---:|---|
| `planner-variants/strict` | 0 | 原严格均衡策略 |
| `planner-variants/direct1050` | 1050 | 可保留原路由的策略 |

Python 使用示意（其余 shmem/group/shape/window/capacity 参数沿用调用方）：

```python
pool = NativeFSDP2Pool(
    ...,
    planner="moonep",
    backward_restore="panel",
    direct_permille=1050,
)
```

启动新进程前，将选定 planner 目录放在 `LD_LIBRARY_PATH` 最前。不能只改 Python 参数或只改源码；
Python 参数必须与实际加载的 SO 导出值一致，否则报错。不能在同一个已加载进程里切换 SO。
同一 EP 组各 rank 必须使用相同参数和构建。

关闭原路由策略：新进程使用 `strict` SO、`direct_permille=0`。
计划校验统一使用融合实现，模式参数仅作兼容，实际始终融合；下文 C/A 等校验对照仅为历史证据。

上述仅是算子 bridge 的显式接口；**本次没有新增整网训练配置开关，也没有做整网训练验收**。
容量仍须在调用前正确分配，以计划校验的 required 为准；本次没有降低 SHMEM 预留，也不能据此宣称显存不再随 EP 增长。

## 4. 如何复现对照

在宿主机按 benchmark README 设置 `LAUNCH` 数组，指定自己的容器 ID、`codex.task` 标签、
宿主机/容器源码路径、CANN/SHMEM/FSDPTurbo/MoonEP 路径。下列命令不会创建或接管容器。

```bash
SHAPE=(--tokens 4096 --hidden 7168 --ffn 3072 --experts 32 --slots 4 --topk 6)

# 四条链路：原始 MegaMoe、原 MoonEP 组件串行、本仓同计划串行、S9。
python3 test/megamoe_planner_test/campaign.py "${LAUNCH[@]}" \
  --name four-way-random --precision-workers 4 -- \
  "${SHAPE[@]}" --route random --planner moonep --warmup 10 --repeat 30

# A/B/C/C/B/A：仅策略、仅校验、两者组合分别归因。
python3 test/megamoe_planner_test/routing_campaign.py "${LAUNCH[@]}" \
  --name routing-random --precision-workers 4 -- \
  "${SHAPE[@]}" --route random --planner moonep --warmup 10 --repeat 30

# 计划切换、阈值两侧、非法计划与集体错误检查，不用于性能。
python3 test/megamoe_planner_test/gate_and_run.py "${LAUNCH[@]}" \
  --out routing-contract --routing-variant direct-fused -- \
  "${SHAPE[@]}" --route balanced_random --planner moonep --mode routing-contract
```

另三场景分别替换 route 为 `balanced_random`、`controlled --load-ratio 1.5`、
`controlled --load-ratio 2`，且使用不同 `--name`。balanced_random 是严格均衡计数；random 是自然轻微不均衡，二者不混用。

每次运行先精度后性能，检查五输出 Y/dX/dP/dW1/dW2、重复 epoch、池覆盖恢复和实际接收量。
同计划改动必须逐位一致，改变 placement 时使用原精度门槛及额外 L2 限制。
每次启动有双次占用检查，退出后复查；Warning/Alarm 仅在显式允许时接受，仍必须空闲。

正式计时为单层完整 online F/B，包括 planner、权重发布/恢复、owner SUM 和完成等待。
不包括离线精度检查、初始化或 FSDP AG/RS。先逐样本取八 rank 最大值，再统计 median/p95/std。
阶段同步诊断和 Timer/profiler 单独运行，不能替代正式完整耗时，也不能相加重叠阶段。

## 5. 历史有效收益：不是本次重新构建的结果

以下来自研究版本 `e78d204` 及结果提交 `3ef8741`，完整数据在
[`planner-megamoe-routing-20261029.md`](planner-megamoe-routing-20261029.md) 与同名 JSON。
历史报告中的实验目录/命令保留原样；**使用当前交付版本时按本说明启动，不照抄历史机器路径**。

这轮已完成 41 次真正 EP8 运行：16 次精度、24 次性能、1 次协议门禁；性能共 720 个 max-rank 样本。
六次计划切换及 17 类错误注入通过。四个大形状场景完整通过；后来未完成的 owner 索引实验不属于这轮。

A = 严格均衡 + 普通校验；B = direct1050 + 普通校验；C = direct1050 + 已有融合校验。
下表为完整 F/B 中位耗时变化 `候选/对照 - 1`；负值更快，正序/反序分开列出。

| 场景 | B/A：仅策略 | C/B：仅校验 | C/A：组合 |
|---|---:|---:|---:|
| 严格均衡 | +1.214% / -2.878% | -4.230% / -2.985% | -3.068% / -5.777% |
| 自然 random | -14.630% / -14.687% | -2.362% / -3.199% | -16.647% / -17.416% |
| 1.5 倍 | -1.015% / -0.136% | -2.627% / -2.172% | -3.616% / -2.304% |
| 2 倍 | -1.053% / +0.824% | -2.308% / -4.170% | -3.337% / -3.381% |

自然 random 的 max/mean 约 1.006714，旧策略仍创建 7 个副本，新策略降为 0；这是主要收益来源。
均衡旧计划本来没有副本；1.5/2 倍仍选择旧严格均衡计划，不能宣称原路由分流在这些场景有稳定收益。
2 倍组合的部分最大值/标准差仍有回退，不能声称所有尾部指标都改善。

**对照是旧严格均衡 S9，不是原始 MegaMoe，不是整网训练，也不是 GMM 主 kernel 的加速。**
早期四链路报告的 E16 和小形状结果另存本目录；不得与本表 E32 直接拼成一次实验。

## 6. 建议如何决定保留

| 项目 | 建议 |
|---|---|
| benchmark、输入生成、回执及占用检查 | 单独保留；即使撤销优化仍可用 |
| 原路由策略 | 保留显式开关，默认 0；当前自然 random 场景可试 1050 |
| 融合校验 | 复用已有实现；用 C/B 和同计划逐位一致证明独立收益 |
| 全局改默认、推广 shape/EP | 当前证据不足，须针对目标工作负载测量 |

采纳新 shape 前，先确认实际 max/mean、迁移副本数及容量，再完成四场景精度和双方向完整 F/B。
若只有一个顺序更快、收益落在波动范围内，或出现重要场景/尾延迟回退，应维持关闭或限制适用场景。
如果希望彻底撤销新增策略，可撤销第二个性能提交；第一个通用 benchmark 提交可独立保留。

以下探索留在 `work/planner-megamoe-bench`，不在本次生产增量中：

- 权重预取 MTE 双缓冲：random 两方向耗时回退约 2.65%/1.67%。
- owner SUM MTE 双缓冲：2 倍中位改善约 0.74%～1.31%，但其他场景及部分 p95 回退。
- owner contributor/地址缓存：已完成场景约 ±0.6%，方向不一致；1.5 倍未完成。
- 新 Native 成本模型、规划后动态缓冲分配、GMM 调度优化仅讨论，未作为已完成代码交付。

## 7. 当前交付回归状态

2026-10-29 在原有自有容器的新目录 `optimization/delivery-20261029-a1` 完成重新构建。
没有安装/替换共享环境，没有启动整网，没有停止其他人的任务。
结构化证据见 [`planner-megamoe-delivery-validation-20261029.json`](planner-megamoe-delivery-validation-20261029.json)。

| 验证 | 本次交付结果 |
|---|---|
| 原生构建 | baseline、S9 worker、strict/direct1050 planner 全部成功，Timer OFF |
| CPU | 18 项 benchmark 合约、7 项 Timer trace 合约、bridge 生命周期、operand ADL、阈值整数边界通过 |
| CPU 并行 checker | 与串行完整报告一致；NaN 和文件篡改拒绝通过 |
| 真正 EP8 计划门禁 | 八 rank 均通过 6 次切换和 17 类错误检查；前后占用门禁通过 |
| 小形状四链路精度 | M2049/H512/F1024/TopK2、E16、home=slots2、hot；四路五输出全部通过 |
| 新版本完整性能对照 | **未完成，不给出新的加速比** |

小形状所有参考对比的最大 relative-L2 为 0.00266814，低于追加门槛 0.05。
这不是声称所有输出都逐位相同；具体逐输出误差和 bitwise 标志在 JSON 中。
240 个源码/测试输入文件已核对远端构建树与本地交付树一致（统一 LF）。
代码指纹为 `e22446e2d4820a6374390ce578a79d81509ef73074b94144ef461f080c594899`。

运行记录共 6 次通过完整门禁、3 次未接受：

1. 首次计划门禁发现测试启动器漏设 planner 优先加载路径，实际加载默认库，阈值检查正确拒绝。
   已修复 `run_planner_megamoe.sh`，重新启动新进程和新结果目录。
2. 修复后的计划功能检查八 rank 均正常退出，但结束时出现外部 Python 设备管理器持有者，后门禁拒绝。
   此次不作为正式通过；后续重新双次检查空闲，重新运行后完整门禁通过。
3. 小形状四路精度及原始路径短计时通过后，原 MoonEP 串行短计时又在退出复查发现外部 Python 持有者。
   该次虽正常退出，仍被拒绝；流程停止，大形状 random/2 倍新回归尚未开始。

未接受的数据与单独原始路径的短采样均不用于性能结论。保持 Draft，后续在稳定空闲窗口完成
random/2 倍 A/B/C/C/B/A 回归后再决定是否正式采纳。第 5 节是可审阅的历史收益证据，不能当作此次新 worker 的重新测量。

| 本次库 | SHA256 |
|---|---|
| S9 worker | `b9ca378491df0f358cb9be6da3975b394cb0f87cadcf9630c95db7717d6a80b4` |
| strict planner | `7f17167f0ec055b37e95f4a4f4d161a93047010a6f99851e1f017e9609afec62` |
| direct1050 planner | `e63aa02d65ef1d90e5f07ef7d6fee68539fe3d7f99f5c590e6599f2f2f28774e` |

两个 planner SO 与历史路由实验哈希一致；worker 已按干净生产源码重新构建，须区分其测量身份。
完整紧凑证据归档 SHA256：`9712067bd47e0bc03398ba1cbc7456e40870831c03909f97be0a419eb34a9b16`。
原始张量保留在自有隔离目录；失败记录也保留，没有删除慢样本、放宽门禁或将失败改记为通过。
