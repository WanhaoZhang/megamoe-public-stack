# 共享专家反向 S0 NPU 验证报告

日期：2026-10-31。分支 `megamoe-balanced-shared-expert`，源码基线 `393174747fe76d4b8acf939aa725cb0a7f80a340`。本轮 **S0 小形状正确性与调用生命周期矩阵通过，功能默认保持关闭**。实施范围和接口见 [IMPLEMENTATION.md](IMPLEMENTATION.md)，运行方法见 [正确性测试说明](../../integration/wuhan/SHARED-EXPERT-BACKWARD-TESTS.md)。

本轮验证的是独立 PyTorch shared forward 提供实际 BF16 `X/rawGU/A` 后，同一 native launch 内执行 shared 串行前缀及原 routed backward。生产 routed forward 保留原实现。没有接入同事的融合前向，也没有完成 S1/P0 重叠、整网、原 rho 大形状、性能或 profiling 验收。

## 环境与构建身份

设备：`rack08os4`，Ascend950DT，world8，32 AIC/64 AIV。使用独立目录和容器，CANN9.2.0；其余原始版本输出归档于项目的 `node84/runtime-versions.txt`。新 host binding 本次重新构建，基础执行库、planner 和 validator 的指纹另见 `node84/BUILD-VERIFICATION.json`。

| 产物 | SHA256 |
|---|---|
| capability1，`shared-backward-s0-a2/libascend_deepep_megaflux_early_bf16.so` | `cc91d22261ed7541b2031317f0a4da98948165f5532ddab4e9a38a2684739fc4` |
| capability0，`shared-backward-default0-a2/libascend_deepep_megaflux_early_bf16.so` | `d9f39984126e5d3b9f449c6b565856f3ca01b56b072d9f06af4bd1fdaf5b39a4` |
| 新 `_C.cpython-312-aarch64-linux-gnu.so` | `3eaf821388e36f086fc039cde6648c50d0fc5b16334c823bfb8c5e9c543ca132` |

两库的 getter、导出入口、TimerOFF 和新 binding 身份检查通过。实测配置为 worker flags0、owner16、compact0、panel restore2、operand-ready1、restore delay0、C1=1、panel acquire cache1。capability1 导出 dedicated shared backward；capability0 不导出该入口，新 shared API 对 capability0 明确拒绝，旧 API 保留原路径。

CPU 构建、shared contract、63组 saved layout 交叉核对、既有 bridge lifecycle 和 early-DW1 identity 检查通过。a1 源码快照曾采用不同 dG FP32 乘法结合顺序，只完成编译，未用于设备验收；本轮设备结果全部来自对齐 Triton 结合顺序后的 a2 实现。a1/a2 是构建编号，下面的 `correctness-node84-a1-*` 是首次实际设备批次编号，两者不能混同。

## 已执行矩阵与分母

所有配置均为 BF16、TP1、无 bias、TopK2、home2/slots2。每个场景连续3次有效调用；表中 rank-call 是一次 backward 在一个 rank 上的调用。EP4/EP2 在同一 world8 作业内分别形成2/4个 EP 组。

| 配置目录后缀 | EP | M/H/routed F/shared Fs | shared limit | capacity | 场景 | 候选 rank-call |
|---|---:|---|---:|---:|---|---:|
| `ep4-tail` | 4 | 257/512/512/256 | 1 | 2176 | balanced、hot、rotate、boundary | 96 |
| `ep8-tail` | 8 | 257/512/512/256 | 1 | 4224 | balanced、hot、rotate、boundary | 96 |
| `ep2-tail` | 2 | 257/512/512/256 | 1 | 1152 | balanced、hot、rotate、boundary | 96 |
| `wide` | 4 | 129/1024/512/768 | 0.5 | 1152 | hot、rotate | 48 |
| `limit6` | 8 | 257/512/512/256 | 6 | 4224 | balanced、hot、rotate | 72 |
| 合计 | | | | | | **408** |

对应同一 capability1 SO 的旧 routed API 对照也是408次 rank-call；它是原 `pool.backward` 路径，不是另一个 feature0 库的设备对照。每次先运行原 routed backward，再对同一 forward 保存态和冻结 placement 以新 plan/epoch 运行 shared backward。固定外部 dY，不把不同 loss 或 forward 数值混入反向比较。

每组均有8个最终 rank 收据，合计40个；所有进程正常退出。每 rank 每组9项非法输入均在 native launch 前拒绝，共 `5×8×9=360` 次。真实输入和独立参考、routed 保存态共272个 PT 文件，逐文件大小和 SHA256 已全部核验。

## 正确性结果

- `dp/dWr1/dWr2` 相对同 SO 旧 routed API 逐位一致；shared 项未改变 routed owner 归并结果。
- 每个固定场景全部七输出连续3次逐位重复一致；热点切换和多 epoch 流程通过。
- `X/rawGU/A`、权重、dY、counts、dispatch 和 placement 等保存 payload 只读检查通过。opaque routed state 只排除原协议允许写入的256个 int32 backward progress counter，其余字节均检查。
- 9项非法输入覆盖 shared dtype、X stride、rawGU/A shape，以及 clip 的0、负值、NaN、Inf和FP32下溢；拒绝后后续有效调用继续通过。
- 尾行、Fs≠routed F、零 tile 核心、空 owner/slot、活动 replica、gate 单侧/up 双侧 clamp 的精确边界及 BF16 邻值均在所列矩阵内检查。

`dx_direct/dWg/dWu/dWd` 使用独立 PyTorch Linear、FP32 clamp/SiLU、BF16 activation、Linear autograd 参考，并直接沿用原 `test_bridge_npu.metrics` 判据，未修改阈值。`dx_direct` 参考为旧 routed BF16 dX 与独立 shared BF16 dX 再相加舍入；它不包含框架 router 对 X 的另一条梯度。下表为各配置所有 rank/场景/重复中的最大相对 L2：

| 配置 | dx_direct | dWg | dWu | dWd |
|---|---:|---:|---:|---:|
| ep4-tail | 8.951662e-5 | 2.238802e-5 | 9.850280e-5 | 0 |
| ep8-tail | 8.950983e-5 | 2.238802e-5 | 9.850280e-5 | 0 |
| ep2-tail | 8.387493e-5 | 2.238802e-5 | 9.850280e-5 | 0 |
| wide | 4.360981e-5 | 0 | 1.728609e-5 | 0 |
| limit6 | 8.888997e-5 | 3.229631e-5 | 1.046912e-4 | 0 |

全矩阵最大绝对误差为 `dx_direct=0.000244140625`、`dWg=0.0009765625`、`dWu=0.001953125`、`dWd=0`。四项均通过既有独立判据；这不表示四项全部与独立参考逐位相同。原始收据还保留完整 metrics，汇总值来源于 `RESULTS.json`。

## 准入、失败编号与清理

设备准入 a1/a2 因外部 `npu-smi` 持有管理设备失败，均保留失败记录且设备测试启动数为0。另用 a3 在同一 boot 完成两次准入后，才启动上述批次。五个配置各自的即时 pre/post 健康、进程、root fuser 检查及批次最终检查均通过，没有绕过占用门禁。

实验完成后仅停止本任务的 idle 容器 `megamoe-shared-backward-20261031-a1`；停止前后门禁通过，清理收据为 `pass_=true`、`stopped=true`、`other_containers_untouched=true`。其他容器未改动。清理证据位于 `node84/evidence/cleanup-node84-a1/`。

## 原始数据与复现索引

项目归档目录：`D:/megamoe_train/shared-expert-backward-impl-20261031`。本地回传根为其 `node84/` 子目录；远端原始项目根为 `/home/z00956592/megaflux/shared-expert-backward-20261031`，容器内映射为 `/project`。大文件、SO、输入和日志保留在项目目录，不纳入源码文档目录。

| 相对本地 `node84/` 的路径 | 内容 |
|---|---|
| `RESULTS.json` | 全矩阵汇总、分母、误差、构建身份和272个输入文件指纹 |
| `BUILD-VERIFICATION.json` | enabled/disabled SO、新 binding、基础库指纹及 getter |
| `runtime-versions.txt`、`preparation.json` | 隔离环境、版本及准备记录 |
| `evidence/correctness-node84-a1/completion.json` | 批次完成及最终准入结果 |
| `evidence/correctness-node84-a1-{ep4-tail,ep8-tail,ep2-tail,wide,limit6}/` | 每组原始 run、pre/post、rank 收据、逐调用 metrics、输入与保存态 PT |
| `evidence/admission-node84-a1/`、`admission-node84-a2/`、`admission-node84-a3/` | 两次拒绝及新编号准入记录 |
| `evidence/cleanup-node84-a1/` | 停止本任务容器前后检查及完成收据 |

每组中的 `rankN.json` 为最终收据，`iterations-rankN.jsonl` 为逐场景/逐 epoch 证据，`rankN-CASE-artifacts.json` 引用 `rankN-CASE-inputs.pt` 与 `rankN-CASE-routed-saved.pt` 的实际指纹。272个 PT 可以 CPU `torch.load(..., map_location='cpu', weights_only=True)` 读取；重新执行须创建新的合法 plan/epoch，不能重用旧地址或序列化 epoch。

本次实测的宿主控制脚本为项目根 `build-a2.sh`、`run-correctness.sh`、`run-queue.py`，分别固定构建参数、runtime 环境和五组参数/端口；源码内的通用启动说明见 [测试复现文档](../../integration/wuhan/SHARED-EXPERT-BACKWARD-TESTS.md)。后续复现使用新编号和空闲机器，保留相同数值判据与前后准入。

本轮不提供加速比、B/FB耗时或流水收益结论。下一阶段应先核对真实前向 `rawGU/A/X` 保存契约，再独立验证流水重叠候选；完整 B/FB 性能和大形状必须另立对照。
