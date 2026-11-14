# 共享专家反向 S0 实施说明

日期：2026-10-31。源码基线：`393174747fe76d4b8acf939aa725cb0a7f80a340`，分支 `megamoe-balanced-shared-expert`。**实验状态：S0 小形状 NPU 正确性矩阵通过，默认关闭**。实测范围和证据见 [NPU 验证报告](NPU-VALIDATION.md)；[原设计稿](BACKWARD-SHARED-EXPERT-DESIGN.md) 中的 S1/P0 重叠仍是后续候选，尚未实现。

当前交付已在 rack08os4 的 Ascend950DT 上完成 world8、EP2/4/8 的 S0 算子数值与调用生命周期验证，共408次候选 rank-call及408次同 SO 原 routed API 对照。尚无性能或整网通过结论。前向和生产框架 autograd 接入均未修改。测试用独立 PyTorch shared forward 生成保存态；它不是同事前向融合实现的替代品。

## 当前执行顺序

新增 torch class 方法 `backward_replica_shared`，通过独立 native 入口 `megaflux_worker_bf16_backward_shared`，在**同一个 mixed AIC/AIV native launch** 中执行：

```text
既有 routed descriptor/epoch/capacity 校验
  → shared S0：dA → 激活反向得到 dG/dU
              → dWd、dWg、dWu、dXg、dXu → BF16 shared dX 合并
  → shared CATLASS/TPipe 资源全部退休
  → 原 routed P0/P1/P2/C1/owner/P5/P6
  → 原 routed dX 在 BF16 边界后加 shared dX
  → 原最终同步和 finish
```

S0 在原 routed P0 之前完成，不与 P0、panel restore 或 routed GEMM 重叠。共享 GEMM 不拆 K，使用独立局部 shape 和完整 K 累加；不改 routed 专家顺序、dW1 cursor、tile publish、owner 分工、归并顺序或 operand-ready 协议。共享 GEMM、激活 VJP 和 dX 合并均在联合 native launch 内，其跨 AIC/AIV 就绪使用私有 cache-line 分离 flags 和精确 epoch；零 tile AIC 也参加就绪协议。host 的 control 清零等初始化仍在该 launch 之外，后续完整 B 计时必须包含这些成本。

共享专家是本地 dense FFN，区别于 planner 复制的 routed expert。`dWg/dWu/dWd` 是本 rank 本地梯度，**本入口不替框架执行共享参数的 DP/其他梯度归约**。routed 权重梯度仍走原 owner 归并。

实现位置：

- [torch class 接口](../../csrc/torch_classes.cpp)、[host 校验与私有分配](../../csrc/ops/mega_moe_bf16.cpp)、[launcher 能力校验](../../csrc/launcher.cpp)。
- [shared ABI/内存常量](../../kernels/mega_moe_bf16/include/mega_moe_shared_expert_types.h)、[S0 数学与同步](../../kernels/mega_moe_bf16/include/mega_moe_shared_expert_backward.h)。
- [独立 shared native 入口](../../kernels/mega_moe_bf16/megaflux_worker_bf16_backward_shared.cpp)、[原反向中的 prefix 和最终 dX 合并接点](../../kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h)。

## 接口与保存态

方法保留旧 `backward_replica` 的完整参数前缀，在尾部追加 7 项：

```python
dx_direct, dp, dWr1, dWr2, dWg, dWu, dWd = op.backward_replica_shared(
    shmem, grad_y, topk_weights, weight1, weight2,
    expert_token_nums, training_saved_state, dispatch_output,
    max_recv_tokens, swiglu_limit, save_gmm1_output, replica,
    X, Wg, Wu, Wd, rawGU, A, shared_limit)
```

| 数据 | 形状/约定 |
|---|---|
| `grad_y`、`X`、`dx_direct` | `[M,H]`，X 必须是 shared forward 使用的原本地 token 顺序，不能传 routed packed X |
| `Wg`、`Wu`、`dWg`、`dWu` | `[Fs,H]`，canonical contiguous |
| `Wd`、`dWd` | `[H,Fs]`，canonical contiguous |
| `rawGU` | `[M,2Fs]`，每行前 Fs 为 raw gate、后 Fs 为 raw up；必须保存 clamp 前值 |
| `A` | `[M,Fs]`，实际 shared forward 产生并供 down projection 使用的 BF16 activation |
| `dp` | 原 `[M,TopK]` 语义；不加入 shared 项 |
| `dWr1/dWr2` | 原 routed canonical 梯度布局；原始方法返回物理专家轴，框架/测试沿用原 owner-prefix 截取方式 |

所有数值输入须为 BF16、连续、与 `grad_y` 在同一 NPU；counts、opaque state、dispatch 保留原 dtype 和布局。`save_gmm1_output` 必须为 `bf16`，`replica` 必须是有效的 26-word S9 early descriptor。新入口消费原 forward 的 counts/state/dispatch 和冻结 placement，新建 replay epoch，不能拿其他调用的状态拼接。

`X/rawGU/A` 只读，必须存活到对应 backward 完成。当前没有新增 production autograd tape、UAM 保存策略、共享重算或 X redispatch；调用方应先确保前向保存数据及 live shared weights 的生命周期。测试适配器独立保持这些张量，不更改生产 `NativeFSDP2Pool`。原 routed saved-state 的 256 个 int32 backward progress counter 仍允许按原协议写入，其余保存 payload 不应改变。

## 数学及 BF16 舍入位置

以下 `B(v)` 表示舍入为 BF16，`float(v)` 表示转 FP32；G/U/A 是上述实际前向保存数据，`L=shared_limit`。

```text
g = min(float(G), L)                         # gate 只有上界
u = min(max(float(U), -L), L)                 # up 为双侧 clamp
s = sigmoid(g)
A = B((g * s) * u)                           # 由 forward 保存，不在 B 中重算

dA = B(dY @ Wd)
dsilu = s * (1 + g * (1 - s))
dG = B((float(dA) * u) * dsilu * [G <= L])
dU = B((float(dA) * (g * s)) * [-L <= U <= L])

dWd = B(dY.T @ A)
dWg = B(dG.T @ X)
dWu = B(dU.T @ X)
dXg = B(dG @ Wg)
dXu = B(dU @ Wu)
dXshared = B(float(dXg) + float(dXu))
dx_direct = B(float(dXrouted_BF16) + float(dXshared))
```

激活反向的 sigmoid/中间乘法在 FP32 中计算，G/U clamp 的等号处保留导数。当前 dG 的 FP32 乘法结合顺序按上式保留：先算 `dsilu`，再算 `(dA*u)*dsilu`。a1 构建快照曾采用不同结合顺序，未用于设备验收；实测使用修正后的 a2 库。不能把 gate 改成双侧 clamp，不能用 clamp 后 GU 判断是否越界，也不能把两次 dX GEMM 改成一个 FP32 累加后才舍入的表达式。原 routed dX 先保持原 BF16 舍入，再合并 shared dX；`dx_direct` 不包含框架 router 对 X 的另一条梯度路径。独立 shared 参考采用未改阈值的数值判据，不声称四项独立参考输出全部逐位相同。

## 私有内存与生命周期

新增内存全为调用私有 NPU 普通 GM tensor，不扩展原 SHMEM window 或原 routed workspace 布局。按当前 ABI1、BF16 每元素 2 bytes：

| 项目 | 大小（bytes） |
|---|---:|
| `dA/dG/dU` scratch | `6*M*Fs` |
| `dXg/dXu` scratch，最终 shared dX 复用 dXg | `4*M*H` |
| `dWg/dWu/dWd` 输出 | `6*H*Fs` |
| control，预留 7 phase × 64 participant × 16 int32 | `28,672`（28 KiB） |
| 新增私有分配合计 | `6*M*Fs + 4*M*H + 6*H*Fs + 28,672` |

这不包含原 routed 分配、allocator 对齐/缓存、模型已有权重和保存态。若单独计算调用方保留 `X/rawGU/A` 的逻辑数据量，则为 `2*M*H + 6*M*Fs` bytes；X 若已经由上层保留，不应重复计作新增拷贝。device 参数结构 152 bytes 按值传入 launch，不另分配持久 GM 参数块。

每次调用创建 fresh scratch/control，control 清零、epoch 从 1 开始；与 routed replay epoch 分属两个协议。指针只用于本次当前 stream，不缓存到下一次调用。shared prefix 完成并退休资源后再进入原 P0；原 dO/packed input/partial/checkpoint 生命周期不变。当前没有跨调用池化、异步复用或双缓冲。

## 默认关闭与构建

`MEGAFLUX_SHARED_EXPERT_BACKWARD=0` 是 CMake 和构建脚本默认值。显式设为 1 才编译 dedicated shared entry。当前只允许 TimerOFF 和原 C1 panel 排程，构建拒绝 TimerON 或 C1 关闭；C1 的互斥约束继续生效。

以下命令只在新的隔离源码/依赖环境中执行。`ROOT` 为本次独立源码目录，CANN、SHMEM、CATLASS 及原基础执行库须已准备；不要指向同事运行目录。构建名必须全新，已有 build manifest 的 variant 不覆盖。

```bash
cd "$ROOT"
export MEGAFLUX_SHARED_EXPERT_BACKWARD=1
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1
export MEGAFLUX_RESTORE_DELAY_CYCLES=0 MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_WORKER_SHARED_PREFIX=0 MEGAFLUX_WORKER_EXTERNAL_REFRESH=0
export MEGAFLUX_OWNER_ROWS=16 MEGAFLUX_OWNER_COMPACT=0
export MEGAFLUX_DISPATCH_READY_LARGEST=0 MEGAFLUX_MOONEP_TIMER=0
export MAX_JOBS=2
bash scripts/build_megaflux_worker_pipeline.sh shared-backward-s0-repro-a1 0 0 0

# 新 torch class/host launcher 同样必须重建；只换 worker SO 不足以增加接口。
python3 setup.py build_ext --inplace

export WORKER_NATIVE="$ROOT/worker-variants/shared-backward-s0-repro-a1"
export LD_LIBRARY_PATH="$WORKER_NATIVE:$ROOT/ascend_deepep/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
```

若隔离目录尚无 MoonEP planner/plan validator，可在同一配置的 `build-worker-shared-backward-s0-repro-a1` 中构建 `ascend_deepep_moonep_planner`、`ascend_deepep_plan_validator` 两个 target，并将生成库放入本次 `WORKER_NATIVE`，与既有测试部署方式一致。上述 worker 脚本的 `build.json` 记录 `shared_expert_backward/shared_expert_backward_abi`、C1 和 native SHA；仍应另记录 binding 与输入指纹。上述 `repro-a1` 是新复现编号；本次已验收的实际库名为 `shared-backward-s0-a2`，其 panel acquire cache 为1，具体指纹见 [NPU 验证报告](NPU-VALIDATION.md)。

加载库须报告 `megaflux_shared_expert_backward_version()==1`。host 还校验 flags0、owner16、compact0、restore2、operand-ready1、delay0、C1=1 和两个 Timer getter 均为 0，以及 dedicated symbol 存在。形状、dtype、stride、device、clip limit 和能力不符时，在清理 routed window/启动 native kernel 前抛错。`shared_limit` 必须在 double 和转成 FP32 后都正且有限，包含下溢至 0 的正 double 也拒绝。

不要使用会同时构建 OFF/ON 配对的 `build_megaflux_timer_capture.sh` 来构建 shared ABI1；当前 shared 入口只支持 TimerOFF，尚未扩展 Timer schema/profiling。

## 回退与支持边界

旧 `backward_replica`、原 forward 和旧 Python 调用协议不变。在 feature0 库上继续调用旧入口即走原 routed 反向；新 shared 方法会明确拒绝 capability0 库，不自动漏掉 shared 梯度。完整回退需要上层恢复“原 routed backward + 原框架 shared autograd”的两条路径，不能仅切回旧 routed 入口就宣称完成 shared backward。

当前代码支持边界是 EP2/4/8、TP1、无 bias、BF16、canonical routed/shared 权重、有效 S9 replica descriptor、保存 BF16 GU、panel restore2/operand-ready1。M 为 1..16384；H/Fs 为正的 256 倍数且不超过16384；Fs 与 routed F 可不同，routed F 和容量仍遵循原算子约束；AIC 数量不超过32。没有新支持 TP>1、共享权重分片归约、MXFP8/none saved mode、bias、其他激活分支、UAM shared-recompute/offload、X redispatch、checkpoint/全网络 tape 接入或新的 profiler schema。

S1/P0 重叠、AIV1 空隙复用、事件优化、内存池化和性能选择均未实现。S0 是否值得保留，要在正确性通过后另做固定条件完整 B/直接 FB 对照，不能凭减少 launch 数推断收益。

## 测试与当前证据

完整参数矩阵和启动命令见 [SHARED-EXPERT-BACKWARD-TESTS.md](../../integration/wuhan/SHARED-EXPERT-BACKWARD-TESTS.md)。入口为：

```bash
python3 integration/wuhan/test_shared_expert_backward_contract.py
torchrun --nnodes=1 --nproc-per-node=8 --master-port=32400 \
  integration/wuhan/runtime_bootstrap.py --native-only \
  integration/wuhan/test_shared_expert_backward_npu.py \
  --out "$RUN_ROOT/s0-ep4-tail-a1" --ep 4 --tokens 257 --hidden 512 \
  --ffn 512 --shared-ffn 256 --limit 1 --repeat 3 --shmem-port 32410
```

设备启动前仍由执行者完成健康、进程和 root fuser 准入；占用或门禁失败不得启动。测试固定外部 dY，以独立 PyTorch Linear/SiLU/clamp autograd 生成参考，沿用旧数值判据；原 routed `dp/dWr1/dWr2` 必须逐位一致。检查 EP2/4/8、尾行/零 tile 核心、Fs≠F、热点切换、正负 clamp 边界/邻值、3 次 epoch、保存态只读和9项 host 拒绝。

| 层次 | 当前状态 |
|---|---|
| Python 语法/CLI | 通过本地 AST、`--help` 检查 |
| 只读排除区间 | 63 组形状与原 saved-layout parser 交叉检查通过 |
| 新 worker/binding 构建 | capability1/capability0 两库及新 binding 构建、getter和身份检查通过；指纹见 [验证报告](NPU-VALIDATION.md) |
| CPU 检查 | shared contract、63组 saved layout、既有 bridge lifecycle/early-DW1 identity 通过 |
| NPU 算子数值/生命周期矩阵 | **S0 小形状矩阵通过**：408次候选 + 408次同 SO 旧 routed API 对照，360次非法输入拒绝 |
| TimerOFF B/FB 性能与 profiling | 未验收；当前无 shared TimerON 支持 |
| 框架前向保存契约联调/整网 | 未接入、未验收 |

每 case 在开始迭代前保存真实 CPU tensor 输入与 shared oracle 四输出为 `rankN-CASE-inputs.pt`；首轮 routed forward 后、两个 B 前保存 counts/state/dispatch/冻结 placement/status 为 `rankN-CASE-routed-saved.pt`。两份文件使用独占创建防覆盖，case manifest、逐轮及 rank 收据记录实际文件 SHA256 和字节数。它们包含复现所需 tensor 和普通 math/seed/version/shape metadata，不包含全环境、凭据或 device 指针；后续接入同事 forward 可直接逐项核对 rawGU/A 保存契约。

逐 rank JSON/JSONL/PT、配置、SO、日志和复现产物归入本次原项目实验目录 `D:/megamoe_train/shared-expert-backward-impl-20261031`；本地回传位于 `node84/`，远端原始路径为 `/home/z00956592/megaflux/shared-expert-backward-20261031`。汇总、矩阵、数值误差、准入失败编号及清理证据见 [NPU 验证报告](NPU-VALIDATION.md)。大文件不放入源码文档目录，不据本轮结果推断 S1、原 rho 大形状、生产前向融合或性能通过。
