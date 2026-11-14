# MegaMoe BF16 Backward V1 — 交接文档（给 Ascend950 验证 agent）

> 2026-09-07 集成说明：本文保留 train 旧训练接口的设计/实现记录。
> 当前 raw 前向已切换三输出 saved-state，旧训练入口暂时明确报错。
> 反向适配请以 [新持久状态与迁移清单](MegaMoe_BF16_前向持久Workspace第一版与反向适配.md) 为准；
> 下文旧 workspace 偏移、接口示例和历史验证结论不能直接用于新 ABI。

> 本文档由实现方（Claude，2026-08-26）写给下一步在 Ascend950 环境执行验证的 agent。
> 配套参考：`docs/mega_moe_bf16_backward_v1_agent_guide.md`（实现契约）、
> `docs/mega_moe_bf16_backward_v1_implementation_note.md`（实现细节与风险点）。
> 请先读这两份再动手。

---

## 1. 任务与交付范围

按 `docs/mega_moe_bf16_backward_v1_agent_guide.md` 实现了 MegaMoe BF16 训练反向算子 V1：

- 训练前向保存上下文（packed_x + routing metadata + count 表）
- 反向 device kernel（P0–P5 单 kernel 融合）
- host launcher / C++ op / Python `autograd.Function`
- EP 多进程精度测试

**代码位置**：

| 项 | 值 |
|---|---|
| 本地 worktree | `C:\MegaMoe\ascend_deepep_hkq\.claude\worktrees\megamoe-bf16-backward-v1` |
| 分支 | `feat/megamoe-bf16-backward-v1`（gff fork） |
| 提交 | `f90dad5`（反向实现）→ `9a5a719`（交接文档）→ `005a500`（编译修复）→ `b2e646b`（本版交接文档更新） |
| 单提交补丁 | `C:\MegaMoe\ascend_deepep_hkq\mega_moe_bf16_backward_v1.patch`（148KB，`git apply` 即可） |

> ✅ **代码已推送**：分支 `feat/megamoe-bf16-backward-v1` 已推送到 gff fork（https://gitcode.com/guofanfeng23/ascend_deepep_hkq.git），含 `f90dad5`（反向实现）、`9a5a719`（交接文档）、`005a500`（编译修复）、`b2e646b`（本次交接文档更新）。验证 agent 直接 `git clone -b feat/megamoe-bf16-backward-v1 https://gitcode.com/guofanfeng23/ascend_deepep_hkq.git` 即可。

---

## 0. 相对上一次交接（9a5a719 → b2e646b）的新动作

> 上一版交接（`9a5a719`）时代码**未编译、未运行**，只做过静态走查。本版（`b2e646b`）新增了**远程实测编译**，并修复了编译期发现的全部问题。

### 0.1 本次做了什么

1. **上远程验证服务器 148 编译**（2026-08-26）：按用户指示"先编译这个机器的"。148 是 910B3 × 8 卡，无 Ascend950/A5 硬件，但**能 a5 跨编译**（CANN 9.0.1，bisheng 2026-04-25，新前端 `-xasc --npu-arch=dav-3510`）。
2. **编译结果**：`[100%] Built target ascend_deepep_mega_moe_bf16`，`libascend_deepep_mega_moe_bf16.so` 生成。前向 `mega_moe_bf16.cpp` 与反向 `mega_moe_bf16_backward.cpp` **均编译通过**。
3. **发现并修复了编译期全部问题**（commit `005a500`，9 个文件）——详见 §0.2。
4. **确认 a2（910B3 原生）无法构建**：`CMakeLists.txt:184` 硬性要求 `ASCEND_DEEPEP_BISHENG_ARCH=a5`，BF16 kernel 只能编 a5。故本机只做跨编译验证。
5. **明确下一步仍需 Ascend950**：编译通过 ≠ 运行正确。本机无 a5 硬件，loss.backward() 精度/同步死锁等运行期结论**仍未成立**，必须到 Ascend950 跑 §4.2 的测试。

### 0.2 编译修复清单（commit `005a500`）

| 文件 | 修复 | 性质 |
|---|---|---|
| `cmake/AscendDeepEPKernel.cmake` | 新前端编译选项加 `-D__ASC_NPU_HOST__`；否则 arch35 路由看不到 `AscendC::Reg::CastTrait`/`__local_mem__` 等 API，`moe_v3_*.h` 全线编译失败 | 工具链兼容（任何 BF16 编译都需要） |
| `kernels/mega_moe/include/op_kernel/math_util.h` | 用 `MEGA_MOE_HOST_AICORE` 宏桥接 `__host__ __aicore__` 与新前端单 token `__host_aicore__` | 工具链兼容 |
| `kernels/mega_moe/include/moe_init_routing_v3/arch35/moe_v3_common.h` | `VECTOR_REG_WIDTH` 在 CANN 9 namespace 作用域不可见，`FLOAT_REG_TENSOR_LENGTH` 改硬编码 `256` | 工具链兼容 |
| `csrc/ops/mega_moe_bf16.cpp/.h` | 去掉 `mega_moe_tiling_host.h` include（其非 inline 定义与 `mega_moe.cpp` ODR 冲突 → multiple definition）；删除无用 `CalcMegaMoeBF16TrainWorkspaceBytes`；train workspace 的 `forward_bytes` 改为调用点传 `hostTiling.workspace_bytes` | 我的代码（host 链接错误） |
| `mega_moe_bf16_backward_catlass_impl.h` | TileCopy 参数化 LayoutA/LayoutB：P1/P2 用 **RowMajor B**、P3/P4 用 **ColumnMajor A**（原先硬编码 B=ColumnMajor，与张量 layout 不匹配，`TileCopyTla` 找不到特殊化） | 我的代码（布局风险点，交接文档 §3 曾预判） |
| `block_epilogue_swiglu_backward_bf16.h` | 平面地址裸 `__gm__` 指针改 `GlobalTensor`+`SetGlobalBuffer`；GM→UB 用 4 参 `DataCopyPad`；修 `CopyPlaneTileToUb` 的 src/dst stride 顺序（原写反） | 我的代码（API 用法） |
| `mega_moe_bf16_backward.h` | workspace layout 类型命名空间限定；5 个 `BackwardGroupMatmulParam` 调用点改新模板签名；`GatherRoutes` 的标量 bf16→fp32 `static_cast` 改位扩展（**A5 backend 无标量 bf16→fp32 指令**，报 "not support bf16 type cast"） | 我的代码（backend 限制，二分定位） |
| `mega_moe_workspace_info.h` | 合并 forward/backward workspace 布局，删除独立 train workspace 头文件 | 我的代码 |

### 0.3 关键经验（供验证 agent 参考）

- **build_mega_moe_bf16.sh 硬性要求 `-soc_type Ascend950`**，且 `CMakeLists.txt` 也强制 a5 —— 在 910B3 机器上是"跨编译 a5、本机不能跑"。
- 报 "not support bf16 type cast"（backend 错误，无源码定位）时，用**注释掉 kernel 的阶段函数做二分**可快速定位（本案例定位到 `GatherRoutes` 的标量 bf16 cast）。
- 148 上已验证的构建流程与脚本：远端克隆 `/home/g00897531/ascend_deepep_bf16_backward_v1`，构建脚本 `a5_build.sh`，日志 `/home/g00897531/a5_build.log` 等，可直接复用。

### 0.4 内部 A5 测试反馈的挂死修复（commit `bcca28d`）

**现象**：内部 A5 测试反馈反向**完全卡死、永不返回**，M=16 也超时；定位到 P3/P4（两个 dW 阶段）。

**根因**：P3/P4 用了 **ColumnMajor-A**（`dW1=dZ^T@Xpacked` / `dW2=dO^T@H` 的转置视图）。catlass A5 对 **A 操作数的原生路径是 RowMajor**（Nd2Nz）；ColumnMajor→nZ 的 `TileCopyTla` 数学正确但非原生，在真硬件上 DMA 挂死。

**修复（"最基础的 gmm"）**：
1. **AIV 转置** dZ→`ptrGradZT`[N,rowsTotal]、dO→`ptrDOT`[K,rowsTotal]（新增 workspace 缓冲，device+host 镜像同步）。
2. **GMM1DW/GMM2DW 改 RowMajor A + RowMajor B**（RowMajor 已在 P1/P2 数据梯度阶段验证），从转置平面按 `expertRowBase` 列带取专家切片。
3. **AIC 在 P3 前等待新增的 `ptrBwFlagTransposeDone`**。
4. **阶段进度标记** `ptrBwProgress`：P0A~P5 及三个 spin 点（AIV1READY/TRANSPOSE/HDONE）写每 block 的进度值。**若仍挂死**，A5 上读该缓冲区最后一个值即可定位卡在哪一阶段/spin。

**状态**：`bcca28d` 已在 148 上 a5 跨编译通过（`Built target ascend_deepep_mega_moe_bf16`），已推送 gff。**运行验证仍需 A5**；若仍挂，读 `ptrBwProgress`（workspace 尾部 256×int32，索引=blockIdx%256，值见 `BW_PROG_*` 常量）回报最后写入值。

### 0.5 致命 shape bug 修复：problemShape 的 N/K 从未初始化（commit `b20afaf` + `618dc82`）

**现象（代码走查发现，未到 A5 即确认）**：反向所有阶段 GEMM 从 `problemShape_`（TupleShape{M,N,K}）读维度，但全仓只有 `UpdateGroupParams` 写 **M** 位置（=每专家 token 数 Me），**N/K 位置从未赋值**（值初始化=0；前向在 `mega_moe_bf16.h` Init 里有 `N_VALUE=n / K_VALUE=k`，反向 Init 漏了这层）。后果：`BlockScheduler` 对 `{m, 0, 0}` 算出 `coreLoops=0`，所有 GEMM 空转 → 全零梯度或非法零尺寸 MMAD 挂死，**任何 A5 精度测试都无法通过**。

**修复**：每个阶段在 GEMM 循环前设置自己的 N/K（DW 阶段额外把 M 覆盖为输出维度、并先存 Me 供 `cumMRows_` 累加）：
- P0-C Recompute：`N=n_, K=k_`（`{m=Me, n=N, k=K}`）
- P1 GMM2Grad：`N=n_/2(F), K=k_`
- P2 GMM1Grad：`N=k_(K), K=n_(N)`
- P3 GMM1DW：`M=n_, N=k_, K=Me`
- P4 GMM2DW：`M=k_, N=n_/2(F), K=Me`

`618dc82` 追加限定 `MegaMoeBF16BackwardCatlassImpl::SWIGLU_N_HALF`（using 指令把两个命名空间的同名常量都引入，编译歧义）。

**状态**：`618dc82` 已在 148 上 a5 跨编译通过（`Built target ascend_deepep_mega_moe_bf16`），已推送 gff。**这是运行正确性的前置修复**——A5 验证必须先含这两个 commit。

---

## 2. 完成情况（已交付）

### 2.1 新增文件

| 文件 | 作用 |
|---|---|
| `kernels/mega_moe_bf16/mega_moe_bf16_backward.cpp` | 反向 kernel 入口 + `mega_moe_bf16_backward_kernel_do` ABI |
| `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h` | `MegaMoeBF16Backward` 主类，P0–P5 各阶段 + 同步 |
| `kernels/mega_moe_bf16/include/mega_moe_bf16_backward_catlass_impl.h` | 反向 GEMM 包装（重计算/GMM2Grad/GMM1Grad/dW1/dW2） |
| `kernels/mega_moe_bf16/include/block_epilogue_swiglu_backward_bf16.h` | 反向 SwiGLU epilogue（RecomputeGMM1/SwiGLUGrad/SwiGLURecompute） |
| `test/megamoe_bf16_test/test_mega_moe_backward.py` / `.sh` | EP 多进程 loss.backward() 精度测试 |

### 2.2 修改文件

| 文件 | 改动 |
|---|---|
| `kernels/mega_moe_bf16/include/mega_moe_bf16.h` | 训练模式 `SaveTokenPerExpert()`（ResetTokenPerExpert 前保存 count 表） |
| `kernels/mega_moe_bf16/include/mega_moe_workspace_info.h` | `ForwardWorkspaceInfo` 持久前向布局与独立 `BackwardWorkspaceInfo` |
| `kernels/mega_moe_bf16/mega_moe_bf16.cpp` | BF16 训练前向统一使用 `mega_moe_bf16_kernel_do`，固定保存 backward context |
| `kernels/mega_moe_bf16/include/mega_moe_base.h` | `GMMAddrInfo::cGlobal`（dW 输出地址） |
| `include/ascend_deepep/kernels/launch.h` | 两个新 ABI 声明 |
| `csrc/launcher.{h,cpp}` | 统一 `LaunchMegaMoeBF16Kernel` + 显式接收 saved/scratch 的 `LaunchMegaMoeBF16BackwardKernel` |
| `csrc/ops/mega_moe_bf16.{h,cpp}` | 统一 `MegaMoeBF16Forward` / `MegaMoeBF16Backward` + saved/backward layout 大小计算 |
| `csrc/torch_classes.cpp` | 原有 `MegaMoeBF16` TorchBind 类新增 `.compute_train` / `.backward` |
| `ascend_deepep/__init__.py` | 新增公开 `MegaMoeBF16Train(nn.Module)` + 私有 autograd bridge |
| `CMakeLists.txt` | BF16 kernel 库加入 `mega_moe_bf16_backward.cpp` |

### 2.3 阶段流水（单 kernel 内）

```
P0-A  加权 gather      grad_y * topk_weights -> peermem.ptrA0          （AIV0+AIV1 分核）
P0-B  恢复 count 表 + Dispatch dO -> ptrDO                             （AIV0）
P0-C  GMM1 重计算      Z = Xpacked @ W1.T -> gate/up 平面              （AIC 算，AIV1 拷出）
P1    GMM2 Grad ‖ SwiGLU Grad：dH = dO@W2 ; dZ = [dGate|dUp]           （AIC+AIV1，单缓冲 tile 流水）
P2    GMM1 Grad ‖ Combine + SwiGLU 重计算：dX = dZ@W1 ; H              （AIC+AIV0 ping-pong；AIV1 并行）
P3    GMM1 dW         dW1 = dZ.T @ Xpacked -> grad_weight1             （纯 AIC，L0C->GM）
P4    GMM2 dW         dW2 = dO.T @ H -> grad_weight2                   （纯 AIC，L0C->GM）
P5    Unpermute       grad_x = sum_slot(ptrD)                          （AIV0+AIV1 分核）
```

### 2.4 Python 用法（验证时直接照抄）

```python
from ascend_deepep import MegaMoeBF16Train, ShmemInstance

shmem = ShmemInstance(rank, ep, ip_port)
shmem.set_data_op_engine_type(0x08)          # ACLSHMEM_DATA_OP_UDMA
shmem.init(local_mem_bytes)
op = MegaMoeBF16Train(shmem, local_mem_bytes)

y, counts = op(x, topk_ids, topk_weights, w1, w2, max_recv)
loss = (y * grad_y_target).sum()
loss.backward()                              # x.grad / w1.grad / w2.grad
```

## 3. 诚实声明（已验证项）

### 3.1 编译状态（2026-08-26 已在 148 上实测）

**已编译通过**：`mega_moe_bf16_backward.cpp` 在 148（910B3 机器，CANN 9.0.1，bisheng 2026-04-25，新前端 `-xasc --npu-arch=dav-3510`）上 **a5 跨编译成功**，`[100%] Built target ascend_deepep_mega_moe_bf16`，`libascend_deepep_mega_moe_bf16.so` 生成。前向 `mega_moe_bf16.cpp` 同时编过。

为达到编译通过，需要以下修复（均已提交到分支 `feat/megamoe-bf16-backward-v1`，commit `005a500`）：

| 文件 | 修复 |
|---|---|
| `cmake/AscendDeepEPKernel.cmake` | 新前端编译选项加 `-D__ASC_NPU_HOST__`（否则 arch35 路由看不到 `AscendC::Reg::CastTrait` 等 API） |
| `kernels/mega_moe/include/op_kernel/math_util.h` | 用 `MEGA_MOE_HOST_AICORE` 宏桥接 `__host__ __aicore__` 与新前端单 token `__host_aicore__` |
| `kernels/mega_moe/include/moe_init_routing_v3/arch35/moe_v3_common.h` | `VECTOR_REG_WIDTH` 在 CANN 9 namespace 作用域不可见，改硬编码 `256` |
| `csrc/ops/mega_moe_bf16.cpp/.h` | 去掉 `mega_moe_tiling_host.h` include（其非 inline 定义与 `mega_moe.cpp` ODR 冲突）；删除无用 `CalcMegaMoeBF16TrainWorkspaceBytes`；train workspace 的 forward_bytes 改为调用点传入 `hostTiling.workspace_bytes` |
| `mega_moe_bf16_backward_catlass_impl.h` | TileCopy 参数化 LayoutA/LayoutB：P1/P2 用 RowMajor B、P3/P4 用 ColumnMajor A（原先硬编码 B=ColumnMajor 与张量 layout 不匹配，`TileCopyTla` 找不到特殊化） |
| `block_epilogue_swiglu_backward_bf16.h` | 平面地址改 `GlobalTensor`+`SetGlobalBuffer`；GM→UB 用 4 参 `DataCopyPad`；修 `CopyPlaneTileToUb` 的 src/dst stride 顺序 |
| `mega_moe_bf16_backward.h` | workspace layout 类型命名空间限定；5 个 `BackwardGroupMatmulParam` 调用点改新模板签名；`GatherRoutes` 的标量 bf16→fp32 `static_cast` 改位扩展（A5 backend 无标量 bf16→fp32 指令） |

### 3.2 仍未验证项（编译通过 ≠ 运行正确）

1. **未运行、未做精度验证**：148 是 910B3，无 Ascend950/A5 硬件，a5 跨编译产物**无法在本机运行**。loss.backward() 正确性、梯度数值、跨核同步死锁等一切运行期结论仍未成立，必须到 Ascend950 环境验证。
2. **布局支持已由编译确认，但运行语义未确认**：RowMajor B（P1/P2）与 ColumnMajor A（P3/P4）已能通过 catlass 编译，但 dW 的转置视图、grad 数值方向是否与参考实现一致，仍需运行验证。
3. **UB 预算**：`block_epilogue_swiglu_backward_bf16.h` 的 `static_assert(3*kTileBytes <= UB_SIZE)` 编译通过（=192KB），A5 UB 足够；若运行期有其他 UB 分配冲突需运行时观察。
4. **forward_workspace 必须按 forward 配对**：host 每次 `compute_train` 新建并返回 forward workspace；backward scratch 每次 backward 重新申请并清零。
5. **device/host layout 必须一致**：`ForwardWorkspaceInfo` / `BackwardWorkspaceInfo` 与 host 对应 layout 是镜像，改任一侧必须同步另一侧。
6. **a2（910B3）原生构建被架构性阻断**：`CMakeLists.txt:184` 硬性要求 `ASCEND_DEEPEP_BISHENG_ARCH=a5`，BF16 kernel 只能编 a5。

## 4. 下一步：Ascend950 环境验证（交接给验证 agent）

### 4.1 前置确认（guide §14.2）

开始前必须确认以下至少一项，否则不能把 910B3 结果当作 Ascend950 反向通过证据：

- 148 上存在可进入的 Ascend950 容器/节点；或
- 目标运行环境需要额外脚本/调度命令进入；或
- 项目方明确允许在 910B3 增加并验证兼容路径。

### 4.2 验证顺序（严格按此）

```bash
# 1) 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
python -c "import torch_npu; print(torch_npu.__version__)"

# 2) 代码同步（三选一，见 §1）
git apply mega_moe_bf16_backward_v1.patch
# 或 scp 整个 worktree / git bundle / 有权限会话 push

# 3) 构建基线（148 上已验证：kernel 编译通过 + pip install -e . 通过）
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950   # 需已含 commit 005a500 的编译修复
#   产物：build_mega_moe_bf16/lib/libascend_deepep_mega_moe_bf16.so

# 4) 前向回归（必须先过）
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1

# 5) 反向精度（小 shape 起）
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1 --mode precision

# 6) 覆盖场景
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1 --mode precision --routing-mode imbalanced   # 空 expert / 负载不均
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1 --mode precision --topk 2 --routing-mode random
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3 --mode precision   # EP=4
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3,4,5,6,7 --mode precision  # EP=8
# 空 expert 用 --routing-mode imbalanced（随机权重，低号 expert 会空）
# topk_weight=0 / capacity 边界：在测试里临时把部分 weight 置 0，或加大 M 触发 cap
```

测试默认参数：M=64, K=512, N=1024, TOPK=8, expertPerRank=4, 容忍度 atol=0.05/rtol=0.15/max 5% 错误比。
参考实现 `reference_backward`（float32 全量路由重放）按 EP 切片比较 grad_x（本 rank M 行）、grad_w1/grad_w2（本 rank E 个 expert）。

### 4.3 必须记录并回报的结果

```text
- 远端 commit/diff（用的哪个方式拿到代码）
- 服务器与设备型号（确认是 Ascend950/A5）
- CANN、torch、torch_npu、SHMEM 版本
- device 列表与 EP topology
- 完整 shape、dtype、TopK、capacity
- 构建命令与退出状态
- 前向回归结果
- 反向各输出误差：grad_x / grad_weight1 / grad_weight2 的 max abs/rel error、错误比
- loss.backward() 是否成功、是否死锁
- 失败日志与复现命令
```

### 4.4 常见失败排查指南

| 现象 | 优先排查 |
|---|---|
| 编译报 layout 不支持 | `mega_moe_bf16_backward_catlass_impl.h` 的 RowMajor B / ColumnMajor A（§3-2） |
| 编译报 UB 不足 | `block_epilogue_swiglu_backward_bf16.h` 的 static_assert（§3-4） |
| 反向死锁/超时 | P0-C/P1 的 value-0 event 配对、P2 的 +16 配对、GM flag 生命周期（§3-3） |
| grad 全零或明显偏小 | 阶段没跑起来（flag 提前通过 / event 提前放行），检查 P0_B dispatch flag 计数 target=EP*blockNumPerEP |
| grad_w1/w2 不对、grad_x 对 | dW 的 ColumnMajor A 转置视图、或 dW2 的 H 地址 |
| 只有 grad_x 不对 | Unpermute 的 expandedRowIdxGather 读取、跨 rank Combine 可见性（CrossRankSyncInWorldSize） |
| 空 expert 死锁 | `UpdateGroupParams` 返回 false 的分支在所有阶段是否一致（AIC/AIV 必须同判同跳） |

### 4.5 验证通过后的收尾

1. 在 `docs/mega_moe_bf16_backward_v1_implementation_note.md` 把「未验证」改为实测结论（含误差数字、设备/版本）。
2. 提交验证代码与结果（若本会话 push 权限不足，用有权限的会话推送分支 `worktree-megamoe-bf16-backward-v1`）。
3. 若期间改动了代码，把新改动也同步进分支并更新本交接文档的提交信息。
4. 正确性全部通过后才允许做性能/profiling 结论（guide §13）。
