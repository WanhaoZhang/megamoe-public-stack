# MegaMoe BF16 Backward V1 — 实现说明

> 2026-09-07 集成说明：本文保留 train 旧训练接口的设计/实现记录。
> 当前 raw 前向已切换三输出 saved-state，旧训练入口暂时明确报错。
> 反向适配请以 [新持久状态与迁移清单](MegaMoe_BF16_前向持久Workspace第一版与反向适配.md) 为准；
> 下文旧 workspace 偏移、接口示例和历史验证结论不能直接用于新 ABI。

本文件记录 `docs/mega_moe_bf16_backward_v1_agent_guide.md` 的 V1 实现落地情况、
关键设计决策、以及远程验证前需要确认的风险点。它是给下一个开发者/agent 的交接说明，
不是对 guide 的替换。

## 1. 实现文件

新增：

- `kernels/mega_moe_bf16/mega_moe_bf16_backward.cpp` — 反向 kernel 入口 + `mega_moe_bf16_backward_kernel_do` ABI
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h` — `MegaMoeBF16Backward` 主类（P0–P5 各阶段）
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward_catlass_impl.h` — 反向 GEMM 包装（P0-C 重计算 / P1 GMM2Grad / P2 GMM1Grad / P3/P4 dW）
- `kernels/mega_moe_bf16/include/block_epilogue_swiglu_backward_bf16.h` — 反向 SwiGLU epilogue（RecomputeGMM1 / SwiGLUGrad / SwiGLURecompute）
- `test/megamoe_bf16_test/test_mega_moe_backward.py` + `.sh` — EP 多进程 loss.backward() 精度测试

修改：

- `kernels/mega_moe_bf16/include/mega_moe_workspace_info.h` — `ForwardWorkspaceInfo` 持久前向布局与独立 `BackwardWorkspaceInfo`
- `kernels/mega_moe_bf16/include/mega_moe_bf16.h` — 训练模式：`SaveTokenPerExpert()`（ResetTokenPerExpert 前保存 count 表）
- `kernels/mega_moe_bf16/mega_moe_bf16.cpp` — 训练与普通前向统一复用 `mega_moe_bf16_kernel_do`
- `kernels/mega_moe_bf16/include/mega_moe_base.h` — `GMMAddrInfo::cGlobal`（dW 输出）
- `include/ascend_deepep/kernels/launch.h`、`csrc/launcher.{h,cpp}` — 两个新 launcher
- `csrc/ops/mega_moe_bf16.{h,cpp}` — 统一 `MegaMoeBF16Forward` / `MegaMoeBF16Backward` + saved/backward layout 大小计算
- `csrc/torch_classes.cpp` — 原有 `MegaMoeBF16` TorchBind 类新增 `.compute_train` / `.backward`
- `ascend_deepep/__init__.py` — 新增公开 `MegaMoeBF16Train(nn.Module)` + 私有 autograd bridge
- `CMakeLists.txt` — BF16 kernel 库加入 `mega_moe_bf16_backward.cpp`

## 2. 阶段与分工

```
P0-A  加权 gather       grad_y * topk_weights -> peermem.ptrA0        （AIV0+AIV1 按 route 分核）
P0-B  恢复 count 表 + Dispatch dO -> ptrDO                            （AIV0）
P0-C  GMM1 重计算       Z = Xpacked @ W1.T -> gate/up planes          （AIC 计算，AIV1 拷出）
P1    GMM2 Grad || SwiGLU Grad：dH = dO @ W2 ; dZ = [dGate|dUp]       （AIC + AIV1，单缓冲 tile 流水）
P2    GMM1 Grad || Combine + SwiGLU 重计算：dX = dZ @ W1 ; H          （AIC + AIV0 ping-pong；AIV1 并行）
P3    GMM1 dW           dW1 = dZ.T @ Xpacked -> grad_weight1          （纯 AIC，L0C->GM）
P4    GMM2 dW           dW2 = dO.T @ H -> grad_weight2               （纯 AIC，L0C->GM）
P5    Unpermute         grad_x = sum_slot(ptrD)                       （AIV0+AIV1 按 token 分核）
```

## 3. 关键设计决策

1. **统一训练前向并拆分生命周期**：BF16 前向共用 `MegaMoeBF16Forward` host 实现和
   `mega_moe_bf16_kernel_do` device ABI；kernel 固定在 `ResetTokenPerExpert()` 前保存物理
   `ptrTokenPerExpert` 表。训练前向返回持久
   `forward_workspace`（包含 count table 和 progress），backward scratch
   则在 backward 调用时单独申请。

2. **反向不重跑 moe_init_routing_v3**：P0-A 用保存的 `expandedRowIdx` 直接做加权 gather。

3. **阶段屏障用 GM flag，tile 握手用前向已验证的 event 协议**：
   - P0-C / P1 单缓冲：完全复用前向 GMM1 的 value-0 协议（NotifyVector(0)/WaitForVector(0) /
     WaitForCube(0)/NotifyCube(0)），跨核 flag 是一次性的，顺序阶段复用 0 安全。
   - P2 ping-pong：完全复用前向 GMM2 的协议（NotifyVector(pp+16)/WaitForVector(pp+16)/
     WaitForCube(pp)/NotifyCube(pp)）。
   - AIV1->AIC（gather 完成 / H 完成）用 peermem 之外的 workspace GM flag（`WriteGmByPassDCache`/`ReadGmByPassDCache` 轮询），
     避免跨核 event 值配对的不确定性。

4. **UB 规划（每 AIV 各 192KB 假设，含 static_assert）**：
   - P0-C：AIV1 UB 0/64KB 放 Z 两半。
   - P1：AIV1 UB 0=dH、64KB=gate、128KB=up；dGate 回写 0，dUp 回写 64KB。
   - P2：AIV0 UB 0/64KB 放 dX（Combine 源）；AIV1 UB 0/64KB=gate/up、128KB=H。

5. **dW 用纯 AIC L0C->GM**（`PackedBF16TileCopyTlaToGm`，参照 W4A8 的 `CopyL0CToGmTla`），
   不占用 AIV，P3/P4 与 P5 Unpermute 重叠。

## 4. 远程验证前必须确认的风险点

1. **layout 支持**：P1/P2 用 RowMajor B；P3/P4 用 ColumnMajor A（转置视图）。**2026-08-26 已在 148 上编译确认**：TileCopy 需参数化 LayoutA/LayoutB（见 `mega_moe_bf16_backward_catlass_impl.h` 的 `BackwardBF16TileCopy`），RowMajor B 与 ColumnMajor A 均能通过 A5 catlass 编译。运行期数值方向仍未验证。

2. **跨核 event 值配对**：P0-C/P1 的 value-0 直接配对、P2 的 +16 配对均照抄前向已验证用法；
   阶段屏障已改 GM flag 规避风险。

3. **UB 大小**：`block_epilogue_swiglu_backward_bf16.h` 有 `3*kTileBytes <= UB_SIZE`
   static_assert（192KB）。**已编译通过**（A5 UB 满足 192KB）；运行期若仍有 UB 冲突再观察。

4. **`ForwardWorkspaceInfo` / `BackwardWorkspaceInfo`（device）与 host
   对应 layout 必须一致**，修改任一侧要同步另一侧。

5. **forward_workspace 按 forward 配对**：每次 `compute_train` 分配新的 `forward_workspace`；每次
   backward 都重新分配并清零 backward workspace，避免同步 flag 跨调用复用。

## 5. 远程验证步骤（Ascend950）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash scripts/build.sh -soc_type Ascend950
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
# 前向回归（确认无回归）
bash test/megamoe_bf16_test/test_mega_moe.sh 0,1
# 反向精度（EP=2 起）
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1 --mode precision
# 覆盖：--routing-mode imbalanced（空 expert）、--topk 不同值、EP=4/8
```

参考（CPU float32 全量路由重放）在 `test_mega_moe_backward.py::reference_backward`，
按 EP 切片比较 `grad_x`（本 rank 的 M 行）、`grad_w1`/`grad_w2`（本 rank 的 E 个 expert）。
BF16 中间量化会带来误差，容忍度默认 atol=0.05/rtol=0.15/max 5%，可调。
