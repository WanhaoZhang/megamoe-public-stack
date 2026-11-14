# MegaMoe BF16 双权重布局与免转置说明

## 1. 修改目的

训练接入使用 Megatron legacy `GroupedMLP` 权重：

- `weight1` 的参数存储为 `[H, E * 2F]`，按 expert 零拷贝 view 后为 `[E, K, N]`；
- `weight2` 的参数存储为 `[E * F, H]`，按 expert 零拷贝 view 后为 `[E, F, K]`。

其中 `K = H`、`N = 2F`。

修改前，`test/bigop_fused` 在每次前向调用前执行：

```text
W1 [E,K,N] -> transpose + contiguous -> [E,N,K]
W2 [E,F,K] -> transpose + contiguous -> [E,K,F]
```

这会复制完整的两块 expert 权重。前向保存复制后的权重供反向使用，因此复制同时增加前向时间和训练期间的显存占用。

修改后，普通 `MegaMoeBF16`/`MegaMoeBF16Train` kernel 直接解释 Megatron legacy 布局，不再生成转置权重副本。原 canonical 布局仍然保留，用于已有低层调用和回归。

## 2. 支持的两种布局

| 名称 | `weight1` | `weight2` | `transW/transW2` |
|---|---|---|---|
| canonical | `[E,N,K]` | `[E,K,F]` | `true` |
| Megatron legacy | `[E,K,N]` | `[E,F,K]` | `false` |

Host 从输入 tensor 的 shape 自动识别布局，不增加 Python 参数、环境变量或构建选项。

识别条件为：

```text
canonical:
  W1.dim2 == K
  W2.dim1 == K
  W2.dim2 == W1.dim1 / 2

legacy:
  W1.dim1 == K
  W2.dim2 == K
  W2.dim1 == W1.dim2 / 2
```

`K` 来自输入 `x/grad_y` 的最后一维。两块权重必须属于同一种布局；混合布局会在 Host 报错。正常非零 Shape 下，两组条件不会同时成立。

识别结果写入已有 `MegaMoeTilingData.transW/transW2`，没有新增 tiling 字段，也没有改变 workspace、saved state 或 launcher ABI。

## 3. 前向怎样直接读取两种布局

前向数学关系不变：

```text
gate_up = X @ W1.logical
H       = SwiGLU(gate_up)
Y       = H @ W2.logical
```

| 布局 | GMM1 的 B | GMM2 的 B |
|---|---|---|
| canonical | `[N,K]` 存储作为 ColumnMajor `[K,N]` | `[K,F]` 存储作为 ColumnMajor `[F,K]` |
| legacy | `[K,N]` 存储作为 RowMajor `[K,N]` | `[F,K]` 存储作为 RowMajor `[F,K]` |

kernel 根据 `transW` 在入口选择对应的 CATLASS 模板。expert 循环和 tile 循环内没有运行时遍历或布局判断。

Dispatch、SwiGLU、GMM1 调度、GMM2 调度、Combine、Unpermute、256-row ready 和保存模式均保持原协议。

## 4. 反向五个 GMM 阶段

设某个 expert 的 token 数为 `Me`，`dZ` 为 GMM1 输出的梯度，`Hp` 为 SwiGLU 输出。

### P0-C：需要时重算 GMM1

```text
gate_up = Xpacked [Me,K] @ W1.logical [K,N]
```

- canonical W1 使用 ColumnMajor B；
- legacy W1 使用 RowMajor B。

该阶段只在 `save_gmm1_output=none` 时运行。

### P1：GMM2 data grad

```text
dHp = dO [Me,K] @ W2.logical.T [K,F]
```

- canonical W2 `[K,F]` 的转置视图使用 RowMajor B；
- legacy W2 `[F,K]` 的转置视图使用 ColumnMajor B。

### P2：GMM1 data grad

```text
dX = dZ [Me,N] @ W1.logical.T [N,K]
```

- canonical W1 的转置视图使用 RowMajor B；
- legacy W1 的转置视图使用 ColumnMajor B。

### P3：直接生成与输入 W1 相同布局的梯度

canonical：

```text
dW1 [N,K] = dZ.T [N,Me] @ Xpacked [Me,K]
```

legacy：

```text
dW1 [K,N] = Xpacked.T [K,Me] @ dZ [Me,N]
```

legacy 路径交换 A/B 地址，并交换 GEMM 的 M/N，不执行额外转置。

### P4：直接生成与输入 W2 相同布局的梯度

canonical：

```text
dW2 [K,F] = dO.T [K,Me] @ Hp [Me,F]
```

legacy：

```text
dW2 [F,K] = Hp.T [F,Me] @ dO [Me,K]
```

因此 `grad_weight1/grad_weight2` 总是和传入 kernel 的权重 shape、存储方向一致。Python autograd 对 legacy 3D view 只需恢复原来的 2D view，不需要复制或转置完整梯度。

## 5. 修改前的前向输入转换和反向梯度恢复

### 5.1 前向进入 MegaMoe 前生成 canonical 权重

修改前，模型参数保持 Megatron `GroupedMLP` 的二维存储：

```text
experts.weight1: [K, E*N]
experts.weight2: [E*F, K]
```

`test/bigop_fused/utils.py::make_weight_tensors()` 先把参数 view 成逐 expert
的 legacy 三维布局，再执行 `transpose(1, 2).contiguous()`：

```python
# f49952c 修改前的代码
def make_weight_tensors(experts, num_local_experts, hidden, ffn):
    w1 = experts.weight1.view(num_local_experts, hidden, 2 * ffn)
    w1 = w1.transpose(1, 2).contiguous()
    # [E,K,N] -> [E,N,K]

    w2 = experts.weight2.view(num_local_experts, ffn, hidden)
    w2 = w2.transpose(1, 2).contiguous()
    # [E,F,K] -> [E,K,F]
    return w1, w2
```

`view` 不复制数据；`transpose` 只改变 stride；最后的 `contiguous()` 会分别申请
完整的 W1/W2 新存储并复制数据。因此旧路径在每次 `BigOpV2.forward()` 中，调用
MegaMoe 前都要付出两块完整权重的转换成本：

```python
# test/bigop_fused/fused.py
w1, w2 = make_weight_tensors(
    experts, num_local, config.hidden_size, config.moe_ffn_hidden_size)
_t.tick("weight_transform")

out = self._fused(
    hidden_flat, topk_ids, topk_weights, w1, w2, max_recv)
```

旧 MegaMoe 收到的是 canonical 权重：

```text
w1 [E,N,K]
w2 [E,K,F]
```

训练 wrapper 还会保存这两个转换后的 tensor，供同一次 backward 使用：

```python
# ascend_deepep/__init__.py::_MegaMoeBF16TrainFunction.forward
ctx.save_for_backward(
    topk_weights, w1, w2, counts, training_saved_state)
```

所以转换成本不仅计入 forward 时间，转换后的两块权重还要存活到 backward 完成。

### 5.2 修改前 backward 的 canonical dW 怎样恢复成模型参数布局

修改前，Host 按传入 MegaMoe 的 canonical 权重 shape 分配梯度：

```cpp
// csrc/ops/mega_moe_bf16.cpp
auto gradWeight1 = torch::zeros_like(weight1);  // [E,N,K]
auto gradWeight2 = torch::zeros_like(weight2);  // [E,K,F]
```

因此旧 backward kernel 直接输出：

```text
dw1_canonical [E,N,K]
dw2_canonical [E,K,F]
```

自定义 autograd Function 把它们作为 `w1`、`w2` 两个输入的梯度返回：

```python
# ascend_deepep/__init__.py::_MegaMoeBF16TrainFunction.backward
dx, dtopk, dw1, dw2 = ctx.op.backward(
    ctx.shmem, grad_y.contiguous(), topk_weights,
    w1, w2, counts, training_saved_state,
    ctx.max_recv, ctx.swiglu_limit, ctx.save_gmm1_output)

return None, None, dx, None, dtopk, dw1, dw2, None, None, None, None
```

这里没有显式调用一次 `dw1.transpose()` 或 `dw2.transpose()`。恢复动作由 PyTorch
沿 forward 中已经建立的 `view -> transpose -> contiguous` 计算图自动完成：

```text
dw1 [E,N,K]
  -> contiguous 的反向
  -> transpose(1,2) 的反向
  -> [E,K,N]
  -> view 的反向
  -> experts.weight1.grad [K,E*N]

dw2 [E,K,F]
  -> contiguous 的反向
  -> transpose(1,2) 的反向
  -> [E,F,K]
  -> view 的反向
  -> experts.weight2.grad [E*F,K]
```

所以从模型参数角度看，旧路径确实把 MegaMoe 输出的 canonical dW 转回了原始
Megatron 参数格式；只是这一步由 autograd 中的布局算子完成，不在 MegaMoe
backward kernel 内。

### 5.3 修改后的零拷贝接入

修改后，Python 只做 view：

```python
w1 = experts.weight1.view(E, K, N)
w2 = experts.weight2.view(E, F, K)
```

两个 tensor 都与原参数共享 storage。MegaMoe 根据 shape 识别 legacy 布局；P3/P4
直接输出 `[E,K,N]` 和 `[E,F,K]` 的 dW。autograd 只需执行 view 的反向，将三维
梯度恢复成原参数的二维 shape，不再对完整权重或完整梯度执行 transpose + copy。

模型参数、checkpoint、optimizer 参数布局和训练模块对外接口保持不变。

## 6. 修改边界

- 普通 BF16 forward/train/backward 支持 canonical 与 Megatron legacy 两种布局。
- canonical 路径保留，作为低层接口兼容和性能回归基线。
- 当前 UDMA、MGroup 接入仍按 canonical Host shape 约定工作；共享 CATLASS 模板的默认参数仍是 canonical，因此它们的既有行为不变。
- workspace 偏移、saved state、SHMEM、路由和通信协议均未改变。

## 7. 验证重点

1. canonical 和 legacy 前向输出分别通过现有 BF16 精度门禁；
2. legacy 的 `dx`、`dtopk`、`dw1`、`dw2` 通过完整训练精度；
3. `none`、`bf16`、`mxfp8` 保存模式全部覆盖，其中 `none` 必须验证 P0-C 重算；
4. `test/bigop_fused` 的 `weight_view` 阶段只做 view，不再包含整块权重转置和复制；
5. 无 Timer 重复测量 forward、backward、forward+backward；
6. 分别观察 RowMajor 前向与 ColumnMajor data-grad 是否引入新的 CATLASS 性能退化。
