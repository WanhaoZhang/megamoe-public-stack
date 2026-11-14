# MegaMoe BF16 Dispatch 输出可选持久化

## 目标

普通 BF16 training 前向把 Dispatch 的两个同序结果从
`training_saved_state` 中拆出，放入一个独立的 `dispatch_output`：

1. packed token，BF16 `[maxRecvTokens, K]`；
2. packed TopK weight，BF16 `[maxRecvTokens]`。

两个区域使用同一个 UInt8 Tensor，框架只需决定是否保留一个对象。Dispatch
直接写该 Tensor，GMM1 和前后向 epilogue 直接读取，没有新增 GM copy。

## 内存布局

`DispatchOutputLayout(K, maxRecvTokens)`：

| 字段 | 偏移 | 字节数 |
|---|---:|---:|
| packed token | `0` | `AlignUp(maxRecvTokens*K*2, 512)` |
| packed TopK weight | `tokensBytes` | `AlignUp(maxRecvTokens*2, 512)` |

常测 `maxRecvTokens=65536, K=7168`：

- token：896 MiB；
- TopK weight：128 KiB；
- `dispatch_output`：896.125 MiB。

普通 `compute_train` 使用 compact `TrainingSavedStateLayout`，其中
`ptrABytes=0`、`dispatchedTopkWeightsBytes=0`。路由元数据、完整 token count
快照、backward progress、可选 gate/up、MXFP8 scale 和截断 mask 仍保留在
`training_saved_state`。

## 接口和生命周期

TorchBind training 前向返回：

```text
y, counts, training_saved_state, dispatch_output, forward_transient_workspace
```

带 timer 版本在末尾再返回 timer。公开 `MegaMoeBF16Train.forward()` 仍只返回
`(y, counts)`。

`retain_dispatch_output=True` 是当前默认值。autograd 保存
`training_saved_state + dispatch_output`，现有反向从独立 Dispatch Tensor 读取
packed token 和 packed TopK weight，算法不变。

`retain_dispatch_output=False` 时，autograd 不保存 `dispatch_output`；前向返回后
该内存可以归还缓存分配器。当前反向尚未实现重新 Dispatch，因此如果随后调用
backward，会抛出明确错误。后续重新 Dispatch 的反向接入时，需要保存原始 `x`、
`topk_ids` 等输入，并删除这项保护。

独立 Tensor 不能改成大 workspace 的 view。保存 view 会保留整块底层 storage，
使 `ptrA2` 等 transient 内存也无法释放。

## 内存结果

以 BF16 gate/up 保存为例：

| 项目 | 拆分前 | 拆分后 |
|---|---:|---:|
| `training_saved_state` | 1920.631 MiB | 1024.506 MiB |
| `dispatch_output` | 内嵌 | 896.125 MiB |
| 默认保留时总量 | 1920.631 MiB | 1920.631 MiB |
| 不保留 Dispatch 时长期驻留 | 1920.631 MiB | 1024.506 MiB |

前向执行期间的峰值基本不变；收益来自前向完成后不再让每层的 Dispatch 输出一直
驻留到反向。后续重新 Dispatch 的反向会在执行该层反向时重新申请临时输出，但不会
同时持有所有层的 Dispatch 输出。

## 兼容范围

- 普通 raw `compute()` 保持三输出和原内嵌布局，避免改变现有精度/性能脚本。
- 普通 `compute_train()` 使用独立 Dispatch 输出。
- 当前普通 backward 只做地址来源适配，仍消费前向 Dispatch 结果。
- MGroup 和 UDMA 本轮保持原布局；UDMA 的 packed token 位于 SHMEM，若以后支持
  可选持久化，需要单独决定是否做条件性快照。

Host 扩展和 BF16 Kernel ABI 同时增加 Dispatch 指针，必须一起重新编译，不能混用
不同版本的 Host 扩展与 BF16 SO。
