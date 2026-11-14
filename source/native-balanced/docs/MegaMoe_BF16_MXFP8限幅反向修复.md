# 普通 BF16：有限幅 MXFP8 保存的反向 mask 修复

## 问题与范围

基于 train `deefdc1b8ab587687f2f3217d217162b5ffd59ed`，仅修复
显式 `save_gmm1_output=mxfp8` 且 `swiglu_limit>0` 时反向使用的限幅 mask。
自动保存模式在有限幅时选 BF16，因此不走这一问题路径。M128 前向、
GMM1 调度、Ready256、GM 保存格式及公开接口保持原样。

前向保存 Gate 上限及 Up 正负限幅标记，反向据此将被截断分量的梯度清零。
原反向按最多16行分片解码，每片覆盖同一个512字节 mask 暂存区，随后却
按完整128行计算梯度。超过首片的行读取了错误位置，导致梯度错误。
整网能执行完成并不能证明这些位置的梯度正确。

## 修改

- 三类 mask 各保留128行，即 `128×32=4096` 字节；解码片段写到
  `chunkRow×32` 偏移，限幅哨兵也使用片段对应的行索引。
- 反向 UB scratch 从215 KiB增至225.5 KiB，低于248 KiB；矩阵分块、
  量化格式、梯度公式和 GM workspace 不变。
- 反向测试按 train 当前紧凑保存状态布局计算偏移。packed X 和路由权重
  位于 dispatch_output，不重复算入 training_saved_state。
- CPU 的 dpPartials 中间参考保留反向重算 H 时 SiLU 的 BF16 舍入。
  两项测试修正不改变 Kernel 行为，也不放宽精度容差。

该 PR 独立基于 train，能够先于 M192 前向优化合入。前向优化另一个 PR
处理 M192 分块、Ready384、MXFP8 分片保存及路由系数的 DCache 读取。

关键的原版失败/修复后通过对照、验证脚本和独立回归见
[反向验证结果](MegaMoe_BF16_MXFP8反向验证结果.md)。

## 验证命令

在仓库根目录执行，先编译；每条 NPU 测试运行前确认设备空闲。
精度验证不要加 `--skip-reference`。

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
npu-smi info
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3,4,5,6,7 \
  --m 513 --k 512 --n 1024 --topk 6 --expert-per-rank 3 \
  --routing-mode random --save-gmm1-output mxfp8 --swiglu-limit 0.5

npu-smi info
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3,4,5,6,7 \
  --m 257 --k 512 --n 1024 --topk 6 --expert-per-rank 3 \
  --routing-mode random --save-gmm1-output bf16 --check-intermediates
```

此前 train 的三专家有限幅 MXFP8 案例出现 grad_x/grad_w1 超容差；修复后
组合版本在三专家、十六专家及 limit=7/0.5 的检查通过。拆分后另验证
本分支的原版 M128 前向与反向修复组合，避免只用 M192 的结果作为证据。

## 独立分支实测

Ascend950、EP8、K512、N1024、TopK6、random，关闭Timer和诊断旁路，
每次运行前确认八卡空闲。只加载本分支独立构建的 M128 Kernel SO：

| BS | 每卡专家 | 保存 | limit | CPU 梯度结果 |
|---:|---:|---|---:|---|
| 257 | 3 | none | 0 | 八卡四类梯度通过 |
| 257 | 3 | bf16 | 0 | 八卡四类梯度通过，中间结果56项通过 |
| 257 | 3 | mxfp8 | 0 | 八卡四类梯度通过 |
| 513 | 3 | mxfp8 | 0.5 | 八卡四类梯度通过 |
| 257 | 16 | mxfp8 | 7 | 八卡四类梯度通过 |

该结果验证反向修复无需依赖 M192 前向优化。现有精度容差未调整。
