# BF16 反向修复：关键验证结果

## 结论与独立性

本 PR 仅修复显式 MXFP8 保存且 SwiGLU 有限幅时的反向 mask 暂存及行偏移，
并修正反向测试的保存状态解析和中间参考。原版前向保持 M128/Ready256。
不依赖 M192 前向优化，也不修改前向接口或 GM 保存格式。
前向 M192 优化同样不依赖本补丁：分块和 UB 暂存是内部行为，反向仍读取
原格式的 Gate/Up、scale 和 mask。可独立审查、独立合入；修复这个已有的
反向错误无需等待前向 PR。

## 用什么脚本、检查什么

设备验证入口为 `test/megamoe_bf16_test/test_mega_moe_backward.sh`，它直接
调用同目录的 `test_mega_moe_backward.py`。测试生成输入、路由、权重和
上游梯度，运行前向保存及反向，按 rank 将 `grad_x`、`grad_topk_weights`、
`grad_w1`、`grad_w2` 与 CPU 参考比较，八卡均 PASS 才视为通过。

mask 验证不是只检查 buffer 大小：显式指定 `--save-gmm1-output mxfp8`
和正的 `--swiglu-limit`，CPU 参考按量化前的 Gate/Up 生成截断判断，
对被截断分量清零相应梯度，再检查最终四类梯度。
`--check-intermediates` 是补充检查，不代替最终梯度检查；Host 的保存
布局测试也不代替设备 mask 精度验证。本次没有使用 `--skip-reference`。

## 原版失败与修复后通过：同配置对照

2026-10-11，Ascend950、EP8、BS257、K512、N1024、每卡3专家、TopK6、
random、显式 MXFP8、limit=7。每次测试前检查八卡空闲，Timer及诊断旁路关闭。
两次使用相同 Host、输入生成规则、随机种子、修正后的反向测试脚本及 CPU
参考，仅替换 Kernel SO；两版前向均为原版 M128。
基线为 train `deefdc1b8ab587687f2f3217d217162b5ffd59ed`，修复 Kernel
为本分支 `f85837407cce698d9c1403b8a4629f7caa443442`。

| 检查项 | 原版 train | 仅反向 mask 修复 |
|---|---:|---:|
| 八卡最终结果 | 0/8 PASS，8/8 FAIL | 8/8 PASS |
| rank4 grad_x 超容差元素 | 42,029 / 131,584，FAIL | 0 / 131,584，PASS |
| rank4 grad_x 最大绝对误差 | 4.83049 | 0.0670986 |
| rank4 grad_w1 超容差元素 | 15,543 / 1,572,864，FAIL | 0 / 1,572,864，PASS |
| rank4 grad_w1 最大绝对误差 | 0.189281 | 0.00249976 |
| rank4 grad_topk_weights 最大绝对误差 | 0.0801897，PASS | 0.0801897，PASS |
| rank4 grad_w2 最大绝对误差 | 0.00247091，PASS | 0.00247091，PASS |

“超容差元素”是现有检查器的逐元素计数；最终 PASS/FAIL 还遵循检查器
原有判定，未放宽容差。这个对照证明原版已存在实际梯度错误，且单独
反向修复能够修好；不是由前向 M192 优化引入，也不是测试参考变化造成
两版差异。未宣称此单组结果覆盖所有 shape 或整网训练。

复现命令（分别在 train 和本修复分支编译后执行同一份修正后的测试）：

```bash
bash scripts/build_mega_moe_bf16.sh -soc_type Ascend950
npu-smi info
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3,4,5,6,7 \
  --m 257 --k 512 --n 1024 --topk 6 --expert-per-rank 3 \
  --routing-mode random --save-gmm1-output mxfp8 --swiglu-limit 7
```

基线 SO SHA256：`aa1267dcae46e1dc3a2de20dab0ceb0533f6ca4e3f4f06d899c6e6a640c6e3d9`。
修复 SO SHA256：`acab5316f5c32b58a0f35c97440d6eaad7f15c26726a2865c76d8f2c84c5260b`。

## 测试脚本修正：独立验证对照

本 PR 的第二类修改是 `test_mega_moe_backward.py`，包含保存状态偏移和
CPU 中间参考舍入两项修正。它们不改变 Kernel、真实输出或检查容差。
上面的 mask 对照使用同一份已修正测试脚本，仅比较两版 Kernel；
下面单独展示测试脚本自身的修正，避免混淆两类收益。

### 保存状态解析偏移

普通 BF16 的 packed X 和路由权重已经存放在 `dispatch_output`，旧测试
仍将这两块大小计入 `training_saved_state`，导致读取错误的保存位置。
采用 BS257、K512、N1024、EP8、每卡3专家、TopK6、BF16 保存，
`max_recv=257×8×3=6168`，分别执行修改前后的 Python 布局函数，
并通过 Host C++ 编译执行 Kernel 的 `TrainingSavedStateLayout`，
使用普通 BF16 的 `embedDispatchOutput=false` 核对偏移：

| 字节偏移 | 旧测试计算 | Kernel 实际布局 | 修正后测试计算 |
|---|---:|---:|---:|
| expandedRowIdxGather | 6,328,832 | 12,800 | 12,800 |
| backwardProgress | 6,358,528 | 29,696 | 29,696 |
| Gate 保存起点 | 6,359,552 | 30,720 | 30,720 |

Gate 偏移原先多算 6,328,832 字节。修正后测试与已有 Kernel 布局一致，
没有移动 Kernel 保存数据。这一项的修改前后证据是 Host 布局核对；
修正后的设备验证还覆盖八卡共56项中间检查，包括 Gate/Up 与路由映射。

测试开始前会打印 `[SAVED_LAYOUT]`。以上 BF16 配置对应的回显示例为：

```text
[SAVED_LAYOUT] rank=0 source=python_test_layout unit=bytes mode=bf16 max_recv=6168 packed_x=dispatch_output routed_weights=dispatch_output progress=29696 gate=30720 up=6346752
```

这里打印测试解析器实际使用的偏移，不是从 Kernel 读回的元数据。
旧测试没有此回显，因此旧偏移来自上面的 Host 对照，不伪造历史运行输出。
带 `--check-intermediates` 时，还会打印 `[INTERMEDIATE] rank0 gate`、
`up` 等数据检查结果；偏移回显与这些 PASS 结果一起用于核对。
只查看相关回显可以执行（完整检查仍会运行）：

```bash
bash test/megamoe_bf16_test/test_mega_moe_backward.sh 0,1,2,3,4,5,6,7 \
  --m 257 --k 512 --n 1024 --topk 6 --expert-per-rank 3 \
  --routing-mode random --save-gmm1-output bf16 --swiglu-limit 0 \
  --check-intermediates \
  | grep -E '^\[(SAVED_LAYOUT|INTERMEDIATE|CHECK|RESULT|SUCCESS)\]'
```

### CPU dpPartials 参考的 BF16 舍入

反向重计算 H 时，SiLU 中间值会先落为 BF16，再乘 Up 并落为 BF16。
旧 CPU 中间参考遗漏第一步舍入，产生 dpPartials 误报。
同样使用 BS257、K512、N1024、EP8、每卡3专家、TopK6、random、
BF16 保存、limit=0，保持原版 train Kernel 和已修正的保存状态解析，
仅修正 CPU 中间参考：

| 检查项 | 修正参考前 | 修正参考后 |
|---|---:|---:|
| rank4 dpPartials 超容差元素 | 2 / 3,168，FAIL | 0 / 3,168，PASS |
| rank4 dpPartials 最大绝对误差 | 0.0879974 | 0.00390625 |
| 八卡中间检查最终判定 | 5/8 PASS，3/8 FAIL | 8/8 PASS，共56项中间检查 PASS |
| 八卡最终四类 CPU 梯度检查 | 全部 PASS | 全部 PASS |
| rank4 grad_x 最大绝对误差 | 0.0714731，PASS | 0.0714731，PASS |

修正前失败的是 rank3、rank4、rank5 的 dpPartials 中间参考，最终四类
梯度检查原本已经通过。这里的 PASS 改善代表消除测试误报，不能解释
为 Kernel 梯度精度提升。修改前后没有放宽容差，也没有改变前向 H 的参考。

## 其他独立回归

EP8、K512、N1024、TopK6、random，使用本分支独立构建，保持 M128 前向。

| BS | 每卡专家 | 保存 | limit | 结果 |
|---:|---:|---|---:|---|
| 257 | 3 | none | 0 | 八卡四类 CPU 梯度 PASS |
| 257 | 3 | bf16 | 0 | 八卡四类 CPU 梯度 PASS，中间结果56项 PASS |
| 257 | 3 | mxfp8 | 0 | 八卡四类 CPU 梯度 PASS |
| 513 | 3 | mxfp8 | 0.5 | 八卡四类 CPU 梯度 PASS |
| 257 | 16 | mxfp8 | 7 | 八卡四类 CPU 梯度 PASS |

修复原因和 UB 容量核算见 [mask 修复说明](MegaMoe_BF16_MXFP8限幅反向修复.md)。
