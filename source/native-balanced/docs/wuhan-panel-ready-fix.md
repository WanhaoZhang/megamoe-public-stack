# Wuhan 反向 panel ready 正确性修复

本次只修复反向权重恢复同步，不改变 planner、GEMM tile/K 累加顺序、BF16 partial、
FP32 固定序归主或默认 worker 分工。性能优化候选未合入。

## 问题与复现

旧 panel 恢复在外层只等首个 W2 panel；CATLASS 一次 MMAD 会遍历完整 K，并预取
后续 operand。H7168 会消费 28 个 W2 panel，首 panel ready 不能代表其余数据可读。

在 pod7-3 的独立容器中，将活动副本的后续 panel 生产延迟 50000000 cycle，关闭
operand-ready 的旧协议控制实际出现 dX、dp、dW 精度断言失败，run exit=1、即时
后门禁通过。该运行是故障复现，不计有效性能，不声称普通长跑必然已出现错误。

修复让 operand 在 CATLASS 内部 GetTile 后继续携带 slot/projection/epoch 和行坐标，
实际 GM→L1B 读取前等待所覆盖的 panel。W1 重算按 column-major 的物理权重行检查，
dH/dX 按 row-major 检查；home 和普通 dW operand 不增加权重恢复等待。

Python FSDP 桥拒绝 panel 模式下 operand-ready getter 缺失或值为0的旧 SO，避免只
更新 Python/源码却继续加载旧库。staged/expert 保持兼容。生产构建 delay 必须为0；
延迟及 ready=0 仅用于独立诊断，不用于训练交付。

## 已完成的设备门禁

| 用例 | 范围 | 结果 |
|---|---|---|
| EP2 无延迟 | H512/F512/M257，MoonEP，真实副本、两层共享池污染/恢复 | 五输出、三次SGD、正常退出/后门禁通过 |
| EP2 延迟50000000 cycle | 同上 | 通过；输入、计划、五输出和SGD与无延迟运行逐位一致 |
| EP8 延迟＋checkpoint | H512/F512/M257，MoonEP，FP32 Parameter | 五输出、三次SGD、8rank退出/后门禁通过 |
| 真实 Expert-FSDP | 8rank、EP4×EFS2，Native，H512/F512/M2049，checkpoint、延迟 | 真实副本、五输出、三次SGD及退出/后门禁通过 |
| 真实 Expert-FSDP | 8rank、EP4×EFS2，MoonEP，H512/F3072/M513，checkpoint、无延迟 | 真实副本、五输出、三次SGD及退出/后门禁通过 |
| 双节点真实 Expert-FSDP | 16rank、EP4×EFS4，Native，H512/F512/M2049，checkpoint、无延迟 | 全16rank五输出、三次SGD；两节点退出0、即时八卡后门禁通过 |
| 双节点真实 Expert-FSDP | 16rank、EP4×EFS4，MoonEP，H512/F3072/M513，checkpoint、无延迟 | 全16rank五输出、三次SGD；两节点退出0、即时八卡后门禁通过 |

所有用例沿用原 golden 判据，没有放宽容差。CPU wrapper 测试覆盖28个内部K panel、
column-major W1、尾部和home；这部分不当作设备验收。

首次EP2 a1因新容器缺Python SHMEM，在算子执行前失败；补齐同源依赖后另编号a2，
原失败保留。旧协议故障的device日志导出因容器GLIBC不兼容失败，改在host只读导出
成功，日志与原退出/后门禁一并归档，没有复位或干预他人作业。

## 二进制与复现入口

设备测试源码为 `0c3c665`，后续只增加旧SO拒绝检查和本文档。通过的无延迟worker：

`38472e762930b9dfa2c1c31cfb56aaf2df7ec97d5e3e53b3bc94415fd634c53b`

延迟诊断worker：`bc31b614e4623750414caf502f1e9a863b39863a73bd92599d5f071bc6779afb`。

重新构建须使用独立目录/名称：

```bash
MEGAFLUX_RESTORE_OPERAND_READY=1 MEGAFLUX_RESTORE_DELAY_CYCLES=0 \
  bash integration/wuhan/build_restore_candidate.sh \
  /project/new-frozen-source /project/frozen-donor \
  /project/evidence/build-panel-ready release-panel-ready 2
```

启动时显式选 `WUHAN_WORKER_VARIANT=release-panel-ready` 和
`BALANCED_RESTORE_MODE=panel`，并核对 build.json、实际加载路径、getter和SO hash。
不要覆盖正在运行的源码或二进制；旧 `.so` 不会因 git pull 自动变成新库。

原始证据：`D:/megamoe_train/moonflux接整网/balanced-8p/panel-ready/collected/`，
包括 `panel-unguarded-ep2-delay50m-a1` 故障、五组成功门禁及真实设备前后检查。
逐位对照：`panel-ready/delay50m-vs-normal-a1.json`。源码/依赖和两套worker的构建hash
保存在远端该项目 `evidence/build-*`、`candidate-panel-ready-a2/` 及部署记录中。

双节点使用交付源码 `e2d8a72`，独立部署 `candidate-source-panel-safe-a1`，保持旧快照
不变。实际加载的无延迟SO与上述验证库相同，32份rank回执均核对ready=1/delay=0、
checkpoint、三次SGD及完整rank集合。原始回收在 `panel-ready/collected-world16-native-a1/`
和 `collected-world16-moonep-a1/`，摘要为 `world16-audit-a1.json`，部署与source hash
在 `panel-ready/release-a1/deployment.json`。EP4位于节点内；这是跨节点Expert-FSDP4
生命周期门禁，不是跨节点EP或完整模型训练性能验收。

## 长跑与交付边界

旧 rho0/rho6 ABCD200步结果来自冻结的旧 panel SO，不能当作本修复版长期验收。
发现问题后先暂停旧队列的CPU协调器，原MegaMoE A自然完成且exit0/即时后门禁通过；
封存A后只退役已确认停止的CPU协调器，没有恢复旧B/C或给NPU作业发信号。
本修复已通过上述world16功能门禁；尚无新200步、多机完整模型或性能收益结论。
接下来以独立编号运行完整模型和长跑，EP4仍位于节点内，不声明跨节点EP或512P
新后端已验收。
