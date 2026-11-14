# 计划校验统一为融合实现

基线：`codex/moonflux-8p-integration@91fa98d`。本修改与 train BF16 调度候选独立。

## 使用

无需配置校验模式。`MoonEPPlanner(...)` 和 `NativeFSDP2Pool(..., planner="moonep")`
始终使用同一个融合校验器。`validation`、`plan_validation` 参数保留用于兼容旧框架，
默认值为fused；旧值generic也映射到融合实现，不保留generic执行回退。`DevicePlanValidator` 是统一实现；旧的
`FusedDevicePlanValidator` 导入名仅为同一类的别名，并不是第二条实现路径。

旧的小算子实现位于 `plan_validation_reference.py`，仅供测试差分，生产包不导入。
Native planner 已在原生规划中检查计划，不额外套一遍 MoonEP 校验；Replay 保留
成功计划的重放和 epoch/lease 检查。这次不改变路由策略、GMM 或权重/梯度搬运。

## 覆盖和代价

沿用整体合法形状：EP2/4/8/16/32/64/128/256；home+slots<=32，逻辑专家、
副本和全局物理槽各不超过2048，M<=16384，路由总计数必须满足现有int32上限。
均衡/随机/偏斜、有副本/无副本、TimerOFF/ON、panel/expert/staged 使用相同校验语义。
MoonEP 自身 slots=home 等限制仍然有效。

每次是 local校验kernel → HCCL all-gather → global校验kernel。路由和placement
快照、跨rank错误传播、capacity/epoch一致性、finish和生命周期检查都保留。
global阶段按32个worker分摊汇总；每个worker独占完整128B输出cacheline，禁止
不同核写同一cacheline后清缓存造成覆盖。两个阶段间均有发布及核间屏障。

校验临时空间按逻辑/物理槽数分配，不按M*K*EP分配大矩阵；all-gather元数据仍随
EP增长，并不宣称所有显存或校验时间与EP无关。

## 构建与框架同步

必须同时重建本仓库 `_C` 与 `libascend_deepep_plan_validator.so`，validator ABI 为2。
ABI1库会显式拒绝，不能仅更新Python源码。Timer配对构建脚本会携带新校验库。

在准备好匹配CANN、SHMEM SDK和子模块的新源码目录中，可沿用本仓构建入口：

```bash
# 先source本机CANN set_env.sh，并设置实际SHMEM_HOME_PATH。
MAX_JOBS=2 bash test/megamoe_planner_test/build_benchmark.sh
# 需要Timer图时，再使用未存在的prefix配对构建。
MAX_JOBS=2 bash scripts/build_megaflux_timer_capture.sh fused-default-a1
```

框架原有`plan_validation="generic"`可以保持不变；无需新增模式开关。
实际采集命令见[TIMER-CAPTURE.md](TIMER-CAPTURE.md)。本次独立验证使用
`HCCL_CONNECT_TIMEOUT=180`；功能门禁需要`WUHAN_POOL_AUDIT=1`记录副本证据，
正式计时入口会关闭该审计，不能把含审计的功能门禁耗时作为性能结果。

本次只修改算子仓。FSDPTurbo/MindSpeed的配置字段及远端分支保持原样。
旧框架默认传generic或显式传fused，都执行同一个融合校验器；输入不在这两个
历史合法值中仍报错。日志/回执记录实际实现fused-v2，不能把旧generic标签当成
仍在运行旧实现。Native保留已有原生检查，不新增额外MoonEP校验。

## 验证口径

CPU检查覆盖兼容参数、默认融合、旧导入名和统一实现，以及EP2至256的正常/异常汇总。
NPU差分验收要求覆盖合法计划、错误路由/副本、不同rank的placement/epoch/capacity不一致、
容量溢出和planner原生错误；拒绝执行非法计划。还需实际前反向、共享池/重算和
TimerOFF/ON五输出对照。大EP的CPU边界验证不等于真实多机NPU验收。

### 2026-10-29 实际结果

- 独立目录中的 `_C` 与 ABI2 validator 构建通过；worker/planner未修改，复用相同源版本的配对二进制并记录哈希。
- CPU默认/兼容参数3项、planner Timer header 3项、Timer capture 5项、trace导出7项、benchmark契约18项通过。
- 新并行汇总与旧串行参考的CPU差分共6848例通过，包含EP2至256的合法边界及错误输入。
- EP2：复用177.8.24.1本轮已通过的同源ABI2差分结果，16类正常/错误计划、前后空闲门禁通过，没有重测。
- EP4/EP8：177.5.23.1独立容器验证通过，每rank各16类正常/错误计划，与旧小算子参考的五个计划张量逐位一致。
- 真正EP8完整前反向：M2049/H512/F1024/E16/Top2，两层共享池、checkpoint，balanced/hot/rotate三次更新，五输出及SGD更新检查通过。
- 实际expert-FSDP2：八卡EP4×FSDP2，M513/H512/F512，checkpoint，hot/balanced/rotate三次SGD，前反向、reshard和参数更新检查通过。
- 大shape：M8192/H7168/F3072/E8/Top2，home=slots=2，capacity=20480，ρ=6，panel、limit=10。
  TimerOFF/ON各重复3次；Y/dX/dProb/dW1/dW2与原基线逐位一致，Timer配对及两张all-rank图的完整性检查通过。
  这是两个EP4组的完整算子采集，不含真实FSDP通信或optimizer；上述实际FSDP门禁单独验证。
- 各项采用成功运行前后的八卡空闲门禁；已通过的项目未重复执行。EP16至256仍只有CPU边界/差分验证，没有多机NPU或收敛结论。

主验证机器：177.5.23.1；自有容器`moonep-plan-default-20261029`，源码/库与提交`05c1159`一致。
原始记录位于该机`/home/x00610564/moonep_plan_default_20261029/optimization/fused-plan-default-20261029-a2/validation-evidence/`。
本地副本在`test/megamoe_planner_test/output/fused-default-20261029/`（忽略目录）。

启动排障记录：初次SHMEM初始化失败的HCCL日志明确指出`HCCL_CONNECT_TIMEOUT=90`
低于合法范围`[120,7200]`，已修正为180；另有测试端口占用及副本审计配置遗漏，均只修改了独立测试启动参数。
177.8.24.1的EP4两次被SIGKILL中断，不作为通过结果，也未停止任何其他任务。

候选版TimerOFF正式计时：warmup 10次、测量30次；每次取八rank完整F/B耗时的最大值，
p50 **34.776 ms**，p95 **35.176 ms**。
没有重跑原基线；机器不同，不能据此给出相对原基线的加速比，也不将本结果当作整网step耗时。

结构化验收结果：[fused-default-validation-20261029.json](fused-default-validation-20261029.json)。
代码提交`05c1159`及本验收说明直接同步到`codex/moonflux-8p-integration`，不创建PR。
