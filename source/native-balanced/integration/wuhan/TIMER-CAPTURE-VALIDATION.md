# 完整 Timer 采集交付验证（2026-10-29）

运行时源码为 `2240cb5`（基于同事最新8087a78，先补五个helper，再补完整诊断链路）。本次重新编译worker/planner，并用仓库自带入口采集；没有把历史冻结库或旧图当成新分支验证。

## 构建与范围

Ascend950/A5，rack07os2隔离目录；CANN9.2.0，Python3.12.13，torch包2.10.0（运行时显示2.10.0+cpu）、torch_npu2.10.0.post6。Python SHMEM与本仓库binding复用同一个libshmem映像，独立入口不导入外部npu_patch/FSDPTurbo/MindSpeed。

两套库均为S9、flags0、owner16、panel restore、operand-ready1、strict MoonEP/direct0；关闭其他优化候选。以下SHA仅标识本次实际构建，其他机器重新编译应生成自己的build回执，不要求二进制SHA相同：

| 库 | SHA256 |
|---|---|
| TimerOFF worker | `aa06bca32ab4847e0ae5d28491dd317f02dfc4fcff876d8c00deb4eec4e94c53` |
| TimerOFF planner | `4993c6573c9f175d0551733923d3238d2e815e0c2eb73d5f27bde5267939892a` |
| TimerON worker | `61cf5d762c30f3a1494af0a86c2d972891c0d9a46f7f6513c349e63ca89f8050` |
| TimerON planner | `a58d640e3e1b0006f0e44b335517ab0ac3d3d3d3d90b8b1767c3d2dd68aaa845` |

新工具只向隔离worker-variants目录写库；没有覆盖生产或此前实验的动态库。planner Timer选项默认OFF，TimerON必须匹配显式扩展header。

## CPU与默认路径检查

- 33项unit合同检查：采集/routing/layout5项、导出7项、planner header3项、原benchmark18项，通过。
- CPU bridge生命周期/重算/保存钩子16组合、restore审计、完整K行列/尾块operand ADL检查，通过。
- Python语法、Bash语法、git diff whitespace检查，通过。
- 对照8087a78确认direct判定仍保留；execution_plan、execution_plan_device、worker_pipeline、moonep_plan、gradient_pipe、replica_mixed六文件内容未回退。没有替换同事的融合校验、路由策略或大EP表。

## NPU采集

每项均使用独立TimerOFF/ON进程、各rank三次完整前反向；从第一次开始记录并检查五输出。TimerON对齐同输入TimerOFF，真实capture模式还核对原始文件/tensor SHA、在线placement和捕获期望。

| 项目 | world / EP | shape | 结果 |
|---|---|---|---|
| small-a1 | 2 / 2 | M257 H512 F512 E4 Top2 home2/slots2/cap1024 | OFF/ON各三次，通过 |
| synthetic-a1 | 8 / 4 | M8192 H7168 F3072 E8 Top2 home2/slots2/cap20480，ρ=6 | OFF/ON各三次，通过 |
| actual-a2 | 8 / 4 | 同完整shape；原step10主层训练capture | OFF/ON各三次，placement及五输出对捕获逐位通过，应用退出0；原post-gate为false，见下文 |

成功运行共108个rank-call；其中TimerON的54个调用从第一次起逐位对齐配对TimerOFF；真实输入OFF/ON共48个调用同时逐位对齐原捕获五输出。合成OFF首调用建立本次参考，并检查后续重复性。

八卡配置为两个EP4组。此入口执行完整专家算子前反向，不包含真实FSDP通信和optimizer；合成输入不替代CPU数学golden，诊断图不用于正式性能或训练质量结论。

## 已完成图的独立回读

独立脚本逐条回读JSON，检查每rank32AIC+64AIV轨道、原始cycle换算、coordinate原始起止保留、scope marker及同轨道零重叠，而非仅信任导出器audit字段。

| 输入 | 前向duration / marker | 反向duration / marker | 同轨道重叠 |
|---|---:|---:|---:|
| 两卡small | 1394 / 582 | 4152 / 2528 | 0 |
| 八卡synthetic | 22472 / 26456 | 656128 / 147232 | 0 |

八卡合成图前向22,590,910字节，反向406,916,099字节。真实输入也成功生成两张完整图，导出audit为全八rank、96轨道及零重叠。图和原始NPY保留在实验项目，不上传源码仓库。

## 失败记录与边界

初始a1/a2构建因隔离目录的CATLASS依赖链接不可达失败，补齐正确相对链接后使用新build prefix delivery-a3完成配对构建。失败日志保留，没有覆盖成成功记录。

真实输入第一次actual-a1在首次HCCL签名all-gather的 `RunAicpuIndOpThreadInit` 报错（0x2a），尚未进入MegaMoE前反向；进程正常退出，post-gate为健康/空闲/root-fuser clean。保留plog、device/shmem日志和失败退出回执。两次新空闲检查间隔>=120秒后，以同源码/同二进制、新输出和新bootstrap端口执行actual-a2；不据此宣称已定位底层HCCL异常根因。

actual-a2应用退出0，所有rank的逐次五输出/placement检查和图导出均通过；但原post-gate发现PID3071046打开davinci_manager，因此严格隔离eligible=false。现场读取/proc确认其为另一目录的64P elastic combine作业，在本次最后阶段进场；本任务采集进程均已退出。原false gate、后续观察和外部holder身份保留，不改写成隔离通过，也未终止或干扰该作业。本次交付声称功能复现通过，不声称此运行具备正式性能统计资格。

所有大文件、源码/构建快照及失败证据留在原实验项目 `D:/megamoe_train/timer-capture-delivery-20261029/` 与对应服务器项目。新同事按[TIMER-CAPTURE.md](TIMER-CAPTURE.md)执行即可，不需要这些历史实验目录的脚本。
