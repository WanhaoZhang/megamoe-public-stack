# 重放与数据口径

2026-11-14已按用户要求收尾，见[CLOSEOUT.md](CLOSEOUT.md)。保留108次主矩阵及12次第四轮；原完整协议没有全部验收。
`evidence/USER-CLOSEOUT-REQUEST.json`会阻止自动设备接入和续跑。恢复任何NPU实验都需要新的明确任务，本轮不再自动执行。
只读重做已保留结果的分析可以使用以下命令，不调用设备：

```powershell
.venv/Scripts/python.exe tools/analyze_matrix.py --campaign primary-existing --campaign primary-existing-recovery --campaign primary-step-time
.venv/Scripts/python.exe tools/analyze_matrix.py --campaign fourth-paired --out analysis/fourth-paired
.venv/Scripts/python.exe tools/analyze_planner_decisions.py
.venv/Scripts/python.exe tools/report_progress.py
.venv/Scripts/python.exe tools/accept_planner_experiment.py
.venv/Scripts/python.exe tools/write_planner_report.py
```

完整协议的`ACCEPTANCE.json`保持未完成；用户收尾及120次数据身份核对见`CLOSEOUT-AUDIT.json`。
轨迹严格检查保留失败结果，不使用观测模式把它改记成通过。

当前任务状态首先看[阶段报告](D:/megamoe_train/planner-executor-ep8-lab/reports/PROGRESS.md)和`STATUS.json`。下文提供已实现入口；未验收的step_time不能直接进入正式成绩。

## 固定源码和依赖

- Native公开基线：`48ef24a6586a633ddbbf265d4eab72e229bf1fae`。
- 继承的EP8/64槽适配：`patches/native-accepted-ep8-vs-48ef.patch`。其私有Git对象为`4439b933ec1be514efc2e06f1a5ca70604c31bfb`。
- 统一接口/已有策略：`patches/native-unified-interface.patch`，在上述继承基线上应用。
- FSDPTurbo：`9c443c614f6aba84908a9eedbaecd50cc15fb022`＋`patches/FSDPTurbo-vs-9c443c6.patch`。
- MindSpeed-LLM：`dd5ee0321c1179aaaded4fba4f119ed310b30639`＋`patches/MindSpeed-LLM-vs-dd5ee032.patch`。
- 私有容器镜像、158个已安装包、Python与编译器身份见`dependencies/ENVIRONMENT.json`；SDK、运行库及实际32/64产物见`dependencies/RUNTIME-CLOSURE.json`与`artifacts/binaries/`。
- 原始构建、实际加载的SO及计时源码身份分别见构建日志、每run的`libraries-rankN.json`、`source-identity.json`及`evidence/BASELINE-FREEZE.json`。正式计算worker沿用已验收SO；统一Planner与Host已重建。Timer9是另一份独立诊断SO。

所有源码patch均做了临时Git index上的应用检查，不改旧项目。大环境、SO、SDK和原始数据保存在本项目，不放进报告仓库。

## 本机分析

PowerShell，在`D:\megamoe_train\planner-executor-ep8-lab`执行：

```powershell
.venv/Scripts/python.exe tools/collect.py
.venv/Scripts/python.exe tools/analyze_matrix.py --campaign primary-existing
.venv/Scripts/python.exe tools/audit_model_trajectories.py --campaign primary-existing
.venv/Scripts/python.exe tools/report_progress.py
```

如存在明确关联的基础设施补跑，分析命令还需加`--campaign primary-existing-recovery`；加入新策略后再加`primary-step-time`及其recovery账本。有效的成功/慢轮均不可替换。允许补测的情况为：零训练步的明确HCCL初始化失败，或有驱动PID/cgroup证据的外部NPU占用窗口。后者按统一时间规则处理全部策略及快/慢尝试；原始成功数据也保留，并记录排除原因。关联补测要求源码身份不变，不用它挑选更快的结果。

正式图由`tools/plot_matrix.py --analysis analysis --out figures`生成，三次独立重复全部完成才连线。PNG/SVG/PDF来自同一份审计数据。最终报告和页面加入后，`overview.html#planner-comparison`是独立Planner章节，旧三后端章节保留。

Planner项目最终验收和报告入口分别为`tools/accept_planner_experiment.py --require-complete`、`tools/write_planner_report.py`。它们只读既有测量并生成验收/报告；缺少阶段会明确列为pending，不能由存在一张图推断完成。不要使用从旧三后端工程继承的`final_acceptance.py`或`write_report.py`来评价本轮Planner矩阵。

## 远端设备入口

远端项目：`/home/z00956592/megaflux/planner-executor-ep8-lab`，专用容器同名。`tools/controller.py`对本任务加设备锁，并检查八卡空闲；不停止其他作业、不reset卡、不做全局清理。

例如单个完整训练进程：

```bash
python3 tools/controller.py --backend balanced --mode train --tag MY-UNIQUE-TAG \
  --planner native --recv-capacity 163840 --maxvio 20 --steps 25
```

标签必须唯一，已有目录拒绝覆盖。后台控制器仍运行时不应另起同一矩阵。正式已有策略矩阵入口为：

```bash
python3 tools/campaign_planners.py --name primary-existing --capacity 163840
```

这个名字当前已有数据，直接重跑会拒绝。重新实验应使用新名字，并保存新配置。不要把初始化、加载和校验结束前的等待算进训练步；也不要把这些进程的总用时当作20步平均步时。

## 校准与新策略

`tools/after_baselines.py`要求90个已有策略计划单元完成并审计，再执行原容量接口诊断、链路测量和独立校准。校准协议已在观察成本数据前写入`configs/calibration-protocol.json`；初始策略按独立银行0.3/7/18的完整步时选择。均衡MaxVio0仅是覆盖无副本分支的额外校准，不属于主矩阵。

`tools/campaign_search.py`要求`evidence/step-time-qualification.json`中的模型SHA已验收；禁止未校准的插件进入成绩。入口分primary、fourth、heldout和条件ablation，各自保存独立配置与账本。模型、插件、主路由与测试条件在正式新策略矩阵前冻结。

本次四个模型为`configs/step-time-full.json`及同目录的no-gradient-service、linear-kernel、uniform-topology变体，
批准SHA和实际插件SHA在`evidence/STEP-TIME-FREEZE.json`。源、21个原始二进制及独立插件均可逐文件校验。
`after_baselines.py`是首次校准控制器；完成后不要直接重跑它来“刷新”本次结果。
它会匹配当前源码另建校准revision并重建诊断link库，不是冻结方案的重放入口。

校准实现细节见`reports/STEP-TIME-IMPLEMENTATION.md`。正干扰网格是在独立校准
暴露稠密副本低估后增加的模型修订；原始模型、全部拟合、步骤3验证残差和
修订记录均保留。`configs/step-time-semantic-*.json`只允许数值门禁，绝不进入正式批准模型列表。

最终验收后，在原Linux环境用新名字重放冻结方案；`--resume`只恢复相同请求，
不覆盖部分失败或另挑更快的运行：

```bash
python3 tools/replay_planners.py --name my_replay --validation --dry-run
python3 tools/replay_planners.py --name my_replay --validation
```

默认重放108次主矩阵；`--validation`再运行12次第四轮、72次独立银行和达到
同一四轮门槛才运行的54次消融。它复用原方案冻结的最强已有基线及模型，不再选型
或拟合。新账本带名字前缀，原数据不改；重放结果不能并入本次主曲线。
严格字节身份重放使用归档源码及产物；源码patch用于审查/接入，换目录重建
可能改变产物SHA，需要重新记录并验证，不冒充本次实际加载的SO。

独立银行和主银行均先生成逻辑ID；设备Planner只接收当前调用的统计，没有访问未来层/步路由的入口。离线计划重放仅用于诊断，不作在线成绩。

`dump_plan_bank.py`另外导出逐层逐步的copies、来源quota、实际反向专家顺序及守恒检查。其隔离规划调用wall/Event区间含当前路由统计、策略和公共编译校验，排除调用前rank对齐及调用后的CPU检查/写盘。这是单独的规划开销诊断，不是完整训练步时间，也不替代主矩阵的在线规划成绩；前5步作为诊断预热。

## 时间、显存与失败

每进程25步，前5预热；每个测量步先取八rank最大完整wall，再对20步取均值，最后对三次独立运行取均值。P50/P90另列。计时覆盖统计/规划/公共编译校验、前向、loss、重计算/反向、通信、clip及Adam，直到设备完成。

整卡HBM来自驱动采样，包含初始化、预热、运行库、缓存与通信池，三次运行/八卡/全进程取最大值。采样会漏掉短瞬态；另存Torch活跃/预留峰值，二者不能相加。共同门槛90GiB，163840容量已通过五策略完整模型门禁。

所有退出、OOM、非法计划、超时和基础设施异常保留。`reports/DECISIONS.md`记录原因、验证及影响。结论仅适用于这套Ascend两层裁剪、受控路由与合成token短程训练，不是官方H20九层图或真实语料收敛复现。
