# 执行与重放

源码位于 `stack/`，原始回执在 `evidence/`。先读 REPORT.md、STATUS.json 和 DECISIONS.md；测试名不可复用覆盖。

## 环境

- 已跑通节点：SSH 配置别名 `pod6-1-root`，单机 Ascend 950DT 八卡。
- 专属容器：`temporal-replica-frontier-20261115`；专属远端目录 `/home/z00956592/megaflux/temporal-replica-frontier-20261115/stack`。
- 基础镜像 `planner-executor-ep8-runtime:20261114`，Python `/usr/local/python3.12.13/bin/python3`。环境指纹和每项实际 SO SHA 保存在各次 source-snapshots runtime JSON。
- 本地工具 Python：`D:\megamoe_train\planner-executor-ep8-lab\.venv\Scripts\python.exe`。本地未装 torch；CPU torch 合约测试也在远端容器使用 CPU 张量运行。
- 新选择器单独构建，不替换冻结 Native worker。构建脚本 `build_temporal_scheduler.py/.sh`、日志 `build-temporal-scheduler.log`，ABI3 冻结配置 `stack/configs/temporal-scheduler-freeze.json`。

## 不互相干扰

`frontier_remote.py gate` 经控制器串行执行，等待设备空闲，记录 driver 与 fuser，冻结源码和 SO，采样整卡 HBM。最长准入等待一小时，训练超时900秒。它不停止其他任务。

上传或构建前应确认当前没有本研究的源冻结运行；不要在运行中调用 sync_frontier.py。该脚本只同步隔离 lane，不能同步到旧正式工程。另一项目不共享本研究锁时也必须检查实际设备占用。

## 小形状回归

在本目录 PowerShell 执行（使用新 tag）：

```powershell
& 'D:\megamoe_train\planner-executor-ep8-lab\.venv\Scripts\python.exe' -X utf8 .\frontier_remote.py gate --metadata-only --tag NEW-TAG --layers 2 --steps 3 --tokens 256 --hidden 512 --ffn 512 --microbatches 4 --forward-window 2 --recv-capacity 4096 --maxvio 10 --planner native --route-bank /home/z00956592/megaflux/temporal-replica-study-20261114/stack/small-routes --temporal-policy easy_home --temporal-scheduler ascendc --dump-grad-step0 --grad-sync
```

该小切片银行不宣称精确 MaxVio10。导出的 step0 梯度在 clip/Adam 前同步，再比较同窗口 `gradient-w2-legacy` 的初始参数/输入/路由身份及全部 FP32 梯度：

```bash
python3 tools/compare_temporal_gradients.py evidence/gradient-w2-legacy evidence/NEW-TAG --step0-only --out evidence/NEW-TAG-comparison.json
```

比较脚本在容器中用 CPU 加载原始 PT。输出/梯度门槛不变；完整整网还检查每个参数≤0.02，不只按大类别平均。metadata-only 不下载数GiB的 PT，远端路径在本地 receipt 中明确记录。

## 主形状资格

同上替换为 `--layers 4 --steps 3 --tokens 16384 --hidden 4096 --ffn 1024 --recv-capacity 262144`，去掉小银行参数。必须确认八rank完成、三次Adam、source_unchanged、实际SO、计划退休、有限性和90GiB共同门槛。

W4 已因内存预算退至 W2，并且四任务整网梯度仍失败；不要将当前入口可接受 W4 解释为已验收。v1主配方的`--hold-fsdp-forward`默认关闭。v3单独登记共同开启该公开FSDP选项的实验族，并保留关闭选项的原始Native对照；不能把两个配方混算成一条曲线。

## 筛查及正式矩阵

`stack/tools/campaign_temporal.py` 默认只生成18次筛查清单：六个现有执行方式×三种时间相关性、MaxVio10。`--execute` 检查 `configs/temporal-gates.json`；未通过不能直接改布尔值绕过。

每进程25次optimizer更新，5预热+20测量；每更新的八rank最大 wall time为一条观测。筛查一轮不等于三轮稳定收益，也不等于108次新方法正式矩阵。

当前未实现 Afore 副本缓存或新联合策略；正式矩阵保持关闭。先观察真实缺陷再决定是否实现，不用离线全知路由、缓存计划重放或未计时的前置计算作为在线成绩。

2026-11-15当前筛查入口为`qualify_strong_baselines_and_screen.py`：先重新验收无探测固定/FIFO，再冻结公共门禁，然后执行已登记的18次。脚本不支持覆盖旧tag，不应对已经启动的批次重复调用。`screening-results.json`保存每次状态。

本地`watch_screening_report.py`只读取已收回回执，调用`analyze_screening.py`及`report_current.py`刷新报告；它不上传代码、不停止训练。分析器复核真实加载SO、25×8更新、路由调用次数和同相关性的身份，只给已完成点作图。

固定/FIFO以当前层输入可用为前沿，不观察路由；日志loads为null。`diagnostics/screening-maxvio-10.npy`是从冻结远端银行复制的离线审计输入，SHA必须与每个运行的source-identity一致，不可加入在线算法。

`--temporal-schedule-replay PATH --diagnostic-only`仅用于保持已观察任务顺序的离线消融；两个参数缺一拒绝，配置显式记录uses_future_derived_schedule=true。该数据的online_speedup为空，不进入screening-runs.csv。

TimerV2 诊断由 `--diagnostic-timer` 显式启用，不能混入正式计时。其跨rank时间不能直接用独立零点拼接。选择 host 包络不等于纯设备规划时间。

## 前缀复用与共同前向驻留v3

`qualify_resident_prefix_v3.py`只接受全新登记，不要对现有登记重复运行。它先在共同驻留配置下做六路三次Adam/逐参数梯度门禁，再做同一主形状的三次Adam/90GiB检查，之后执行七路25更新pilot。状态与严格核验分别是`resident-prefix-v3.json`、`resident-prefix-v3-summary.json`；本地报告由`watch_resident_v3_report.py`刷新。

前缀复用显式参数为`--temporal-prefix-reuse --hold-fsdp-forward`。它使用真实prefix输出做选择，改变checkpoint划分为prefix与FFN/shared finish。共同Native基线不使用prefix复用但开启同一驻留选项；原始Native也另保留一个新进程对照。性能数据必须确认`prefix_diagnostic_sync=none`、`prefix_stage_audit=false`、router每任务原始/重放合计两次，并验证实际加载正常worker。

初版prefix v2在Easy-home梯度门禁失败，原始失败和五路隔离不能删除。`--prefix-diagnostic-sync`、`--prefix-stage-audit`和离线同顺序重放只用于受限小形状数值诊断，不得进入性能。通过v3不等于修复了W4，不等于根因已确认，也不等于实现了Afore副本缓存。

CPU Afore风格语义参考可独立执行`validate_afore_reference.py`，不占用NPU、不上传代码。其10个测试含120个公共编译器用例；该结果只证明规划合同，不证明设备权重迁移、训练数学或性能。接入边界见`AFORE-ADAPTATION.md`。

## 作者核心MaxUtil基线与本地重放材料

旧v1/v3使用work_tie适配，负结果不能冒称作者原规则的结论。`qualify_upstream_maxutil.py`已登记独立upstream_first规则，先过设备、完整梯度和主形状门禁，再运行9点screen；不要重复启动已有登记或复用tag。

新规则显式参数为`--temporal-policy easy_home --maxutil-rule upstream_first --temporal-prefix-reuse --hold-fsdp-forward --temporal-scheduler ascendc --temporal-scheduler-library /path/to/stack/source/native-balanced/ascend_deepep/lib/libascend_deepep_temporal_scheduler_v4.so`。旧ABI3不支持新模式，入口会拒绝，不会自动退回旧选择器。固定home控制使用fixed_home、相同prefix/驻留和ABI4；Native控制使用legacy及相同驻留，单调用Native规划固定beta768、反向罚项2、最小收益4。

新SO的实际字节及指纹在`implementation/upstream-selector-v4.{so,json}`，原始24路径/22去重SO在`frozen-build-export.tar.gz`。`stage_frozen_binaries.py --include-upstream`已在纯staging目录重建并复核25个路径；它不覆盖运行中的库。性能必须通过`EP8_WORKER_VARIANT=ep8-cap64-aligned`选取46772ebc，不能误用根目录的其他worker或TimerV2变体。

源代码按`implementation/source-base.tar.gz`＋`source-changes.tar.gz`恢复，覆盖文件SHA在source-manifest.json；实际每个运行的精确冻结源码仍以evidence/source-snapshots为准。根目录的replay-tools.tar.gz还保留运行器、归档、分析器、Afore CPU原型、作者源快照及LICENSE。

路由npy未进入源码Git归档。`stage_replay_inputs.py`只复制已实测的4MiB MaxVio10银行及原始JSON，并对照每个新screen的源指纹，默认放在`implementation/replay-inputs/routes`。远端重放可用该目录作为`--route-bank`；不会生成替代路由。其他五档和108矩阵尚未运行。合法合成输入/next-token标签由固定源码seed42重建，按运行中的输入SHA复核；不要把该合成短跑称为自然路由预训练。

## 报告与代码快照

`report_current.py` 从回执重建 REPORT.md/STATUS.json；`plot_qualification.py` 仅画资格/显存图，不能把3次更新画成性能曲线。源码 diff、未跟踪新增源码和配置需要一起归档，不能仅交付 git diff 而遗漏新文件。

失败、OOM、准入失败、参考修正前后的数据保留。原始张量放原项目专属目录；D:\WORK只放报告与路径索引。
