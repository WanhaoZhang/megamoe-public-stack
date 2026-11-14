# Python owner 2:1 实验复现

> 2026-11-02 合入状态：本报告对应实现及最终说明已合入 `megamoe-balanced`；文中“未push”描述2026-11-01原实验交付时状态。默认P1cache0、owner1:1不变，本次不新增设备验收。相对node/analysis等原始证据链接属于文中技术项目归档，大文件不随Git分发；当前可复用入口见[合入说明](20261102-balanced-python-integration.md)。

本地项目 `D:/megamoe_train/paper-python-owner-tail-20261101`；远端节点71/73/75均为 `/home/z00956592/megaflux/paper-python-owner-tail-20261101`。私有源码分支 `codex/python-owner-tail-20261101`，基线 `b9147416010cc3c1e72b260dd22ba938b9247b34`，未 push。代码 commit/补丁/SO/输入指纹见 CODE-DELIVERY.json、DELIVERY.json；**交付三组有效短验证，原四组正式保留条件未完成**。

## 1. 固定输入与来源

沿用上轮新采样，不重采：官方预训练 OLMoE-1B-7B-0924（ModelScope 权重与固定 HF SHA 匹配）在真实官方 Python 文本上，原顺序 pack4096、BF16 train+grad/checkpoint/causal backward、无 optimizer 更新；event0/layer9、32768 tokens、E64/Top8，source κ=3.0070735886693。

直方图映射至 world8、两个 EP4 组、E8/Top2/M8192/H7168/F3072/home2/slots2/cap20480，目标 κ=2.481662403792143；BF16 saved、canonical W1/W2、clip10、p=.5，X/dY/W/p/路由共现为构造输入。每组原 owner 行数 `[44819,4067,6672,9978]`，planner 复制 E1 到 rank1/2、E0 到 rank3，最后每 rank 16384 行。没有真实 FSDP collective。

不是论文作者 P95 trace、不是 DeepSeek 自然 router/训练轨迹、不是整网 FSDP/router/shared/optimizer 加速。原采样及映射来源见上一项目 SCENARIO.md/MODEL-ASSETS.json/node71/PLAN-FACTS.json；REMOTE-LARGE-ARTIFACT-INDEX.json 索引八份 capture PT、模型及环境。模型/大 PT 不进 Git 或报告仓库。

## 2. 环境与构建

节点71 `pod7-1-root` / rack07os1；节点73 `pod7-3-root` / rack07os3。镜像相同：`sha256:2754601232edf86a4b7dc6e2aca87cf86648e14491e173458ebbf3d5cbd6d2f8`；CANN9.2、torch2.10.0+cpu、torch_npu2.10.0.post6。SHMEM SO SHA `246ed665479f676b608d2f318824c02cd1b9eba935720b8efb373c2095bc7112`。镜像/SDK/Python-runtime/原 `_C` 扩展和编译输入分开固定；155 项清单不代表完整第三方传递依赖闭包。

`/project` 是本项目可写目录，SDK/Python-runtime 原件 `/frozen` 只读；节点71 `/prior` 为上一项目，节点73 `/prior` 为本项目 `frozen-prior` 迁移副本，全部八份 PT/输入/原输出收据 SHA 已匹配。不得照抄节点71原绝对路径声称节点73具备同路径。

执行 `build-owner-a2.sh` 构建三库；配置 earlyDW1=1、DX acquire cache=1、P1 acquire cache=1、restore2、operand-ready1、worker flags0、owner16、compact0、delay0、其他实验关。新增 `MEGAFLUX_OWNER_EARLY_RATIO=1/2`，默认1，原 P1 cache默认0；ratio1回退到原 Task 映射。

- owner1-off-a2：新控制接线，ratio1，TimerOFF；与旧 P1 OFF 五输出等价验证。
- owner2-off-a2：ratio2，TimerOFF。
- owner2-on-a2：ratio2，TimerON。
- 正式控制实际使用旧 `prior-off-a1`（SHA `8385529b7b1f637ebad082338f734c8df692e1108491d703743ebbe81b355851`）；ON 控制 `prior-on-a1`（SHA `baaf8ce9ea19c8a423c21771391aed84cb0271956589fba094b881a6ae72bfe3`）。

只读 getter `megaflux_worker_owner_early_ratio` 和 manifest `owner_early_ratio` 必须核验。CPU `test_owner_ratio_partition.cpp` 使用真实 Task、O2且不禁用assert；304组检查。`scripts/check-cmake.py` 检查10组 default1/1/2/非法配置，`scripts/verify-builds71.py` 核实际库/155项源码。

ratio1: `worker+ordinal*workers`；ratio2: 每 AIC 成对worker分三个residue，晚偶worker占0，早奇worker占1/2。tile唯一/完整，贡献者SUM次序/完整K/BF16 partial/ready/epoch/barrier/UB/workspace及planner均不变；dW1早/晚56/28、dW2 28/14。

## 3. 准入与正确性

沿用 `source/integration/wuhan/gate_and_run.py`：阶段启动前两次全八卡健康/进程/root-fuser准入；每进程pre/post；`scripts/run-and-retire.py`只允许已死PID的短暂记账退休，不容许活外部持有者。任一占用/失败停止当前设备阶段，不kill外部任务、不reset；失败编号独占保留，修正必须新编号。

有效正确性是节点71 `OWNER-CORRECTNESS-QUEUE-a2.json`（9项全0）、`correctness-queue-a2.exit`，验证入口 `scripts/verify-correctness71-a2.py`，结果 `LIFECYCLE-COMPARISON.json`：

1. ratio1-OFF、ratio2-OFF/ON各8rank三次 Y/dX/dp/dW1/dW2 SHA 对冻结 base-capture-a2；
2. EP2/4/8 ×两版，原独立参考，M257/H1024/F512、两层、balanced/hot/rotate、pool权重投毒、checkpoint、SGD；
3. 两版14rank对五输出/参数更新SHA再比较。

既有数值判据未放宽。未覆盖legacy/其他EP/整网。准备a1缺父目录、build a1遗漏untracked CPUtest、首次correctness a1缺非Git `_C` 扩展的失败都保留，均未执行候选算子；a2补齐冻结扩展65248416bytes并核SHA/导入后全量重做。

## 4. 原 TimerV2 图

原 `source/integration/wuhan/export_timer_trace.py`；实际导出命令保存在 `scripts/verify-correctness71-a2.py`。控制取上一项目 P1cache1 原NPY，候选取本项目 `evidence/owner2-trace-a2`。

本地四份 JSON 在 `node71/analysis/{control,candidate}-no-waits/all_8_rank_{forward,backward}_trace.json`；原NPY和全部逐条区间已回传，96轨道/rank、同轨零重叠、不绘wait块，trace-export-audit.json两版pass。`scripts/compare-pipeline.py`、`audit-owner-effect.py` 和 `plot-results.py` 离线生成同rank对比、所有worker/tile审计及图。ON数据节点71/原队列2，与节点73队列1 OFF预检分开描述；不能跨节点/rank时间相减。

## 5. 性能入口与当前受阻点

原 `run-owner.sh` + `performance-a3.py` 的 TASK_QUEUE_ENABLE=2，在节点71/73正式轮 HCCL初始化报AICPU ret0x2a，MegaMoE未调用。计时外先建HCCL的 `performance-a4-hccl-first.py` 也未稳定恢复。独立HCCL功能检查及预检不能代表根因已修复；失败细节见 RUNTIME-ISSUE.md。

节点73隔离实验使用 `run-owner-tq1-a2.sh` + `performance-a5-tq1.py`，双方统一 TASK_QUEUE_ENABLE=1；其他native/完整计时loop不变、原DevicePlanValidator/HCCL仍执行。两版 `node73-tq1-preflight-{c0,c1}-a2`均0/pre/post/hash通过，但**它们不是正式样本**。

已停止的正式/准入记录：

- `OWNER-PERFORMANCE-QUEUE-node73-a2.json`：首C0 post外部8个活Python持有者，exit77，作废；
- `evidence/node73-tq1-admission-a3/first.json`：新外部占用，0测试启动，controller未创建；
- `evidence/node71-tq1-admission-a4/first.json`：外部占用，0测试启动，容器未重新启动。

旧 exclusive 控制器**不可再次直接运行**，它们故意使用独占文件/目录。续测前先新编号、重新两次准入，确保旧控制器无活进程、私有容器idle，双方同节点同队列。可参考 `scripts/start-node73-tq1-a3.py`（从两C0自重复开始）或 `start-node71-tq1-a4.py`（含两版预检）；根据实际新节点设置 SSH/只读绑定，不复用旧attempt ID。不能跳过root-fuser、validator、HCCL或改GLOO来凑通过。

完整正式计划：两C0自重复，然后C0→C1、C1→C0重复两次，共10进程。每进程B/FB各10预热+30测量，每个样本取八rank最大wall；Timer/profiler/诊断回读关闭、五hash在计时外；每次F在线MoonEP/home刷新，B含restore/compute/归主/finish/autograd。计时期间不做重型导出/打包。

统计参考 `scripts/analyze-performance73-tq1-a3.py`，必须匹配真实新queue/exit/receipt任务队列字段。P90为(n-1)q线性插值；C0波动=所有有效C0轮中位数极差/其中位数。四B均改善且配对收益中位>max(2%,2×B控制波动)，FB配对中位正、逐对FB及B/FB P90无超max(1%,对应控制波动)回退才接受。当时未完成，不运行统计器伪造最终结果；原队列2仍无验收。

## 6. 交付与回退

`code/owner-ratio.patch`、`.bundle`、`source-final.tar.gz`、`reproduction-scripts.zip`；真实SO与155项编译输入快照在node71证据包。LOCAL-DELIVERY-VERIFICATION.json核Git与构建字节，所有包逐文件SHA。大PT/模型/完整依赖仍远端原件/迁移副本，只提供索引，不称本地全量镜像。

两自有容器已证明idle后停止，node71停止当时pre/post通过；node73停止时外部设备占用仍在，pre/post均false如实保存，仅停自有idle容器，不干预外部进程。临时白名单迁移服务已关闭。若后续任一数值/生命周期/性能失败，保留默认ratio1实验并定位，不自动叠其他优化。生产P1默认0、ratio默认1不变；未push共同分支。


## 7. 最终交付：节点75三组有效结果（覆盖此前进行中状态）

用户明确选择“先交付三组有效结果”，不再启动设备实验。节点 `pod7-5-root` / rack07os5（177.28.25.1），同镜像私有容器 `megamoe-python-owner-tail-20261101-node75-a1`；`/frozen` 为本项目 `frozen-runtime`、`/prior` 为 `frozen-prior`，10项迁移约4.07GB、683源码/SO/扩展/输入文件与3467运行依赖文件全部SHA检查。迁移服务已关闭。实际SO/扩展/torch身份见node75/NODE75-BUILD-IDENTITY.json。

准备a1缺evidence父目录、a2外部占用，均0设备测试；a3重新两次准入后启用 `run-owner-tq1-a2.sh` + `performance-a5-tq1.py`，双方统一TASK_QUEUE_ENABLE=1，原DevicePlanValidator/HCCL不跳过。八个有效进程为原 `OWNER-PERFORMANCE-QUEUE-node75-a1.json` 的前8项：两C0自重复+AB/BA/AB三组，各10warm30measure/逐样本8rankmax，前后hash和pre/post全部通过。C0=prior-off-a1、C1=owner2-off-a2，SO未改；两preflight不纳入正式样本。

第四组a1 C1 post77，a2 C0 post77，a3 C0 post76，均因退出设备门禁不满足而停止，新编号重新准入。整组失败则整组排除，所以a2/a3各自eligible的C1也不计入；未按耗时选样本，未把跨占用分段单版硬拼成第四组。旧queue/controller/门禁/日志均保留。

最终统计入口 `scripts/summarize-three-pairs75.py`（host纯CPU，不依赖正在运行的容器），逐项检查前8eligible、TimerOFF、TQ1、10/30、五hash、八rank maxima并重算P90/控制波动；输出 `node75/OWNER-PERFORMANCE-THREE-PAIRS.json` 与 `OWNER-PERFORMANCE-QUEUE-THREE-PAIRS.json`。`formal_retention_accepted=false` 因缺原计划第四组，三组观察性指标均改善；不要运行旧四组analyze/finish生成器来强行接受。

节点75未启动TimerON，最终图使用节点71已经通过的原TimerV2/TQ2四份前反向JSON；图与OFF条件差异必须保留在报告旁。每rank96lane、零重叠不绘wait。三组性能图在analysis/paired-performance-three-pairs.png。未把ON缩短当作OFF预算。

最终只停证明idle、无在途自有队列/等待/分析进程的容器，真实前后八卡门禁保留（外部占用可以为false，不能写成最终空闲）。代码/报告/复现私有commit，不push共同仓；默认ratio1/P1cache0不变。原始数据/SO/指纹在技术项目，CodexWork只简明索引。完整链条见DELIVERY.json/LOCAL-DELIVERY-VERIFICATION.json。


配套采样/映射/运行/统计脚本交付于 `D:/megamoe_train/paper-python-owner-tail-20261101/code/reproduction-scripts.zip`；大模型、PT与原始trace不进Git，位置及范围见技术项目DELIVERY.json。
