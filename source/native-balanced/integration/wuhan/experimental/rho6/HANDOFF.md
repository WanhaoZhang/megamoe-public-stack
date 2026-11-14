> 本文件保留历史实验状态。当前校验已统一、原 generic/fused 开关删除，最新用法见 [FUSED-PLAN-DEFAULT.md](../../FUSED-PLAN-DEFAULT.md)。

# rho6 优化代码交接（2026-10-28）

## 2026-10-28 后续门禁结果：P4确定性整网失败，禁止晋级

P4宏1仍是默认关闭的实验源码，**当前不能用于已验收训练或性能结论**。新节点原条件下C0自重复及同编译器宏0对照C0R的10步训练数值、最终local模型参数分片均逐位一致；宏1P4虽八rank各10步、退出0且即时八卡后门禁通过，确定性数值门禁失败。首步rank6本地loss已有细微差异，各rank grad norm由C0的19.234207153320312变为19.233402252197266；10步reduced LM相对C0最大6.808470584544066e-5，最终local模型参数不逐位。不能将这种差异直接判作正常噪声；原因尚未定位。

此前H7168主层/MTP各实际step10捕获与固定/在线计划回放共12case/96rank全部退出/后门禁通过、五输出逐位；该有限覆盖没有替代全网门禁。本轮停止性能筛查、延迟producer构建和200步，不重跑失败原编号。立即device导出退出0，plog/shmem与完整失败证据已归档；原生产SO38472e...未变。后续先定位首步rank6的具体层/阶段及候选自重复，再修复并以新编号完整验证。独立已验收P1和宏0路径不因宏1失败被宣称失效。

证据：项目`rho6-optimization-20261027/p4-det-node8-failure-a1-summary.json`、`collected-p4-det-node8-failure-analysis-a1/`、完整`collected-p4-det-P4-failure-a1/`；失败archive SHA256 `9c2584cf5c8c27fd453875523cab3bc2dbe403df771f6cb9433a8bd139f04ca3`。本补充优先于下方旧“待验证”描述。

本次按用户要求提前交接源码，供同事继续修改。目标配置仍为单机8卡、EP4×EFS2、E8、Top2、H7168/F3072、S8192、rho6、重计算、Muon。所有新优化保持默认关闭；这不是整网性能或原始A精度验收版本。已知数值不逐位的CPU控制tape不作为本次已验收优化交付。

|部分|代码与开关|已验证|仍未完成|
|---|---|---|---|
|P1融合计划校验|`execution_plan_fused.py`、`mega_moe_plan_validator.cpp`；训练参数`--optimization.megamoe_balanced_plan_validation fused`，默认`generic`|修复版EP2/EP8正常/错误元数据差分；真实FSDP/checkpoint/3SGD；H7168真实main/MTP五输出；确定性10步loss/grad norm/最终参数分片逐位C0|整步正反筛查仅0.138%/0.446%，低于3%，不晋级、不作为默认|
|P2发布同步|已有`WUHAN_PUBLISH_SYNC=device / stream`，默认device|小EP stream功能及原node5确定性10步逐位|新节点性能尚未配对；CPU tape原始不逐位且旧队列缺终态，不能称通过|
|P3 owner peer缓存|本目录`p3-owner-peer-prototype.patch`，以0fd8661为base，**未应用到默认源码**|CPU9000贡献者case/27274地址检查，关闭路径源码token相同|无完整设备验收；先compact独立过门禁再peer，不能与P4直接组合后宣称通过|
|P4输入行发布|`MEGAFLUX_DISPATCH_READY_LARGEST=0 / 1`，默认0；device头文件`mega_moe_input_ready_selection.h`|独立同编译器C0R与原C0、P4与C0R在EP2 MoonEP/Native、EP8 MoonEP、真实EP4×EFS2的五输出/plan/输入/SGD hash逐位；共12case/60rank回执，退出0/即时全8后门禁|H7168真实层、延迟权重producer、确定性整网/最终参数、Timer OFF正反性能、前后profiling与最终200步|

P4将唯一现有row-ready计数器组给最大有效本地专家，tie保留首个；不增加GM计数器，不改专家/GEMM顺序、完整K、BF16 partial或每tile归主序。显式限制临时数组宽度，Python入口只允许EP2/4/8；构建拒绝与旧worker、compact或owner32直接组合。已测核心device头文件与本次交接字节相同，当前PR的大EP基础和新增host守卫不代表重新完成多机验收。

P4证据见`p4-lower-gates.json`。桥接门禁H512/F512，包含balanced/hot/rotate及额外skew、checkpoint和4次SGD；真实EFS2为3次SGD。EP8实际execution counts证明选到了非首个最大专家。独立测试时生产C0库为38472e...、新编译器宏0对照00dcf60f...、宏1候选5ddb765d...；这些SHA只标识已测独立快照，不能冒称同事重建产物具有相同SHA。

## 构建与使用

必须同步三个代码仓，并重建相应扩展/SO；单独git pull不会更新已经加载或磁盘上的旧库。P1需要包含`validate_plan_local/global`的重建Python扩展和`libascend_deepep_plan_validator.so`；旧P1失败版库不可复用。源码已修复packed/counts共享cacheline多writer覆盖：packed和header/counts单writer，分片scratch继续并行。

P1的generic路径不加载validator库。P1与native planner组合fused会拒绝，EP>8的fused也拒绝；大EP继续generic。将上述`--optimization.*`参数放进实际torchrun参数列表，原整网shell不转发末尾任意参数，不能简单追加到shell末尾。

P4在Native仓根目录、新的构建名下分别编译对照和候选。先source匹配的CANN，并指定匹配SHMEM_HOME_PATH：

```bash
export MEGAFLUX_WORKER_SHARED_PREFIX=0 MEGAFLUX_WORKER_EXTERNAL_REFRESH=0
export MEGAFLUX_OWNER_ROWS=16 MEGAFLUX_OWNER_COMPACT=0
export MEGAFLUX_BACKWARD_RESTORE=2 MEGAFLUX_RESTORE_OPERAND_READY=1
export MEGAFLUX_RESTORE_DELAY_CYCLES=0
MEGAFLUX_DISPATCH_READY_LARGEST=0 bash scripts/build_megaflux_worker_pipeline.sh rho6-control 0 0 0
MEGAFLUX_DISPATCH_READY_LARGEST=1 bash scripts/build_megaflux_worker_pipeline.sh rho6-largest 0 0 0
```

检查各`worker-variants/<name>/build.json`和`input-ready-selection.json`，并对实际加载库调用`megaflux_dispatch_ready_largest()`核0/1。整网runner使用`WUHAN_WORKER_VARIANT=rho6-control`或`rho6-largest`选择库，配套`BALANCED_RESTORE_MODE=panel`；planner仍默认MoonEP、plan validation仍generic、publish仍device、控制tape仍npu。不要同时打开P1/P2/P3后把组合结果记作单项P4。

接整网使用现有`integration/wuhan/run_training_case.sh`与主使用指南，继续原EP4×EFS2/E8/rho6/home2/slots2/capacity20480。正式计时关闭profiler/Timer/捕获和诊断hash；先C0/C0R/P4同机正反筛查，入围再同轮A/C0/C*/D正反30步warm5，主分母A÷方法，>=3%超过波动且HBM不增才最终200步。

## 同事接续入口

- Native：P1是独立提交；P4默认关闭的实验实现与小门禁摘要另一个提交。P3仅作为原0fd8661基线的原型补丁参考，不要整文件覆盖当前大EP源码。
- FSDPTurbo：MegaMoeBalancedConfig的plan_validation字段及pool key/构造参数传递。
- MindSpeed-LLM：OptimizationArguments与ModelFactory传参。
- 本轮新机基线A/C0/D10步warm2：3136.74138/2939.82980/2981.20647ms；只短筛查，非正式优化收益。C0/D对A reduced LM仍超2e-5，不默认晋级。
- 完整原始数据/脚本/冻结源码位于`D:/megamoe_train/moonflux接整网/balanced-8p/rho6-optimization-20261027/`；当前新节点项目`/home/z00956592/megaflux/balanced-rho6-node8-a1`，不要覆盖在途/已验收快照。看STATUS.md获取实时作业，任何新启动先全8健康/进程/root fuser，不抢占/复位/干预别人。
