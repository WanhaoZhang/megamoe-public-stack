# Python 高偏斜优化与场景合入 megamoe-balanced

2026-11-02，按用户要求将此前独立分支的两项反向优化、最终实验报告，以及固定Python场景生成/复现工具合入共同分支。原提交历史完整保留；生产默认不改变，本次不新增设备/整网验收。

## 代码与实验分支关系

| 内容 | 原实验分支 | 保留提交 |
|---|---|---|
| P1 completed-panel acquire cache | `codex/paper-python-v4pro-20261101`，base `7c41ebf` | `b9147416010cc3c1e72b260dd22ba938b9247b34` |
| owner早AIV1:晚AIV0=2:1 | `codex/python-owner-tail-20261101`，base `b914741` | `847a1f9178793886ae7bbd7abadd936b8e1af4da` |
| 三组有效OFF及原四组计划未完成的最终说明 | 同上 | `ea98005f876b0ba5544533f8c22797de4bef2293` |
| 可复用固定场景/输入生成器 | 共同分支新增 | `integration/wuhan/scenarios/python-olmoe-high-skew/` |

此前 `megamoe-balanced@7c41ebf` 是两轮实验的共同基线，不是实验分支名。现在两项实现与场景一起合入该共同分支。不合入 `megamoe-balanced-shared-expert`，也不改变框架接口或shared前反向实现。

## 优化、默认和回退

P1复用调用内已经完成的panel acquire：首个真实operand load仍执行精确epoch WaitPanel+DSB，再记录完成状态；不跨AIC/专家调用。只减少重复检查，不改变完整K累加/BF16 partial/同元素贡献者SUM次序、ready发布和最终同步。

owner2:1把更多独立tile交给较早加入的AIV1：每worker dW1早/晚42/42→56/28，dW2早/晚21/21→28/14。复用原Task映射，不改变归约字节、贡献顺序、planner和workspace。

| 构建开关 | 默认 | 历史owner候选 |
|---|---:|---:|
| `MEGAFLUX_BACKWARD_RESTORE` | 依入口配置 | 2 |
| `MEGAFLUX_RESTORE_OPERAND_READY` | 1 | 1 |
| `MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY` | S9 panel兼容配置默认1 | 1 |
| `MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE`（DX） | 0 | 1 |
| `MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE` | **0** | 1 |
| `MEGAFLUX_OWNER_EARLY_RATIO` | **1** | 2 |
| owner rows / compact / worker flags / delay | 依入口配置 | 16 / 0 / 0 / 0 |

需要配套CANN、binding、SHMEM及CATLASS；生成新库名，不覆盖运行库。候选与控制使用独立进程，按下列配置各构建TimerOFF/ON配对库：

```bash
export MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_OWNER_EARLY_RATIO=2
bash scripts/build_megaflux_timer_capture.sh python-owner2-a1

# 同源码owner1:1控制，保持其他配置相同。
export MEGAFLUX_OWNER_EARLY_RATIO=1
bash scripts/build_megaflux_timer_capture.sh python-owner1-a1

# P1实验C0/回退：owner1:1，P1cache0，仍保留已验收DXcache1。
export MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=0
bash scripts/build_megaflux_timer_capture.sh python-p1off-a1
```

ratio2依赖earlyDW1=1、DXcache1、owner16/compact0/row-ready0等隔离条件；P1cache1也要求earlyDW1/DXcache1。CMake仍拒绝非法组合。运行前核对worker `build.json`、实际SO SHA及只读getter `megaflux_restore_p1_panel_acquire_cache`、`megaflux_worker_owner_early_ratio`；仅pull源码不会改变已有SO。

## 场景与测量

详细 [构造/采集/完整B和直接FB命令](../../integration/wuhan/scenarios/python-olmoe-high-skew/README.md) 与 [历史性能及流水](../../integration/wuhan/scenarios/python-olmoe-high-skew/PERFORMANCE.md) 已随代码携带。CPU生成路由只需NumPy，八rank密集输入另需固定torch2.10.0+cpu；逐tensor SHA强校验，不下载模型也能重建目标case。

源为官方预训练OLMoE与真实Python文本的新采样，源κ3.0070736；八卡直方图派生case目标κ2.4816624。world8/两个EP4组/E8/Top2/M8192/H7168/F3072、home2/slots2/cap20480。初始owner行数[44819,4067,6672,9978]；历史MoonEP复制E1到rank1/2、E0到rank3，最终每rank16384行。它不是作者P95原trace或DeepSeek自然逐token路由。

每次F在线MoonEP/home刷新；完整B/直接FB用autograd bridge分别计时、每进程10预热30测量、逐样本八rank最大wall，hash在计时外。原TimerV2每rank96轨道、保留逐条区间、不画wait块，不跨rank相减。设备实验依然需要全八卡两次准入、逐进程pre/post门禁和失败停止。

## 已有证据与本次核验边界

- 历史P1四配对：B/FB降幅中位7.0930%/4.6805%，当时固定case判据通过。
- 历史owner2:1三有效配对：B/FB降幅中位4.306%/2.838%；缺第四组，原正式保留未完成。原队列2初始化问题未修复，OFF节点75/队列1与ON节点71/队列2必须分开。
- 两轮节点/队列不同，收益不相加；两轮C0也不是完全原版MegaMoE，无该固定case相对完全原版的已测性能比。
- 历史正确性包含五输出逐位、大形状三次、EP2/4/8生命周期/checkpoint/SGD；原BF16 partial判据不变。routed算子不等于整网/shared融合接受。
- 本次重新完成13项Python CPU检查及语法/输入重建/链接/统计检查；将合入CMake、构建脚本和kernel等155项输入与历史实际编译原件逐文件匹配（规范LF），并核匹配SO SHA。详见 [build-reference.json](../../integration/wuhan/scenarios/python-olmoe-high-skew/build-reference.json)。这是历史构建身份复核，不冒充新的CANN构建或设备运行。

大模型、PT、原始JSON/NPY、SO、SDK和运行日志仍在原技术项目，代码仓只带轻量配置/指纹/生成器与报告。根仓在Native先push后更新native gitlink；两个框架子仓本轮无变更。
