# 新采样 Python → 当前八卡算子：复现说明

> 2026-11-02 合入状态：本报告对应实现及最终说明已合入 `megamoe-balanced`；文中“未push”描述2026-11-01原实验交付时状态。默认P1cache0、owner1:1不变，本次不新增设备验收。相对node/analysis等原始证据链接属于文中技术项目归档，大文件不随Git分发；当前可复用入口见[合入说明](20261102-balanced-python-integration.md)。

技术根目录为 `D:/megamoe_train/paper-python-v4pro-20261101`；设备原件在 rack07os1 (`pod7-1-root`) 的 `/home/z00956592/megaflux/paper-python-v4pro-20261101`。容器挂载该目录为 `/project`。所有运行编号独占，复测须使用新编号、独立输出目录和空闲端口，不能覆盖已有成功或失败证据。

## 固定输入与边界

- 官方预训练 `allenai/OLMoE-1B-7B-0924`，HF revision `6d84c48581ece794365f2b8e9cfb043c68ade9c5`；用户指定使用 ModelScope，镜像 revision `d799cc18394603b9bd1abc813be923515bbb5bee`。三 safetensor 分片的大小/SHA 均与 HF LFS 一致，见 MODEL-ASSETS.json。采样前再逐文件核验。
- 官方 `allenai/OLMoE-mix-0924` revision `1e44595eaffc7491dfab23947ea4d5a62b33aff3`，`data/starcoder/python-0000.json.gz` 前2048文档，保持原顺序；抽取脚本 scripts/download-sampling-inputs.py，原文/清单在 sampling/。
- sample-routing-a2.py：原顺序tokenize+EOS，原生4096上下文；固定BF16权重、train、梯度开启、checkpoint、零optimizer更新。每事件8上下文，共32768tokens。按事件顺序找首个任一层κ在[2.7,3.3]，同事件选最接近3、再选层号小者，最多32事件。
- 接受的 source 是 sampling-search-a2 event0/layer9，κ=3.0070735886693。sampling-verify-a2 对选中8上下文各做真实causal loss backward，检查有限梯度并逐位核IDs/prob。该选择不是作者原P95，也不是训练过程复现。
- scripts/map-routes.py 从上述 selected.json 的64专家直方图积分为8专家，再按Top2唯一性约束/最大余数分配，seed17专家置换和token打散；生成 inputs/e8-ep4-m8192-k2。映射κ=2.481662403792143，不能称目标DeepSeek自然路由。
- 当前目标 world8、EP4×两组，E8/Top2/M8192/H7168/F3072/home2/slots2/cap20480，canonical BF16、saved BF16、clip10；合成X/dY/W1/W2，p=.5。实际副本和行数以 PLAN-FACTS.json 为准。算子bench没有整网FSDP/router/shared/optimizer成本。

## 环境、源码与构建

容器image SHA为 `2754601232edf86a4b7dc6e2aca87cf86648e14491e173458ebbf3d5cbd6d2f8`，CANN9.2、torch2.10.0+cpu、torch_npu2.10.0.post6、Transformers5.17.0。SDK/Python依赖位于本项目dependencies/python-runtime的已固定链接，原依赖只读挂载 `/frozen`；原件路径与SHA在迁移/构建清单。不得替换为其他shmem映射。

本轮独立源码 `source`，base `7c41ebf938363995838056677e73e5ea6a289d02`，私有分支 `codex/paper-python-v4pro-20261101`。C0 为原有 earlyDW1=1 / DX panel cache=1，不含本轮P1 cache。

构建入口 build-p1-cache.sh；默认新开关 `MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=0`。候选设1；其余要求 panel restore2 / operand-ready1 / DX cache1 / earlyDW1=1 / worker flags0 / owner16 / compact0 / delay0 / 其他实验候选关闭。三库分别为 p1-cache0-off-a1、p1-cache1-off-a1、p1-cache1-on-a1。原C0的base-off-a1/base-on-a1冻结保留；OFF旧SO与历史已验收C1逐字节一致。

CPU验证入口 check-p1-cmake.py 与 source/integration/wuhan/test_p1_panel_acquire_cache.cpp。后者直接测试真实缓存类型的首次miss/完成后hit、全部metadata key、epoch和panel边界，不替代设备正确性。只读getter与155个实际编译输入核验见 P1-BUILD-VERIFICATION.json；9组配置检查见 P1-CMAKE-CHECKS.json。

## 设备运行

所有NPU进程必须由原 source/integration/wuhan/gate_and_run.py 包裹，启动前两次健康/进程/root-fuser准入，每个进程独立pre/post。scripts/run-and-retire.py只允许退出后已死PID的SMI计账短暂收敛，活PID立即失败；不杀外部进程，不重置设备。失败停止队列、保留编号，新编号重做。

主入口 run-a4.sh：加载 selected-p1-entry.py 核实际SO getter，再原 runtime_bootstrap.py；master与shmem错开32，每EP组shmem再错开32，分配96端口并预探测。所有端口低于32768，避免旧a1命中临时端口。

正确性控制器脚本 scripts/start-p1-correctness71.py，完整B/FB控制器 scripts/start-p1-performance71.py。这些是当次独占启动脚本，原样重跑会拒绝已有目录；新实验须换输出编号。前者大形状flag0、flag1 OFF、flag1 ON各三次五输出逐位比base-capture-a2，再EP2/4/8×flag0/1小形状生命周期。small使用 bridge-entry.py 包装原独立参考，M257/H1024/F512、两层、均衡/热点/切换热点、pool权重投毒、checkpoint和SGD，必须实际产生副本，不能删require-replicas。

原独立无副本MegaMoE reference用cap65536以避免逻辑热点容量截断；候选/C0同cap20480。Y/dX/dp最大数值差0（该参考只记录误差，没有存储SHA），dW沿用既有BF16 partial参考判据；C0/C1始终五输出SHA逐位，不能因原参考舍入差异而放宽C0/C1阈值。

性能入口 performance-a3.py 包装原 replay_captured_layer.py，固定 capture/rank*.pt 的全部输入SHA，在计时前后核五输出。每次F仍在线MoonEP并刷新home，B使用对应F的保存placement。完整B计入权重恢复/计算/owner归并/finish/autograd梯度成本；直接FB另含对应F的home刷新/规划/计算。每进程B、FB各预热10次、测30次；取每样本八rank最大wall，不平均rank，不把分段相加当FB。Timer/profiler/audit/capture与诊断回读关闭；hash在计时区外。

顺序为两轮C0自重复，再C0→C1、C1→C0、C0→C1、C1→C0四对。C0=base-off-a1，C1=p1-cache1-off-a1；每个进程独立门禁。统计脚本 scripts/analyze-p1-performance.py，P90使用(n-1)*q线性插值；控制波动为六个有效C0中位的极差/中位。四对B全改善、配对收益中位大于max(2%,2×控制波动)、FB收益中位为正，逐对FB和两阶段P90无超过max(1%,对应控制波动)回退才通过固定case。新开关默认仍0。

## 打点导出与分析

原 AscendTimerV2，不加新打点/时钟对齐/额外profiling模式。CPU导出在正式OFF性能前完成，不与性能测量并行：

```bash
python3 /project/source/integration/wuhan/export_timer_trace.py \
  --input /project/evidence/base-trace-a3 \
  --out /project/analysis/base-no-waits --world-size 8
python3 /project/source/integration/wuhan/export_timer_trace.py \
  --input /project/evidence/p1-cache1-trace-a1 \
  --out /project/analysis/candidate-no-waits --world-size 8
python3 /project/scripts/analyze-trace.py base-trace-a3 base
python3 /project/scripts/analyze-trace.py p1-cache1-trace-a1 candidate
```

输出每rank96轨道、逐原始区间、同轨零重叠；默认不画wait块，trace-export-audit.json必须pass。分析时间均为rank本地归零，不能跨rank相减推断通信延迟；TimerON分段仅解释依赖链，最终收益由OFF完整B/FB决定。

## 回传与失败索引

本地 scripts/remote71.py 通过已授权SSH别名执行host Python；scripts/upload71.py 上传本项目相对路径，逐文件核SHA；scripts/fetch71.py取回时统一加 node71/ 前缀并保存传输包SHA。模型权重、capture/output大PT和完整依赖原件可能仅留远端同名项目，以最终 DELIVERY.json/清单的实际范围为准，不将索引称为全量回传。

已知失败保留：rack08采样准入外部占用；sampling-search-a1即时post未过；base-capture-a1端口冲突、base-perf-a2适配脚本断言失败。修正后分别新a2/a3重做，均不计入性能。任何后续失败同样保留，不能覆盖。


配套采样/映射/运行/统计脚本交付于 `D:/megamoe_train/paper-python-v4pro-20261101/code/reproduction-scripts.zip`；大模型、PT与原始trace不进Git，位置及范围见技术项目DELIVERY.json。
