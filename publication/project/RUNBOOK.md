# Runbook

Source: stack/, branch research/step-lifetime. Primary project report: REPORT.md (watch_report.py refreshes it from guarded receipts).

## Isolation

A lane/container: /home/z00956592/megaflux/step-lifetime-study-20261115.
B lane/container: /home/z00956592/megaflux/step-lifetime-kernel-20261115.
Both use the existing global device-jobs.lock under temporal-replica-study-20261114. No private competing hardware lock. Controllers check fuser, driver processes/utilization, and foreign ownership during each run. Only our jobs are timed out.

Never deploy to a lane during its frozen run. Changing local code does not change a submitted source snapshot. Reuse neither registration files nor evidence tags; restarts use fresh campaign prefixes and preserve failures. HOLD-B delays the next B submission while a completed checkpoint is patched; it is not a device lease.

## Current finite campaigns

A: run_track_a.py, registration lifetime-a3.json. Six small gates, per-parameter FP32 gradient comparisons, a main shape/memory gate, then18 screened processes.
B: run_track_b.py through remote_b.py, registration lifetime-b2.json. Anchor plus B0-B3 small gates/comparisons, a main shape gate, then15 qualification screening processes. On any gate failure the campaign stops; it never falls back.
Each screened process:25 updates,5 warmup+20 measured. Complete-update wall max across8ranks. Formal108 matrix is not started by these scripts.

## Validation

Local: python stack/tools/test_step_lifetime.py.
Torch CPU: python implementation/validate_cpu.py; logs diagnostics/delivery-cpu.log and JSON. Includes two ranks, four gradient boundaries, three Adam updates and an empty-owner update.
Worker build: implementation/build-lifetime.sh; build logs preserve every attempt. Final fingerprint: stack/configs/lifetime-runtime-freeze.json. Actual loaded SO and source identities are in each evidence run.

## Results

python analyze.py refreshes REPORT.md, STATUS.json, results.csv and figures/{A,B-qualification}.{png,svg,pdf}. Only completed25-update runs with all8ranks, unchanged guarded sources, no foreign overlap and <=90GiB are eligible. Diagnostic/qualification status is explicit. One point is one independent process, not20 repeats.
Large gradient tensors stay in the task-owned remote evidence path in each *-receipt.json; metadata is collected locally. No raw data goes to D:/WORK.

## Reproduction

Replay source entrypoints are tracked in stack/research/step_lifetime/. These expect the checkout at PROJECT/stack and place artifacts in PROJECT. Use fresh tags. Remote deployment currently needs the frozen predecessor runtime and CANN/SHMEM SDK paths recorded in build-lifetime.sh; it does not modify those inputs.

## Outstanding science gates

Independent output/dx/dp/owner-dW checks passed64records with forced legal replicas and positive remote/cache payload; see diagnostics/independent-lifetime-math.json. Matched-prefix Timer qualification, invalid-cache collective rejection, online device-resident control, stable multi-repeat gains, held-out/natural routes and a calibrated joint policy remain required. No top-conference or innovation claim follows from the integration alone.

2026-11-16：follow_up.py已登记并等待A/B两矩阵完成。候选与基线按三类路由的进程中位步时几何均值选取；任一点单轮改善达到3%才补R2/R3，三轮同向且中位改善>=3%再补R4/R5。区间含筛选轮，仅为机制筛查，独立银行/自然路由和正式新planner验证仍必需。发生门禁失败停止，不覆盖旧回执。

字节统计是从有效缓存来源/参数平面及发送缓冲长度推导的逻辑payload，包含预热，不是链路带宽计数器。B3是更新末尾优先；银行满会提前flush，当前实现会flush全局pending集合，需与更细粒度服务策略区分。

2026-11-16：audit_completed.py读取已完成点，核对八rank的初始参数、输入、逻辑路由、重计算次数及B实际任务顺序；输出diagnostics/completed-audit.json和lifetime-payloads.csv。没有逐调用copies/quota的计时日志不能作为该字段运行期完全相等的直接证明。

原Timer诊断使用第三独立lane/container step-lifetime-prefix-timer-20261116，以及implementation/prefix-timer-stack的research/prefix-timer-check分支。仅允许<=3更新且导出step0梯度；不进入性能曲线。该诊断源码与A/B正式筛查快照分开。

主形状copies/quota摘要：run_main_plan_gates.py。先在共同锁内发布诊断代码，然后五方式各三次更新，验证非空副本与逐调用计划一致。--plan-hash-qualification同步读回只在<=3更新诊断允许，主形状不导出巨大梯度PT；相应数值正确性由独立门禁提供。原run_diagnostic_gates.py保留小形状资格尝试及共享发布/对照工具，不能把其零副本记录冒称主形状覆盖。
