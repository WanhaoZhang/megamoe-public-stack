# AIV1 全包归主候选

基线是 11 月 6 日普通版实际采集的 Native `7582a9c3b235f1b729fa58bda11872c72d46ca07`。候选只在 S9 panel、flags0、BF16 saved 且存在梯度归主计划时生效。`MEGAFLUX_OWNER_AIV1_TAIL` 默认 `0`。打开后，每组 AIV1 按 tile 编号处理原 AIV1 和配对 AIV0 的任务并集，先完成该组全部 dW2，再完成该组全部 dW1。各组独立推进。AIV0 做完 `combine dX` 后参与原完成同步。每块仍有唯一写回者，等待同 epoch 的本地和副本贡献，按原顺序 FP32 累加并写回 BF16。

直接对照是四轮版本 `8b617b5`，其顺序为原 AIV1 的 `dW2 → dW1`，再处理原 AIV0 的 `dW2 → dW1`。新版逐组验证 tile 并集、单调顺序及唯一覆盖。设备端在相同冻结输入下核对五个输出。

在有匹配 CANN、SHMEM、CATLASS 和普通版绑定的 Linux 构建环境，分别构建两套对照，每套含独立的 TimerOFF/ON worker 和配套库。

```bash
export MEGAFLUX_OWNER_EARLY_RATIO=2
export MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1
export MEGAFLUX_RESTORE_PANEL_ACQUIRE_CACHE=1
export MEGAFLUX_RESTORE_P1_PANEL_ACQUIRE_CACHE=1
MEGAFLUX_OWNER_AIV1_TAIL=0 bash scripts/build_megaflux_timer_capture.sh owner2-control-a1
MEGAFLUX_OWNER_AIV1_TAIL=1 bash scripts/build_megaflux_timer_capture.sh aiv1-tail-candidate-a1
```

用 `build.json`、`timer-capture-build.json` 和 worker 的只读 `megaflux_worker_owner_aiv1_tail()` 区分实际运行库。新版诊断使用 TimerV2 schema 8，在 schema 7 的槽位后追加 `OWNER_EXPERT_PHASE`。每个有实际归主的专家按投影在原 AIV1 轨道记录一次起止时间，包含等待和归并。归主父窗口用于详情。单 tile 循环沿用 summary 模式，逐 tile 打点不会写入。

设备验证先用 [原捕获复放入口](../../integration/wuhan/TIMER-CAPTURE.md) 对同一 BF16 输入分别核验基线/候选的 Y、dX、dp、dW1、dW2，以及各自 OFF/ON 的一致性。通过后，在同一节点用同一固定高偏斜输入做 TimerOFF 正反序配对，比较逐样本八 rank 最大完整 B 与直接 F+B 的 P50/P90。TimerON 单独查看后接归约的 ready 等待、AIV0 combine 完成和最终同步位置。设备结果记录在实际运行目录。
