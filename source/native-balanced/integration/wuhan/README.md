> 当前计划校验默认融合（保留兼容参数），统一实现及旧框架迁移见 [FUSED-PLAN-DEFAULT.md](FUSED-PLAN-DEFAULT.md)。

> train BF16 优化与默认融合校验的配套构建、使用和组合验收见 [TRAIN-BF16-INTEGRATION.md](TRAIN-BF16-INTEGRATION.md)。

> 共享专家仍以 `framework` 为默认；算子bridge保留 `saved`/ABI2，并提供显式 `redispatch`/ABI3。当前17项独立算子验收通过，8卡两个EP4组的完整专家F/B耗时降低keep 6.27%、recompute 4.54%；真实FSDP整网训练尚未验收。见[使用与复现](../../docs/内核X重派发实现与验证-20261101.md)、[最终结果与范围](../../docs/内核X重派发八卡验证结果-20261101.md)及[接口契约](SHARED-EXPERT-FORWARD-ABI.md)。框架的 `shared_expert_mode=fused` 当前接V2共享计算，不会自动选择ABI3。

> 2026-10-31：已验收的DX panel acquire cache（9d036d2，新开关默认0）及原rho6/rho3
> 在线MoonEP四配对结果，见[实验报告与复现](reproductions/rho63-panel-cache-20261031/README.md)。

> 2026-10-30：兼容的S9 panel构建默认开启复制专家DW1前移；新建构建时设
> `MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=0` 可回退。须重建配套扩展/SO，
> 默认值变更不代表最新组合已通过整网验收。见[构建与验证边界](../../docs/backward-replicated-dw1-early.md)。

# Wuhan eight-card integration

## Routed saved state and selective replay

[三种保存格式、框架选择性重放与 EP 同步点视图](SAVED-GATE-UP.md)：
BF16/none/MXFP8 是可选的路由专家保存策略；UAM 存储策略独立配置。
包含真实 expert-FSDP2 的单层验证入口与支持边界。

## Fixed Python high-skew case

共享专家分支合入双方代码后的冲突处理、诊断兼容性和验证边界见
[shared分支合入说明](../../docs/experiments/20261102-shared-python-integration.md)。

[Python高偏斜固定场景：构造、复现与历史性能](scenarios/python-olmoe-high-skew/README.md)：
官方预训练OLMoE/Python文本的新采样直方图，源κ3.0071、八卡映射后κ2.4817；
轻量配置与CPU生成器，无需模型或约4GB输入capture入库。支持原TimerV2入口的
`--scenario-inputs`和在线MoonEP完整autograd B/FB对照。历史P1-cache与owner2:1
结果单独注明版本；两项优化现已合入、默认保持关闭/1:1，本次未新增设备验收。
合入范围、历史构建指纹与显式复现/回退见
[Python优化合入说明](../../docs/experiments/20261102-balanced-python-integration.md)。

## Kernel Timer trace default

[既有routed五输出采集入口](TIMER-CAPTURE.md)：配对构建TimerOFF/ON，支持默认八卡EP4×EFS2、ρ=6的合成负载和真实capture，五输出逐位校验后自动生成两张all-rank图。共享专家的Y＋七梯度及内核X重派发使用本文末尾的独立benchmark入口。

[TimerV2 export guide](TIMER-TRACE.md): kernel diagnostics now default to two
all-rank Chrome/Perfetto files, forward and backward, with exactly 32 AIC + 64
AIV tracks per rank. Planner, dispatch, weight movement and gradient reduction
share those tracks. Export runs offline after successful collection; ordinary
training and the frozen compute libraries are unchanged.

## Current user entry

[Planner selection and full-network usage (Chinese)](https://gitcode.com/zhangwanhao/wuhan/blob/codex/moonflux-8p-integration/docs/megamoe-balanced-usage.md)
documents the default `moonep` policy, the explicit `native` selection,
matching panel-ready builds, and 8P / EP16-DP1 full-model launch examples.
`megamoe_balanced` uses the same S9 executor for both policies; the author
`moonep_megamoe` dispatcher is a separate Hybrid control.
Large EP requires a rebuilt `_C`, GM planner library and worker together;
see the linked guide for explicit GM build commands and measured limits.

This branch starts from Wuhan `c3d38b9`, with framework bases MindSpeed-LLM
`9dfc5d8d`, FSDPTurbo `f2e56d8`, MoonEP `5f53a44`, and SHMEM `f281664`.
The S9/planner donor is `39349f4805a6904ff58781c0073d2466fc7a19b6`.

The implementation plan and raw evidence live in the local project
`D:/megamoe_train/moonflux接整网/balanced-8p/`.
First freeze an unmodified A/D control build in its own container and source
snapshot. Candidate code and libraries use another snapshot. Do not replace
the original repositories, SHMEM install, framework branches, or 512p scripts.

`snapshot.py` exports committed sources plus mandatory dependencies and writes
a SHA256 manifest. `build_control.sh` builds the original kernel, binding and
MoonEP against one SHMEM install. No training is launched by either script.

The fast 8P delivery uses four complete backends, targeted FSDP/recompute
gates, a 40-step ABCD/DCBA matrix (first 10 warmup), and a UAM short gate.
The original 200-step/full matrix remains future validation; short runs
cannot establish convergence or 512P acceptance. Source scaffolding or a
successful build alone does not establish numerical/performance acceptance.

## Optional bridge publication synchronization

`WUHAN_PUBLISH_SYNC=device` is the conservative default. The opt-in `stream`
mode completes the calling stream after both live-home copies, keeps the EP
barrier and completes its stream before launching SHMEM consumers. FSDP must
already order the live Parameters onto the calling stream, as required by
the expert forward/backward hooks. This avoids waiting for unrelated device
streams without removing the peer publication boundary. Never enable it
solely from a CPU ordering test: compare same-plan five outputs and updated
parameters under real FSDP, recompute and UAM before making performance claims.

`run_fast_queue.py` accepts explicit per-case `publish_sync` and restoration
mode; its host gate refuses occupied devices and existing evidence paths.
`fast_memory_command.py` samples all-device HBM independently of Torch's
allocator. `summarize_fast.py` reports same-node speedups against A only when
initial shards and fetched data match, and combines results only after both
process orders exist. For B/C versus Hybrid, report step-time percent change
separately from the A-based speedup.

The paired training runner optionally applies WUHAN_STEP_RECORD_DIR; its MindSpeed patch is patches/MindSpeed-LLM-step-recorder.patch (tip in frameworks.json). It records complete-step device wall time only when explicitly enabled. Fetched-data hashes retain labels before PT replaces that entry; payload hashing, reduced-loss recording and JSON writes are outside the timing boundary. The analysis omits speedups if initial parameters or actual batches lack matching fingerprints.

`deploy_snapshot.py` checks every native input against the immutable build donor before reusing its fingerprinted artifacts. For native changes, `build_restore_candidate.sh` preserves the explicit core/planner donor lineage and rebuilds the worker SO with an independent build manifest.

Backward restoration provides isolated `expert` (`MEGAFLUX_BACKWARD_RESTORE=1`)
and `panel` (`=2`) candidates; neither is promoted by functional acceptance.
AIV1 starts after the all-AIV count-table barrier and uses its upper UB to
copy W2 then W1. In expert mode it publishes projection epochs per replica;
in panel mode W1 `[2F,H]` and W2 `[H,F]` are split into 256-row panels. Every
worker publishes only after its assigned panel writes retire from MTE3,
including zero-task workers. Panel flags are indexed by slot, projection,
panel and worker, with the current backward epoch. AIC checks the panel at
the actual GEMM operand access, including recomputation and dH/dX K blocks;
dW does not consume weights. AIV1 retires both copy buffers before its
original SwiGLU duties. BF16 partials, fixed FP32 owner SUM, and final
synchronization retain their original order.

Choose `s9-ported`, `restore-expert`, or `restore-panel` with
`WUHAN_WORKER_VARIANT`, and pass the matching `--restore` for bridge/FSDP
gates or `BALANCED_RESTORE_MODE` for training. A selection/SO mismatch fails
before execution. All worker/owner candidate switches stay off. The serial
training queue accepts an explicit `restore` per case and records the SO hash.
Run restoration modes in separate processes, then use
`compare_bridge_restoration.py` to check actual plans and five-output hashes.
Explicit input and post-SGD parameter hashes were added after `e8095f6`:
old receipts can prove five-output equality but cannot prove bitwise SGD
updates. The report marks that evidence absent instead of inferring it.

On pod7-5, `e8095f6` panel passed the real eight-rank EP4 x expert-FSDP2
checkpoint gates under both planners, with hot/balanced/rotating routes and
three SGD updates. Each run has exit 0 and immediate all-device pre/post
gates. Same-source staged controls have identical plans and five outputs.
Evidence directories are `panel-fsdp2-{moonep,native}-ep8-a1` and
`panel-compare-{moonep,native}-staged-e809-a1` in the original project.
Panel SO SHA256:
`f5c58906766f793dc01db5ce28c3db276e68310133381da0bb5f02dedd29fe39`.
These are functional gates, not end-to-end performance acceptance.

## Delivery, provenance and training reports

- [Composite integration guide](https://gitcode.com/zhangwanhao/wuhan/blob/codex/moonflux-8p-integration/docs/megamoe-balanced-integration-guide.md)
- [Eight-card training report](https://gitcode.com/zhangwanhao/wuhan/blob/codex/moonflux-8p-integration/docs/experiments/megamoe-balanced-8p-report.md)
- Upstream draft dependencies: [native !1](https://gitcode.com/bingobb/ascend_deepep_hkq_wuhan/merge_requests/1),
  [FSDPTurbo !8](https://gitcode.com/bingobb/FSDPTurbo_wuhan/merge_requests/8),
  [MindSpeed !8](https://gitcode.com/bingobb/MindSpeed-LLM_wuhan/merge_requests/8),
  [composite !4](https://gitcode.com/bingobb/wuhan/merge_requests/4).

Before formal aggregation, run `seal_evidence_origin.py --runs <completed run directories>`
on the known execution host. It records collection-time host/boot identity and hashes
the original gates, configuration, parameter/batch/step records. It does not rewrite
or backdate the original gates. Archive `origin.json` together with the evidence.
`summarize_training.py` refuses a speedup without matching host and training settings;
`summarize_fast.py` additionally requires the same implementation/SO/configuration
for each method's two process orders. Its combined ratio is the ratio of aggregated
process times; per-order ratios remain available. Capture runs cannot supply a
performance denominator. `summarize_optimization.py` separately compares device/stream
publication against frozen A/D on that same node; reused controls are labeled.

## Real-input capture and fixed-plan restoration replay

`WUHAN_CAPTURE_REPLAY=1` enables a diagnostic training run. Each process saves
exactly the first expert backward it reaches (before expert-FSDP reduction),
including local X, logical routes/probabilities, live home W1/W2, dY, placement,
and reference Y/dX/dp/owner-dW hashes. Files are `layer-capture/rank*.pt` and
SHA256 manifests under the run evidence directory. This identifies a layer
by the actual reverse execution order, not by an assumed layer number.
The saved-tensor schema stays constant during checkpoint recomputation.
The default leaves capture off and saves no extra X/Y on the tape. Captured
training times are diagnostic and excluded from reported step speedups.

Run `replay_captured_layer.py` using `run_bridge_gate.sh` with
`WUHAN_GATE_ENTRY=replay_captured_layer.py`, the matching isolated worker SO,
`--capture <directory> --restore staged|expert|panel --warmup 10 --repeat 30`.
Use separate processes in both restoration orders. The replay first checks
all five output storage hashes against the capture, then measures F, B, and
direct F+B. It includes both live-home refreshes, placement replay, weight
restoration, native compute/owner reduction, clones and finish. It excludes
online policy selection, CPU capture loading and FSDP communication. The
original captured plan is fixed outside timing; rebuilding its epoch/header
and checked executor state remains in the bridge. Report this as a single
captured-layer experiment, never as the complete training-step speedup.
Timer builds are rejected for this performance entry.

## Multi-node lifecycle and training checks

`run_training_case.sh` keeps its single-node defaults. Explicit multi-node
launches use `WUHAN_NNODES`, `WUHAN_NODE_RANK`, `WUHAN_MASTER_ADDR`,
`WUHAN_MASTER_PORT`, `WUHAN_EP_SIZE=4`, and
`WUHAN_EXPERT_FSDP=WORLD_SIZE/4`; each node still runs eight local processes.
`validate_launch.py` rejects unsupported mesh sizes and a localhost multi-node
master. These tests keep EP4 within a node: cross-node FSDP acceptance is not
cross-node EP transport acceptance.

Use `run_multinode.py --manifest <JSON> --out <new directory>` with explicit
`name`, `timeout`, and `nodes` entries containing `ssh`, `project`, and a
Docker-exec `command` array. Deploy `multinode_node.py`, `subprocess_bounds.py`,
`gate_and_run.py`, and `seal_evidence_origin.py` to each project's `snapshots/`.
The coordinator releases the job only when every node's resource pre-gate
passes. Every node owns its lock, actual exit, and immediate post-gate. For
Docker, GNU `timeout` runs inside the container and owns the child process
group; terminating only the host Docker client cannot prove cleanup. Missing
post-gates or a failed peer invalidate the distributed result. Failures and
timeouts require checking the actual container processes before another run.

Before training, run `test_expert_fsdp2.py` through `run_bridge_gate.sh` with
`WUHAN_GATE_ENTRY=test_expert_fsdp2.py`, positional local process count `8`,
matching operand-ready=1, delay=0 SO selected with
`WUHAN_WORKER_VARIANT=restore-panel-ready` (or its explicit isolated build name)
and `--restore panel --checkpoint --slots 2 --require-replicas
--expect-operand-ready 1 --expect-delay-cycles 0`. Native needs enough tokens to actually make copies:
the validated world16 case uses `--tokens 2049 --ffn 512`; M257 fails the
replica assertion and must not be counted as a restoration test. For MoonEP,
use an independently validated shape, not a claim derived from Native.

For training evidence, collect all nodes and use `summarize_multinode.py
--case <all-node-A-directories> --case <all-node-B-directories> ... --steps N
--warmup W --out report.json`. It verifies all global ranks, distinct physical
nodes, matched configuration/SO and sealed evidence. It computes the world
maximum for each step before calculating the median; it does not combine
node medians. Capture runs are rejected. Follow with the opposite method
order on the same topology, keeping all raw samples. Run CPU checks with
`test_multinode_cpu.py` and `test_multinode_summary_cpu.py` before deployment.

For isolated local data mounts, use `WUHAN_DATA_NO_SHARED_STORAGE=1` so that each
node builds its own GPT sample indices. This flag is recorded in `launch.json`;
the author's shared-filesystem default is preserved. Coordinator SSH processes
have closed stdin, avoiding competing Windows console readers. A timed-out
readiness query is unknown and is retried within the readiness deadline;
abort requests are best effort and never replace the original error. Always
inspect remote workers after a coordinator failure before using a new run ID.

## 200-step single-node stability campaign

`run_long_queue.py` runs explicit single-node EP4 x Expert-FSDP2 cases. It fixes
200 steps, eight layers, recompute, S9/panel, device publication, NPU control
tape, two replica slots and 20480 receive rows. This bounded capacity is for
the validated E8/Top2 rank-skew quotas only. It does not support arbitrary
natural routing without a separate capacity audit. All methods use the same
200-step LR schedule; do not reuse a 10/40-step denominator.

Example manifest: `[{"name":"long8-rho6-A-a1","method":"A","rho":6},
{"name":"long8-rho6-B-a1","method":"B","rho":6},
{"name":"long8-rho6-C-a1","method":"C","rho":6},
{"name":"long8-rho6-D-a1","method":"D","rho":6}]`.

```bash
python3 /project/long-tools/run_long_queue.py \
  --project /project --source /project/frozen-source \
  --container isolated-container --worker-variant release-panel-ready \
  --manifest /project/long-cases.json \
  --record /project/long-queue.jsonl
```

Run the host queue as root with the host project path, not literally inside
the container in this example. The tools directory must contain the queue,
gate, HBM sampler, subprocess bounds, evidence sealer and summary modules.
Each case acquires the same project device lock, checks all eight devices,
runs a 2400-second timeout inside Docker, saves actual exit and immediate
post-gate, seals evidence and checks all 200 x 8 step records. Any failure
stops the queue; no automatic restart or overwrite is allowed.

The queue defaults to `restore-panel-ready`; `--worker-variant` selects the
independent build name created for deployment. Rebuild or deploy that SO,
verify operand-ready=1, delay=0 and the actual hash before the campaign.
Updating source alone does not update an old SO. The bridge rejects unguarded
panel libraries. A variant name selects a path; it does not prove its binary
capabilities or correctness. Preserve old run snapshots and evidence.

After collecting a matched A/B/C/D group:

```bash
python3 summarize_long.py --runs <A-dir> <B-dir> <C-dir> <D-dir> --out long.json
```

This produces a JSON summary plus per-step CSV, verifies initial and actual
batch identity, computes rank-max wall before aggregation and reports segments
11-50/51-100/101-150/151-200, LM and grad differences and allocator/HBM memory.
It reports evidence rather than declaring convergence or numerics acceptance.
CPU checks: `test_long_cpu.py`, `test_training_summary.py`.

## Candidate: panel readiness at actual operand loads

Static inspection found that an outer output-tile wait covers only one panel,
while CATLASS's MMAD traverses complete K and prefetches later B tiles. The
candidate `MEGAFLUX_RESTORE_OPERAND_READY=1` wraps B operands through the internal
ADL `GetTile`, waiting the actual physical row panels immediately before each
GM-to-L1 copy. It preserves CATLASS's K accumulation and events; canonical W1
recompute uses the column coordinate, dH/dX use the row coordinate. Plain dW
operands and homes carry no ready wait. Existing running binaries are unchanged.

`MEGAFLUX_RESTORE_DELAY_CYCLES>0` is a diagnostic producer delay before panel1,
to expose a first-panel-only consumer. It and the readiness capability are
exported by the SO and recorded in build.json. Never include delayed builds in
performance comparisons. Keep an isolated operand-ready=0 control; passing
normal timing does not prove the old protocol safe under delayed production.

Device bridge and expert-FSDP gates accept `--expect-operand-ready 0|1` and
`--expect-delay-cycles N`. They check the loaded SO's exported capabilities,
save its exact path/hash in every successful rank receipt, and can require
actual replicas with `--require-replicas`. The two-layer bridge poisons pooled
home/replica weights after forward; backward must restore the current layer.
Inputs, plans, five outputs and SGD hashes allow same-plan comparisons between
no-delay, delayed, and staged controls. A deliberate old-protocol failure is
diagnostic evidence only, never eligible performance or a successful gate.

`test_restore_operand_cpu.py` compiles the real wrapper with small host tensor
stubs and checks internal-K ADL, next-panel waits, column-major W1, multi-panel
tails and homes. This proves CPU template/read-set behavior only. Ascend build,
delayed-producer five-output and real-FSDP/checkpoint/SGD gates remain necessary
before any candidate promotion or performance claim.

## 共享专家与 X redispatch 的独立验收

`benchmark_shared_expert_fb.py` 从总Y触发真实autograd，按AB/BA交替测量完整专家前反向及最终同步。输入模式和共享激活策略是两个独立选择：

| benchmark `--routed-input-mode` | framework对照 | fused候选 |
|---|---|---|
| `saved`（默认） | 外置共享，保留packed X | 融合共享，保留packed X（ABI2） |
| `redispatch` | 外置共享＋外置X/prob重建 | 融合共享＋同样外置X/prob重建 |
| `native` | 外置共享＋外置X/prob重建 | 融合共享＋kernel内X/prob重建（ABI3） |

`--shared-save-mode keep|recompute` 只决定共享G/U/A是否跨前反向保存；`--saved-dispatch-reference` 对同输入saved结果的Y、七梯度和冻结计划做逐位回归。TimerON必须与OFF配对，采集和离线导图见[校时说明](CLOCK-CALIBRATION.md)。Timer包络包含等待和诊断开销，性能采用NoTimer完整F/B。

算子bridge的ABI3入口是 `forward_shared(..., routed_input_mode='redispatch')`，对应benchmark的 `native`；bridge默认仍为 `saved`。当前框架配置层尚未转发这个新参数，`shared_expert_mode=fused` 只选择共享融合/V2，后续需单独接入并验证真实FSDP、checkpoint/UAM生命周期。

已验证组合显式开启shared forward/backward、`MEGAFLUX_SWIGLU_DECOUPLE=on`、构建时 `MEGAFLUX_GM_SWIGLU_TILE_READY=1` 和运行时 `MEGAFLUX_SHARED_EXPERT_TILE_PIPELINE=1`，并使用S9 panel restore。默认auto/关闭组合不能直接视为这份性能结果。完整构建、Triton依赖检查、NoTimer/Timer命令统一见[使用与复现](../../docs/内核X重派发实现与验证-20261101.md)；精度、内存及EP4×2边界见[最终验证结果](../../docs/内核X重派发八卡验证结果-20261101.md)。
