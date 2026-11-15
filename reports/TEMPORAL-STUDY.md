# Temporal training study

This branch implements a training adaptation of ready-task scheduling on the ordinary MegaMoE executor. It does not claim a new planner contribution or reproduce the original inference system's performance. Original GA1 data remains a separate experiment family.

The contract is `configs/temporal-study.json`. Evidence and failure history live in `D:/megamoe_train/planner-executor-ep8-lab/temporal-replica-study/`: `REPORT.md`, `STATUS.json`, `DECISIONS.md`, `evidence/`. Experiment tensors are not stored in Obsidian.

## Common experiment

- V4 attention/mHC, framework shared FFN, L4/H4096, routed/shared F1024, EP8/E256/Top8.
- Four complete16K sequences per rank/update: microbatch1, GA4, global32 sequences/524288 sequence tokens.
- FP32 parameters/gradients/Adam, BF16 expert computation, fixed optimizer owners and existing AdamW recipe.
- Non-reentrant logical-layer group checkpointing. Easy arms observe actual routing with an extra prefix probe; fixed grouping/FIFO use already available layer inputs without a routing probe. All required work is timed.
- Common receive capacity262144 (2×S×K), maximum admitted W2,90GiB whole-card limit,89.5GiB allocator target. W4 failed this memory gate. Model and global batch were not shrunk.
- Native32-home/32-replica and64-home/0-replica pools are preallocated for every method. Fewer active replicas does not imply lower HBM.

## Implemented entrypoints

`tools/train.py` adds `--microbatches`, `--forward-window`, `--temporal-pattern`, `--hidden`, `--ffn`, `--temporal-policy` and `--temporal-scheduler`.
For GA4, every original forward and replay uses `(step,microbatch,layer,invocation)`.
Microbatch losses are averaged once, while the existing EP divide factor8 remains unchanged.
Clip and Adam run once after all four microbatches' gradients. Tickets retire only after exactly one backward each.

Legacy retains W complete model graphs, then performs one backward on the summed window losses. This is the ordinary Native reference.
Fixed grouping/FIFO consume the layer-input frontier; Easy arms expose observed routing with Attention/router probes. The grouped path checkpoints complete logical layers in frozen task groups, then performs one backward per window. All GA4 windows finish before clipping and Adam.

Policies are `legacy`, `fixed_native`, `fifo_native`, `easy_home`, `easy_independent`; `fixed_home` is a numerical diagnostic arm. Legacy/fixed/FIFO have no routing probe; their unknown loads are null, not fabricated measured zeros. Easy-independent selects tasks then uses separate full16K Native calls and independent replication; it does not assume two Native calls can execute freely in parallel. Afore persistent replica reuse and a new joint policy are not implemented.

`tools/temporal_scheduler.py` contains deterministic FIFO, same-layer grouping and EasyBalance-style MaxUtil ready-set selection.
The selection concept follows [EasyBalance](https://arxiv.org/html/2608.07964v1); the training adaptation, fairness rule and executor constraints are this study's implementation, not a claim to reproduce the original inference system.
It consumes only observed tasks; minimum group ceil(W/2), forced inclusion after two deferrals.
FIFO selects oldest arrivals at a common minimum batch size. MaxUtil admits at most two layer domains, matching64 home slots. `device_temporal_scheduler.py` dispatches the separate AscendC ABI3 plugin. It evaluates subsets with integer ratios and deterministic ties; Python reads the small chosen mask to drive model control flow. That fence is timed.

`grouped_experts.py` reversibly maps `(layer,expert)` into one virtual domain without changing owner rank, reads live owner weights and invokes Native. `home_group_planner.py` compiles fixed-home quotas through the existing device validator; it does not expand the E256 search arrays.

`temporal_model.py` compiles each selected set into waves containing chunks from every selected task. Cyclic row partitions support three-task sets without padding, duplication or lost rows. Attention remains complete16K. Pool serialization and kernel barriers remain in force.

FSDP replay acquires public `unshard()` leases for layer/HC ownership groups. Original Parameters remain intact. Valid zero-probability initialization installs the existing divide8 gradient hook once. Zero-valued dependencies enforce reverse Native wave order across ranks.
Framework shared FFN runs after routed FFN, matching the original model. A second public layer forward obtains its FSDP-owned parameters. Its extra communication is charged. JSON ready-set persistence is outside the measured wall-time boundary; plan checks, ticket retirement and device completion remain inside.

`tools/test_temporal_contract.py` checks eager versus checkpointed gradients, reversed backwards, route identity and selection constraints.
`tools/gate_pool_replay.py --reference torch --microbatches 4` uses differentiable EP collectives and explicit PyTorch expert math as a separate reference.
The reference is eager; the native path retains whole-block checkpoint replay and poisons the reusable weight pool before backward.

## Qualification boundaries

- A per-step diagnostic finite-gradient scan handles empty FSDP shards.
- Nonfinite loss/norm rejects the update before Adam.
- Runtime SOs must match `configs/runtime-freeze.json`; no directory-name-only admission.
- TimerV2 metadata now includes microbatch/invocation; rank clocks remain separate.
- Three synthetic temporal patterns preserve logical expert loads under consistent global ID permutations. Small sliced fixtures are qualification-only and do not retain calibrated MaxVio10.
- CPU9 tests pass, including FIFO fairness and full-runset row conservation. Eight ranks each passed506 AscendC/reference cases for ABI3.
- The original CPU-frontier small model passed three Adam updates and full same-weight owner-gradient comparison. Category relative L2 was0.00564–0.00797; maximum individual parameter error0.01962≤0.02. New wave compilation is requalified separately.
- Main-shape W2 passed8×3 Adam updates, source guard unchanged, sampled driver peak82720MiB/80.781GiB. Smoke timings are not performance scores.
- Original graph-order W2 fixed-native, Easy-home, FIFO and Easy-independent each pass8×3 Adam plus same-weight full step0 owner gradients, including every parameter≤0.02. Earlier early-shared scalar-gradient failures remain archived. W4 fixed-home still fails full-model gradients and is not admitted.
- Fixed/FIFO were requalified after removing their unneeded probes: three Adam updates and every owner gradient≤0.02 again passed. Both now execute router twice/task (original+replay); online Easy arms execute it three times/task.
- Initial capacity163840 overflowed (required up to179072) and Adam was refused. Capacity262144 is the newly common condition. W4 then hit its allocator limit; the failure is preserved.
- Dropout must be0 and router auxiliary-loss collection disabled. Task-specific stochastic RNG and regrouped auxiliary statistics are not implicitly supported.

The formal matrix remains disabled: the complete strong-baseline screening, Afore cache, conditional joint strategy and independent validation are unfinished. A single MaxVio10/random 25-update pair is a separately labelled feasibility pilot, not the formal matrix.
No reference/simulated schedule cost is a measured training time.

Fresh MaxVio10/random observation controls gave Native W2 mean4.34644s and online Easy-home4.68762s. Replaying the same observed263-group task order without probing/selection gave4.14667s, but uses a future-derived schedule and is explicitly offline-only, never an online speedup. Each point has one independent process. Source/initial/input/route/loaded-SO checks passed; no stable gain is claimed.

`--temporal-schedule-replay PATH --diagnostic-only` exposes that isolated ablation; both flags are required. It retains online Native quota compilation from actual routing. `tools/task_schedule_replay.py` rejects incompatible shapes, ordering, duplicate or unready tasks instead of falling back.

The normal worker is Timer schema0 (SHA46772ebc...). A separately frozen schema9 diagnostic worker (SHA6933b647...) passed a matching numerical gate. Diagnostic traces are excluded from normal wall/HBM scores. Framework Events and hardware ticks retain independent rank-local clocks. The registered18-run online screen excludes all offline task-order data.

`tools/campaign_temporal.py --out <manifest.json>` only preregisters screening. `--execute` checks recorded gates; it does not enable the108-run joint-policy matrix. No future route bank entries may influence online scheduling. Natural-router evidence, artificial banks and old GA1 results remain separate.
