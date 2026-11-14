# Decisions and validation log

## 2026-11-14 — isolation and interface

- Isolated the accepted ordinary Native/framework source and route bank in this
  lab. New private Native branch codex/planner-executor-ep8 at48ef24a6; accepted
  64-slot and public-framework patches retained. Previous results untouched.
- Committed the stopped prior runtime container to a private image and created
  a dedicated labelled container. No other workload was stopped or removed.
- Reuse histogram/prefix exchange, not a full MoonEP planning pass, for every
  policy. The planner algorithm has no weight, gradient or optimizer pointer.
- All five policies now produce the same source-aware proposal; a common
  compiler and fused collective validator provide the executor lease.
- Immutable snapshots are separate from epoch/header/lease. Explicit framework
  step/layer/checkpoint identity is used; no tensor-pointer guessing or implicit
  sitecustomize. Originals submit once, recomputation rebinds, backward retires.
- Shared expert remains in the framework; only ordinary executor is present.
- Source port numerical/locality adaptations are listed in references/POLICY-PORTS.md.
- CPU oracle480 pass; unified64 AscendC/Host build pass. Device correctness and
  memory gate are in progress. No formal run or performance claim yet.

## Failure receipts retained

- Provision attempt0: Windows newline translation corrupted bash pipefail;
  fixed by sending bytes. No remote mutation had run in that failed attempt.
- First CPU oracle: EPLB float64 promotion/reference sum association mismatch;
  corrected independent oracle to the intended FP32 scalar semantics and reran
  the complete480 cases successfully.

## 2026-11-14 — final contract and baseline preservation

- Added a public device-proposal compiler. Its shared gate checks source
  marginals, destination/contributor counts, capacity and cross-rank proposal
  agreement for built-in policies and external providers.
- Frozen EPLB reroute uses cyclic global ordinals in physical-instance order.
  Corrected NumPy/Ascend source quotas accordingly before common materialization.
- Added retired training tickets separately from immutable bodies and epochs.
  CPU test rejects duplicate replay, retired bind and body mutation; snapshots
  survive original scratch poisoning and keep independent runtime headers.
- Added an optional policy-module hook before baseline timing. Later step_time
  can be added without replacing the baseline SO, Host extension or worker.
- Both32/64 Host/planner builds pass. Calibration Timer9 and SHMEM/MTE link
  diagnostics build; no calibration data has been collected yet.
- Independent calibration/heldout banks use seeds20261114,31415,27182. The
  primary six-point bank remains unchanged.

Additional failure receipts: a final-interface gate caught a Python method
binding mistake during compiler refactoring; corrected before formal timing,
then CPU lifetime checks passed. The link diagnostic needed AscendC namespace
visible before the original Timer header; fixed and build passed. A Windows
GBK log-print failure happened after successful remote builds; exit0 build
receipts were retained and the runner now prints UTF8.

## 2026-11-14 02:24 CST — full-model qualification and baseline freeze

- Final v4 seven-device gates passed, including32-slot actual binary,
  external-proposal collective refusal/recovery, ticket replay, singleton
  gradient scaling and global clipping. All five full-model3-Adam-step gates
  passed at163840; largest sampled HBM90350MiB (88.2324GiB). No capacity fallback.
- Froze baseline sources/actual SO identities in BASELINE-FREEZE.json and
  started the90 planned existing-policy runs. No Native/framework or measured
  harness source is edited while this matrix runs. New postprocessing and
  diagnostic tools are inactive in formal training.
- Exported Native/accepted EP8/framework patches and checked their application
  in temporary Git indexes. Working sources/HEAD and old projects were not
  changed. Preserved upstream notices/licenses with the source. Archived the
  binary/SDK/libstdc++ closure and158 installed package/compiler identities.

## 2026-11-14 03:20–03:50 CST — independent calibration preparation

- Preregistered initial-policy choice from the independent seed20261114 bank
  at0.3/7/18, with10 steps and5 warmup, geometric-mean complete-step score.
  Primary/heldout routes are not fitting inputs. Fourth-round pairs alternate
  order across MaxVio. One old-capacity interface pair at each endpoint is
  diagnostic only, not an extra main-matrix repeat or stable-gain claim.
- Source audit corrected the dependency record: BF16 saved gate/up skips
  internal backward GMM1; P2 does each expert's dW2 before dX, followed by
  tail dW1, with odd/even owner service overlapping. Earlier broad wording
  associating dW2 with P1 was inaccurate; code/SOs did not change.
- Before any cost data was observed, added one uniform calibration-only bank
  to cover the source-confirmed no-replica home32/gradient.ready branch. The
  earlier protocol is retained as calibration-protocol-before-uniform-20261114.json.
  This bank does not participate in initial-policy selection or formal results.
- Timer9 samples column0; those owner tasks use even static residues at this
  shape. The model must use full per-lane phase envelopes for odd/even overlap
  and disclose the sparse-release limitation. Rank-local clocks are not merged.

## 2026-11-14 03:46 CST — pre-training HCCL initialization failure

- primary-existing-vio20-moonep-r1 exited1 before any training-step record.
  Rank3 reported HcclBroadcast/RunAicpuIndOpThreadInit AICPU exception in
  WeightLoader.load. Peak sampled HBM16947MiB; no Planner execution or timed
  step exists. The next Native process completed normally; no card/container
  reset, other-workload cleanup or model/policy change was performed.
- Original failed row/stdout/plog remain. A separately linked supplemental
  fresh-process run is allowed only after the scheduled90 attempts, and only
  for this zero-step initialization signature with identical source identity.
  No successful/slow timing can be replaced; policy/training failures, OOMs
  and invalid plans cannot use this recovery path. Up to3 explicitly logged
  initialization retries, then stop if still failing. Charts and CSV retain
  the failed attempt and the retry link rather than hiding the failure.
- This failure paused the calibration waiter as intended. The waiter now
  requires all90 qualified planned cells, including explicit infrastructure
  recovery, before interface diagnostics or cost collection. The root cause
  inside HCCL/AICPU is not established beyond its initialization location.

## 2026-11-14 04:23–05:20 CST — external device contention and queue hold

- The driver process monitor identified GPU PIDs in another container,
  `agy-live-a5` (cgroup container e4ec55cc554d...), running fabric_bench.py and
  later mc2_bench.py. This is independently recorded driver/PID/cgroup evidence,
  not an inference from slow results. Other workloads were not signalled,
  stopped or reset. No migration to a differently occupied node was made.
- Held only this task's exact primary-existing scheduler PID1353855 between
  jobs, after obtaining its own device-jobs lock. The experiment remains active;
  CPU documentation, evidence and diagnostic preparation continue. A read-only
  waiter resumes only this scheduler after120s of eight idle, process-free cards
  and an unchanged baseline source fingerprint.
- Conservatively excluded every original process window overlapping the other
  container's start1794598975.742943 through our queue hold1794601391.3439374.
  This affects11 of35 original attempts:24 remain valid. The exclusion rule is
  uniform across policies, success/failure and measured speed. All raw steps and
  original statuses remain unchanged. Explicitly linked fresh-process reruns
  must use identical arguments/source identities after exclusivity returns.
- Four affected attempts failed in WeightLoader.load/HcclBroadcast before any
  training step. Their location is established; the foreign workload is not a
  proven root cause of those HCCL/AICPU initialization faults. Excluded timing
  and HBM values must not become claims about intrinsic Planner performance.
- Evidence: EXTERNAL-CONTENTION.json, OWN-QUEUE-INFRASTRUCTURE-HOLD.json,
  driver-process-observations.jsonl and EXCLUSIVITY-WAIT.json. The original
  campaign and supplemental recovery ledger are separate. The calibration
  waiter requires90 resolved, qualified, uncontaminated planned cells.

## 2026-11-14 05:20 CST — dependency audit and inactive diagnostic checks

- Confirmed that backward ExpertAt uses the existing GradientKey order:
  replicated logical experts first by global ID, then local unreplicated
  experts, empty slots last. Copy insertion/removal can therefore change the
  release order of other experts. A physical-slot-order model is insufficient.
- With this shape each nonzero replica owes full BF16 dW1/dW2 delivery of
  56MiB+28MiB, independent of its token fraction. Owner service keeps the actual
  static64-worker dW2-then-dW1 order and odd/even entry boundaries.
- Added the existing backward X/dY/probability redispatch Timer to diagnostic
  exports. It is nested in the original backward kernel span, not an additive
  extra kernel cost. Added explicit diagnostic-only checks that persistent
  parameters/gradients/Adam moments are FP32 on NPU. These tools are not loaded
  by formal training; no frozen source or measured binary changed.

## 2026-11-14 05:22–05:28 CST — repeated foreign activity after first idle gap

- The first resume followed125s of eight process-free, zero-utilization driver
  samples and an identical source fingerprint. Foreign mc2_bench.py workers
  then reappeared. The next Native MaxVio1/r2 process failed before training in
  WeightLoader.load/HcclBroadcast. Held the same own scheduler between jobs
  again; no live training or foreign process was interrupted.
- Appended a second conservative exclusion interval, from the own resume
  timestamp1794604965.8094013 through hold1794605224.037672. Both windows retain
  independent driver/PID/cgroup proof. There are now36 original attempts,
  24 valid and12 excluded, including5 zero-step initializer failures. No valid
  slow process has been discarded or retried.
- Corrected the resource waiter's Docker query: `docker top ... -eo comm`
  fails because Docker requires a PID column. The earlier driver-idle evidence
  remains intact, but its receipt's foreign-container-absence assertion was
  not independently established by that failed query. New code uses
  `-eo pid,comm`, treats inventory errors as nonexclusive, checks active foreign
  commands as well as driver state, and requires600s sustained quiet after this
  second collision. The original receipt is preserved, with this correction.
- A suffix extension in the hold helper initially reused its argparse variable
  for a process argument list. The exact own scheduler was stopped correctly,
  but writing the second receipt failed. Fixed the variable name, verified the
  same PID/command, then wrote the separate -2 receipt; original receipts and
  foreign workloads were not changed. Neither issue touched measured sources.


## 2026-11-14T00:29:19.001137+00:00 — user selected return to original node

The user explicitly chose to wait for pod6-1's other tests to finish. Retain the
old sealed checkpoint and all40 raw attempts,13 exclusions and27 clean runs.
Reopen this same-node experiment only after sustained600s eight-card idle and
no active foreign benchmark coordinator. Actual source fingerprints and all21
frozen SOs have been rechecked equal. Resume at the canonical next cell; do not
replace any valid slow run. No pod5-8 timing enters this matrix.

The in-flight pod5-8 ECHO capacity gate completed successfully (90333MiB). That
host has four passing capacity gates and zero formal runs. Its own idle
container and read-only monitor were stopped; other workloads untouched.
Its source/gradient/replay gates remain supplemental engineering evidence.


## 2026-11-14 12:45–13:06 CST — existing-policy matrix and calibration admission

- Completed all90 clean planned cells on the original pod6-1/rack06os1 node,
  after the user's explicit request to wait for that node. Original90 attempts
  and13 linked external-contention replacements are all retained (103 attempts).
  No pod5-8 training times enter these results. The peak eligible sampled HBM
  is88.2627GiB, below the common90GiB gate. Independent trajectory audit passed.
- Four original131072-capacity interface endpoint jobs passed. Independent
  calibration remains pending; no step_time implementation/gain is claimed.
- Replaced the inactive diagnostic wrappers' per-call global synchronize and
  CPU exports with on-stream Timer/plan clones and per-step deferred export.
  Source-aware quota is retained, backward identity comes from its actual saved
  placement, and read-only tensor gradient hooks replace full-backward view
  hooks. Extra Event/snapshot/export overhead is explicitly diagnostic only.
- The link-only diagnostic failed its strict Timer completeness assertion on
  two attempts. The second retained raw NPZ shows all-zero records for one
  receiver. Raw C ABI launches are outside the PyTorch NPU task queue; a queued
  zero_ is a possible overwrite race. The next revision synchronizes queued
  initialization/Event commands before isolated C launches and preserves the
  same full32-lane validity gate. This is a hypothesis under test, not a proven
  hardware failure. Failed measurements are excluded from fitting, retained as
  separate attempts, and have not modified formal source/SO results.
- Four PNG/SVG/PDF figures from the90 audited runs are added to the separate
  Planner phase chapter of the offline overview. Browser checks passed at
  1500/760/390px, all downloads and source hashes match, no network requests or
  JS errors, and the prior three-backend chapter is byte-identical.


## 2026-11-14 13:10–13:50 CST — link validity, topology and dependency capture

- Completing the queued initialization/Event commands before raw C ABI launches
  fixed the link Timer completeness failure. All32 lanes, six iterations,
  eight ranks and every directed/fanout measurement passed. The original two
  failures and raw all-zero receiver NPZ remain in the attempt ledger.
- First valid directed-link p10–p90 cost spread is1.369%, below the prespecified
 5% resolution threshold. Treat topology heterogeneity as unresolved and shrink
  per-link noise, without claiming a topology benefit. Fanout contention remains
  a measured resource cost. A provenance supplement was frozen before rerunning
  the unchanged link protocol with per-process actual mapped-SO fingerprints;
  both valid datasets will be retained, rather than selecting the faster one.
- Read-only gradient hooks pass a CPU in-place/keyword/checkpoint test with
  bitwise identical output and all gradients. They record dependency envelopes,
  which can contain recomputation or sibling-branch work; reducers export both
  complete envelopes and measured union-subtracted remainders. Same-device
  Event axes and original GetSystemCycle axes remain separate for every rank.
- Both formal and Timer workers have OWNER_AIV1_TAIL=0, ratio1 and largest-ready
  selection0. The original forward input tile-ready path requires<=7 experts,
  so this32/64-wide shape waits full-expert input readiness. AIC performs all
  GMM1 experts before GMM2. These actual resource/order constraints supersede
  optimistic per-expert alternating or freely streamed cost assumptions.
- Subsequent formal controllers now retain direct foreign PID/cgroup proof,
  admit only idle cards without the known foreign coordinator, and wait600s
  after a collision. They can adopt matching completed receipts after a parent
  interruption. They never signal another workload or overwrite partial data.


## 2026-11-14 13:53–13:57 CST — frozen initial policy and provenance supplement

- All15 timer-OFF independent selection runs and the additional uniform branch
  run completed, with the unchanged common model/capacity. The preregistered
  three-target geometric-mean selector chooses Native:1157.551ms versus
  MoonEP1168.406, UltraEP1160.687, ECHO1170.046 and EPLB-fixed1618.047ms.
  This independent-bank choice is frozen before cost/search implementation;
  it is not the strongest-baseline selection from the formal six-point matrix.
- The explicit mapped-library provenance supplement passed all eight ranks,
  preserving the original successful link dataset and both failed attempts.
  It changes no transfer size, repetition count or measurement arithmetic.
  Timer-on training diagnostics are next; step_time remains unimplemented.


## 2026-11-14 14:00–14:08 CST — diagnostic OOM isolated and corrected

- The first timer training attempt completed three steps then hit the unchanged
  allocator's71.34GiB limit when requesting a1.75GiB gradient allocation.
  Driver sample peak was89.481GiB. This is an instrumented-only failure, not a
  failed formal matrix cell; all original statuses and144 partial NPZ remain.
- Found strong input-tensor capture and retained multi-grad-hook node lists in
  diagnostic hooks. The former capture version was recovered byte-for-byte
  against its recorded837b5536 SHA. A CPU regression with cyclic GC disabled
  reproduces its activation retention. Revised hooks capture only a boolean
  and remove all input/output handles after gradient delivery. Six independent
  CPU checkpoint steps free all observed activations without cyclic GC and
  remain bitwise equal to the uninstrumented output/all gradients.
- The corrected eight-rank four-step MoonEP-policy diagnostic passed, with
  peak90518MiB (88.397GiB), identical model/capacity and no allocator-budget
  change. Its numerical trajectory is checked against the timer-OFF reference.
  New diagnostic requests preserve instrument source bytes and verify they
  remained unchanged during each process. The remaining seven diagnostics use
  this same capture version; no new cost/search policy has been enabled.

## 2026-11-14 15:27 CST — calibrated policy, portable DAG and device proposal gates

- All29 accepted independent calibration jobs completed:4 original-capacity
  interface runs,16 Timer-OFF selection/uniform runs,8 TimerV2 training captures
  and1 provenance-complete link measurement. The corrected instrumentation's
  dtype, checkpoint identities, quotas, memory and trajectories passed.
- The device policy is a separate plugin. The frozen Host, common planner and
  ordinary worker binaries remain unchanged. Generated plugin source preserves
  the complete common compiler/materializer text; only source-proposal dispatch
  skips redundant quota rewriting when the new policy supplies its own quota.
- Portable CPU search and an independently coded explicit task DAG passed48
  cases, including24 adopted plans. The independent NumPy compiler verifies
  all assignment identities and capacities. First device gates passed12 small
  scenarios on8 ranks and3 true-geometry calibration points on8 ranks, including
  exact copies/quota/physical IDs, decision statistics, replay and reject/recover.
- Initial provisional dense-replica predictions underpredicted EPLB backward
  substantially. Before any new-strategy formal run, declared a125-point positive
  interference grid on accepted calibration snapshots, trained on steps1/2,
  validated on step3, then refit1/2/3. Final data/copy/owner scales are0.5/4/8;
  these are fitted interference corrections, not newly measured link bandwidth.
  The owner scale reaches the search-grid boundary; model inadequacy remains.
- Whole-step differential guard fell from215.74ms in the first provisional
  model to47.40ms after the calibration-only correction. All original fits and
  errors remain. No primary, heldout or new-strategy training measurements were
  used for fitting. No performance improvement is claimed.
- True-shape warm plan-only measurements add roughly4–5ms/invocation over the
  Native initial policy. Freeze the smallest four-class neighbourhood at1 round
  and4 candidates (limits remain4/64), charging all extra scores and both layers.
  Four measured models and separate forbidden-in-formal semantic fixtures are
  frozen before final numerical/pool/three-update qualification. The unused
  baseline link SO was restored byte-for-byte; its measured diagnostic variant
  remains in the diagnostic directory. Formal step_time timing has not started.

## 2026-11-14 15:41–15:54 CST — final qualification and formal start

- Final plugin664c65d6 passed both device plan gates, all five numerical cases
  on8 ranks (including physical index511), two-layer checkpoint/cache poisoning
  with FP32 parameters and replica restoration, and three full-model Adam steps.
  The actual loaded plugin and model SHAs, eight-rank plans, parameter updates,
  loss/norm trajectory and plan retirement are checked in the qualification
  receipt. Full-model qualification driver peak was90350MiB, below90GiB.
- STEP-TIME-FREEZE captures final source identity and the four approved model
  SHAs before the new18-cell matrix starts. First new formal process started
  at07:41:38UTC. The formal five-policy sources remain recoverable from the
  original archive; the three frozen core binaries remain byte-identical.
- Correction record: the first assembly smoke check mistakenly indexed the
  static3-D route bank as a multi-step bank. It only checked model arithmetic,
  never entered fitting or performance data. Replaced it with the exact
  RouteProvider expert permutation before calibration. Both smoke logs remain.
  The first final-preparation helper then failed127 because docker exec used
  the wrong working directory after freezing models. Retried with an explicit
  task workdir, without overwriting or changing those frozen models; build and
  all final gates passed. These are preparation errors, not excluded timings.

## 2026-11-14 17:20–17:45 CST — retained trajectory warning and fixed-state review

- All108 primary and12 fourth-round training processes finished normally on
  pod6-1, with unchanged frozen sources/binaries. The auxiliary25-step trained
  trajectory comparison stopped the driver: MaxVio5 StepTime repeat4 differs
  from MoonEP repeat1 in global gradient norm by2.415% at step22. Its loss
  difference is0.068%. Neither this valid run nor a slow valid run is replaced.
- Full initial parameter/input/route identities match across the nine checked
  runs. All four StepTime repeats make identical decisions and restore Native.
  Ordinary Native repeats also exhibit cumulative trajectory differences.
  These observations do not prove a numerical root cause.
- The user-prescribed0.004/0.02 limits concern outputs and gradients at the
  same mathematical input/weights. The additional global-norm observation
  compares independently updated weights after multiple Adam steps; it is not
  gradient relative L2. Preserve its strict failed result and original limits
  in evidence/trajectory-audit-strict-fourth-pause.json. Subsequent trajectory
  audits retain all warnings and enforce complete finite observations, rather
  than mislabelling a trained-trajectory norm as the independent gradient gate.
- Before resuming, require a separate same-weight, same-input/IDs/RNG full-model
  diagnostic: each of25 steps computes StepTime and Native gradients before
  one shared Adam update. Check routed outputs, dx/dp, routed/shared gradients,
  aggregate parameter-gradient L2 and same-weight norm, retaining0.004/0.02.
  CPU reference copies are diagnostic only; formal no-offload timing is intact.
  Frozen train.py, model, optimizer, Native sources and binaries are untouched.
  Status remains pending until all200 rank-step comparisons and pool retirement
  pass. Failure will keep formal training stopped for investigation.

## 2026-11-14 17:52–17:56 CST — diagnostic host-rank alignment correction

- The first paired diagnostic passed32 rank-step pairs (steps0–3) before a
  notify-wait/HCCL507001 failure in clip_grad_norm at step4. The device plog
  reports a290s notify wait. No numerical assertion failed. Slow CPU reference
  copying/comparison and rank skew are a plausible diagnostic scheduling cause,
  not yet proof of the precise root cause. This is not a formal-timing failure.
- Retain its full stdout/plog/status and original generated driver/source SHA
  under evidence/diagnostic-fixed-state-pair-vio5. A cleanup helper found that
  the elastic parent had already exited naturally; it sent no signals and did
  not reset cards. The continuation correctly refused to start formal work.
- Retry under a new tag with a Gloo host-only barrier before each policy pass
  and before the next HCCL norm collective. No HCCL timeout setting, formal
  training source, numerical threshold, model or frozen binary is changed.
  Increase this diagnostic controller's outer limit to7200s to accommodate
  CPU reference comparisons, without changing formal25-step training limits.

## 2026-11-14 18:08–18:14 CST — user-requested negative-result closeout

- User instructed “没效果可以收尾一下了”. Stop the waiting continuation first,
  then signal only the verified task container's paired diagnostic parent/ranks.
  Retire the task observer after its final sample. The release receipt confirms
  all8 cards have no running processes; no device reset or foreign process signal.
- Retain108 primary runs and12 fourth-round runs. All120 runs pass complete
  original parameter/input/route and actual-runtime identity checks. Main900
  policy calls adopted no searched plan. The new policy is0.34%–0.85% slower
  than Native; the MaxVio0.2 gain against UltraEP inherits the Native fallback.
- The auxiliary2.415% trajectory warning remains unresolved. The original
  paired diagnostic completed4/25 steps before HCCL failure; the aligned retry
  completed4/25 then was canceled. CPU copies and host barriers are diagnostic
  only. A read-only stack snapshot found a slow rank copying/comparing gradients
  while another waited on the host barrier; it does not prove the numerical
  trajectory root cause. The reader was installed under task dependencies only.
- Cancel72 heldout,54 conditional-ablation runs and the isolated plan bank.
  Keep original full-protocol acceptance false. A separate120-run closeout audit
  verifies only retained evidence and resource release, not canceled validation.
  Add a persistent admission/driver guard against automatic device continuation.
- Preserve source/model/SO freezes, clean-applying patches, all failures, raw
  timings and stage figures. Update the existing page as an explicitly closed
  negative-result chapter, preserving the previous three-backend chapter.

## 2026-11-14 18:23 CST — archive member accounting correction

- First local collection stopped before downloading the five new archives:
  the original measured-source tar has2030 source files,21 binaries and one
  embedded BASELINE-FREEZE.json. Its whole-file SHA matches the original
  3d3feb15 archive; no measured source or binary is missing or altered.
- Correct the collector to require exactly those2052 members and verify the
  embedded manifest SHA as well. Retain2051 explicitly as source/binary count.
  Preserve the first closeout archive/index; issue an r2 archive containing the
  corrected bookkeeping tool. No numerical or performance data are changed.
