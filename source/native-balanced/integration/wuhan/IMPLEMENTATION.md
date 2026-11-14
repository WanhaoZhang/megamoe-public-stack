# Wuhan eight-card integration (in progress)

Base native: `c3d38b93b2bdb3641b15793a9862f815292835f7`.
S9 donor: `39349f4805a6904ff58781c0073d2466fc7a19b6`.
Framework changes and exact bases: `frameworks.json`, `patches/`.

## Implemented, full integration acceptance pending

- Isolated `megamoe_balanced` backend on ordinary expert-FSDP Parameters;
  Native(768,2,4) or MoonEP planner, one shared SHMEM runtime and one S9 pool.
- New replica bindings and isolated S9 kernel build; workers/prefix/external
  refresh/compact are all off and owner rows is 16. Historical SO hashes do
  not identify these ported binaries.
- Split-storage `restore_weights` mode restores W1/W2 from republished live
  homes before backward, preserving forward placement. `staged` only at this
  milestone; expert/panel reject explicitly and remain required work.
- All per-layer routes, numerical state and successful control status are on
  the autograd saved-tensor tape. A checkpoint recomputation's status travels
  with its regenerated state rather than pairing it with the first epoch.
- No per-layer pooled weight references are saved. Owner gradients returned
  to FSDP are private; no second EP/FSDP gradient reduction is added.
- Wuhan GM SwiGLU's W1 consumer now checks replica stripe readiness before
  blockMmad. Ordinary forward is unchanged by the conditional guard.

## Evidence so far

CPU mathematical lifecycle gate: 4/4 configurations passed (two layers,
ordinary/checkpoint, direct/packed saved tensors, two optimizer updates).
This proves Python tape/lifecycle behavior with a mathematical mock, not
Ascend precision, actual expert-FSDP2, UAM or training performance.

The first CPU test exposed a stale Python control epoch during checkpoint
recompute; fixed by saving control metadata on the same tape. Failed and
passing logs stay in the project evidence directory.

The original control build/import gate passed on pod7-3 (control-build-a7,
exit 0). Python SHMEM is built from f281664; all four installed runtime
library hashes remain unchanged. The combined process maps exactly one
SHMEM image. Importing MegaMoE before MoonEP without this canonical Python
runtime reproduced double-free at normal interpreter teardown; failed
import probes are preserved. `runtime_bootstrap.py` establishes the common
import order for all four experiment backends without changing their math.

The first candidate build found an omitted common MoonEP algorithm header;
the header was restored byte-for-byte from the donor. The worker target now
uses the same Wuhan SwiGLU decoupling configuration as the ordinary BF16
target (auto by default); H7168 stays on the original coupled path. Device
acceptance remains pending. Full P0–P4, expert/panel, 48 training runs and
same-plan analysis remain outstanding under the project PLAN.md.

## First actual device gate

pod7-5 `bridge-ep2-a3` passed (both rank receipts, exit0, immediate all-eight
pre/post gates). H512/F512/E8/Top2/M257, BF16 Parameters, MoonEP planner,
two layers sharing one pool, balanced/hot/rotated routing and three SGD
updates. Homes and replicas were deliberately poisoned after forward;
backward restored from live layer Parameters. Five outputs passed original
staged predicates; cold owner gradients were required to be exactly zero.
This is an operator bridge gate, not actual expert-FSDP2/UAM or full training.

## Real expert-FSDP2 and first complete-network gates

pod7-5 `expert-fsdp2-moonep-a1` and
`expert-fsdp2-moonep-checkpoint-a1` both exited 0 with eight rank receipts
and immediate healthy/idle eight-card post-gates. These use the actual Wuhan
fully_shard, EP4 x EFS2 Shard(1), BF16 unshard, FP32 master/RS, divide8,
two layers and three SGD updates (hot/balanced/rotated). Five-output and
updated-weight predicates passed. This does not yet establish UAM acceptance.

Eight two-step full DeepSeek runs (A/B/C/D x rho0/6, recompute) exited 0.
They establish executable paths but are excluded from paired accuracy and
performance: SCRATCH_INIT_BEFORE_SHARD=1 in the first runner changed A/B/C
to full-model BF16 initialization while EFS2 Hybrid retained post-shard FP32
initialization. Step1 loss differed before any update. The corrected runner
uses SCRATCH_INIT_BEFORE_SHARD=0 for all four methods and saves actual local
Parameter SHA256 identities before timing. These corrected controls must be
rerun before reporting cross-backend accuracy or timing ratios. The existing
default/512p scripts and backend math are unchanged.

`summarize_training.py` checks eight rank/step records, real exit and pre/post
gates, and initial parameter identities; it withholds speedup ratios without
matching identities. Two-step probes do not replace the required 10-step
short gates, UAM gates, restoration candidates or 200-step paired matrix.

The first launch used a nonresolvable hostname rendezvous (stopped before
operator execution). The next exposed a test bug requiring every rank to
change weights even when it owned no hit experts; failed runs and successful
post-gates are preserved. The corrected assertion requires a genuine global
update while permitting cold owners to stay unchanged. Successful runs use
normal teardown; assertion failures exit promptly so peers cannot hang in
SHMEM finalizers. All tests are correctness/diagnostic runs, not timing data.
