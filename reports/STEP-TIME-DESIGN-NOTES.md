# Deferred implementation notes

This is a design record only. Implement the policy after the existing90-run
comparison and independent calibration, as requested. No new policy result is
claimed here.

- Preserve baseline SO/Host/worker identities. Use the already prepared policy
  extension entry and frozen common compiler; add only the separate policy module.
- Keep search arithmetic in calibrated integer nanoseconds (link costs in
  picoseconds per byte). This avoids repeated scalar floating division in an
  online AscendC search. Quantization/error must be checked against the CPU DAG.
- Represent phase costs as nonlinear functions of row tiles and active experts.
  A dense table indexed by ceil(rows/128) can avoid hot-loop interpolation/division.
- Model actual64-slot backward: after P1 dH/SwiGLU gradients, each expert's dW2
  precedes its dX in P2; tail dW1 follows. Do not assume the <=32-expert early-dW1 path applies. Check the
  no-replica fast path separately against the source and diagnostic trace.
- Current BF16 saved gate/up skips the backward's internal GMM1 recompute
  fallback (mega_moe_bf16_backward.h:2344 onward). Whole-layer checkpoint still
  replays the ordinary MoE forward. Count these separately; do not invent a
  second backward GMM1 cost. No-replica entry selects home tiling32 but also
  disables gradient.ready, so earlyDW1's readiness guard remains false.
- Owner service must keep its real projection/expert/tile order and contributor
  waits. A nonzero replica always contributes its complete gradient plane; do not
  multiply gradient-delivery bytes by its fraction of token rows.
- Sparse column0 owner-tile samples fall on even static task residues for this
  shape (both projection column counts are even). They are not direct samples
  of early odd-worker service. Use original per-lane reduction phase envelopes
  to constrain the odd/even overlap, and report the sampled-release limitation.
- Use logical barrier dependencies and local phase durations. Never merge raw
  rank-local time-zero points into an invented global trace.
- Include weight/token traffic, measured link/fanout contention, framework and
  original/recompute/backward/optimizer dependencies. An overlapping resource
  model is needed, not the sum of every kernel duration.
- Bound neighborhoods to128-row transfers with tails, copy removal and migration;
  deterministic tie breaks, <=4 rounds and <=64 candidates each. Contributors
  and owner pressure change whenever a quota crosses zero.
- Preserve the frozen calibration-selected initial policy exactly on fallback,
  including its source quota mode and physical-copy order. A source-locality
  change without changing global row quotas is a separate candidate decision.
- Calibrate prediction uncertainty and planning overhead independently. Require
  predicted improvement to cover both; count the search and fallback cost in
  actual complete-step timing.
- Freeze initial policy/model before step_time's primary18 runs. Do not select
  routes or tune to primary/heldout results. Select the strongest existing policy
  for the final comparisons using the predefined six-point aggregate.
- Existing policy ports are first implementations. Diagnose planning vs execution
  cost separately; a slow port is not evidence that the original author's system
  is slow. The retained Native policy is an additional strong baseline.
- On an independently calibrated route, retain actual device plans and compare
  full/no-gradient-service decisions. Explain replica/quota changes using
  measured release/owner-service intervals; predicted scores alone do not prove
  a reduction of the real critical path. No changed decision or no stable
  measured benefit must remain an explicit result.
