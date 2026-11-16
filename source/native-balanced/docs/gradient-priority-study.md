# Matched gradient producer and owner priority experiment

`MEGAFLUX_GRADIENT_PRIORITY=0` retains the original replicated-first logical-ID
order. Value1 prioritizes replicated experts with more remote contributors,
using logical ID to break ties identically across participating ranks. The
producer and both standalone/cooperative owner paths use the same priority.

Every logical gradient tile keeps a unique writer. The owner still loads and
sums contributions in ascending rank/slot order. Ready checks, epochs, fences,
payload addresses and numerical casts are unchanged. Readiness lookahead maps
through the same expert permutation as actual execution.

This is an experimental pipeline candidate, not a proven optimization. The
first hypothesis is reduced gradient convergence tails under many-to-one owner
service. It can also worsen dX latency or shift the tail to a large expert;
full-update timing and per-rank diagnostics decide acceptance. The inherited
step_time model currently assumes logical-ID order and must not be claimed as
calibrated for this variant. No online planner benefit is established yet.

Before use, build an independent worker variant and run plan, reference-gradient,
checkpoint, multi-update and full-model gates. Preserve the original binary as
the baseline. Compare original/new planner x original/new executor separately.

The frozen measured worker reports SwiGLU mode2, flags0, restore2, operand-ready1,
and replicated-DW1-early1. Current CMake's legacy auto resolver would choose
mode0 for this target. `MEGAFLUX_S9_SWIGLU_OVERRIDE=2` reproduces the observed
mode explicitly without changing auto behavior. Build matched priority0 and
priority1 controls, and compare their exported capabilities before timing;
do not attribute a SwiGLU-mode change to priority scheduling.

The existing early-DW1 path is additionally guarded by `expertPerRank_<=32`.
The primary experiment uses 64 physical slots, so the compiled capability1 does
not imply that early production actually runs. `MEGAFLUX_EARLY_DW1_WIDE=1`
uses a 64-bit invocation-local completion mask and extends that guard to64.
Default0 preserves the original mask and guard. All shifts use the selected
mask type. Test slots31/32/63, checkpoint lifetime and every home/replica
gradient before measuring this separate candidate. Early DW1 may delay dX;
exposed owner time and full update time, not capability flags, decide benefit.
