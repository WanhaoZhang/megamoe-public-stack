# Redispatch readiness cut: unqualified diagnostic

The existing routed-only ABI3 path calls PrepareXRedispatchSources,
TransportXRedispatch, RetireXRedispatchSources, then publishes/acquires all AIV
arrivals before P1. Generic saved-input comments about streaming dO do not
describe this particular path. This is a code dependency observation, not a
measurement of removable time.

`MEGAFLUX_REDISPATCH_P1_RUNAHEAD=0` is the original control. Modes 1, 2 and 3 are
experimental and require a new build, capability audit, independent numerical
reference, checkpoint replay and repeated full-update comparisons.

The narrow enabled case is routed-only ABI3, saved BF16, GM output, gradient
pipe active, panel restore, standalone flags0. Other paths keep their fence.
Modes never depend on a planner name.

| Operation | Reads | Writes | Required dependency retained |
|---|---|---|---|
| dHp AIC GEMM | dispatched raw dY, W2 | GM Hp scratch (temporarily dHp) | actual per-expert/producer dY flags; exact epoch W2 panel acquire |
| AIV SwiGLU epilogue | dHp, saved gate/up, reconstructed P | Hp, dZ, dP partials | original global transport/read-retirement lease; all AIC dHp arrivals |
| dW2/dX/dW1 | Hp/dZ, dO/W1/X | gradients and returned dX | original read-retirement lease at AIC P1 exit; original per-expert dZ waits |

The source X gather borrows symmetric `ptrD`. `CombineTokens` returns dX to
that same source's expert-sorted row ranges. Their offsets match, but matching
addresses do not prove that every remote reader has retired. This prototype
does **not** relax the global read-retirement lease for dX. It only moves
independent dHp production in front of that lease. No new activation copy or
memory-capacity allowance is introduced.

Mode 1 retains the original dO and P1 order. Mode 2 uses natural physical order
(home slots before replica slots) for dO dispatch and P1 only. It can supply
home-W2 work before remote panels are restored. Fixed prefix offsets and
natural per-expert dispatch ordinals remain in use. X/P transport, weight
restore, dW production and owner service retain the original ordering. The
GM P1 producer does not wait for its AIV consumer; all AICs finish/publish one
expert's dHp before that expert's AIV epilogue reads it. Each expert has a
disjoint row band, so the original completion protocol already protects Hp's
in-place replacement.

Mode 3 additionally restores W2 panels before X/P transport and W1 panels
after it. It uses the same single-visit panel copier and unchanged exact-epoch
publication. Each projection is copied exactly once, including publishing
empty slots. This lets the home-first P1 producer proceed to remote replicas
while XP is still in flight. It may also delay XP and contend for bandwidth;
neither its net benefit nor free overlap is assumed.

This deliberately keeps full X/P transport before epilogue. A more aggressive
expert/tile lease requires a different progress protocol and additional proof;
it is not assumed here. Neither a freed AIV nor free overlap is assumed.

Research decision: first measure whether this dependency cut creates useful
time on Native and the fixed strong UltraEP baseline. A future public schedule
descriptor may choose a P1 order independently from dW/owner order, jointly
with source quotas and copy placement. This compile-time diagnostic is **not**
that final planner or a versioned production execution-plan extension.

Do not fit a planner to the old model's assumed dO overlap: that model does not
encode this ABI3 fence correctly. Source-ready dHp runway, complete dW fan-in,
checkpoint replay and actual source traffic would all need native calibration
and held-out ranking checks. Novelty versus MegaFlux/Lancet/PlaceMoE remains to
be established; merely considering backward or overlapping transport is not
a novelty claim.
# Unqualified asymmetric extensions

Modes4/5 are source prototypes only. Mode4 preserves AIC/dY home-first but
keeps AIV epilogue in shared-first order. Whole-expert dHp completion is the
GM handoff; direct-UB is excluded. Mode5 also permits the home-only regime,
keeping natural order there. Both retain the XP read lease before dW/dX.
The shared CPU predicate and tile/DAG oracle passed288 cases. These tests do
not prove NPU liveness, numerical equivalence or speed. Measured modes0–3
remain the archived a1722b6 sources, without these extensions.
