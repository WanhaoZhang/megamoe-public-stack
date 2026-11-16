# Owner-service window intervention (unqualified experiment)

The first20 full-update screen cells found no2% gain from contributor-count
priority or64-slot early dW1. In the original MaxVio7 trace, owner service was
already complete before the last local dW in42/48 Native and42/48 UltraEP
rank-calls. This does not make overlap free: communication could contend with
dispatch or dX on other parts of the call.

`MEGAFLUX_OWNER_LATE_ONLY=1` is a causal intervention, not the new planner.
Odd AIV1 workers publish P1 completion and skip both early reduction and their
unused P2 role. At P5, each even AIV0 reduces the union of its original static
share and its paired odd share, using the existing `Reduce<true>` task mapping.
It still overlaps with AIC tail dW1; it does not defer gradients across layers,
microbatches or optimizer steps. Non-GM execution retains its original mapping.

This intervention also changes service concurrency: the original odd/even
shares execute on both lane classes at different times, whereas late-only
places their union on the even lanes. Therefore a loss does not isolate
bandwidth interference from service parallelism, and cannot establish that
contention is absent. An all-lane late variant would need a separate P2
completion rendezvous and numerical qualification.

Ready epochs, partial addresses, producer writes, per-tile contributor summation
order and final completion barriers remain the same. No producer waits for the
owner consumer to recycle partial storage. The80-case CPU test checks exclusive
coverage of the original task union, including small non-divisible shapes and
the896/448 gradient-tile dimensions. It does not prove device liveness or math.

Default0 retains the original behavior. Build both0/1 from the same source with
the complete frozen recipe. Keep flags0/panel/owner16/ratio1/compact0/priority0/
wide0; reject untested combinations. Require independent gradients, checkpoint
and main-shape multi-update gates before any timing. Compare Native, UltraEP,
bounded EPLB and full EPLB with identical plans. A win only over full EPLB does
not establish a competitive planner contribution.

If moving service later helps, add a versioned scheduling descriptor and jointly
choose assignment and service windows from current-microbatch dependencies.
If it loses, retain the result and investigate source-quota / preparation costs.
The eventual executor must consume a schedule, without branching on policy names.
