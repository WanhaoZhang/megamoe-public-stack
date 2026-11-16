# Routed GM transport batching: unqualified prototype

The saved BF16 path forces one row per dispatch ping. At H7168 a64KiB
static stage can hold two rows per ping, and a256KiB arena can hold nine,
subject to the tiling's existing limit. Fewer DMA submissions may reduce
transport overhead; neither a bandwidth gain nor step speedup is assumed.

`MEGAFLUX_REDISPATCH_TRANSPORT_BATCH=0` preserves legacy batching. Mode1
uses the original disjoint stage window. Mode2 temporarily binds the full
AIV UB ONLY between source preparation and transport completion, restricted
to routed-only saved-BF16 GM with qualified dHp-only runahead enabled.
Gather is unchanged. The direct-UB, shared-expert and save=none paths retain
their original layout. Unsupported row alignment uses the original layout.

Ownership argument: source gathers and the probability TPipe have returned
before the temporary binding. AIC dHp writes GM. Every AIV epilogue remains
behind the complete XP/read-lease fence, so its P1 vector tiles are not live.
Each dY ring drains both slots before return; each X ring drains before P
borrows the same bytes. W2 panel copy occurs before the ring, W1 after XP;
CopyExpertsPanel drains each chunk and MTE3 publication before returning.
The original stage base and ping layout are restored before retirement/P1.
No source read lease, remote completion handshake or gradient order changes.

The shared C++ byte helper and CPU span checks cover aligned, oversized,
empty and residual rows. CPU checks do not prove event reuse on hardware,
NPU numerical correctness, liveness or performance. Required next gates:
independent references, checkpoint and repeated Adam updates; empty experts,
257-token residual shapes; delayed source/pull; matched fast-metadata
executor controls. All variants remain research-only until those pass.
