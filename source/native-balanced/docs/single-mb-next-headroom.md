# Remaining native experiments after the current runahead stage

This is a design queue, not implemented/qualified functionality or a novelty
claim. Run the existing 29 gates and Native/UltraEP endpoint screens first.
Do not edit the deployed tools while their controller is active.

## If dHp runahead is numerically valid but does not reach the engineering gate

1. **Separate AIC and AIV P1 order.** Current modes2/3 use home-first for both.
   Home-first can expose home W2 work while X/P moves, but may delay the shared
   experts' dZ and then stall the original shared-first P2/dW2 order. In GM P1,
   AIC writes disjoint dHp row bands and publishes all-AIC completion per expert;
   AIV reads only after that expert's count is complete. AIC and AIV tile-lane
   cursors therefore need not necessarily assign the same tiles to the same
   cores, unlike the direct-UB path. A candidate should keep AIC/dY home-first
   and AIV epilogue shared-first. Prove every tile's unique coverage, dP partial
   indexing and arrival counts; qualify with a deliberately delayed producer.
   Do not assume the order is safe for direct-UB, save=none or worker flags1.

2. **Borrow transport UB only inside the proven routed GM interval.** Current
   saved-mode `dispatchPingRows_` is forced to1. At H7168, the existing64KiB
   stage region could hold two rows/ping, while a temporarily borrowed256KiB
   UB can hold nine with the existing256-byte margin. AIV epilogue cannot start
   before the current global XP retirement lease, and AIC dHp stores GM rather
   than fixpiping into AIV UB. This suggests a legal temporary larger ring for
   dY/XP transport, after source preparation and before P1. W2-before-XP and
   W1-after-XP copy buffers at[224,240)KiB must be completely drained before
   borrowing/releasing their bytes. Restore original staging layout before
   P1/P2/empty-gradient clearing. This is an unproven optimization; inspect
   intrinsic transfer limits, event reuse and all small/remainder shapes.
   Keep source gather unchanged initially to isolate transport batching.

3. **Obtain a lighter original TimerV2 timeline.** The old snapshots include
   per-tile record/clone/export overhead and inter-step CPU export can perturb
   next-call arrival. Their long source-publish wait is not a formal speedup
   bound. A separate diagnostic build could retain original per-expert AIC/AIV
   TimerV2 and the four XP phase pairs but disable coordinate stage records.
   A capture-only driver should retain just original small timers and exact
   plans for a selected final post-warmup call, exporting after all timed
   updates; no per-call synchronize or large raw stage clone. Audit timer SO,
   buffers and event scope. Its step time remains diagnostic, with TimerOFF
   controls required for any performance claim.

Do not turn the first negative screen into a claim that all single-microbatch
headroom is exhausted. Conversely, do not add each speculative variant to the
main planner before a native effect is demonstrated.

## Planner contract if the dependency cut creates useful native headroom

Joint candidates must retain copies/source-quota assignment and checkpoint
replay. A future versioned schedule may carry separate AIC dHp order, AIV P1
order, dW production order and owner service order. Validate permutations,
fixed prefix offsets, bounds, version/epoch and immutable replay in the common
compiler; kernel behavior must not branch on a planner name. The current
compile-time modes are diagnostic, not this final descriptor.

Use current source×expert counts, shape/tile arithmetic and actual transfer
granularity. Include F plus checkpoint F replay, dY/XP/W2 readiness, full84MiB
nonzero gradient contributions, fixed-owner service and blocking dX. Count
per-expert setup/output costs without pretending every AIC has an expert-level
barrier: its start cursor rotates. Consider deletion, token quota transfer,
source transportation and slot/restore order with bounded online work.

Train and validate ranking on forced native candidates; the inherited47ms
absolute-error model and all-fallback decisions are insufficient. Paired delta
errors and coverage must be reported separately from absolute errors. A search
can fail because it misses candidates, misranks them, costs too much online, or
the executor cannot realize them. Keep those outcomes distinct.

Host allocation uses64 physical experts, but the actual forward kernel and
routed-only V3 backward override to homeTiling32 when the header has no active
copies. They also skip gradient-pipe initialization and backward restore.
See megaflux_full_bf16.cpp and megaflux_worker_bf16_backward_shared.cpp, not only
the Host wrapper. Current runahead modes require the active gradient pipe, so
Native at low skew (no copies) is a no-op control, not an exercised negative.
An explicit no-copy saved-GM runahead extension needs its own qualification;
natural P1 order must be retained when gradient prefix addressing is inactive.

Before declaring EPLB's overhead inherent, inspect EnableGradientPipe's sort:
each GradientKey comparison rescans all copies using uncached Fresh reads.
The original frozen implementation already has that structure. Cached exact
keys and single-pass contributor counts should be an additional matched
executor baseline, keeping every order and gradient contract identical. Only
native timings can quantify this cost; do not equate operation counts with ms.

Also inspect panel restore's inactive-slot work before fitting an irreducible
replication penalty: CopyExpertsPanel visits every slot/panel/worker and
publishes epochs even for empty slots. A sparse service must first prove all
flag consumers, forward/backward resets, exact-epoch reuse and inactive-to-active
checkpoint transitions. Do not remove those publications based only on a
comment or on nonzero GEMM rows. Per-rank gradient participation might use the
union of replica destinations and fixed owners, but all global routing and
retirement rendezvous must remain valid. These are unimplemented hypotheses,
separate from the exact-key cache experiment.

Overlap, locality, backward awareness and contributor-count penalties already
have strong related work. Establish a specific missing dependency/tradeoff
against MegaFlux, Lancet, PlaceMoE, FlexMoE and TEMPO before making a paper claim.

## Required after the fast-pipeline screen: test planner interactions

The24 forced deletion scores used runahead0. They do NOT decide which plan is
best under a changed P1/transport schedule. If a new pipeline passes numerical
gates and the2% engineering screen, rerun Native/Ultra/proxy/sparse on both
matched original and selected pipeline workers. First qualify each changed
plan's composition with that worker. Keep all arms in the same prepared-body
protocol; these are interaction/headroom diagnostics, never online scores.
Only then refit and port a current-route rule; a forced oracle win cannot be
used as the final four-cell online ablation. Include source-quota candidates
on the new pipeline if source readiness is the exposed cut.

The existing MEGAFLUX_OWNER_COMPACT=1 builds a per-call contributor list and
preserves ascending rank/slot SUM order, but current framework and Host shared
backward contracts explicitly accept only compact0. Do not silently bypass
those capability checks or call this pre-existing optimization novel. If light
timelines identify owner scalar scanning as exposed, adapt/qualify it with an
explicit supported topology and accurate capability/Host contracts. Full-list
constant arrays are not automatically safe at EP>8.

## Inspect the true dX publication boundary before a larger redesign

The old instrumented Native/Ultra snapshots have about25.14/25.54ms per-rank
between the last dX GEMM and kernel exit. This is NOT final input-gradient
readiness or removable time: remote combine, unpermute, route gradients,
empty clears, owner work and retirement still intervene. The current code
places unpermute behind the all-AIV rendezvous after owner reduction; DW1
also follows the per-expert DW2/dX loop. First use the new light original
TimerV2 scopes to distinguish DW1's envelope, unpermuted-dX completion and
remaining exit work on the fast executor. Never sum independent ranks.

A possible fallback is a separate dX-critical phase and deferred DW1 phase,
but postponing weight gradients itself is already Lancet's scheduling idea;
PlaceMoE already overlaps replica GradSync with attention backward. Any
claim would need something additional: current-route joint replica/quota
choices, full56MiB DW1 partial admission/owner deadlines, bounded activation
and partial-bank leases, and measured accelerator resource interference.
The current API returns all gradients together. A safe split needs explicit
autograd/FSDP stream dependencies and optimizer completion, not an unready
gradient tensor returned through the existing bridge. Full32-AIC persistent
kernels cannot be assumed to leave compute resources for another layer.

Memory is a hard gate: retaining the whole existing backward workspace plus
packed X can add multiple GiB per layer. Views do not release the base
allocation. Any selective defer scheme needs actual compact buffers or a
versioned arena, and must include packing, extra launches, source leases,
owner partial storage and concurrent-layer peaks under the common90GiB cap.
No split implementation or speedup is claimed at this checkpoint.
