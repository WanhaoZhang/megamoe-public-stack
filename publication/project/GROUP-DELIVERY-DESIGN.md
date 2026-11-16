# Next causal control, conditional on the common executor baseline

Status: design only; not implemented or measured. This does not replace the full joint-planning objective.

Prior diagnostic: B1/B3 actual GEMM and X/probability pull are close; source-publication/table rendezvous differs strongly. This does not prove asynchronous gradient service causes the delay. A pre-Native service drain may merely move unavoidable waiting from the kernel to host. A fixed group flush may merely be ordinary gradient aggregation. Test these explanations before proposing policy novelty.

## Minimal future controls

1. Immediate delivery, unchanged.
2. Identical sends/bytes/placements, but complete outstanding service flights before the next Native entry. Keep existing stream/buffer leases; never delete the serial-pool guard. Measure complete updates, not just reduced in-kernel wait.
3. Accumulate remote dW across the original group's two FFN waves and deliver after all its backwards finish. Distinguish this from GA2-window or GA4-update delivery. Keep capacities and deterministic capacity-pressure fallback common.

Use the best qualified common raw-output schedule, same copies/quota/task order and same FP32 public path. Original native anchor and fixed-home remain required stronger controls before any policy claim.

## Authority belongs to the plan lifecycle

The prefix key currently is `(step,group,wave,1)`, not `(step,mb,layer,call)`. Do not infer group meaning from generic tuple positions inside the device executor. The framework already knows the original wave set before execution and should bind explicit group identity/member count alongside the existing invocation context. PlanTicket owns this immutable membership; checkpoint replay reuses it. Backward binds a unique live ticket before producing an obligation. The common lifecycle marks one completed backward per ticket; a delivery policy consumes the group-complete event. No strategy-name-dependent routing decisions belong in the executor.

Group completion must count every original member once, including zero-work replicas; zero-work replicas create no dW obligation. Version/layer/expert/rank identify weights, separate from ready epoch and physical slot generation. Pending gradient debt stays bounded and cannot cross clip/Adam. Weight eviction and gradient-bank reclamation remain separate actions.

All current B modes can be represented by common settings: cache-enabled, flush boundary (call/group/window/update), and service-admission rule. A future joint policy emits these actions through one executor. This settings refactor is common infrastructure, not a claimed novel planner.

## Qualification before timing

Independent reference: W1/W2, frozen checkpoint replay, empty owners/all-remote dW, zero replicas, and two waves whose disjoint hotspot sets force an8-entry bank to flush mid-group. Validate reverse backward order, exactly-once contributions, same global submit order and deterministic capacity fallback. Three actual Adam updates must invalidate old weights and finish all gradient debt before clipping. Keep outputL2<=.004, gradients<=.02 and whole-card<=90GiB.

If a control only relocates a wait, report no benefit. If a fixed boundary/caching fix explains all gains, open it to baselines and do not attribute it to joint task selection. Only after showing a residual conflict between grouping, copies and gradient service should search a new online policy. Never read future layers/microbatches from the pre-generated bank.
