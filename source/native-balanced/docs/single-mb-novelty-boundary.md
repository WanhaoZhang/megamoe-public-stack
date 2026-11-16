# Boundary of the proposed single-microbatch contribution

The target is a measured change to assignment and execution, with a fixed
parameter/optimizer owner. Forward, checkpoint recomputation and backward use
the same logical assignment. Cross-microbatch scheduling is outside this study.

Existing mechanisms already cover much of the broad pitch:

| Primary source | Existing mechanism | Required distinction in this study |
| --- | --- | --- |
| [MegaFlux, appendix A](https://arxiv.org/html/2610.00671v1) | A reduction-aware owner penalty, full remote dW contributions, replicated-first gradient production and tile reduction overlapping compute. Its fixed heuristic penalty is explicitly not a precise time model. | Replacing the penalty alone is insufficient. Test whether contributor release times and owner service change assignment choices after compute is already balanced; show a residual win over the same optimized executor. |
| [FlexMoE, section 3](https://arxiv.org/html/2304.03946v1) | Models compute, four token exchanges, replica gradient synchronization and placement changes. | “Considering backward” and gradient bytes are existing ideas. Identify dependencies and nonseparable tails that a sum-of-costs model misses. |
| [Lancet](https://proceedings.mlsys.org/paper_files/paper/2024/file/339caf45a6fa281cae8adc6465343464-Paper-Conference.pdf) | Whole-graph overlap, including scheduling weight-gradient computation against communication. | Early/late dW scheduling is existing. Establish why persistent, fixed-owner contributor joins require a different placement decision. |
| [PlaceMoE](https://arxiv.org/html/2610.14798v1) | Joint placement and source-aware token mapping with hierarchical deduplication; replica-gradient synchronization overlaps attention backward, with a completion boundary before clipping/update. | Source-aware assignment and gradient overlap are existing. Our current executor has no token deduplication and waits at a different boundary; do not claim it dominates PlaceMoE without a fair adaptation. |
| [EasyBalance](https://proceedings.mlr.press/v306/wu26o.html) | Cross-layer microbatch workload combination for inference. | Relevant to the earlier broad proposal, but this work optimizes one microbatch and includes backward/update dependencies. |

The two current kernel interventions are diagnostic candidates: matched
contributor-count priority and enabling an existing early-DW1 mechanism for64
physical slots. Neither is a publication claim. In particular, widening a mask
is an implementation limitation fix. If it eliminates the apparent headroom,
the new planner must still beat the strongest existing planner on that fixed
executor. A large comparison only against EPLB-fixed cannot establish this.

Candidate planning objective: evaluate each replica/quota perturbation through
its forward/recompute cost, rounded GEMM work, dX completion and the **maximum
release time of all contributors for each owner-served gradient tile**. Service
backlogs couple experts on an owner. Removing a contributor may save full dW
traffic but increase the final contributor's work; extra early dW may delay dX.
The evaluator must represent both effects instead of a per-replica scalar.

The old absolute step-time model's 47.4ms uncertainty rejected every candidate.
Before making a new online search, compare native timings of identical prepared
candidate plans with their predicted **paired deltas** and ranking. Recalibrate
only with training/calibration banks; keep validation banks independent. If the
predictor is wrong, report that failure rather than quietly removing its margin.

This document records a hypothesis and the required evidence. It does not
establish remaining headroom, model accuracy or a novel qualified planner.
