# Initial hardware screening contract

Baseline: the original 2-layer MindSpeed/FSDPTurbo model, H7168/F2048,
E256/K8, 16384 tokens/rank, 32 home +32 replica slots, receive163840.
Precision, full-layer recomputation, optimizer, global batch, complete-update
wall-clock boundary and the90GiB ceiling are retained.

The original frozen strongest fixed planner is **UltraEP** (geometric mean
1159.322ms over the original six points, Native1159.803ms). These historical
figures select the comparator; they are not current-node measurements.
Keep both in screening. Later comparisons also retain pointwise best results.

`co_design_pilot.py` tests Native and UltraEP with five workers: frozen legacy,
matched rebuilt priority0/1, and wide early-DW1 priority0/1. High skew20 precedes
low skew0.2, using a declared mixed order. Each cell is a fresh process group
with25 updates; discard the first5. For each remaining update take the maximum
of eight rank wall times, then average. This is one independent repeat/cell,
not20 repeats. No confidence or stable2% claim comes from this screen.

Before screening, both matched workers must pass8 qualification jobs and both
wide workers10 jobs. Every update records loaded SO hashes, unchanged source
hashes, model/input/route identities, plan lifetime checks and driver HBM. No
compilers run during screening. Foreign NPU activity only stops this project.

Prepared-plan diagnostics are a separate namespace. `quota` excludes policy
generation while still running the public compiler; `body` excludes compilation
and validation too, rebinding a checked immutable assignment on each epoch.
Six paired numerical gates cover original online/quota/body, fixed placement with
local source quotas, reversed replica slots, and smallest-copy removal.
All are fixture-specific and `online_score=false`. They are never substituted
for online screening figures. Preparation storage also changes the allocator
budget slightly and must be kept in the diagnosis scope.

The model performs int32→int64→int32 index conversion. Prepared-policy v1
correctly rejected the resulting new pointer, exposing an overly strict fixture
identity assumption before expert execution. V2 verifies complete ID values on
device and the fixture's original version; the check synchronizes. The guarded
online, quota and body arms all pay this same check and retain the same prepared
storage. They are instrumented diagnostics, so their differences are not a
claim about ordinary online training, nor a promise that subtracting two means
isolates planner cost without any scheduling interaction.

Budget eligibility uses all50 actual step/layer fixture permutations, not just
the base histogram. Some low budgets pass the first3 steps but fail later at
high skew; skip/report infeasible cells instead of changing capacity or silently
switching policy. Budget32 and above are legal across the current six bins.

Continue with the full six bins, original five policies, old StepTime, budget
EPLB and the new planner. Promotion requires paired independent process repeats
(at least5), approximately2% beyond noise, the strongest fixed comparator,
pointwise comparisons, planner × pipeline ablations and numerical acceptance.
Held-out banks, other shapes and natural training quality remain subsequent
requirements. A pipeline fix alone does not qualify a new planner contribution.
