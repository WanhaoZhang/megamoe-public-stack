# Additional old-Native controls and numerical provenance

No new online planner is qualified here. All tuning results below are CPU
structure checks or future prepared-plan diagnostics, not online performance.

## Old rule parameter controls

Across6 original banks x50 layer/step permutations x7 configurations,
2100 exact Python/C++ comparisons pass. They verify quota conservation,
active copy/gradient contributor correspondence and common receive capacity.
The old Native rule's default is beta768,penalty2,gain4. Compare beta384,p4,g8
and beta1536,p1,g2 to change allocation granularity while retaining a1536-row
copy penalty and3072-row fallback threshold. Also independently scan old
penalty0/1/4/8 at beta768,gain4. These do not encode real gradient deadlines.

Three preselected bank families (384,1536,penalty4) are frozen for all6 bins,
18 archives. All50 calls per bank fit common163840 capacity. No measured
result was used to select them. High20 mean active copies/default/384/1536/p4
are6.54/7.00/6.54/6.46; max rank rows are133455/132325/135065/133558.
Global128- and192-row tile totals match across these variants in the tested
fixtures. Native already uses a common multiple of forward/backward row tiles.
Remaining imbalance, launch shape, source traffic and full-gradient service
still differ; those tile counts are not a whole-step lower bound.

The next bounded native diagnostic is high20 paired3-update gates, then
prepared-body Native/Ultra and3 bank timings on matched r0 and the eligible
exploratory pipeline selected in the prior screen. If a tuned old rule wins,
it is promoted to an additional strong baseline. No paper novelty follows
from tuning its constants. Candidate banks and the C++ oracle DLL stay in
project artifacts/analysis; compact scan metadata is in the research kit.

## Unresolved numerical warning

Per-parameter energies from three existing same-weight diagnostics were
re-audited. Native self:13 of1392 comparisons exceed relative L2.02, all HC;
MoonEP/Native:9 exceed, all HC; incoming-dY Native self:19 exceed, all HC.
The respective non-HC maxima are.00712/.00580/.00536. The largest Native
self HC relative error is.08434, with reference norm2.40e-5 and difference
norm2.03e-6. Inputs into earlier MoE backward are already unequal, while the
last MoE's input/output gradients match in that self control.

This does not clear the warning or justify relaxing the gate. Near-zero
gradients make relative tests sensitive, but Adam normalization can amplify
small absolute differences. Existing25-step StepTime/Native diagnostic lacks
per-parameter energies, so its6.50 maximum cannot be attributed from these
controls. Long-trajectory and natural-quality validation remain required.

A separate default-off diagnostic sets
`torch.use_deterministic_algorithms(True,warn_only=False)` before device work,
records its actual enabled state, and requests deterministic HCCL. Ascend's
[950DT support list](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/determin_API_list.md)
identifies fused attention backward among affected APIs. This is a hypothesis
for the upstream dY variation, not a measured cause. The [PyTorch contract](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)
does not guarantee full application reproducibility or audit custom native
kernels. Unsupported deterministic operations must remain recorded failures,
not warnings silently accepted. These runs are excluded from performance.

Data: analysis/native-parameter-scan.json, analysis/numerical-error-provenance.json.
