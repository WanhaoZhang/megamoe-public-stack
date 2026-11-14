# Research boundaries for the later step_time policy

Primary sources rechecked on2026-11-14. These references constrain the claim;
they do not establish a new strategy's benefit before experiments.

|Work|Already addressed|What this experiment must distinguish|
|---|---|---|
|[MegaFlux](https://arxiv.org/abs/2610.00671)|Device-side tile-aligned replication, forward weight pipelines, backward gradient reduction overlap|A calibrated release/service model and actual complete-step improvement, beyond the existing Native tile/gradient penalty.|
|[PlaceMoE](https://arxiv.org/abs/2610.14798)|Joint expert layout and source-aware mapping; hierarchical communication and gradient/attention overlap|The current executor's actual assignment traffic and fixed-owner synchronization. This lab cannot assume PlaceMoE's cross-layer overlap.|
|[TopoEP](https://arxiv.org/abs/2609.35481)|Device-side deterministic topology-aware placement/refinement|A single-node eight-card measurement can only support the topology differences actually observed.|
|[TAOT](https://arxiv.org/abs/2608.03676)|Topology flow hints, integer placement/assignment, weight transfer overlap|Topology alone or weighted bytes are not a new contribution. Costs must reflect this kernel's timing and gradient obligations.|
|[EasyBalance](https://arxiv.org/abs/2608.07964)|Cross-layer inference scheduling without changing expert-device mapping|Not a first-round training baseline: this single-microbatch training loop does not supply its cross-layer inference work pool.|

MoEScheduler's frozen Draft PR8500 is a layering reference, not an Ascend runtime
replacement. ECHO/EPLB are policy ports from its frozen source. UltraEP's frozen
quota solver is ported without its weight/gradient runtime. See MANIFEST.json,
the preserved license texts and POLICY-PORTS.md for exact versions/adaptations.

The proposed claim is conditional: show a calibrated gradient release/service
model changing replica/quota decisions, then show a complete-step gain under the
fixed protocol and independent route banks. If the stability threshold fails,
report no stable benefit. Finer quota granularity or different replica count
alone does not prove that the gradient model caused a gain.
