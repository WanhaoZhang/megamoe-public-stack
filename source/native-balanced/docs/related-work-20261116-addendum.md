# Primary-source recheck while native candidates qualify

These sources restrict novelty claims; none is newly measured in this lab.
Downloaded code stays in project artifacts with exact hashes. The original
five planner ports remain the executed methods, not claims about entire
upstream systems.

* [Megatron VE implementation at pinned0be53c7](https://github.com/NVIDIA/Megatron-LM/blob/0be53c791ffb70fe746bc0cdeaedb1409b1e23d9/megatron/core/transformer/moe/virtual_expert_load_balancer.py)
  explicitly derives its planner from MoonEP. It already retains per-forward
  plans and shares virtual buffers, pushes backward weights at a layer-output
  hook, starts FC2 gradient reduction after its weight GEMM, and later hands
  owner gradients into framework edges. Therefore neither split FC1/FC2
  synchronization nor shared arenas nor backward hooks alone are new.
  Its [pinned supported-configuration notes](https://github.com/NVIDIA/Megatron-LM/blob/0be53c791ffb70fe746bc0cdeaedb1409b1e23d9/megatron/core/transformer/moe/README.md)
  exclude Megatron-FSDP and whole-MoE/full recomputation. These are specific
  implementation limitations, not proof the mechanisms cannot extend there.
  We did not execute that GPU stack on NPU or count it as an extra baseline.
  Its planner also supports aligned counts; Native here already uses768-row
  blocks, divisible by both forward192 and backward128. Tile alignment itself
  is already present, so do not claim the old policies ignore all tile effects.

* [LAER-MoE, ASPLOS2026, section3](https://arxiv.org/html/2602.11686v1)
  integrates expert re-layout with FSDP-like parameter gathering and gradient
  resharding using FSEP. It also delays gradient communication to overlap
  later expert work. Hence integrating expert placement with FSDP, or delaying
  communication by itself, is not new. Its parameter-sharded execution differs
  from our fixed EP owner/expertFSDP1 contract. A fixed-owner heuristic port
  would not reproduce the full FSEP system and must be labelled accordingly.

* [ReLibra, sections3 and5](https://arxiv.org/html/2605.08639v1)
  uses RL routing replay available before training, inter-batch reordering and
  intra-node microbatch replica/token-split optimization. Shared single-layer
  buffers and attention-overlapped synchronization are already there. Its
  preknown-route MILP is an offline planning comparator, not an online score
  for current-route pretraining. Our prepared-bank experiments must use the
  same distinction. It supplies oracle loads to strengthened baselines and
  shares the executor overlap machinery across methods, an evaluation rule
  we should retain.

Together with MegaFlux, PlaceMoE, FlexMoE and Lancet, these sources narrow the
hypothesis to measurable current-route decisions under actual producer and
owner service dependencies, plus the executor that realizes those choices.
Home mirror reuse and metadata caching remain common executor repairs.
No novelty or speedup follows merely from mentioning forward/backward costs,
source locality, shared buffers, replay or gradient communication overlap.

Source receipts: analysis/megatron-ve-source-audit.json; raw copies:
D:/megamoe_train/single-mb-fbw-planner-study/artifacts/related-work-20261116/.
