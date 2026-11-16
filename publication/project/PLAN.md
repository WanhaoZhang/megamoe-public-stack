# Temporal scheduling and expert replication study

User-authorized implementation, 2026-11-14. Base: public stack 84a3363d95e11096de95332f9b691f736be752d1.
Isolated branch: research/temporal-replica, checkout: stack/.

## Contract

- Ordinary balanced executor only; framework shared expert; fixed route/owner/optimizer semantics.
- Primary shape L4 EP8 E256 K8 H4096 F1024, full 16K sequences, four microbatches/rank/update, 524288 sequence tokens/update.
- FP32 parameters, gradients and Adam states; BF16 expert compute; existing whole-layer checkpointing.
- W1/2/4 with fixed GA4 and data. Whole-card limit 90GiB; common W2 fallback if W4 fails, stop if W2 fails.
- Existing GA1 experiments are regression data, not a speed denominator for GA4.
- No new planner before actual ready-task complementarity and execution costs are established.

## Implementation gates

1. Microbatch-specific routing, replay identity and frozen plans; GA4 single-update boundary; multiple retained graphs.
2. CPU contract tests plus small-shape output/gradient/update and native saved-state coexistence gates.
3. Device admission, matched binary fingerprints, primary-shape finite-update and memory checks.
4. Ready-task scheduling reference (fixed groups, FIFO, EasyBalance MaxUtil), common task and resource lease contract.
5. Group-capable native execution and framework layer-level integration, verified without removing existing required barriers.
6. TimerV2 mechanism evidence; screening of same/alternating/random hotspots before formal runs.
7. Only after repeated failure of independent composition: bounded joint time/replication search.

## Measurement and completion

25 optimizer updates/process, 5 warmup +20 measured; rank-max full wall time, three independent repeats, rotated order.
Failures remain explicit. No simulated durations enter performance figures. Formal paired matrix: 2 methods x6 MaxVio x3 temporal patterns x3 repeats=108 runs, conditional on earlier gates.
Independent held-out routes, sensitivity and ablations follow only after a repeatable mechanism.

## Artifacts and risks

Raw data stays here; previous closed experiment is not resumed or overwritten. Status and decisions record partial completion explicitly.
Current native pool is serial and per-call kernels have collective barriers; saved-graph coexistence is not concurrent execution.
Previous 25-step numerical trajectory warning remains unresolved and is not inherited as a correctness pass.
