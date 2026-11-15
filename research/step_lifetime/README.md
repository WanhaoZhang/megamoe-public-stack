# Step-lifetime research

Independent ordinary MegaMoE branch based on44b4e0d. Canonical raw artifacts: D:/megamoe_train/planner-executor-ep8-lab/step-lifetime-study.

Track A uses a frozen baseline lane. Serial means the same ready prefixes/checkpoint partition with independent full-S FFN calls, rather than coexecuted half-S waves; it does not change GA or serialize all Attention.
Track B uses the private ABI1 candidate lane and a dedicated owner-service stream/HCCL group. Metadata inspection is online and CPU-observed; this is not an AscendC joint planner.

The candidate worker exports megaflux_step_lifetime_version=1. Sidecar fields6/7 select per-slot cached BF16 sources/raw per-copy dW. Public Host descriptor ABI is unchanged. Ordinary pools leave these fields zero.

A weight hit saves remote traffic but still performs a local copy into the execution slot and publishes fresh stripe/panel epochs. Eight cache entries and eight FP32 gradient entries are shared across layers, with one additional FP32 receive plane pair. Recompute reuses the immutable route plan. Current-stream ordering protects weight readers; gradient service events protect sends and bank reuse.

Raw routed gradients bypass autograd weight returns: the framework parameter hook does not execute on None. The adapter accumulates into the original FP32 Parameter.grad layout, applying EP=8 scaling once. No Parameter, owner or optimizer is replaced. GA scaling remains in the loss. clip/Adam follow the explicit finish boundary.

CPU contract test: python tools/test_step_lifetime.py. Two-rank CPU math oracle: python tools/test_lifetime_delivery.py in the Torch runtime. Actual8-rank math and SO/HBM/identity gates are required before interpreting performance.

Run the transport/campaign scripts in this folder from a checkout at PROJECT/stack. Campaign registrations and receipts stay in PROJECT, not Git. Every restart uses a fresh prefix. Controllers share the original global device-jobs.lock, check real device owners, and reject overlaps.

Formal108-run matrix and joint planner remain conditional on the registered gain gate. Failed gates stop their campaign; no silent fallback. Only completed numerical-qualified measurements enter figures.

Every external-path update initializes zero FP32 routed gradients even for an empty owner, so Adam/weight decay is not skipped. The Gloo oracle additionally checks an all-empty update. Raw tensors used on the owner stream are recorded there, and send/receive completion events are observed before bank reuse.

Qualification status and live results are project artifacts in REPORT.md/STATUS.json, never inferred from this source snapshot.
