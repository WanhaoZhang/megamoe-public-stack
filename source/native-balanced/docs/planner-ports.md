# Planner policy port boundaries

All reported systems in this lab are **MegaMoE + a planner policy**. CUDA/NCCL,
UltraEP communication, original Echo kernels and original EPLB transports are
not used. No result here is the original author's full-system performance.

|Name|Frozen algorithm|Ascend/fixed-owner adaptation|
|---|---|---|
|moonep|Native 48ef24a6 accepted MoonEP port|Preserves counts, target, stable global ordinal and slot ordering; new common source quota and validator.|
|native|Native 48ef24a6 online beta768, penalty2, gain4|Preserves block rounding/saturation, copy insertion order and return-to-home rule.|
|ultra_ep|Dots-Infra/UltraEP 1474022d7e19b30d931d43364fe814c478b9fb3a, placement.cu stage1|Scalar AscendC quota oracle: dynamic-floor fast path, three retries, four-probe fallback. Defaults: balance1, min1024, keep master>=1, oracle eps0.01. No CUDA warp/TMA runtime. Source-local fill is retained; a joint largest-remainder transportation repair preserves exact source AND destination marginals (upstream independently rounds source rows). Canonical replica slot order is ascending expert ID.|
|echo|MoEScheduler 0dca1919e0c77cc342eb25ca88a8bd63f395d3b1|Home-spillover, sorted one-shot interval assignment, top slots per receiver, simultaneous proportional floor then source-prefix residual. Ascend stable expert/rank ID tie breaks replace unspecified GPU topk ties. Canonical replica slot order is ascending expert ID.|
|eplb_fixed|Same MoEScheduler commit|Greedy load/current-copies replication; LPT placement with fixed homes, home update disabled. Uses sequential FP32 sums on Ascend and independent oracle to define deterministic ties. Global cyclic ordinal counts (ordered physical instances) produce exact per-source quotas; multiple instances on one rank are then coalesced, including home instances. Freed physical slots are not refilled. Common materialization makes those per-source quotas consecutive. Canonical replica slot order is ascending expert ID.|
|step_time|This lab|Not yet enabled. Uncalibrated selection fails explicitly; it never silently selects MoonEP.|

The three imported policies are mathematical ports with explicit adaptations,
not instruction-for-instruction replicas. CPU oracle tests compare the public
source-count/copies/quota contract independently against the portable Ascend
algorithm. Device gates additionally compare physical IDs and rejection states.

Reference sources and blob identities are in MANIFEST.json. Upstream license
texts are preserved next to the sources. The common histogram exchange is
adapted from the accepted Native source and remains separate from policies.

## Initial implementation evidence (2026-11-14)

- CPU semantic oracle: 480 cases pass over EP2/4/8 and physical width8/16/32/64;
  includes empty experts, repeated assignment, hotspots and moving hotspots.
- Common compiler rejects malformed source totals, invalid replicas and bad
  capacity; every physical assignment is reconstructed independently.
- Unified AscendC planner and matching Host extension built successfully with
  max local experts64, ABI1. This is build evidence, not device correctness.
- A first oracle mismatch exposed NumPy float64 promotion and reduction order
  in EPLB; corrected reference to explicit FP32 scalar arithmetic. No formal
  timings existed when this was corrected.
