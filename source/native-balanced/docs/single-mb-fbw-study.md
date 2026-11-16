# Single-microbatch planner research

Research branch: `research/single-mb-fbw-planner`. No new NPU performance or
device qualification is claimed yet. The common compiler and worker contract
remain shared across policies.

`eplb_budget_N` is an experimental **total physical instance** budget, not a
per-rank slot limit. Supported registry values are 0,1,2,4,8,16,32,64,128,256.
All policies retain the same configured slot and receive capacities. The
original `eplb_fixed` still allocates the complete budget. Coalescing may make
the active remote-copy count smaller than N. Canonical slot ordering and cyclic
source quotas remain unchanged.

Budget codes 100+N are collectively checked as part of the algorithm identity.
The host rejects budgets above EP*slots. The planner SO advertises budget ABI1;
Python rejects a stale SO without this capability. Existing codes retain their
previous behavior. The extension is an experimental strong baseline, not the
new owner-service-aware research contribution.

CPU qualification: 544 portable C++/NumPy/assignment compiler comparisons.
Before training, rebuild the Host extension and planner SO in the isolated
project, then run the original device, lifetime and fixed-state numerical gates.

Experiment scripts and source/data indexes:
`D:/megamoe_train/single-mb-fbw-planner-study`.
