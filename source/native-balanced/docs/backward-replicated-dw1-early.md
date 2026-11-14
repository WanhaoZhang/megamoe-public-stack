# Experimental replicated-expert DW1 before DX

Default: **enabled for compatible S9 panel builds** as requested on 2026-10-30.
Explicit `MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=0` selects the previous schedule.
Staged/non-panel builds default to 0; incompatible explicit combinations still
fail the existing build guards. Fixed-capture canonical operator validation is complete;
this does not qualify full-model training. The experiment project is
`D:/megamoe_train/backward-dw1-early-20261029`.

For experts with an active replica in the current immutable plan, the candidate
runs DW2, DW1, then DX on both the owner and replica. Non-replicated DW1 remains
in the final phase. The original expert order, numerical reductions, full K,
panel readiness and owner schedule are unchanged.

## Build

Use separate new prefixes on Linux in the existing matched CANN/SHMEM build
environment:

```bash
MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=0 bash scripts/build_megaflux_timer_capture.sh dw1-c0-NEW
MEGAFLUX_BACKWARD_REPLICATED_DW1_EARLY=1 bash scripts/build_megaflux_timer_capture.sh dw1-c1-NEW
```

The builder emits OFF/ON worker and planner variants, with the new flag in
`build.json` and `timer-capture-build.json`. Read the exported
`megaflux_backward_replicated_dw1_early()` getter as well as checking SO SHA.
The normal pipeline flags remain zero: this candidate does not enable the
cooperative-worker backward implementation.

The build rejects worker flags, shared-prefix/external-refresh candidates,
restore other than panel2, operand-ready other than1, restore delay,
owner other than16, compact, input-row selection, DB/scan/copy/ratio changes.
At runtime the original path is used unless saved BF16, gradient pipe, panel2,
EP2/4/8 and <=32 physical experts apply.

## Correctness and lifecycle

The existing all-AIV DZ/Hp counter and all-AIC DO-reader retirement remain
before the early DW1. Its local shape cannot alter DX dimensions. Each AIC
keeps an invocation-local completion mask and a DW1 tile cursor; the tail
continues the cursor and skips only completed experts after restoring the
original packed-row offsets. Empty experts follow original zero-fill rules.
No global barrier was added and no final barrier was removed.

Use `integration/wuhan/early_dw1_entry.py ENTRY ...` under the normal runtime
bootstrap with `WORKER_NATIVE` and explicit `EXPECTED_EARLY_DW1=0/1` to validate
the selected experimental build before the device entry initializes an NPU.

The public FSDP2 pool accepts canonical `[home,2F,H]` / `[home,H,F]` weights.
`test_early_dw1_layouts_npu.py` checks canonical no-replica numerics and the
existing full-pipeline rejection of legacy weights. The host launcher explicitly
requires canonical BF16 weights even without replicas. Legacy reference execution
uses the ordinary native operator and is not candidate legacy qualification.

## Validation status and rollback

Integration on 2026-10-30 carries C1 from `7c8e769` onto the newer
`codex/moonflux-8p-integration` base `dfaa1c3`. The merge retains its prepared
dispatch ordinals, per-producer row readiness, vector paths and fused planner
validation. At that merge the scheduling flag remained **0** by default; C2 WGrad-priority
changes are excluded. The fixed-capture device results below apply to the
original tested `91fa98d` base. They do not qualify a newly built combined SO:
repeat device correctness and paired performance checks to qualify this newer
integration base. The merge itself ran CPU/source checks only. The subsequent
default change is an explicit user configuration decision, not a new device or
whole-model acceptance result. C2 remains excluded.

New panel builds resolve the default to 1 in both CMake and the worker/Timer
builders. The shared backward header keeps its fallback at 0 for ordinary BF16
targets; the worker target receives the resolved compile definition explicitly.
Existing CMake caches and frozen variants are not overwritten. Use a fresh
variant/build directory and rebuild the matching host binding, worker, planner
and ABI2 validator; updating Git alone does not update an old loaded SO.

Read the experiment project's REPORT.md and STATUS.json for actual completed
checks. Compilation or CPU checks alone do not establish numerical correctness
or speedup. A failed device admission must not be bypassed.

The 2026-10-30 pod6-1 validation retained all four paired B improvements, with
median paired gains of 6.025% for B and 3.837% for direct FB. All four B/FB P90s
improved. These results use the original fixed rho6 placement, warmup10/repeat30,
eight-rank per-sample maximum wall time, and the full single-layer bridge costs.
Legacy full-pipeline numerics remain unqualified because the existing host
entry rejects that layout; its rejection contract was checked separately.

The [fixed-capture validation record](backward-replicated-dw1-early-results.md)
includes the correctness scope, all paired timings and artifact fingerprints.

Rollback selects a separately built flag0 SO. Never replace a loaded library.
