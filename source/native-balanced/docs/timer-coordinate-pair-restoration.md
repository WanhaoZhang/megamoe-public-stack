# Restore the original TimerV2 coordinate pair interface

The frozen PR head `0618eeb` has stage-timer calls to `TikPair`, `TokPair`,
`MarkPair`, `TikPairAsync` and `TokPairAsync`, but its `AscendTimerV2_device.hpp`
does not define those methods. Production TimerOFF builds do not compile the
calls. Building the worker with TimerON fails at the first `TikPair` call.

Restore the five original methods from the previously validated diagnostic
snapshot, retaining their exact method tokens. Synchronous methods keep the
original `PIPE_ALL` barrier; asynchronous methods do not add a barrier. Each
coordinate slot still has one writer and records its original `[start,end]`
pair. This does not add a profiling backend or reinterpret an envelope as work.

The interface is guarded by `ENABLE_TIMER || DEEPEP_BF16_STAGE_TIMER`. An
includes-stripped preprocessing check confirms the TimerOFF header still has
the same 1,067 tokens as the frozen header. This is a source check, not proof
of object-code identity or runtime correctness.

## Earlier isolated research validation on 2026-10-29

- The five method bodies match the existing diagnostic snapshot exactly.
- Diagnostic OFF/ON worker builds pass with the same compiler, source manifest,
  owner16/flags0/panel restore and only the independent ready-bank flag changed.
- TimerOFF source and libraries used by the ongoing numerical campaign remain
  frozen; the diagnostic source is a separate copy with this header restoration.
- Diagnostic numerical, null-pointer and real-input trace checks are queued
  behind successful completion of the TimerOFF wholemodel comparisons. They
  are not yet runtime acceptance or performance results.

At that earlier research revision the planner did not expose the diagnostic
timer schema. A separate failed build also showed that inheriting the global Timer
block-local-init option is unsuitable for that planner target. The current
diagnostic campaign therefore reuses the exact qualified, uninstrumented
planner binary for both worker variants and records that planner timing is
unavailable. It does not claim a planner profile or end-to-end timing from
the kernel traces.

Local experiment evidence is indexed in
`D:/megamoe_train/megaflux-kernel-research-20261029/STATUS.md`; raw logs, failed
attempts and compiled libraries stay in the experimental project.

## Portable capture delivery

The complete delivery now adds opt-in planner timing with
`TIMER_WITHOUT_BLOCK_LOCAL_INIT`, schema6 source/restore coordinates, an
extended header owned by the explicit Timer planner, and a standalone paired
capture entry. The earlier queued/reused-planner statements above describe
the research snapshot, not this new delivery. See
[capture commands](../integration/wuhan/TIMER-CAPTURE.md) and
[current validation](../integration/wuhan/TIMER-CAPTURE-VALIDATION.md).
