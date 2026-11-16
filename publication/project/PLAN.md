# Host-only boundary diagnosis

Independent sibling research/step-lifetime-host-boundary, based on b77bb06 and the explicitly frozen priority-cache worker e2868fd2. Old source, SOs and completed runs remain immutable. Same common node admission lock; isolated container/install/artifacts/port range.

Question: do the old rich-capture 188¨C206ms cross-rank backward entry gaps persist after warmup with the TimerOFF worker and no new device Events/snapshots/synchronizations? Measure host calls before any diagnostic resource creation; do not infer device completion from host return.

Implement explicit --host-boundary-trace, off by default. Buffer CPU timestamp records, dump after training only. Trace existing pool/operator, planner/publish, FSDP, optimizer, lifecycle/drain/P2P submission and Python GC boundaries. No tensor clone, CPU read or new device synchronization in the tracer.

Gate: small L2 H512 F512 S256 GA4 W2, three complete Adam updates and full step0 per-parameter gradient comparison against retained qualified reference. Then bounded main-shape B1/B3 controls (11 updates, inspect warmed updates5¨C10); no traced wall time enters performance figures. Actual worker, guarded source, initial/input/route identity, finite updates, exclusive admission and90GiB budget must all pass.

Do not implement a new gradient policy solely from pre-warmup rich-capture waits. Interpret a disappearing gap as a difference between measurement/run conditions, not proof of one specific Event-limit cause. This lane is diagnostic infrastructure, not a new planner.
