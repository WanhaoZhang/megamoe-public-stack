# Decisions

2026-11-15: separate clone (--no-hardlinks), branch research/step-lifetime. Old study read-only. Two pilots before choosing main mechanism; no presumed novelty or speedup. Node observed idle; admission must recheck.

Attempt1: first gate launched before deployment finished and failed before device initialization (missing lane). Retained receipt/log and track-a.json. Added new a2 tags; restart only after successful deploy.

Private worker build3 passed:7023c5d8. CPU Gloo oracle passed. B1 controller CLI mistakenly dispatched into A lane and rejected the unrecognized lifetime flag before NPU training; preserved, fixed using dedicated remote_b gate CLI and new b2 tags.
B transport now uses a separate HCCL group and owner service stream, drains actual completion events before bank reuse and clip/Adam. Eight accumulation entries plus one bounded receive pair are included in measured memory; same allocation offered to anchor.

Empty-owner audit: external dW path now creates explicit FP32 zero gradients at each update, including fully empty owner matrices, preserving Adam/weight-decay semantics. Extended two-rank oracle passed the empty-owner case. B source was updated only after anchor controller completed with source_unchanged=true, while next B job was held locally; no active snapshot was changed.

Independent v2 FFN errors passed but all remote/cache byte counts were zero: Native chose no small-shape replicas, so it was explicitly rejected as remote-path qualification. v3 constructs an arbitrary legal proposal with hot8 replicas/rank via the public common compiler, uses original forward plan for checkpoint/backward, and passes64records with positive logical cache/gradient payload. Outputmax0.00030364, gradientmax0.00275919. This gate is semantic only, never a performance planner. Core-source fingerprints are bound to the proof before B timings.
Future remote patches are staged outside guarded source then atomically published while holding the shared run lock, avoiding source mutation during any active frozen run.

2026-11-16: Timer diagnostic clone initially exceeded Windows path limits. Restoring only the worktree left a partial index; an unpublished diagnostic commit accidentally recorded deletions. Restored the full index and moved only the private diagnostic branch back to272d83e before recommitting the intended5-line change as3808cf9. Verified7725tracked files and a train.py-only diff. Main research/step-lifetime, old study and active A/B snapshots were unaffected. The incomplete first deploy never created a runnable container; corrected deploy uses a fresh full Git archive and the global run lock.

Timer v1 finished three updates and passed full-gradient comparison, but coordinate records were zero. Snapshot inspection confirmed Native selected zero replicas in every small-shape call. This qualifies local-path numerical instrumentation only, not replica lifecycle timing. Preserve numerical proof and reject replica-coverage claim. A second Timer diagnostic uses MoonEP legal replica planning as a coverage stimulus, not a performance result. No causal kernel-tail conclusion is drawn from v1.

The small Native plan-hash campaign was superseded after its first anchor was submitted: empty copies cannot qualify main replica consistency. Only its local waiting scheduler was stopped; the admitted/queued child gate was allowed to finish. Main-shape three-update hash diagnostics replace further small hash gates, in the isolated diagnostic lane. They capture actual Native copies/source_quota/IDs/counts for anchor and B0-B3, without multi-GiB gradient dumps; numerical math and full-gradient gates remain separate evidence. All prior registration and receipts are retained.
