# Do not turn an unobserved diagnostic gap into a policy premise

The existing one-node diagnostic gives a shared-host observation without merging device clocks: B1 has three backward entry gaps of188–206ms at `(step2,group1/2/3,wave0,1)`, whereas B3's maximum is5.03ms. Median B1 entry skew is only.353ms; a median over ranks/calls hides these outliers. Full records: diagnostics/host-operator-entry-skew.json.

The recorded lifetime/framework CPU envelopes do not cover more than5ms of the late rank's gap in these cases. P2P submission calls themselves have no duration over10ms across the624B1 submissions; maximum reused-pattern submission is.969ms. This excludes a large HOST batch_isend_irecv call in the recorded data, not device-side communication initialization or pending work.

There is a measurement blind spot: capture_framework.begin creates/records its timing Event before assigning metadata.host_begin_ns; corresponding wrapper bodies also start their host envelope after begin. Event-record/allocation cost or runtime/GC work outside wrappers can therefore be absent from existing host scopes. We have not measured such a stall or established a particular hardware Event limit. Do not claim either as the cause.

The three-update detailed capture also precedes the standard five-update warmup. Its source-publication waits cannot automatically explain the steady-state TimerOFF score. Keep the measured waits as observations, not proven avoidable critical path.

Before implementing a gradient policy from this evidence, use a separate explicit host-only trace on the same TimerOFF worker: no new device Events, no tensor clones and no new synchronizations; record original function entry/return before any diagnostic resource creation and dump after the run. Compare ordinary/warmed updates, including the two waves of a group. Separately instrument Event creation/record envelopes if repeating the rich capture. Numerically qualify wrappers and keep all traced wall times separate from formal performance.

Group-delivery / pre-Native-flight-drain controls remain useful hypotheses, but may only relocate unavoidable work. A proposed joint planner must ultimately outperform the strongest common runtime after these observer/startup effects are excluded.
