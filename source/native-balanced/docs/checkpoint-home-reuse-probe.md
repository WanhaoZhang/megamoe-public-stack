# Immediate recompute to backward home-weight handoff: unqualified probe

This is an executor-cost probe, not a new planner, not a replica restoration
optimization, and not a claimed paper contribution. Default is OFF.

The current framework converts current FP32 expert parameters to BF16 for
forward, checkpoint recomputation and backward, and the serial pool copies
them into its persistent home mirrors every time. Immediately after the
matching checkpoint recomputation, backward may consume those same mirrors.
No extra parameter, BF16 replica, activation or gradient storage is retained.
The standard backward replica restoration, communication, owner reduction,
plan compilation, gradient scaling and optimizer ownership are unchanged.

Eligibility requires unified EP2/4/8, routed redispatch, BF16 saved operands,
a live unrepeated PlanTicket, the matching full placement, immediate pool
operation order, the same NPU stream, and identical home tensor identity,
version, pointer, shape and dtype. An intervening routed/shared pool call,
retirement, visible home write/replacement or stream change prevents a hit.
A proof once issued must be consumed exactly once by the very next backward;
stale or foreign proofs are errors. Failures invalidate leases. Normal misses
execute the original live parameter callback and home refresh. No device
header bits or worker ABI are modified. Arbitrary external raw-pointer writes
are unsupported, as are parameter updates between checkpoint recomputation
and its matching backward. Buffer stamps do NOT establish master parameter
equality; normal checkpoint parameter immutability remains a prerequisite.

`checkpoint_home_reuse` is a static pool/config signature field, false by
default. Autograd still returns owner gradients to the original Parameter
edges with the original recorded dtypes. The optional optimization bypasses
only the backward `current_weights()` callback and two home `copy_` calls.
Counters record actual hits/misses and skipped BF16 home-copy bytes. They
must show exactly one hit per checkpointed layer backward in the main probe.

Qualification so far:16 standalone lease-state cases plus23 tests executing
actual bridge control methods through AST with fake buffers/streams and the
real configuration validator. These cover callback bypass/fallback, retained
gradient dtypes, aliases, mutation, retirement, stream/order errors and invalid
configurations. These are NOT real Torch storage, FSDP, stream execution or
numerical kernel tests. Python compilation and diff checks also pass.

The new pool gate additionally poisons homes/replicas after original forward,
checks ordinary noncheckpoint fallback, and can poison homes immediately
after recomputation to require a miss. Next: independent reference/replay
numerical gates; three Adam updates at canonical shape; then paired full-step
screens for Native/Ultra under matched fast metadata control and, if eligible,
an exploratory selected pipeline. Original checkpoint/strict numerical
warnings remain open and are not inherited as passes.

Deployment must wait until the sealed fast-pipeline, source-quota and clock
queues retire. Never replace remote audited Python files during those runs.
No NPU correctness, runtime hit rate, memory reduction or speedup is claimed.
