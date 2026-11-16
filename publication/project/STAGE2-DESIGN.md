# Online ready-frontier prototype

Current source keeps whole-layer non-reentrant recomputation. Exposing routes halfway through an FSDP layer is not equivalent to launching another thread/stream.

The first prototype will preserve complete-layer checkpoint coverage with an explicit routing prepass:

1. Under no_grad, run attention/HC/router for an available layer input and observe that task's actual route histogram. No future task routes are read. This adds real attention/materialization work and must be charged to the step.
2. Select a ready runset using fixed groups, FIFO or EasyBalance MaxUtil.
3. A non-reentrant checkpoint function computes complete selected logical layers: attention/HC/router/shared FFN, grouped native experts, then HC merge. Its immutable wave membership is captured at original forward; replay does not schedule again.
4. Two tasks use two 8K+8K FFN waves for full16K sequences, preserving Attention context. A wave has at most two layer domains and at most64 physical experts.
5. Sum the losses of a bounded W window into ONE backward graph task; accumulate across windows to fixedGA4, clip/update once. This avoids replaying one group checkpoint separately for each output.

Original Native without prepass is an essential baseline. The prepass is overhead, never excluded from measured time. If it dominates, that is a negative finding; do not hide it by comparing only prepass-enabled methods.

Parameter names/ownership remain unchanged by assembling a decoder subclass from the original component modules before FSDP initialization. Framework expert gradient divide hooks must be installed via valid explicit warmup forwards before bypassing experts.forward; otherwise grouped direct-owner execution would miss the /8 hook. Warmup has no optimizer update and no route-plan tickets.

The legacy32-home/32-replica pool and64-home/0-replica pool share one SHMEM runtime with a jointly budgeted heap. All methods allocate the same catalog. Whole-card90GiB admission must be repeated for this runtime.

The CPU reference prototype passed three complete small-model Adam updates and full same-weight owner-gradient comparison against sequential execution. Public FSDP unshard leases restore the layer and HC ownership groups at replay; no Parameters are replaced. Native waves use an explicit zero-valued autograd order dependency so all ranks execute reverse collectives in the same order.

An AscendC plugin implements FIFO, fixed-layer grouping and bounded MaxUtil using integer ratio comparisons, forced admission and deterministic tie breaking. It is separately built without changing the frozen worker. Device-vs-reference validation and full-shape qualification remain gates before performance use. Python must still read the small selection result to execute decoder control flow; this fence is charged to update wall time.

Neither CPU prototype correctness nor a successful plugin build qualifies the full performance matrix. Afore replica-cache integration and a new joint policy remain conditional work after the basic comparison.
