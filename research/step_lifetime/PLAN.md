# Step-lifetime study

Independent ordinary MegaMoE exploration, starting at 44b4e0d.

Order: isolate -> matched correctness -> Track A task/replication screening and Track B weight/gradient lifetime screening -> independent repeats -> choose a mechanism -> conditional joint strategy.

Fixed: L4 EP8 E256 K8 H4096 F1024 S16384/rank, GA4 W2, FP32 persistent states, BF16 compute, FSDP8/E-FSDP1, fixed owners, common recv262144, whole-card90GiB. Native32home+32replica; two-layer64home+0replica.

Track A: serial_home, serial_native, fixed_home, fixed_native, fifo_home, easy_home. 3 patterns at MaxVio10, 25 updates with5 warmup.
Track B: original anchor, immediate uncached, immediate cached, window flush, step flush. Eight cache and gradient entries/rank. Identical copies/quota and FP32 external path for B0-B3.

No future routing, no external process termination, no old workspace mutation. Only numerical-qualified completed runs enter plots. Full matrix requires >=3% full-update improvement in3 independent runs.

Required gates: identity, actual SO capability, assignment conservation, cache version/epoch/lease, exact-once gradients, owner scaling once, output relativeL2<=.004 and each gradient<=.02, three Adam updates, <=90GiB. Failures remain recorded.
