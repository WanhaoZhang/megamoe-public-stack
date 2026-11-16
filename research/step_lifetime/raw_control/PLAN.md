# Preserve native compute schedule while exporting raw dW

Independent ordinary MegaMoE child branch from research/step-lifetime@6e7dd58. The parent pilot and causal-v1 source/results stay read-only. No new policy or positive result is assumed.

Confirmed issue: legacy raw dW sets gradient_.ready=0. That also switches ExpertAt, prefix bookkeeping and eligibility of native early/cooperative scheduling. We need an independent raw-output flag to isolate external gradient delivery from computation order. Preserve gradient ready publication while disabling native owner consumption, direct replica raw outputs to torch-owned dW, and clear empty outputs safely.

Private ABI2 supports both sidecar7=1 (legacy) and sidecar7=2 (preserved schedule) in the SAME new SO. Compare these modes on identical copies/quota, numerical data, owners and budgets. Old ABI1 may be used only with legacy mode. Main current worker forward/back flags remain0; unsupported cooperative variants must reject raw-preserve until explicitly implemented and qualified.

Sequence: source change -> all raw/owner/release call-site audit -> CPU contracts -> isolated new build -> forced-replica output/dx/dp/dW gates with both schedules and empty experts -> >=3 full-model updates with matched gradients -> bounded same-shape paired diagnostic/performance. Never submit TimerON time as performance. Device admission uses the shared lock plus actual ownership checks. No other task's processes, SOs or paths are changed.

Success is correctness plus evidence of which schedule dependency changes. Any speedup is common executor groundwork available to all B lifecycle strategies, not a claimed new planner contribution. The full research goal remains online joint task/replica/gradient lifetime optimization.
