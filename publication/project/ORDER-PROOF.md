# Why key caching preserves the original execution priority

For one legal invocation, let `g(e,P)` be GradientKey for physical expert `e` and immutable placement table `P`. It reads no token counters, ready flags, delivery epochs, gradient values or RNG. Empty slots map to100000+e; valid slots use their logical identity and membership in the fixed replica table. Current legal layouts have at most64physical experts and logical IDs below the sentinel range.

Both implementations initialize order[e]=e. In original insertion-sort iteration i, the condition is g(order[j],P)>g(v,P). The candidate precomputes priority[e]=g(e,P) for every e. Its condition is priority[order[j]]>priority[v], which is identical whenever P is unchanged. By induction, every shift and final insertion position is identical; strict greater-than retains the same tie order. The existing expert/row prefix and ready protocol are untouched.

The premise is the common plan/executor contract: this call's checked placement stays unchanged until its consuming serial stream has finished. Planner scratch may be reused only in stream order after that consumer. Checkpoint/backward bind the saved immutable placement. The change does not cache polling reads from ready/epoch locations. Invalid concurrent mutation is not a supported plan and is not made valid by this optimization.

The384actual-plan integer oracle is supporting evidence of correct encoding and sentinels, not the general proof. NPU output/dx/dp/dW and three-update qualification separately check implementation behavior. Build audits compare the real baseline and candidate source/flags. Reduced source-level table tests are not a predicted or measured wall-time gain: register/stack allocation and compiler behavior can change and must be measured.
