# Candidate: reuse original plan's gradient obligations

Status: design/CPU semantic proof, not an implemented or timed training policy.

The complete-update objective remains task selection, replica residence and gradient delivery. This candidate is common interface plumbing, not a claimed paper contribution. The existing negative A/B pilots remain unchanged.

Evidence: all 576 saved plans from the old forced-replica capture, plus 2304 records from the first four new small captures, exactly reproduce native execution counts from current copies and source quota. No future routing is used. New main captures will independently check all records.

Hypothesis: checkpoint replay and backward need not observe copies/counts on the host and exchange contributor masks again if the validated original immutable PlanTicket carries those obligations. Copy/quota observation once at original compilation is still online cost; this is not precomputed-plan replay timing.

Prototype boundary: isolated child source branch only. Metadata must be bound to a live original ticket, checked against version and immutable body identities, and retired with that ticket. Never cache mutable planner scratch pointers or infer task identity from allocation reuse. Source quota must retain source rank. Zero-work copies have no gradient obligation. CPU reference must reject duplicate copies, unavailable destinations, negative quotas and receive-capacity overflow.

Go/no-go before any timing: (1) main timeline demonstrates an exposed observation/launch bottleneck; (2) original/replay/backward binding and stale/mutated-ticket rejection pass; (3) independent three-update FP32 gradient comparison across W1/W2 and positive replicas passes; (4) actual counts match predicted counts in qualification, with comparison disabled only for explicitly identified TimerOFF performance experiments; (5) matching reference and candidate share model, routes, SO, path and resource budget.

Formal performance still includes the initial online compilation/observation cost. Any improvement is opened to all applicable existing baselines before evaluating a new policy. If the observation merely waits for unavoidable compute and removing it does not shorten the update, retain that negative result and stop this candidate.
