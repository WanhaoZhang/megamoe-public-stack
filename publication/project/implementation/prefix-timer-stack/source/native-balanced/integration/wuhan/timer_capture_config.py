"""CPU-only shape/routing contract for the portable diagnostic entry."""
import math


def validate(world, ep, tokens, hidden, ffn, experts, topk, slots, capacity, rho, hot):
    # Source/partition coordinate schema6 has been qualified on small EP only.
    if ep not in (2, 4, 8) or world not in (2, 4, 8) or world % ep:
        raise ValueError('single-node world2/4/8 and EP2/4/8 dividing world required')
    if experts % ep or experts <= 0 or experts // ep > 16:
        raise ValueError('experts must be divisible by EP and home<=16')
    home = experts // ep
    if slots != home or not 0 < topk <= home or not topk <= hot <= home:
        raise ValueError('synthetic rank_skew requires slots=home and TopK<=hot_experts<=home')
    if tokens <= 0 or hidden <= 0 or ffn <= 0 or hidden % 256 or ffn % 256:
        raise ValueError('positive M and H/F divisible by256 required')
    if capacity < tokens * topk or capacity % 128:
        raise ValueError('capacity must cover M*TopK and be aligned to128')
    if not math.isfinite(rho) or rho < 0:
        raise ValueError('rho must be finite and nonnegative')


def rank_skew_ids(tokens, ep, home, topk, rho, hot_experts):
    """Deterministic unique TopK; rho=hot-card/cold-card load - 1 before rounding.

    This standalone generator has its own recorded quota rule; use a capture
    for bit-identical reproduction of a previous framework router invocation.
    """
    weights = [rho + 1] + [1.] * (ep - 1)
    quotas = [tokens * w / sum(weights) for w in weights]
    counts = [int(q) for q in quotas]
    order = sorted(range(ep), key=lambda i: (-(quotas[i] - counts[i]), i))
    for i in order[:tokens - sum(counts)]:
        counts[i] += 1
    ids = []
    for rank, count in enumerate(counts):
        width = hot_experts if rank == 0 else home
        ids.extend([[rank * home + (row * topk + k) % width for k in range(topk)]
                    for row in range(count)])
    return ids, counts


def window_bytes(tokens, hidden, topk, ep, physical_experts, capacity):
    # Mirror CalcPeermemSize + BF16 early topk area; 2MiB large-page alignment.
    align = lambda n, a: (n + a - 1) // a * a
    size = 72 * 1024 + ep * align(ep * physical_experts, 128) * 4
    size += max(tokens * topk, capacity) * hidden * 4
    size += align(tokens * topk * 2, 512)
    return max(256 * 1024**2, align(size, 2 * 1024**2))
