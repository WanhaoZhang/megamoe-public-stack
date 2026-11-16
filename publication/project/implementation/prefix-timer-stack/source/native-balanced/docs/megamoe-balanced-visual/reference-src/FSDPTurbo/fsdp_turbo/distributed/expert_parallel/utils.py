# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import math
import os
from collections import OrderedDict

import torch


def normalize_expert_args(top_k_index, top_k_weights):
    """
    Ensure top_k_index is integer tensor (indices) and top_k_weights is float tensor (weights).
    Swap if necessary and adjust dimensions if needed.

    Args:
        top_k_index: Tensor that could be either indices or weights
        top_k_weights: Tensor that could be either weights or indices

    Returns:
        (correct_top_k_index, correct_top_k_weights)
    """
    # Swap if top_k_index is floating point (actually weights)
    if torch.is_floating_point(top_k_index):
        top_k_index, top_k_weights = top_k_weights, top_k_index

    # Ensure weights have the same shape as indices
    if top_k_weights.size() != top_k_index.size():
        # Gather weights using indices (assume top_k_weights has shape [batch_size, num_experts])
        # and top_k_index has shape [batch_size, top_k]
        top_k_weights = top_k_weights.gather(1, top_k_index)

    return top_k_index, top_k_weights


def _resolve_router_imbalance(imbalance: float | None) -> float:
    if imbalance is not None:
        return max(0.0, float(imbalance))
    return max(0.0, float(os.environ.get("ASCEND_ROUTER_IMBALANCE", "0") or "0"))


def _resolve_num_hot_experts(num_experts: int, num_hot: int | None = None) -> int:
    """How many hotspot experts for ρ>0 (production-like: a few hot, rest cold)."""
    if num_hot is not None:
        value = int(num_hot)
    else:
        try:
            # Default ≤2 hot experts on the hot card (card-level ρ metric).
            value = int(os.environ.get("ASCEND_ROUTER_NUM_HOT_EXPERTS", "2") or "2")
        except ValueError:
            value = 2
    # Need ≥1 cold expert so min>0 and ρ = max/min - 1 is well-defined.
    # Cap at 2 for the card-level imbalance experiments unless explicitly larger.
    return max(1, min(value, num_experts - 1))


def _resolve_router_distribution() -> str:
    """``rank_skew`` / ``hotspot`` / ``linspace``."""
    value = (os.environ.get("ASCEND_ROUTER_DISTRIBUTION", "hotspot") or "hotspot").strip().lower()
    if value in ("linspace", "linear", "ramp"):
        return "linspace"
    if value in ("rank_skew", "rank", "ep_skew"):
        return "rank_skew"
    return "hotspot"


def _rank_skew_counts(
    num_slots: int,
    num_experts: int,
    imbalance: float,
    ep_size: int,
    align: int,
    max_per_expert: int | None,
    num_hot: int | None = None,
) -> list[int]:
    """Card-level skew: ρ = max_rank_load / min_rank_load - 1.

    Contiguous ownership ``[r*local_e, (r+1)*local_e)``. Hot card is EP rank0;
    its hot experts are the first ``num_hot`` (≤2 by default). Remaining load
    is shared equally across cold ranks so the coldest card defines ``min``.
    """
    if ep_size <= 1 or num_experts % ep_size != 0:
        # Fall back to equal-cold hotspot on a single group.
        num_hot = _resolve_num_hot_experts(num_experts, num_hot)
        num_hot = min(num_hot, 2, max(1, num_experts - 1))
        ratio = 1.0 + float(imbalance)
        weights = [ratio] * num_hot + [1.0] * (num_experts - num_hot)
        return _distribute_by_weights(num_slots, weights, align=align)

    local_e = num_experts // ep_size
    # Hot experts on hot card: at most 2 (user metric), and ≤ local_e.
    num_hot = _resolve_num_hot_experts(local_e, num_hot)
    num_hot = max(1, min(num_hot, 2, local_e))
    ratio = 1.0 + float(imbalance)
    # Target: hot_card / cold_card = ratio, cold cards equal.
    # hot : each_cold = ratio : 1  => hot = S * ratio / (ratio + cold_ranks)
    cold_ranks = ep_size - 1
    r0 = int(round(num_slots * ratio / (ratio + cold_ranks)))
    r0 = _floor_align(r0, align) if align > 1 else r0
    r0 = max(align if align > 1 else 0, min(r0, num_slots))
    rem = num_slots - r0

    max_per = None
    if max_per_expert is not None:
        max_per = _floor_align(max_per_expert, align) or int(max_per_expert)

    counts = [0] * num_experts
    # Put hot-card load onto ≤num_hot experts; leftover on other local experts.
    hot_ids = list(range(num_hot))
    rank0_cold_ids = list(range(num_hot, local_e))
    if max_per is not None:
        hot_cap = min(max_per, r0)
        per_hot = _floor_align(r0 // max(num_hot, 1), align) if align > 1 else (r0 // max(num_hot, 1))
        per_hot = min(per_hot, hot_cap)
        for e in hot_ids:
            counts[e] = per_hot
        left = r0 - per_hot * num_hot
        fill_order = hot_ids + rank0_cold_ids
        for e in fill_order:
            if left <= 0:
                break
            room = (max_per - counts[e]) if max_per is not None else left
            if room <= 0:
                continue
            add = _floor_align(min(room, left), align) if align > 1 else min(room, left)
            if add <= 0 and left > 0 and room > 0:
                add = min(room, left)
            counts[e] += add
            left -= add
        rem += left
    else:
        # Prefer concentrating on the ≤2 hot experts (equal among them).
        if num_hot > 0:
            part = _distribute_by_weights(r0, [1.0] * num_hot, align=align)
            for i, e in enumerate(hot_ids):
                counts[e] = part[i] if i < len(part) else 0
        elif rank0_cold_ids:
            part = _distribute_by_weights(r0, [1.0] * len(rank0_cold_ids), align=align)
            for i, e in enumerate(rank0_cold_ids):
                counts[e] = part[i] if i < len(part) else 0

    # Split rem equally across experts on cold ranks.
    cold_ids = list(range(local_e, num_experts))
    if cold_ids and rem > 0:
        part = _distribute_by_weights(rem, [1.0] * len(cold_ids), align=align)
        for e, c in zip(cold_ids, part):
            if max_per is not None:
                c = min(c, max_per)
            counts[e] = c
        # Fix sum drift.
        drift = num_slots - sum(counts)
        cursor = 0
        donors = cold_ids + hot_ids + rank0_cold_ids
        while drift != 0 and donors:
            e = donors[cursor % len(donors)]
            if drift > 0:
                room = (max_per - counts[e]) if max_per is not None else drift
                if room > 0:
                    step = min(drift, room)
                    if align > 1 and step >= align:
                        step = _floor_align(step, align) or step
                    counts[e] += step
                    drift -= step
            else:
                if counts[e] > 0:
                    step = min(-drift, counts[e])
                    if align > 1 and step >= align and counts[e] - step >= 0:
                        step = _floor_align(step, align) or step
                    counts[e] -= step
                    drift += step
            cursor += 1
            if cursor > len(donors) * num_slots:
                break
    return counts


def _resolve_count_align(ep_size: int | None = None) -> int:
    """Optional local per-expert quota alignment for debug routers.

    Default is **1** (no align): recent ascend_deepep MegaMoe BF16 no longer
    requires host-side 128-row tile quantization of route quotas.

    Legacy behavior (``ep * align`` multiple of 128) can be restored with
    ``ASCEND_ROUTER_COUNT_ALIGN`` (e.g. ``64`` for ep=2) or
    ``ASCEND_ROUTER_COUNT_ALIGN=auto`` to derive ``128 / gcd(ep, 128)``.
    """
    env = os.environ.get("ASCEND_ROUTER_COUNT_ALIGN")
    if env is not None and str(env).strip() != "":
        value = str(env).strip().lower()
        if value in ("auto", "tile", "128"):
            tile = 128
            ep = int(ep_size) if ep_size is not None and int(ep_size) > 0 else 1
            return max(1, tile // math.gcd(ep, tile))
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 1


def _distribute_by_weights(
    num_slots: int,
    weights: list[float],
    align: int = 1,
) -> list[int]:
    align = max(1, int(align))
    # Distribute in ``align``-sized quanta so each expert quota (and therefore
    # ep*quota recv rows) stays on MegaMoe's 128-row tile boundary.
    if align > 1 and num_slots >= align * len(weights) and num_slots % align == 0:
        unit_slots = num_slots // align
        unit_counts = _distribute_by_weights(unit_slots, weights, align=1)
        return [c * align for c in unit_counts]

    total_w = sum(weights)
    raw = [w / total_w * num_slots for w in weights]
    counts = [int(x) for x in raw]
    rem = num_slots - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - counts[i], i), reverse=True)
    for i in range(rem):
        counts[order[i % len(weights)]] += 1
    if num_slots >= len(weights) and min(counts) == 0:
        rich = sorted(range(len(weights)), key=lambda i: counts[i], reverse=True)
        for e in range(len(weights)):
            if counts[e] == 0:
                donor = next(i for i in rich if counts[i] > 1)
                counts[donor] -= 1
                counts[e] += 1
    return counts


def _floor_align(value: int, align: int) -> int:
    align = max(1, int(align))
    return (int(value) // align) * align


def _hotspot_counts_with_cap(
    num_slots: int,
    num_experts: int,
    imbalance: float,
    num_hot: int,
    align: int,
    max_per_expert: int,
) -> list[int]:
    """Hotspot quotas under per-expert packing cap (≤ one hit per token).

    When equal-cold ``hot=(1+ρ)*cold`` would exceed ``max_per_expert``, saturate
    the ``num_hot`` hotspots at the cap and set the cold-tail minimum to
    ``hot/(1+ρ)`` so target ρ remains realizable (other cold experts absorb the
    remainder). This matches production-like "few hot, long cold tail".
    """
    align = max(1, int(align))
    max_per = _floor_align(max_per_expert, align)
    if max_per <= 0:
        max_per = max_per_expert
    num_cold = num_experts - num_hot
    ratio = 1.0 + float(imbalance)

    # Ideal equal-cold plan (ignore cap first).
    weights = [ratio] * num_hot + [1.0] * num_cold
    ideal = _distribute_by_weights(num_slots, weights, align=align)
    if max(ideal) <= max_per:
        return ideal

    # Saturate hotspots; realize ρ via cold-tail minimum.
    hot = max_per
    rem = num_slots - num_hot * hot
    if rem < 0:
        # Degenerate: shrink hot equally (should not happen when slots=N*K, hot≤N).
        return _distribute_by_weights(num_slots, weights, align=align)

    cold_min = max(align, _floor_align(int(hot / ratio), align))
    if cold_min * num_cold > rem:
        cold_min = max(align, _floor_align(rem // num_cold, align)) if rem >= align * num_cold else max(1, rem // num_cold)

    # One (or more) cold experts sit at cold_min (= global min); dump remainder
    # onto the other cold experts without exceeding hot.
    counts = [hot] * num_hot + [0] * num_cold
    cold_ids = list(range(num_hot, num_experts))
    # Prefer a single min-tail expert when possible for a sharp ρ.
    n_min = 1 if num_cold >= 1 else 0
    while n_min < num_cold and rem - n_min * cold_min > (num_cold - n_min) * hot:
        n_min += 1
    for i, e in enumerate(cold_ids):
        counts[e] = cold_min if i < n_min else 0
    rem_left = rem - n_min * cold_min
    rich = cold_ids[n_min:]
    if not rich:
        rich = cold_ids
        rem_left = rem - sum(counts[e] for e in cold_ids)
    # Distribute rem_left onto rich cold experts in align quanta.
    if rich and rem_left > 0:
        if align > 1 and rem_left % align == 0:
            units = rem_left // align
            base, extra = divmod(units, len(rich))
            for i, e in enumerate(rich):
                add = (base + (1 if i < extra else 0)) * align
                counts[e] = min(hot, counts[e] + add)
        else:
            base, extra = divmod(rem_left, len(rich))
            for i, e in enumerate(rich):
                counts[e] = min(hot, counts[e] + base + (1 if i < extra else 0))
        # Fix sum drift from hot caps.
        drift = num_slots - sum(counts)
        cursor = 0
        while drift != 0 and rich:
            e = rich[cursor % len(rich)]
            if drift > 0 and counts[e] < hot:
                step = align if drift >= align and (hot - counts[e]) >= align else 1
                step = min(step, drift, hot - counts[e])
                counts[e] += step
                drift -= step
            elif drift < 0 and counts[e] > cold_min:
                step = align if (-drift) >= align and (counts[e] - cold_min) >= align else 1
                step = min(step, -drift, counts[e] - cold_min)
                counts[e] -= step
                drift += step
            else:
                cursor += 1
                if cursor > len(rich) * hot:
                    break
                continue
            cursor += 1
    return counts


def _target_expert_counts(
    num_slots: int,
    num_experts: int,
    imbalance: float,
    num_hot: int | None = None,
    align: int = 1,
    max_per_expert: int | None = None,
    ep_size: int | None = None,
) -> list[int]:
    """Integer route counts with max/min - 1 ~= imbalance.

    Distributions (``ASCEND_ROUTER_DISTRIBUTION``):
      * ``rank_skew``: ρ is **ρ_card** = max_card/min_card - 1; hot card=rank0 (≤2 hot).
      * ``hotspot``: few hot experts at ``(1+ρ)*cold``.
      * ``linspace``: legacy ramp.

    ``align`` quantizes quotas (see ``_resolve_count_align``) so MegaMoe's
    per-expert recv segments remain 128-row aligned.
    ``max_per_expert`` (usually ``num_tokens``) caps each expert under unique
    top-k packing.
    """
    if num_experts <= 1 or num_slots <= 0:
        return [num_slots] if num_experts == 1 else []
    if imbalance <= 0.0:
        base, rem = divmod(num_slots, num_experts)
        return [base + (1 if i < rem else 0) for i in range(num_experts)]

    dist = _resolve_router_distribution()
    if dist == "linspace":
        # L[i] = 1 + ρ * i / (E-1)  =>  max/min = 1 + ρ
        weights = [1.0 + float(imbalance) * (i / (num_experts - 1)) for i in range(num_experts)]
        counts = _distribute_by_weights(num_slots, weights, align=align)
        if max_per_expert is not None:
            # Soft clamp only; linspace cannot always preserve ρ under a hard cap.
            max_per = _floor_align(max_per_expert, align) or max_per_expert
            overflow = sum(max(0, c - max_per) for c in counts)
            counts = [min(c, max_per) for c in counts]
            if overflow:
                for i in range(len(counts)):
                    if overflow <= 0:
                        break
                    room = max_per - counts[i]
                    if room <= 0:
                        continue
                    add = min(room, overflow)
                    counts[i] += add
                    overflow -= add
        return counts

    if dist == "rank_skew":
        ep = _resolve_ep_size(ep_size, num_experts)
        return _rank_skew_counts(
            num_slots,
            num_experts,
            imbalance,
            ep_size=ep,
            align=align,
            max_per_expert=max_per_expert,
            num_hot=num_hot,
        )

    num_hot = _resolve_num_hot_experts(num_experts, num_hot)
    if max_per_expert is not None:
        return _hotspot_counts_with_cap(
            num_slots, num_experts, imbalance, num_hot, align, max_per_expert,
        )
    num_cold = num_experts - num_hot
    ratio = 1.0 + float(imbalance)
    weights = [ratio] * num_hot + [1.0] * num_cold
    return _distribute_by_weights(num_slots, weights, align=align)


def _resolve_ep_size(ep_size: int | None, num_global_experts: int) -> int:
    if ep_size is not None and int(ep_size) > 0:
        return int(ep_size)
    env = os.environ.get("ASCEND_EP_SIZE") or os.environ.get("PARALLEL_EP_SIZE")
    if env:
        try:
            value = int(env)
            if value > 0:
                return value
        except ValueError:
            pass
    # Contiguous expert ownership requires E % ep == 0; prefer largest feasible ep.
    for candidate in (8, 4, 2):
        if num_global_experts % candidate == 0:
            return candidate
    return 1


def _rebalance_counts_across_ep(counts: list[int], ep_size: int) -> list[int]:
    """Optional: pair heavy/light quotas so each EP rank gets ~equal routes.

    Default hotspot placement intentionally concentrates hot experts on the
    first EP ranks. Enable with ``ASCEND_ROUTER_REBALANCE_EP=1`` only when
    ablating inter-rank skew while keeping per-expert ρ.
    """
    num_experts = len(counts)
    if ep_size <= 1 or num_experts % ep_size != 0:
        return counts
    local_e = num_experts // ep_size
    order = sorted(range(num_experts), key=lambda i: (counts[i], i), reverse=True)
    rank_load = [0] * ep_size
    rank_slots: list[list[int]] = [[] for _ in range(ep_size)]
    for src in order:
        candidates = [r for r in range(ep_size) if len(rank_slots[r]) < local_e]
        rank = min(candidates, key=lambda r: (rank_load[r], r))
        rank_slots[rank].append(src)
        rank_load[rank] += counts[src]
    rebalanced = [0] * num_experts
    for rank in range(ep_size):
        # Stable slot order: lighter quota first within the rank.
        chosen = sorted(rank_slots[rank], key=lambda i: (counts[i], i))
        for slot, src in enumerate(chosen):
            rebalanced[rank * local_e + slot] = counts[src]
    return rebalanced


def _router_rebalance_ep_enabled() -> bool:
    return os.environ.get("ASCEND_ROUTER_REBALANCE_EP", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _imbalance_route_cache_enabled() -> bool:
    return os.environ.get("ASCEND_ROUTER_IMBALANCE_CACHE", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _imbalance_route_return_clone() -> bool:
    return os.environ.get("ASCEND_ROUTER_IMBALANCE_RETURN_CLONE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


# imbalance>0 assignment depends only on shape and router settings, not on the
# live gate output. Layers share one table; cached device tensors are read-only.
_IMBALANCE_ROUTE_CPU: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_IMBALANCE_ROUTE_DEVICE: OrderedDict[tuple, torch.Tensor] = OrderedDict()
_IMBALANCE_ROUTE_CACHE_MAX = max(1, int(os.environ.get("ASCEND_ROUTER_IMBALANCE_CACHE_MAX", "8") or "8"))


def clear_imbalance_route_cache() -> None:
    """Drop cached imbalance route tables (tests / dynamic env changes)."""
    _IMBALANCE_ROUTE_CPU.clear()
    _IMBALANCE_ROUTE_DEVICE.clear()


def _cache_put(cache: OrderedDict, key: tuple, value: torch.Tensor) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _IMBALANCE_ROUTE_CACHE_MAX:
        cache.popitem(last=False)


def _imbalance_plan_key(
    num_tokens: int,
    top_k: int,
    num_global_experts: int,
    imbalance: float,
    ep_size: int | None,
    num_hot_experts: int | None,
) -> tuple:
    return (
        int(num_tokens),
        int(top_k),
        int(num_global_experts),
        float(imbalance),
        None if ep_size is None else int(ep_size),
        None if num_hot_experts is None else int(num_hot_experts),
        _resolve_router_distribution(),
        _router_rebalance_ep_enabled(),
        os.environ.get("ASCEND_ROUTER_NUM_HOT_EXPERTS", ""),
        os.environ.get("ASCEND_ROUTER_COUNT_ALIGN", ""),
        os.environ.get("ASCEND_EP_SIZE", ""),
        os.environ.get("PARALLEL_EP_SIZE", ""),
    )


def _build_imbalance_assignments(
    num_tokens: int,
    top_k: int,
    num_global_experts: int,
    imbalance: float,
    ep_size: int | None,
    num_hot_experts: int | None,
) -> torch.Tensor:
    """Deterministic quota placement. CPU int64, built once per cache key."""
    ep = _resolve_ep_size(ep_size, num_global_experts)
    align = _resolve_count_align(ep)
    counts = _target_expert_counts(
        num_tokens * top_k,
        num_global_experts,
        imbalance,
        num_hot=num_hot_experts,
        align=align,
        max_per_expert=num_tokens,
        ep_size=ep,
    )
    if _router_rebalance_ep_enabled():
        counts = _rebalance_counts_across_ep(counts, ep)

    assignments: list[list[int]] = [[] for _ in range(num_tokens)]
    placed_counts = [0] * num_global_experts
    token_ptr = 0
    budget = num_tokens * top_k * 2
    for expert_id, quota in enumerate(counts):
        placed = 0
        attempts = 0
        while placed < quota and attempts < budget:
            t = token_ptr % num_tokens
            token_ptr += 1
            attempts += 1
            row = assignments[t]
            if len(row) >= top_k or expert_id in row:
                continue
            row.append(expert_id)
            placed += 1
            placed_counts[expert_id] += 1
    for row in assignments:
        if len(row) >= top_k:
            continue
        need = sorted(
            range(num_global_experts),
            key=lambda e: (placed_counts[e] - counts[e], placed_counts[e], e),
        )
        for expert in need:
            if len(row) >= top_k:
                break
            if expert in row:
                continue
            row.append(expert)
            placed_counts[expert] += 1
    return torch.tensor(assignments, dtype=torch.int64)


def _cached_imbalance_index(
    num_tokens: int,
    top_k: int,
    num_global_experts: int,
    imbalance: float,
    ep_size: int | None,
    num_hot_experts: int | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    plan_key = _imbalance_plan_key(
        num_tokens,
        top_k,
        num_global_experts,
        imbalance,
        ep_size,
        num_hot_experts,
    )
    plan = _IMBALANCE_ROUTE_CPU.get(plan_key)
    if plan is None:
        plan = _build_imbalance_assignments(
            num_tokens,
            top_k,
            num_global_experts,
            imbalance,
            ep_size,
            num_hot_experts,
        )
        _cache_put(_IMBALANCE_ROUTE_CPU, plan_key, plan)
    else:
        _IMBALANCE_ROUTE_CPU.move_to_end(plan_key)

    device_key = plan_key + (str(device), str(dtype))
    resident = _IMBALANCE_ROUTE_DEVICE.get(device_key)
    if resident is None:
        resident = plan.to(device=device, dtype=dtype)
        _cache_put(_IMBALANCE_ROUTE_DEVICE, device_key, resident)
    else:
        _IMBALANCE_ROUTE_DEVICE.move_to_end(device_key)
    if _imbalance_route_return_clone():
        return resident.clone()
    return resident


def fixed_router_for_debug(
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    num_global_experts: int,
    imbalance: float | None = None,
    ep_size: int | None = None,
    num_hot_experts: int | None = None,
):
    """
    Debug-only helper that replaces router output with a deterministic route plan.

    ``imbalance`` meaning depends on ``ASCEND_ROUTER_DISTRIBUTION``:
      * ``rank_skew`` (preferred for Hybrid vs MegaMoe): **ρ_card** =
        max_card_load/min_card_load - 1; hot card = EP rank0 with ≤2 hot experts.
      * ``hotspot``: ρ = max_tokens/min_tokens - 1 over experts; few hotspots
        at ``(1+ρ)*cold`` (``ASCEND_ROUTER_NUM_HOT_EXPERTS``).
      * 0 -> fully balanced (legacy round-robin).

    When ``imbalance`` is None, read ``ASCEND_ROUTER_IMBALANCE`` (default 0).
    ``ASCEND_ROUTER_REBALANCE_EP=1`` optionally redistributes quotas across EP
    ranks (off by default — card skew is intentional for Hybrid vs MegaMoe).
    ``ASCEND_ROUTER_COUNT_ALIGN`` defaults to 1 (no host 128-align); set
    ``auto``/``64`` only for legacy MegaMoe tile constraints.

    When ``imbalance`` > 0 the assignment depends only on token count, top-k,
    expert count and router env. It is cached across layers and steps
    (``ASCEND_ROUTER_IMBALANCE_CACHE=0`` disables). Dispatchers copy dtype
    after this call; the default is to return a shared read-only cache tensor
    (``ASCEND_ROUTER_IMBALANCE_RETURN_CLONE=1`` for a per-call clone).
    """
    imbalance = _resolve_router_imbalance(imbalance)
    is_1d = top_k_index.dim() == 1
    num_tokens = top_k_index.shape[0]
    top_k = 1 if is_1d else top_k_index.shape[1]
    device = top_k_index.device
    dtype = top_k_index.dtype

    with torch.no_grad():
        if imbalance <= 0.0 or num_global_experts <= 1 or top_k <= 0:
            base = torch.arange(num_tokens, device=device, dtype=dtype)
            offset = torch.arange(top_k, device=device, dtype=dtype)
            fixed_index = (base.unsqueeze(1) + offset.unsqueeze(0)) % num_global_experts
        else:
            if top_k > num_global_experts:
                raise ValueError(
                    f"fixed_router imbalance requires top_k ({top_k}) <= num_experts ({num_global_experts})."
                )
            if _imbalance_route_cache_enabled():
                fixed_index = _cached_imbalance_index(
                    num_tokens,
                    top_k,
                    num_global_experts,
                    imbalance,
                    ep_size,
                    num_hot_experts,
                    device,
                    dtype,
                )
            else:
                fixed_index = _build_imbalance_assignments(
                    num_tokens,
                    top_k,
                    num_global_experts,
                    imbalance,
                    ep_size,
                    num_hot_experts,
                ).to(device=device, dtype=dtype)

        if is_1d:
            fixed_index = fixed_index.squeeze(1)
        fixed_weights = torch.full_like(top_k_weights, 1.0 / top_k)

    return fixed_index, fixed_weights
