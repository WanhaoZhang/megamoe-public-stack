# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Benchmark-only BF16 original-X redispatch for a checked MoonEP placement.

Adapted from FSDPTurbo's ``mega_moe_redispatch._recompute_dispatch_segment``.
The caller passes *physical* IDs, with ``local_experts=home+replica_slots``.
Logical IDs would silently put replica tokens back on their home ranks.

This eager reference intentionally recomputes histogram, exchanges counts,
materializes host splits, sorts, transfers tokens/probabilities, reorders and
packs on every invocation. There is no fixed-input metadata cache and no
comm-stream overlap. All of this work belongs inside the complete-F/B timer.
It does not implement or claim a kernel-fused original-X redispatch.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path


def dispatch_output_layout(hidden, capacity):
    """Match MegaMoeBF16Training::DispatchOutputLayout's 512-byte alignment."""
    if type(hidden) is not int or type(capacity) is not int or min(hidden, capacity) <= 0:
        raise ValueError('positive integer hidden and capacity required')
    align = lambda n: (n + 511) // 512 * 512
    tokens_bytes, weights_bytes = align(capacity * hidden * 2), align(capacity * 2)
    return dict(tokens_offset=0, tokens_bytes=tokens_bytes,
                weights_offset=tokens_bytes, weights_bytes=weights_bytes,
                total_bytes=tokens_bytes + weights_bytes)


@dataclass(frozen=True)
class SplitMetadata:
    input_splits: tuple
    output_splits: tuple
    source_expert_chunks: tuple
    expert_major_chunk_order: tuple
    actual_rows: int


def resolve_split_metadata(counts_by_source, local_experts, rank, capacity):
    """Pure-host collective geometry; every rank checks every destination.

    Counts are [source, physical_global_expert]. All participants already have
    the same gathered counts, so an overflowing destination is rejected on all
    ranks before either payload all-to-all, without another collective.
    """
    ep = len(counts_by_source)
    if (not ep or type(local_experts) is not int or local_experts <= 0
            or type(rank) is not int or not 0 <= rank < ep
            or type(capacity) is not int or capacity <= 0):
        raise ValueError('invalid EP/rank/expert/capacity geometry')
    global_e = ep * local_experts
    if any(len(row) != global_e or any(type(v) is not int or v < 0 for v in row)
           for row in counts_by_source):
        raise ValueError('counts must be nonnegative [EP,EP*local_experts] integers')
    destination_rows = [sum(sum(row[d*local_experts:(d+1)*local_experts])
                            for row in counts_by_source) for d in range(ep)]
    if max(destination_rows) > capacity:
        raise RuntimeError('redispatch receive capacity exceeded on at least one EP rank: '
                           f'demand={destination_rows}, capacity={capacity}')
    current = counts_by_source[rank]
    input_splits = tuple(sum(current[d*local_experts:(d+1)*local_experts]) for d in range(ep))
    chunks = tuple(value for row in counts_by_source
                   for value in row[rank*local_experts:(rank+1)*local_experts])
    output_splits = tuple(sum(chunks[s*local_experts:(s+1)*local_experts]) for s in range(ep))
    order = tuple(s*local_experts + e for e in range(local_experts) for s in range(ep))
    return SplitMetadata(input_splits, output_splits, chunks, order, destination_rows[rank])


def recompute_dispatch_segment(x, physical_ids, prob, *, local_experts, ep_group, max_recv):
    """Return (packed UInt8 dispatch_output, actual_rows) on the current stream.

    Requires certified valid physical routes from this forward's immutable
    placement. TopK IDs within a token are distinct, as in the benchmark.
    No gradient is recorded for this backward reconstruction operation.
    """
    import torch
    import torch.distributed as dist

    if (x.ndim != 2 or physical_ids.ndim != 2 or physical_ids.shape != prob.shape
            or x.shape[0] != physical_ids.shape[0] or x.dtype != torch.bfloat16
            or prob.dtype != torch.bfloat16 or physical_ids.dtype not in (torch.int32, torch.int64)
            or x.device != physical_ids.device or x.device != prob.device):
        raise ValueError('expected colocated BF16 X/prob and integer [M,K] physical routes')
    ep, rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    m, hidden = x.shape
    topk = physical_ids.shape[1]
    num_out, global_e = m * topk, ep * local_experts
    if num_out <= 0 or global_e * num_out >= 2**24:
        raise ValueError('float32 stable composite-sort keys require 0 < EP*width*M*TopK < 2**24')
    layout = dispatch_output_layout(hidden, max_recv)
    with torch.no_grad():
        flat = physical_ids.reshape(-1).long()
        # Integer argsort can fall back to AiCpu on NPU. Composite float32 keys
        # are exact under the guard, and preserve token/TopK order per expert.
        key = (flat * num_out + torch.arange(num_out, device=x.device)).to(torch.float32)
        order = torch.argsort(key)
        permuted = x.index_select(0, order // topk)
        carry = prob.reshape(-1).index_select(0, order)
        counts = torch.histc(flat.to(torch.float32), bins=global_e, min=0, max=global_e).long()
        if ep > 1:
            gathered = [torch.empty_like(counts) for _ in range(ep)]
            dist.all_gather(gathered, counts, group=ep_group)
            host_counts = torch.stack(gathered).cpu().tolist()
        else:
            host_counts = [counts.cpu().tolist()]
        metadata = resolve_split_metadata(host_counts, local_experts, rank, max_recv)
        if ep > 1:
            recv = x.new_empty((metadata.actual_rows, hidden))
            carry_recv = prob.new_empty(metadata.actual_rows)
            dist.all_to_all_single(recv, permuted,
                output_split_sizes=list(metadata.output_splits),
                input_split_sizes=list(metadata.input_splits), group=ep_group)
            dist.all_to_all_single(carry_recv, carry,
                output_split_sizes=list(metadata.output_splits),
                input_split_sizes=list(metadata.input_splits), group=ep_group)
            recv_chunks = torch.split(recv, metadata.source_expert_chunks, dim=0)
            carry_chunks = torch.split(carry_recv, metadata.source_expert_chunks)
            recv = torch.cat([recv_chunks[i] for i in metadata.expert_major_chunk_order], dim=0)
            carry = torch.cat([carry_chunks[i] for i in metadata.expert_major_chunk_order])
        else:
            recv = permuted
        # Match the framework reference's explicit zero padding and packed
        # copies. Only actual rows are read by the kernel; padding is not part
        # of the saved-forward bitwise comparison.
        if metadata.actual_rows < max_recv:
            recv = torch.cat((recv, x.new_zeros((max_recv-metadata.actual_rows, hidden))), dim=0)
            carry = torch.cat((carry, prob.new_zeros(max_recv-metadata.actual_rows)))
        buf = torch.empty(layout['total_bytes'], dtype=torch.uint8, device=x.device)
        buf[:max_recv*hidden*2].view(torch.bfloat16).view(max_recv, hidden).copy_(recv)
        start = layout['weights_offset']
        buf[start:start+max_recv*2].view(torch.bfloat16).copy_(carry)
        return buf, metadata.actual_rows


def verify_dispatch_output(saved, rebuilt, counts, hidden, capacity):
    """Untimed verification of initialized token/prob prefixes, with hashes."""
    import torch
    layout = dispatch_output_layout(hidden, capacity)
    actual = int(counts.sum().item())
    if not 0 <= actual <= capacity:
        raise ValueError('invalid actual dispatch row count')
    for value in (saved, rebuilt):
        if value.dtype != torch.uint8 or value.numel() < layout['total_bytes'] or not value.is_contiguous():
            raise ValueError('invalid packed dispatch buffer')
    tok_bytes, weight_bytes = actual*hidden*2, actual*2
    start = layout['weights_offset']
    old_parts = (saved[:tok_bytes], saved[start:start+weight_bytes])
    new_parts = (rebuilt[:tok_bytes], rebuilt[start:start+weight_bytes])
    if not all(torch.equal(a, b) for a, b in zip(old_parts, new_parts)):
        raise AssertionError('original-X redispatch token/prob initialized prefix differs from saved forward')
    digest = lambda p: hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
    return dict(pass_=True, actual_rows=actual, capacity=capacity,
                token_prefix_sha256=digest(old_parts[0]),
                probability_prefix_sha256=digest(old_parts[1]),
                padding_compared=False)


def reference_identity():
    path = Path(__file__).resolve()
    return dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                physical_routes=True, fixed_split_cache=False, overlap=False,
                algorithm='histogram+count_all_gather+host_splits+sort+token_a2a+prob_a2a+reorder+pack',
                scope='external original-X redispatch; native dY dispatch remains inside backward')
