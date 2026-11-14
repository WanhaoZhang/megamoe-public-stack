# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MegaMoe BF16 dispatch_output 框架侧重算（re-dispatch）。

前向只保留 training_saved_state（路由元数据 + gate/up 等，save_gmm1_output
默认 bf16），丢弃独立的 dispatch_output（packed token + packed TopK weight，
定版形状约 896MiB/层/卡）；backward 时在 HCCL all_to_all_single 上重放
hidden -> permute -> a2a -> 按本地专家/源 rank 重排，重建与算子前向 ptrA
行序逐位一致的 dispatch_output 喂给 raw backward —— 算子 host 侧只校验
dtype/bytes 不问来源，算子仓零改动（无需重编 .so）。

行序契约（与算子前向 ptrA 一致；参考 MindSpeed_7941 megamoe-recompute-muonfix
的 bit-verified 实现，``ASCEND_DEEPEP_MEGAMOE_REDISPATCH_VERIFY=1`` 可在线验证）：
  行 = 本地专家升序分段；段内 = 源 rank 升序；rank 内 = token 升序。
  dropless 下 (token, expert) 唯一，topk 槽位序不影响行序。

与共享专家 backward 的重叠（通信掩盖计算）：modeling 侧在
DeepseekV4SparseMoeBlock 输出（routed + shared_out）外包 _RedispatchLauncher
（identity Function）。该节点是 block 输出节点、反向最先执行：在专用 comm
stream 上发射重算 a2a；随后共享专家分支的 backward（计算流）与 a2a 并行；
experts 的 _MegaMoeRedispatchFunction.backward 以 wait_event 消费重算结果。
launcher 缺席（旧 modeling / 无 hook）时 backward 在当前流串行自算，行为退化
为参考实现的串行重算，正确性不变。

降级保护：torch checkpoint（recompute）下 op forward 反正会在 backward 里
重放，重算是纯浪费 —— ``megamoe_redispatch_active()`` 检测
is_checkpointing/is_recomputing 自动回退旧路径（算子封装 op(...)）。
"""

from fsdp_turbo.utils.env import native_env, launcher_env, ensure_shmem_session

import torch
import torch.distributed as dist

_REDISPATCH_ENV = "ASCEND_DEEPEP_MEGAMOE_REDISPATCH"
_VERIFY_ENV = "ASCEND_DEEPEP_MEGAMOE_REDISPATCH_VERIFY"
# 通信掩盖（comm stream launcher）开关。历史挂死根因已定位并修复：launcher 曾把
# dispatch buffer 以裸 Tensor 存入 cell["outputs"]，消费端按 (buf, actual) 二元组
# unpack 触发 Tensor.__iter__ -> self.cpu() 整块 D2H（353MB/块），backward 自旋在
# 拷贝内且宿主 RSS 膨胀至 100GB+（164 py-spy 现场）。修复后仍保留 env 门控，
# 验证轮通过前默认串行重算。
_OVERLAP_ENV = "ASCEND_DEEPEP_MEGAMOE_REDISPATCH_OVERLAP"
_ALIGN_512 = 512
_SIZE_BF16 = 2

# per-device comm stream（domino_dispatcher 同款缓存模式）
_COMM_STREAMS = {}


def _get_comm_stream():
    device_index = torch.accelerator.current_device_index()
    stream = _COMM_STREAMS.get(device_index)
    if stream is None:
        stream = torch.accelerator.Stream()
        _COMM_STREAMS[device_index] = stream
    return stream


def megamoe_redispatch_active():
    """特性开关：env 开启，且当前不在 torch checkpoint 的前向/重放中。"""
    if native_env("ASCEND_DEEPEP_MEGAMOE_REDISPATCH") != "1":
        return False
    try:
        from torch.utils.checkpoint import is_checkpointing, is_recomputing

        if is_checkpointing() or is_recomputing():
            return False
    except ImportError:
        pass
    return True


def _verify_enabled():
    return native_env("ASCEND_DEEPEP_MEGAMOE_REDISPATCH_VERIFY") == "1"


def megamoe_redispatch_overlap_enabled():
    """通信掩盖开关：默认关闭（串行重算），见 _OVERLAP_ENV 注释。"""
    return native_env("ASCEND_DEEPEP_MEGAMOE_REDISPATCH_OVERLAP") == "1"


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment



# Host metadata only; bounded cache retains no device tensors. This opt-in
# requires identical fixed-router token/chunk shapes on every EP rank.
_FIXED_SPLIT_CACHE = {}


def _resolve_split_metadata(counts, local_e, ep_group, ep_size, cache_key=None):
    ep_rank = dist.get_rank(ep_group)
    key = (ep_group, ep_size, ep_rank, local_e, str(counts.device), cache_key)
    if cache_key is not None and key in _FIXED_SPLIT_CACHE:
        return _FIXED_SPLIT_CACHE[key]
    gathered = [torch.empty_like(counts) for _ in range(ep_size)]
    dist.all_gather(gathered, counts, group=ep_group)
    local = torch.stack(gathered)[:, ep_rank * local_e:(ep_rank + 1) * local_e]
    metadata = torch.cat((counts.reshape(ep_size, local_e).sum(dim=1),
                          local.reshape(-1))).tolist()
    input_splits = metadata[:ep_size]
    chunk_sizes = metadata[ep_size:]
    output_splits = [sum(chunk_sizes[i * local_e:(i + 1) * local_e])
                     for i in range(ep_size)]
    result = (input_splits, output_splits, chunk_sizes)
    if cache_key is not None:
        if len(_FIXED_SPLIT_CACHE) >= 32:
            _FIXED_SPLIT_CACHE.pop(next(iter(_FIXED_SPLIT_CACHE)))
        _FIXED_SPLIT_CACHE[key] = result
    return result


def _recompute_dispatch_segment(seg_x, seg_ids, seg_w, local_e, ep_group, max_recv, split_cache_key=None, quantize=None):
    """重算单个 token 分块的 dispatch_output。

    入参为前向喂算子的同款分块原料：seg_x [m,K] bf16、seg_ids [m,topk] 全局
    专家 id、seg_w [m,topk] bf16。max_recv 用前向实际传给算子的值（注意本地
    ``_max_recv_tokens = m*ep*min(local_e, topk)``，非参考实现的 m*topk*ep），
    补零与 DispatchOutputLayout 布局与之对齐。返回 (UInt8 buffer, actual_rows)。
    """
    m, k = seg_x.shape
    topk = seg_ids.shape[1]
    ep_size = (
        dist.get_world_size(ep_group)
        if dist.is_available() and dist.is_initialized()
        else 1
    )
    num_out = m * topk
    e_global = local_e * ep_size

    # 1) 源端 permute：expert-major、专家内 token 升序。复合 key（expert*num_out
    #    + 平铺序号）保证排序确定性，不依赖后端 stable-sort 支持。key 必须用
    #    float32：NPU ArgSort kernel 不支持 int32/int64（会降级到 AiCpu，在
    #    backward comm stream 上挂死并使宿主内存膨胀至 OOM —— 167/188 均复现）；
    #    复合 key 值域 < 2^24，float32 尾数精确表示，排序结果与 int64 逐位一致。
    assert e_global * num_out < 2**24, (
        f"sort key may lose exactness in float32: e_global={e_global} "
        f"num_out={num_out}"
    )
    flat_ids = seg_ids.reshape(-1).long()
    key = (
        flat_ids * num_out + torch.arange(num_out, device=seg_x.device)
    ).to(torch.float32)
    order = torch.argsort(key)
    scales = None
    if quantize is None:
        permuted = seg_x.index_select(0, order // topk)
    else:
        payload, source_scales = quantize(seg_x)
        permuted = payload.index_select(0, order // topk)
        scales = source_scales.index_select(0, order // topk)
    carry = seg_w.reshape(-1).index_select(0, order)

    # 2) 路由计数（本 rank 各全局专家的 token 数，eager dispatcher 同款 histc）
    #    -> a2a splits（目的 rank 按 [rank*local_e,(rank+1)*local_e) 连续归属）
    #    histc 输入 cast 到 float32：整型输入在 NPU 上同样存在 AiCpu 降级风险。
    # A hit needs neither histc, count exchange, nor D2H metadata conversion.
    if split_cache_key is None or ep_size == 1:
        counts = torch.histc(
            flat_ids.to(torch.float32), bins=e_global, min=0, max=e_global
        ).long()
    else:
        key = (ep_group, ep_size, dist.get_rank(ep_group), local_e,
               str(seg_ids.device), split_cache_key)
        counts = None if key in _FIXED_SPLIT_CACHE else torch.histc(
            flat_ids.to(torch.float32), bins=e_global, min=0, max=e_global
        ).long()
    if ep_size > 1:
        if counts is None:
            input_splits, output_splits, chunk_sizes = _FIXED_SPLIT_CACHE[key]
        else:
            input_splits, output_splits, chunk_sizes = _resolve_split_metadata(
                counts, local_e, ep_group, ep_size, split_cache_key)
        recv = permuted.new_empty(sum(output_splits), k)
        dist.all_to_all_single(
            recv, permuted,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits, group=ep_group,
        )
        if scales is not None:
            scale_recv = scales.new_empty((sum(output_splits), *scales.shape[1:]))
            dist.all_to_all_single(scale_recv, scales,
                output_split_sizes=output_splits, input_split_sizes=input_splits, group=ep_group)
            scales = scale_recv
        carry_recv = carry.new_empty(sum(output_splits))
        dist.all_to_all_single(
            carry_recv, carry,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits, group=ep_group,
        )
        carry = carry_recv
    else:
        recv = permuted
        local = counts.reshape(1, local_e)

    # 3) 按（本地专家, 源 rank）重排：a2a 到达序是 (源rank, 本地专家) 连续块，
    #    与算子前向 ptrA 的段内序契约对齐（参考实现 step3 同款）。
    if ep_size > 1 and local_e > 1:
        chunks = torch.split(recv, chunk_sizes, dim=0)
        w_chunks = torch.split(carry, chunk_sizes)
        sort_ids = [source * local_e + expert
                    for expert in range(local_e) for source in range(ep_size)]
        recv = torch.cat([chunks[i] for i in sort_ids], dim=0)
        carry = torch.cat([w_chunks[i] for i in sort_ids])
        if scales is not None:
            scale_chunks = torch.split(scales, chunk_sizes, dim=0)
            scales = torch.cat([scale_chunks[i] for i in sort_ids])

    # 4) 补零到 max_recv + 组装 DispatchOutputLayout（算子
    #    mega_moe_training_workspace.h：packed token 512 对齐 + packed TopK weight
    #    512 对齐，同一 UInt8 Tensor）。kernel 按 actualRows 读，零行不参与计算。
    actual = recv.size(0)
    if actual > max_recv:
        raise RuntimeError(
            f"MegaMoe receive capacity exceeded: actual={actual}, "
            f"max_recv={max_recv}; fix the forward capacity calculation"
        )
    if quantize is not None:
        if actual < max_recv:
            recv = torch.cat((recv, recv.new_zeros((max_recv-actual, k))))
            scales = torch.cat((scales, scales.new_zeros((max_recv-actual, *scales.shape[1:]))))
            carry = torch.cat((carry, carry.new_zeros(max_recv-actual)))
        return {"xq": recv.contiguous(), "sx": scales.contiguous(), "packedP": carry.contiguous()}, actual
    if actual < max_recv:
        recv = torch.cat([recv, seg_x.new_zeros(max_recv - actual, k)], dim=0)
    if carry.numel() < max_recv:
        carry = torch.cat([carry, carry.new_zeros(max_recv - carry.numel())])
    tokens_bytes = _align_up(max_recv * k * _SIZE_BF16, _ALIGN_512)
    weights_bytes = _align_up(max_recv * _SIZE_BF16, _ALIGN_512)
    buf = torch.empty(
        tokens_bytes + weights_bytes, dtype=torch.uint8, device=seg_x.device
    )
    buf[: max_recv * k * _SIZE_BF16].view(torch.bfloat16).view(max_recv, k).copy_(recv)
    buf[tokens_bytes: tokens_bytes + max_recv * _SIZE_BF16].view(
        torch.bfloat16
    ).copy_(carry)
    return buf, actual


def _verify_dispatch_output(op_buf, re_buf, counts, k, max_recv, ci):
    """verify 钩子：重算结果 vs 算子前向保留值逐位比对。

    只比对 actual 行（max_recv 尾部补零区算子无写出保证，比对无意义）。
    行序/数值必须完全一致（saved_state 的 packed 元数据按前向打包序索引，
    序不对则梯度错位），不等时打印首个不一致行并 raise。
    """
    actual = int(counts.sum().item())
    tokens_bytes = _align_up(max_recv * k * _SIZE_BF16, _ALIGN_512)
    a_tok = op_buf[: actual * k * _SIZE_BF16].view(torch.bfloat16).view(actual, k)
    b_tok = re_buf[: actual * k * _SIZE_BF16].view(torch.bfloat16).view(actual, k)
    a_w = op_buf[tokens_bytes: tokens_bytes + actual * _SIZE_BF16].view(
        torch.bfloat16
    )
    b_w = re_buf[tokens_bytes: tokens_bytes + actual * _SIZE_BF16].view(
        torch.bfloat16
    )
    if not (torch.equal(a_tok, b_tok) and torch.equal(a_w, b_w)):
        diff_rows = (a_tok != b_tok).any(dim=-1).nonzero().flatten()
        first = int(diff_rows[0]) if diff_rows.numel() else -1
        max_diff = (
            (a_tok.float() - b_tok.float()).abs().max().item() if actual else 0.0
        )
        raise AssertionError(
            f"[megamoe-redispatch-verify] chunk={ci} MISMATCH: first_row={first} "
            f"n_diff_rows={diff_rows.numel()} max_abs_diff={max_diff}"
        )
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        print(
            f"[megamoe-redispatch-verify] chunk={ci} bit-identical "
            f"(actual={actual} max_recv={max_recv})",
            flush=True,
        )


class _MegaMoeRedispatchFunction(torch.autograd.Function):
    """框架自持 autograd 桥：直调 raw TorchBind compute_train/backward。

    绕过算子 Python 封装 _MegaMoeBF16TrainFunction 的 retain_dispatch_output
    保护（retain=False 时 raise）—— 算子仓零改动。调用约定与该封装逐字对齐
    （grad_y.contiguous()、save 列表、counts mark_non_differentiable）。
    """

    @staticmethod
    def forward(
        ctx, op_wrapper, ep_group, seg_x, seg_ids, seg_w, w1, w2,
        seg_max_recv, swiglu_limit, save_gmm1_output, local_e, cell, chunk_index,
    ):
        raw_op = op_wrapper._op
        shmem = op_wrapper._shmem
        if (native_env("ASCEND_DEEPEP_FWD_SYNC") == "1" or
            native_env("ASCEND_DEEPEP_FWD_PRE_STREAM_SYNC") == "1"):
            torch.npu.current_stream().synchronize()
        y, counts, training_saved_state, dispatch_output, _transient = (
            raw_op.compute_train(
                shmem, seg_x, seg_ids, seg_w, w1, w2, seg_max_recv,
                float(swiglu_limit), save_gmm1_output,
            )
        )
        # dispatch_output 此后无人引用 -> 缓存分配器回收（本特性的全部收益）；
        # verify 模式保留以做逐位比对。transient 只服务本次 forward，同样即弃。
        ctx.op_wrapper = op_wrapper
        ctx.ep_group = ep_group
        ctx.seg_max_recv = seg_max_recv
        ctx.swiglu_limit = float(swiglu_limit)
        ctx.save_gmm1_output = save_gmm1_output
        ctx.local_e = local_e
        ctx.cell = cell
        ctx.chunk_index = chunk_index
        ctx.verify = _verify_enabled()
        # Use saved tensors so UAM can rematerialize norm output / offload
        # routing inputs. Ordinary ctx attributes would bypass its hooks.
        if ctx.verify:
            cell.setdefault("op_outputs", {})[chunk_index] = dispatch_output
        ctx.save_for_backward(seg_w, w1, w2, counts, training_saved_state, seg_x, seg_ids)
        ctx.mark_non_differentiable(counts)
        return y, counts

    @staticmethod
    def backward(ctx, grad_y, grad_counts=None):
        seg_w, w1, w2, counts, training_saved_state, seg_x, seg_ids = ctx.saved_tensors
        cell = ctx.cell
        ci = ctx.chunk_index
        event = cell.get("done_event")
        if event is None:
            # launcher 未发射（降级路径）：当前流串行重算本分块
            dispatch_buf, _actual = _recompute_dispatch_segment(
                seg_x, seg_ids, seg_w, ctx.local_e,
                ctx.ep_group, ctx.seg_max_recv,
                **({"split_cache_key": cell["split_cache_keys"][ci]}
                   if cell.get("split_cache_keys") else {}),
            )
        else:
            torch.accelerator.current_stream().wait_event(event)
            dispatch_buf, _actual = cell["outputs"][ci]
            # comm stream 上分配、当前流消费：防止缓存分配器过早复用
            dispatch_buf.record_stream(torch.accelerator.current_stream())
        if ctx.verify:
            _verify_dispatch_output(
                cell["op_outputs"][ci], dispatch_buf, counts,
                seg_x.shape[1], ctx.seg_max_recv, ci,
            )
        raw_op = ctx.op_wrapper._op
        shmem = ctx.op_wrapper._shmem
        # Materialize a strided gradient before the native launch dependency fence.
        grad_y = grad_y.contiguous()
        if native_env("ASCEND_DEEPEP_BWD_STREAM_SYNC") == "1":
            # 显式 current-stream 保护（原始包装器另用 BWD_SYNC 控制全局同步）：融合
            # backward host 侧直接取设备指针并做跨 rank SHMEM 屏障，若当前
            # 流上 grad_y 生产 kernel（及对端 rank 对本 rank 窗口的读）尚未
            # 排空即发射，会以 acl 507015 aicore 异常浮现（实测 8L attn
            # 首个 backward 即崩；包装路径早已默认开启本同步，直调路径
            # 此前裸奔，成功纯属竞态侥幸）。
            torch.npu.current_stream().synchronize()
        dx, dtopk, dw1, dw2 = raw_op.backward(
            shmem, grad_y, seg_w, w1, w2, counts,
            training_saved_state, dispatch_buf, ctx.seg_max_recv,
            ctx.swiglu_limit, ctx.save_gmm1_output,
        )
        if event is not None:
            cell["outputs"][ci] = None  # 释放重算 buffer 引用
        return (None, None, dx, None, dtopk, dw1, dw2,
                None, None, None, None, None, None)


class MegaMoeRedispatchModule(torch.nn.Module):
    def forward(
        self, op_wrapper, ep_group, seg_x, seg_ids, seg_w, w1, w2,
        seg_max_recv, swiglu_limit, save_gmm1_output, local_e, cell, chunk_index,
    ):

        return _MegaMoeRedispatchFunction.apply(
            op_wrapper, ep_group, seg_x, seg_ids, seg_w, w1, w2,
            seg_max_recv, swiglu_limit, save_gmm1_output, local_e, cell, chunk_index,
        )


class _RedispatchLauncher(torch.autograd.Function):
    """block 输出上的 identity Function：backward 一到就发射重算。

    必须包在 DeepseekV4SparseMoeBlock 的输出（routed + shared_out）上 —— 该
    节点是 block 输出节点，反向先于 AddBackward 与两个分支执行，重算 a2a
    得以与共享专家分支的 backward（计算流）重叠；experts 的
    _MegaMoeRedispatchFunction.backward（上游节点）以 wait_event 消费。
    """

    @staticmethod
    def forward(ctx, out, cell):
        ctx.cell = cell
        chunks = cell["chunks"]
        ctx.chunk_metadata = [(chunk[3], chunk[4]) for chunk in chunks]
        ctx.save_for_backward(*(value for chunk in chunks for value in chunk[:3]))
        # No tensor-bearing cell survives forward: both this launcher and the
        # expert backward now restore through saved-tensor hooks.
        cell["chunks"] = []
        # 前向完成事件：重算读的 seg 张量（experts forward 产物）就绪保证，
        # 兼容任意 fwd/bwd 流拓扑。
        cell["fwd_event"] = torch.accelerator.current_stream().record_event()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        cell = ctx.cell
        comm = _get_comm_stream()
        # FSDP's pre-backward hook orders freshly unsharded parameters on the
        # compute stream. Producer replay on comm may read those parameters,
        # so transfer that dependency before unpacking. Record at launcher
        # entry, before shared-expert backward and before any restore/replay.
        # Waiting after compute-side restoration would also fence its kernels.
        cell["backward_entry_event"] = torch.accelerator.current_stream().record_event()
        with torch.accelerator.stream(comm):
            comm.wait_event(cell["fwd_event"])
            comm.wait_event(cell["backward_entry_event"])
            # Unpack on the consuming stream: UAM H2D completion is handed
            # directly to comm, and producer replay is enqueued on comm.
            # A compute-stream event here also fences unrelated backward
            # kernels and prevents overlap with the shared expert branch.
            saved = ctx.saved_tensors
            for ci, (seg_max_recv, local_e) in enumerate(ctx.chunk_metadata):
                seg_x, seg_ids, seg_w = saved[3 * ci:3 * ci + 3]
                # Protect inputs (including retained forward tensors) until
                # the communication stream has finished reading them.
                for value in (seg_x, seg_ids, seg_w):
                    value.record_stream(comm)
                buf, actual = cell.get("redispatch_fn", _recompute_dispatch_segment)(
                    seg_x, seg_ids, seg_w, local_e, cell["ep_group"], seg_max_recv,
                    **({"split_cache_key": cell["split_cache_keys"][ci]}
                       if cell.get("split_cache_keys") else {}),
                )
                # 消费端按 (buf, actual) 二元组解包：直接存裸 Tensor 会使
                # unpack 走 Tensor.__iter__ -> self.cpu() 整块 D2H（353MB/块），
                # backward 自旋在 libtorch_npu 拷贝内且宿主 RSS 膨胀至 100GB+
                # （164 py-spy 现场），此处必须与串行路径返回值同构。
                cell["outputs"][ci] = (buf, actual)
            cell["done_event"] = comm.record_event()
        return grad_out, None
