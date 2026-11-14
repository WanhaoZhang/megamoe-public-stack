# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""MegaMoe BF16 融合 EP dispatcher。

用 ``ascend_deepep.MegaMoeBF16Train`` 融合可微算子替换 eager 的
dispatch -> GMM1 -> SwiGLU -> GMM2 -> combine 流水线：token 经
CANN SHMEM 对称窗口跨 rank 直接路由（替代标准 all-to-all），两个
GMM 与 SwiGLU 激活均在设备上完成。

权重布局与 dsv4 的 ``DeepseekV4Experts`` 直接对齐：
    w1 = gate_up_proj -> [local_experts, 2 * intermediate_size, hidden_size]
    w2 = down_proj    -> [local_experts, hidden_size, intermediate_size]
``topk_ids`` 携带全局专家 id（``[0, num_global_experts)``），每个 EP
rank 拥有专家 ``[rank * local_e, (rank + 1) * local_e)`` —— 与
eager dispatcher 的归属约定一致。

CANN SHMEM runtime 是进程单例，窗口池同样如此：融合算子的对称窗口是
纯瞬态 scratch —— backward 从 ``training_saved_state`` 恢复 token
计数表（RestoreTokenPerExpert），从不读取任何 forward 遗留的 payload。
因此只要某窗口的上一个使用者在所有 rank 上都已完成，该窗口即可跨层
复用；这一点由两处顺序保证（见 ``_acquire_window_pool_op``）：
dispatcher 在 forward 前的逐调用 HCCL 屏障，以及 backward kernel 在
首次写对端窗口前的入口 ``CrossRankSyncInWorldSize``。由
``shmem_window_pool`` 个窗口组成的轮转池（experts 模块上的配置属性，
默认 2；``ASCEND_DEEPEP_SHMEM_WINDOW_POOL`` 环境变量可覆盖）服务
所有层，backward 经由 ``ctx.op`` 复用其 forward 的槽位，从而把
SHMEM 堆的上限钉在“池大小个全序列窗口”，而非每个 MoE 层一个窗口
（旧 one-window-per-layer 设计中的 acl 507015 类冲突由复用距离 2
加上上述入口同步覆盖）。堆大小由池大小与窗口形状自动推导
（pool + 1 个窗口的余量）；``ASCEND_DEEPEP_SHMEM_TOTAL_BYTES``
仍可覆盖。

注意：融合 BF16 kernel 在其 SwiGLU 内部应用 dsv4 的
``swiglu_limit`` clamp（``silu(min(gate, L)) * clamp(up, -L, L)``，
L 取自专家模块的 ``limit`` 属性），与 eager 路径的激活语义一致。
"""

import math
import os
import threading

import torch
from torch.distributed.tensor import DTensor

from fsdp_turbo.distributed.expert_parallel.mega_moe_redispatch import (
    _MegaMoeRedispatchFunction,
    megamoe_redispatch_active,
    megamoe_redispatch_overlap_enabled,
    MegaMoeRedispatchModule,
)
from fsdp_turbo.distributed.expert_parallel.utils import (
    fixed_router_for_debug,
    normalize_expert_args,
)

# ACLSHMEM_DATA_OP_UDMA —— 与 test/megamoe_bf16_test/
# test_mega_moe_backward.py 中 BF16 基础训练路径保持一致。
_ACLSHMEM_DATA_OP_UDMA = 0x08
_DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
_BUFFER_ALIGNMENT = 512
# CANN SHMEM runtime 初始化时预留的进程级对称堆。窗口池生效时堆大小
# 自动推导为 (pool_size + 1) 个窗口（保留一槽余量）；
# ASCEND_DEEPEP_SHMEM_TOTAL_BYTES 可覆盖。legacy per-module 模式
# （池大小 <= 0，窗口数随层数增长）则回退到该平铺默认值。
_DEFAULT_SHMEM_TOTAL_BYTES = 4 * 1024 * 1024 * 1024

# 对端内存布局镜像 kernels/mega_moe/include/mega_moe_tiling_host.h：
#   CalcPeermemSize + CalcBf16EarlyTopkWeightPeermemSize（BF16，非 UDMA 路径）。
# fce2f40 统一 SHMEM 头布局后，payload 起点为 72 KiB（kPeerDataOffset）。
_PEERMEM_DATA_OFFSET = 72 * 1024
_ALIGN_128 = 128
_SIZE_INT32 = 4
_SIZE_BF16 = 2


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _resolve_save_gmm1_output():
    mode = os.environ.get("MEGAMOE_SAVE_GMM1_OUTPUT", "bf16").strip().lower()
    if mode not in ("bf16", "mxfp8", "auto", "none"):
        raise ValueError(
            f"MEGAMOE_SAVE_GMM1_OUTPUT must be bf16, mxfp8, auto or none; got {mode!r}"
        )
    return mode


def _ceil_align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _peer_mem_bytes(
    m: int,
    k: int,
    topk: int,
    ep_size: int,
    local_e: int,
    max_recv_tokens: int | None = None,
) -> int:
    """单个 MegaMoe BF16 算子在该形状下所需的对称窗口字节数。

    镜像 host 侧 tiling 计算（CalcPeermemSize + BF16 early topk 权重区）。
    UDMA 路径的 sharedActivation 与对称 payload 区复用，长度按
    ``max(m*topk, max_recv_tokens)`` 计：ρ>0 时 per-rank recv 可超过
    ``m*topk``，若仍按均衡尺寸开窗会写穿对称堆（表现为 loss=nan /
    ACL 507001）。
    """
    payload_rows = m * topk
    if max_recv_tokens is not None:
        payload_rows = max(payload_rows, int(max_recv_tokens))
    total = _PEERMEM_DATA_OFFSET
    total += ep_size * _ceil_align(ep_size * local_e, _ALIGN_128) * _SIZE_INT32
    total += payload_rows * k * _SIZE_BF16  # dispatch/recv / GMM1 staging
    total += payload_rows * k * _SIZE_BF16  # combine staging
    total += _ceil_align(m * topk * _SIZE_BF16, _BUFFER_ALIGNMENT)  # early topk 权重
    return total


def _local_mem_bytes(
    m: int,
    k: int,
    topk: int,
    ep_size: int,
    local_e: int,
    max_recv_tokens: int | None = None,
) -> int:
    return _align_up(
        _peer_mem_bytes(m, k, topk, ep_size, local_e, max_recv_tokens),
        _DEVICE_LARGE_PAGE_SIZE,
    )


# 进程级 CANN SHMEM runtime（runtime 是进程单例）。加锁是因为
# torch.multiprocessing worker 在首次 forward 时仍可能竞争；单线程
# 训练循环中只触发一次。池化的融合算子（见 _acquire_window_pool_op）
# 在该堆内分配各自的对称窗口。
_MEGA_MOE_STATE = {"lock": threading.Lock(), "shmem": None}

# 同一个 SHMEM 世界内所有融合算子调用共享的轮转对称窗口池（安全论证见
# _acquire_window_pool_op）。池大小是 mega_moe 特性参数，由
# _resolve_window_pool_size 逐调用解析：experts 模块的
# ``shmem_window_pool`` 配置属性（默认 2 —— 保证复用距离的最小值，
# 使 forward 的 pre-handshake 计数表写入不会与对端对同一槽位的
# exit-reset 竞争）；ASCEND_DEEPEP_SHMEM_WINDOW_POOL 环境变量可覆盖。
_WINDOW_POOL_STATE = {"lock": threading.Lock(), "pools": {}}


def _resolve_window_pool_size(module):
    """解析对称窗口池大小（特性参数，默认 2）。

    优先级：ASCEND_DEEPEP_SHMEM_WINDOW_POOL 环境变量覆盖（``0`` 选择
    legacy per-module 的逐层一窗口模式，供 A/B 调试）> experts 模块的
    ``shmem_window_pool`` 配置属性 > 默认值 2。
    """
    env = os.environ.get("ASCEND_DEEPEP_SHMEM_WINDOW_POOL")
    if env is not None:
        return int(env)
    value = getattr(module, "shmem_window_pool", None) if module is not None else None
    if value is not None:
        return int(value)
    return 1


def _ensure_mega_moe_runtime(ep_group, window_pool_size=None, window_bytes=None, *, bootstrap_ip_port=None):
    """一次性创建进程级 ShmemInstance runtime。

    对称堆大小由窗口池自动推导：N 个窗口的池对应 (N + 1) 个窗口的堆
    —— 池本身加一槽余量，堆随池配置联动，无需手工调参。
    ASCEND_DEEPEP_SHMEM_TOTAL_BYTES 可覆盖；缺少池信息时使用平铺的
    _DEFAULT_SHMEM_TOTAL_BYTES（legacy per-module 模式）。
    """
    with _MEGA_MOE_STATE["lock"]:
        if _MEGA_MOE_STATE["shmem"] is not None:
            if bootstrap_ip_port is not None:
                existing=_MEGA_MOE_STATE["shmem"]
                if (existing.world_size()!=torch.distributed.get_world_size(ep_group) or
                        existing.rank()!=torch.distributed.get_rank(ep_group) or
                        _MEGA_MOE_STATE.get("ep_group") is not ep_group):
                    raise RuntimeError('balanced large EP cannot reuse a different SHMEM world/group')
            return _MEGA_MOE_STATE["shmem"]

        from ascend_deepep import ShmemInstance

        ep_rank = torch.distributed.get_rank(ep_group)
        ep_size = torch.distributed.get_world_size(ep_group)
        ip_port = bootstrap_ip_port or os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", "tcp://127.0.0.1:8899")
        # 单节点多个 EP group（ep_size < world_size 且 ep_fsdp_size > 1）时
        # 不能共享同一个 CANN SHMEM bootstrap ip_port / UID 会话：每个
        # group 是独立的 SHMEM 世界（各自 n_pes），在同一控制通道上
        # 混合声明会以 aclshmemx_init_attr status=-3/-4 失败。在
        # ('edp','efsdp','ep') mesh 中每个 EP group 是 ep_size 个连续
        # 全局 rank 的块，因此 per-group 端口偏移直接由全局 rank 推导。
        # 单 group 运行（ep_size == world_size）保持配置端口不变。
        global_rank = torch.distributed.get_rank()
        if ep_size > 1 and bootstrap_ip_port is None:
            group_id = global_rank // ep_size
            if group_id > 0:
                host, _, port = ip_port.rpartition(":")
                ip_port = f"{host}:{int(port) + group_id}"
        # CANN SHMEM UID bootstrap 以该会话 id 为键。
        os.environ.setdefault("SHMEM_UID_SESSION_ID", ip_port.removeprefix("tcp://"))
        env_total = os.environ.get("ASCEND_DEEPEP_SHMEM_TOTAL_BYTES")
        if env_total is not None:
            total_mem = int(env_total)
        elif window_pool_size is not None and window_pool_size > 0 and window_bytes:
            total_mem = (window_pool_size + 1) * window_bytes
        else:
            total_mem = _DEFAULT_SHMEM_TOTAL_BYTES
        total_mem = _align_up(total_mem, _DEVICE_LARGE_PAGE_SIZE)

        shmem = ShmemInstance(ep_rank, ep_size, ip_port)
        shmem.set_data_op_engine_type(_ACLSHMEM_DATA_OP_UDMA)
        shmem.init(total_mem)
        _MEGA_MOE_STATE["shmem"] = shmem
        _MEGA_MOE_STATE["ep_group"] = ep_group
        print(
            f"[MegaMoe] shmem runtime ep_rank={ep_rank} ep_size={ep_size} "
            f"total_mem_bytes={total_mem} (window_pool={window_pool_size} "
            f"window_bytes={window_bytes}) ip_port={ip_port}",
            flush=True,
        )
        return shmem


def _max_recv_tokens(
    m: int, ep_size: int, local_e: int, topk: int,
    fixed_router: bool = True,
) -> int:
    # Optional hard override for diagnostics / capacity pinning.
    override = os.environ.get("ASCEND_DEEPEP_MAX_RECV_TOKENS")
    if override:
        try:
            return _align_up(int(override), 128)
        except ValueError:
            pass
    # General top-k routing: each source token can hit at most this many
    # experts owned by one destination rank.
    worst = m * ep_size * min(local_e, topk)
    try:
        imbalance = float(os.environ.get("ASCEND_ROUTER_IMBALANCE", "0") or "0")
    except ValueError:
        imbalance = 0.0
    # Production skew is unknown at buffer allocation time: always budget the
    # general top-k worst case. Do NOT shrink capacity with a skewed estimate
    # derived from ASCEND_ROUTER_IMBALANCE (debug-only).
    if fixed_router and imbalance <= 0.0:
        # Balanced fixed_router_for_debug (round-robin): every top-k offset
        # contributes at most ceil(m / global_e) rows to any one expert.
        global_e = ep_size * local_e
        per_expert = ((m + global_e - 1) // global_e) * topk
        capacity = min(worst, ep_size * local_e * per_expert)
    else:
        capacity = worst
    return _align_up(capacity, 128)


def _acquire_window_pool_op(
    shmem, m: int, k: int, topk: int, ep_size: int, local_e: int, module=None, fixed_router: bool = False,
):
    """同一个 SHMEM 世界内所有融合算子调用共享的轮转窗口池。

    对称窗口是纯瞬态 scratch（见模块 docstring），因此槽位跨层回收，
    而非每层一个窗口。安全性建立在复用距离 2 加两处入口同步之上，
    二者共同把跨 rank 的错位（skew）约束在不足一次融合调用内：

    * forward：``experts_forward`` 中的逐调用 HCCL 屏障确保所有 rank
      上一次融合调用排空后，本次调用才写对端窗口；
    * backward：kernel 以 ``CrossRankSyncInWorldSize`` 开局，先于其
      首次对端窗口写入，即所有 rank 都已进入同一个 backward。

    两者都保证某槽位的上一个使用者在所有 rank 上完成后，该槽位才会
    被再次发出。池由此把 SHMEM 堆上限钉在解析出的池大小个窗口
    （48 + MTP 层此前需要 49 个）。每个 forward 的 backward 经由
    ``ctx.op`` 复用该 forward 的槽位，因此只有 forward 调用推进
    轮转计数。

    池大小由 ``_resolve_window_pool_size`` 逐调用解析，并被该
    SHMEM 世界的首个 experts 模块钉死；后续模块解析出不同大小视为
    配置错误（堆是按首个配置分配的）。
    ``ASCEND_DEEPEP_SHMEM_WINDOW_POOL=0`` 选择 legacy per-module
    模式：每个 experts 模块（层）保留自己的 op 和窗口，与池化前的
    dispatcher 完全一致，用于窗口复用数值的 A/B 调试。
    """
    pool_size = _resolve_window_pool_size(module)
    key = id(shmem)
    with _WINDOW_POOL_STATE["lock"]:
        pool = _WINDOW_POOL_STATE["pools"].get(key)
        if pool is None:
            pool = {"ops": [], "next": 0, "modules": {}, "size": None}
            _WINDOW_POOL_STATE["pools"][key] = pool
        if pool["size"] is None:
            pool["size"] = pool_size
        elif pool_size != pool["size"]:
            raise ValueError(
                f"[MegaMoe] inconsistent shmem_window_pool config: this experts "
                f"module resolved {pool_size} but the window pool for this SHMEM "
                f"world was created with {pool['size']}"
            )
        max_recv = _max_recv_tokens(m, ep_size, local_e, topk, fixed_router=fixed_router)
        if pool_size <= 0:
            mod_key = id(module)
            op = pool["modules"].get(mod_key)
            if op is None:
                from ascend_deepep import MegaMoeBF16Train

                local_mem_size = _local_mem_bytes(
                    m, k, topk, ep_size, local_e, max_recv_tokens=max_recv,
                )
                op = MegaMoeBF16Train(shmem, local_mem_size)
                pool["modules"][mod_key] = op
                print(
                    f"[MegaMoe] per-module op module_id={mod_key} "
                    f"local_mem_bytes={local_mem_size} "
                    f"max_recv_tokens={max_recv}",
                    flush=True,
                )
            return op
        if len(pool["ops"]) < pool_size:
            from ascend_deepep import MegaMoeBF16Train

            local_mem_size = _local_mem_bytes(
                m, k, topk, ep_size, local_e, max_recv_tokens=max_recv,
            )
            pool["ops"].append(MegaMoeBF16Train(shmem, local_mem_size))
            print(
                f"[MegaMoe] window pool slot={len(pool['ops']) - 1} "
                f"local_mem_bytes={local_mem_size} "
                f"max_recv_tokens={max_recv}",
                flush=True,
            )
        op = pool["ops"][pool["next"] % len(pool["ops"])]
        pool["next"] += 1
        return op


def get_mega_moe_forward_fn(ep_group, fixed_router=False):
    """返回绑定到专家模块的 MegaMoe BF16 EP forward。"""

    def experts_forward(self, hidden_states, top_k_index, top_k_weights):
        save_gmm1_output = _resolve_save_gmm1_output()
        top_k_index, top_k_weights = normalize_expert_args(top_k_index, top_k_weights)
        hidden_states_shape = hidden_states.shape
        hidden_dim = getattr(self, "hidden_dim", None) or hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(-1, hidden_dim)

        ep_size = torch.distributed.get_world_size(ep_group)
        if fixed_router:
            top_k_index, top_k_weights = fixed_router_for_debug(
                top_k_index, top_k_weights, self.num_global_experts, ep_size=ep_size,
            )

        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        m = hidden_states.shape[0]
        k = hidden_states.shape[1]
        topk = top_k_index.shape[1]
        local_e = gate_up_proj.shape[0]

        # token 维度分块：每次融合算子调用最多携带 ``_TOKEN_CHUNK_M``
        # 个 token。窗口池（见 _acquire_window_pool_op）生效后，49 层
        # 构建只需预留 2 个窗口（全序列下堆约 0.8 GiB），分块不再为
        # 余量所需；仅作为 HBM 紧张配置的逃生通道（连 pool_size 个
        # 全序列窗口都放不下时使用；小分块同时会缩小每块 GEMM 的 M
        # 并增加逐块屏障，均损失性能——除非 OOM 否则保持 999999）。
        # ``ASCEND_DEEPEP_TOKEN_CHUNK_M`` 覆盖分块大小（设大值即恢复
        # 全序列单次调用行为）。
        _TOKEN_CHUNK_M = int(os.environ.get("ASCEND_DEEPEP_TOKEN_CHUNK_M", "1024"))
        chunk_m = min(_TOKEN_CHUNK_M, m)

        # SHMEM 堆由池大小和该形状的窗口字节数推导（pool + 1 槽余量）；
        # ASCEND_DEEPEP_SHMEM_TOTAL_BYTES 可覆盖。在此解析是为了让
        # runtime 初始化与下方 _acquire_window_pool_op 看到同一份池配置。
        window_pool_size = _resolve_window_pool_size(self)
        chunk_max_recv = _max_recv_tokens(
            chunk_m, ep_size, local_e, topk, fixed_router=fixed_router,
        )
        window_bytes = _local_mem_bytes(
            chunk_m, k, topk, ep_size, local_e, max_recv_tokens=chunk_max_recv,
        )
        shmem = _ensure_mega_moe_runtime(
            ep_group,
            window_pool_size=window_pool_size,
            window_bytes=window_bytes,
        )
        op = _acquire_window_pool_op(
            shmem, chunk_m, k, topk, ep_size, local_e, module=self, fixed_router=fixed_router,
        )

        # 跨 rank step 对齐：融合 forward 把路由 payload 直接写进
        # 对端 rank 的对称窗口。若无屏障，快的 rank 可能进入 step N+1
        # 的 forward，而慢的对端仍在同一 per-layer 窗口上跑 step N 的
        # backward -> 数据竞争 -> acl 507015。HCCL 屏障在本 rank
        # 待处理的 backward kernel 之后入队到 NPU 流上，因此只有当
        # 所有 rank 排空上一步后它才完成，使下一次 forward 安全。
        # 注意：不能替换为设备侧 all-reduce —— host 不阻塞会提前发射
        # 下一次 forward 的写窗口 kernel，而 backward 部分在 FSDP2
        # 非默认流上，all-reduce 只等当前流，无法保证对端 backward
        # 完成，实测触发 acl 507015。
        # torch.distributed.barrier(ep_group)

        # 融合 BF16 kernel 要求所有输入为 BF16。ep_fsdp_size=1 时
        # 专家权重加载后保持 FP32（对齐 qwen3_moe 在融合计算前 cast
        # 到 bf16 的模式）。
        topk_ids = top_k_index.to(torch.int32).contiguous()
        w1 = gate_up_proj.to(torch.bfloat16).contiguous()
        w2 = down_proj.to(torch.bfloat16).contiguous()
        x = hidden_states.to(torch.bfloat16).contiguous()
        topk_w = top_k_weights.to(torch.bfloat16).contiguous()
        # 融合 kernel 保持 x / 权重 / 路由权重为 BF16，SwiGLU 在其
        # 内部执行，并应用 dsv4 swiglu_limit clamp
        # （gate = min(gate, L), up = clamp(up, -L, L)）以对齐 eager
        # 路径。self.limit 镜像 eager dispatcher 的 act_limit 来源。
        swiglu_limit = getattr(self, "limit", None)
        if swiglu_limit is not None and swiglu_limit > 0:
            swiglu_limit = float(swiglu_limit)
        else:
            swiglu_limit = 0.0

        def _run_op(seg_m, seg_x, seg_ids, seg_w):
            if (os.environ.get("ASCEND_DEEPEP_FWD_SYNC", "0") == "1" or
                os.environ.get("ASCEND_DEEPEP_FWD_PRE_STREAM_SYNC", "0") == "1"):
                torch.npu.current_stream().synchronize()
            return op(
                seg_x,
                seg_ids,
                seg_w,
                w1,
                w2,
                _max_recv_tokens(seg_m, ep_size, local_e, topk, fixed_router=fixed_router),
                swiglu_limit=swiglu_limit,
                save_gmm1_output=save_gmm1_output,
            )

        if megamoe_redispatch_active():
            # 重派发模式：框架自持 autograd 直调 raw compute_train/backward，
            # 前向丢弃 dispatch_output（-896MiB/层/卡 @ 定版形状），backward
            # 在 HCCL a2a 上重算（见 mega_moe_redispatch 模块 docstring）。
            # cell 由同 block 的 forward 末尾 hook（modeling_deepseek_v4.
            # DeepseekV4SparseMoeBlock）消费，挂 _RedispatchLauncher 实现
            # 重算 a2a 与共享专家 backward 的重叠。
            # 分块仅在 _TOKEN_CHUNK_M < m 时生效（HBM 紧张逃生通道，见上方
            # _TOKEN_CHUNK_M 注释）；默认 999999 单块直达，循环退化为单次。
            cell = {
                "chunks": [], "outputs": [], "done_event": None,
                "fwd_event": None, "ep_group": ep_group,
            }
            self.megamoe_redispatch_cell = cell
            ys = []
            for start, end in (
                (i, min(i + _TOKEN_CHUNK_M, m)) for i in range(0, m, _TOKEN_CHUNK_M)
            ):
                seg_x = x[start:end]
                seg_ids = topk_ids[start:end]
                seg_w = topk_w[start:end]
                seg_max_recv = _max_recv_tokens(
                    end - start, ep_size, local_e, topk, fixed_router=fixed_router,
                )
                ci = len(cell["outputs"])
                # Only the deterministic debug router may reuse metadata.
                # Uniform EP token shapes are an explicit opt-in contract.
                if fixed_router and os.environ.get("ASCEND_DEEPEP_FIXED_ROUTER_SPLIT_CACHE", "0") == "1":
                    cell.setdefault("split_cache_keys", []).append(
                        ("round_robin_v1", m, start, end - start, topk, self.num_global_experts))
                if megamoe_redispatch_overlap_enabled():
                    cell["chunks"].append((seg_x, seg_ids, seg_w, seg_max_recv, local_e))
                cell["outputs"].append(None)
                # y, _counts = _MegaMoeRedispatchFunction.apply(
                if not hasattr(self, "megamoe_module"):
                    self.megamoe_module = MegaMoeRedispatchModule()
                y, _counts = self.megamoe_module(
                    op, ep_group, seg_x, seg_ids, seg_w, w1, w2,
                    seg_max_recv, swiglu_limit, save_gmm1_output, local_e, cell, ci,
                )
                ys.append(y)
            y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=0)
        elif m <= _TOKEN_CHUNK_M:
            y, _counts = _run_op(m, x, topk_ids, topk_w)
        else:
            # _TOKEN_CHUNK_M < m 的 HBM 逃生通道：按 token 维切分调用融合
            # 算子（窗口字节随分块缩小），路由与 GMM combine 均按 token
            # 独立，拼接各分块输出复现全序列结果。
            y = torch.cat(
                [
                    _run_op(end - start, x[start:end], topk_ids[start:end], topk_w[start:end])[0]
                    for start, end in (
                        (i, min(i + _TOKEN_CHUNK_M, m)) for i in range(0, m, _TOKEN_CHUNK_M)
                    )
                ],
                dim=0,
            )
        # 诊断用途：FSDP2 在非默认流上调度 backward；forward 后强制
        # 设备同步，用于测试 backward aicore 故障（acl 507015）是否
        # 由未同步的 forward workspace 引起。
        if os.environ.get("ASCEND_DEEPEP_FWD_SYNC", "0") == "1":
            torch.npu.synchronize()
        return y.view(*hidden_states_shape)

    return experts_forward
