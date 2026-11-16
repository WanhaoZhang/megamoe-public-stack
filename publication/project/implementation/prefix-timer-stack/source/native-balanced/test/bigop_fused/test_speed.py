"""BigOpV2 vs BigOp 性能对比（test_precision 风格：逐次同步）。

与 mega_moe_bf16_train_bench.py（深队列异步 bench）不同，本脚本照
test_precision.py 的节奏：**每轮迭代前后都 torch.npu.synchronize()**，
"跑一次同步一次"，避免深队列下多轮 SHMEM forward/backward 争用同一块
workspace 导致的 launch kernel failure。用来验证"卡住是深队列问题、
算子本身不卡"。

对比两条路径（同输入、同 seed、同 config）：
  - megamoe : BigOpV2（router 手写 + 可插拔融合算子，MegaMoeBF16Train，
              EP 通信走 SHMEM）
  - bigop   : BigOp.apply（纯 torch 手写 10 步，EP 通信走 HCCL alltoall）
三组计时（每轮重建图，Event 圈流时间）：
  - fwd     : 仅前向
  - bwd     : 仅反向（每轮先 forward 建图不计时，Event 只圈 backward）
  - fwd+bwd : 完整前反向
speedup = bigop/megamoe（>1 表示 megamoe 更快）。
event 含 host 准备期间的流空闲、kernel 外设备初始化和 kernel 执行；纯 kernel
时间用 MEGAMOE_TRACE=1 的 kernel_us。event - kernel_us 是 kernel 外总贡献，
不能直接等同于纯 CPU 时间。

bwd 口径（MEGAMOE_SAVE_GMM1_OUTPUT，与 run_precision.sh 同一套，默认 bf16）：
  bf16 / mxfp8 / auto : 前向保存 gate/up，backward 直接用 → 双方 12 RFK
                        （bigop 自动联动到保存中间量口径）。
  none                : 不保存，backward 重算 P0-C → 双方 16 RFK
                        （bigop 自动联动到重算口径）。
  bench 保证 v2 和 bigop 做同样多的 GEMM 工作；mxfp8 无 bigop 等价物时
  bigop 退回 bf16 保存口径（精度差异属 kernel 的 mxfp8 压缩，速度口径仍对齐）。

用法（EP2，2 卡，seq 逗号分隔可多个）：
  bash bigop_fused/run_speed.sh [seq] [warmup] [iters]
    # seq 默认 4096；warmup 默认 0；iters 默认 1（单轮逐次同步，
    # 最小化 SHMEM 交互，验证"多轮深队列卡住"假说）
    # 例：bash bigop_fused/run_speed.sh 4096,8192,16384
    #     bash bigop_fused/run_speed.sh 8192 3 10   # 想测多轮时显式传

shape 由环境变量注入（BENCH_HIDDEN/BENCH_FFN/BENCH_EXPERTS/BENCH_TOPK，
BENCH_LOCAL_MEM_BYTES 对称窗口），EP 并行度 = torchrun 进程数。例：
  NPROC=8 BENCH_HIDDEN=7168 BENCH_FFN=3072 BENCH_EXPERTS=64 BENCH_TOPK=8 \
    bash bigop_fused/run_speed.sh 4096 3 10
"""
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist

import timing
from golden import BigOp, _bwd_recompute_enabled
from fused import build_bigop_v2
from make_fake_moe_layer import make_fake_moe_layer
from utils import DEFAULT_IP_PORT

ACLSHMEM_DATA_OP_UDMA = 0x08

# ---- shape / 容量（环境变量注入，与 test_precision.py 同名同默认）----
#   BENCH_HIDDEN           K（hidden size，须 %256==0）
#   BENCH_FFN              F（SwiGLU 输出宽，不乘 2；kernel 侧 N = 2*FFN，须 %512==0）
#   BENCH_EXPERTS          E（总专家数，须整除 EP）
#   BENCH_TOPK             topK（须 <= E）
#   BENCH_LOCAL_MEM_BYTES  SHMEM 对称窗口（默认 1 GiB；大 topK shape 需要更大，
#                          启动时会打印需求估算）
HIDDEN = int(os.environ.get("BENCH_HIDDEN", "1024"))
FFN = int(os.environ.get("BENCH_FFN", "2048"))
NUM_EXPERTS = int(os.environ.get("BENCH_EXPERTS", "16"))
TOPK = int(os.environ.get("BENCH_TOPK", "2"))
LOCAL_MEM_BYTES = int(
    os.environ.get("BENCH_LOCAL_MEM_BYTES", str(1024 * 1024 * 1024)))
BENCH_OUT_DIR = os.environ.get("BENCH_OUT_DIR", "")  # CSV/trace 输出目录
WARMUP = 0


def _out_path(name):
    """返回输出文件路径：有 BENCH_OUT_DIR 时写到该目录，否则当前目录。"""
    if BENCH_OUT_DIR:
        return Path(BENCH_OUT_DIR) / name
    return Path(name)


def _case_tag():
    """返回 case 标识（用于 trace 文件命名，包含 shape 参数避免覆盖）。"""
    return (f"s{os.environ.get('BENCH_SEQS', 'unk')}"
            f"_K{HIDDEN}_F{FFN}_E{NUM_EXPERTS}_t{TOPK}")
ITERS = 1

# ---- kernel 侧打点（MEGAMOE_TRACE=1）----
# 经 diagnostic_holder 走 MegaMoeBF16Train 的 compute_train_with_timer 和
# backward_with_timer 路径，捕获 AscendTimerV2 原始 buffer。前向报告 kernel
# 总时长及 kernel 外差值；反向打印分阶段表（五个 GEMM 阶段、AIV 搬运与等待）。
# 前提：算子以 scripts/build_mega_moe_bf16.sh --enable-timer 构建（打点本身
# 有开销：归因用 timer 构建，speedup 报数用正式构建）。
MEGAMOE_TRACE = os.environ.get("MEGAMOE_TRACE", "0").lower() in ("1", "true", "on", "yes")

# ---- 反向模式（MEGAMOE_SAVE_GMM1_OUTPUT，与 run_precision.sh 同一套）----
# 算子的保存模式：none=不保存（backward 重算 P0-C，16 RFK）；
# bf16/mxfp8=前向保存 gate/up（backward 直接用，12 RFK）。默认 bf16。
# bench 自动把 bigop 标杆同步到同一口径（BIGOP_BWD_RECOMPUTE 联动），
# 保证 v2 和 bigop 做同样多的 GEMM 工作。mxfp8 无 bigop 等价物时
# bigop 退回 bf16 保存口径并打提示。
SAVE_GMM1 = os.environ.get("MEGAMOE_SAVE_GMM1_OUTPUT", "bf16").strip().lower()
if SAVE_GMM1 not in ("auto", "none", "bf16", "mxfp8"):
    raise SystemExit(
        "[ERROR] MEGAMOE_SAVE_GMM1_OUTPUT must be auto, none, bf16 or "
        f"mxfp8; got {SAVE_GMM1!r}")
if SAVE_GMM1 == "none":
    BWD_RFK = 16.0                     # 重算：P0-C 加入反向
    os.environ.setdefault("BIGOP_BWD_RECOMPUTE", "1")
    BWD_MODE = "recompute"
else:                                   # auto/bf16/mxfp8：保存 gate/up
    BWD_RFK = 12.0
    os.environ.setdefault("BIGOP_BWD_RECOMPUTE", "0")
    BWD_MODE = f"saved-{SAVE_GMM1}"


def _validate_shape() -> None:
    """shape 合法性（与 test_precision.py 同一套约束）。违规 exit 2。"""
    world = int(os.environ.get("WORLD_SIZE", "2"))
    if HIDDEN % 256 != 0 or (2 * FFN) % 512 != 0:
        raise SystemExit(
            f"[ERROR] requires HIDDEN%256==0 and 2*FFN%512==0, got "
            f"H={HIDDEN} F={FFN}")
    if NUM_EXPERTS % world != 0:
        raise SystemExit(
            f"[ERROR] NUM_EXPERTS {NUM_EXPERTS} not divisible by EP {world}")
    if not 1 <= TOPK <= NUM_EXPERTS:
        raise SystemExit(
            f"[ERROR] TOPK {TOPK} out of range [1, {NUM_EXPERTS}]")


def _print_capacity(seqs: list[int]) -> None:
    """启动前容量估算（每 rank），对称窗口超限时给出建议值再退出前提示。"""
    world = int(os.environ.get("WORLD_SIZE", "2"))
    epr = NUM_EXPERTS // world
    k, f, n = HIDDEN, FFN, 2 * FFN
    m = max(seqs)
    sym = 60 * 1024
    sym += world * ((world * epr + 127) // 128 * 128) * 4   # 计数表
    sym += 2 * m * TOPK * k * 2                              # A0 + D 两个 m*topk*K 区
    sym += m * TOPK * 2 + (m * TOPK * world) * 2             # early-topk 权重双侧
    weights = epr * (n * k + k * f) * 2                      # 本 rank 权重
    permuted = m * TOPK * world * k * 2                      # permutedInput[ maxRecv, K ]
    fwd_ws = m * TOPK * world * (k + f) * 2                  # 前向 workspace 平面
    bwd_ws = 2 * m * TOPK * world * (k + n + 3 * f)          # 反向 workspace：maxOutputSize
                                                             # = maxRecv = m*topk*world（dropless
                                                             # 最坏口径，对齐 C++
                                                             # MegaMoeBF16BackwardWorkspaceLayout）
    gi = 1024 ** 3
    timing.rank0_print(
        f"[shape] M={seqs} K={k} F={f} N={n} E={NUM_EXPERTS} "
        f"EP={world} EPR={epr} topk={TOPK} bwd_mode={BWD_MODE}"
        f"({BWD_RFK:.0f}RFK)")
    timing.rank0_print(
        f"[mem/rank] 对称窗口需≈{sym / gi:.2f}GiB "
        f"(LOCAL_MEM_BYTES={LOCAL_MEM_BYTES / gi:.2f}GiB) | "
        f"权重+梯度≈{2 * weights / gi:.2f} | permuted≈{permuted / gi:.2f} | "
        f"fwd_ws≈{fwd_ws / gi:.2f} | bwd_ws≈{bwd_ws / gi:.2f} GiB")
    if sym > LOCAL_MEM_BYTES:
        suggest = 1 << (sym - 1).bit_length()   # 向上取 2 的幂
        timing.rank0_print(
            f"[ERROR] 对称窗口超限：{sym} > {LOCAL_MEM_BYTES} 字节。"
            f"export BENCH_LOCAL_MEM_BYTES={suggest} 后重跑")
        raise SystemExit(2)


@dataclass
class BenchContext:
    """一次测试的全部输入/环境对象（setup 阶段一次性构造）。"""

    rank: int
    world: int
    moe_layer: object       # 共用同一份 moe_layer（v2 与 bigop 同源路由）
    ip_port: str
    K: int


def setup(ctx: BenchContext):
    """0. 环境初始化 + 输入。dist 启动、seed、fake MoELayer。"""
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    # torchrun 单机/多机统一：LOCAL_RANK 是本节点设备下标（单机=rank，
    # 双机=每台各自的 0-7），rank % device_count 的旧写法在多机时恰好
    # 碰巧对但语义错误（依赖隐式取模，节点多于 2 时会映射错）。
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", world_size=world, rank=rank)
    torch.manual_seed(123)

    moe_layer, config = make_fake_moe_layer(
        hidden_size=HIDDEN, ffn_hidden_size=FFN, num_moe_experts=NUM_EXPERTS,
        router_topk=TOPK, aux_loss_coeff=0.0)
    print(f"[rank{rank}] shape H={HIDDEN} FFN={FFN} N={2 * FFN} "
          f"E={NUM_EXPERTS} EP={world} topk={TOPK}", flush=True)

    ctx.rank, ctx.world = rank, world
    ctx.moe_layer = moe_layer
    ctx.K = HIDDEN


def build_v2(ctx: BenchContext):
    """构造 SHMEM + BigOpV2（train 模式：MegaMoeBF16Train，fwd+bwd 一体）。"""
    from ascend_deepep import ShmemInstance

    shmem = ShmemInstance(ctx.rank, ctx.world, ctx.ip_port)
    shmem.set_data_op_engine_type(ACLSHMEM_DATA_OP_UDMA)
    shmem.init(LOCAL_MEM_BYTES)
    v2 = build_bigop_v2(shmem, LOCAL_MEM_BYTES, mode="train")
    return shmem, v2


def bench_once(fn, warmup: int = WARMUP, iters: int = ITERS,
               build_fn=None, clear_timing_tag=None) -> timing.BenchStats:
    """validate 风格逐次同步计时。

    event_ms：NPU Event 圈 fn() 的流时间。包含 host 准备期间的流空闲、
    kernel 外设备初始化（zero/memset）以及 kernel 执行。event - kernel_us
    （MEGAMOE_TRACE=1 时有）是 kernel 外总贡献，不等于纯 CPU 时间。

    build_fn 非空时为"仅反向"口径：每轮先 build_fn() 建前向图（不计时），
    sync 等前向 SHMEM 通信收尾后，Event 只圈 fn(out)（backward）。fused
    backward 每次原地清零 forward_workspace（save 的张量），不能 retain_graph
    复用同图，故每轮重建（节奏同 timing.bench_event_with_build）。
    """
    for _ in range(warmup):
        if build_fn is not None:
            out = build_fn()
            torch.npu.synchronize()
            fn(out)
        else:
            fn()
        torch.npu.synchronize()
    if clear_timing_tag is not None:
        timing.clear_tag(clear_timing_tag)
    event_pairs = []
    for _ in range(iters):
        torch.npu.synchronize()
        if build_fn is not None:
            out = build_fn()          # 前向建图不计时
            torch.npu.synchronize()   # 前向 SHMEM 通信完成后再计 backward
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        if build_fn is not None:
            fn(out)
        else:
            fn()
        end.record()
        torch.npu.synchronize()
        event_pairs.append((start, end))
    return timing.stats_from_ms(
        [s.elapsed_time(e) for s, e in event_pairs])


def _report_bwd_decomposition(seq: int):
    """bigop bwd 分解行：GEMM / alltoall / other 三列（含占比）+ cube:非cube 比例行。

    仅 BIGOP_TIMING=1 且本 seq 跑过 bigop backward 后有数据。
    - GEMM = gmm1_recompute + dhp + dw2 + dx + dw1（重算口径；BIGOP_BWD_RECOMPUTE=0
      时不含 recompute），并换算 TFLOPS 与 roofline 百分比；W1/W2 转置 copy
      单列为 w1_transpose/w2_transpose，归入 other
    - alltoall = alltoall_dispatch + alltoall_combine，小算子链两次串行 HCCL
      通信的实测耗时
    - other = 其余所有分步桶之和（permute/unpermute、swiglu_bwd、router 等）
    - 占比 = 各列 / 分步合计；比例行给出 cube : 非cube(alltoall+other) 与
      理论加速上限 = 合计/GEMM——即"非GEMM 全部被融合算子的 AIC/AIV 流水线
      吸收"时（cube-bound 单机、双方 GEMM 同效率、v2 自身 overhead≈0）的
      理想上界。已知偏置：async HCCL 尾部泄漏使 GEMM 桶偏小 -> 上限偏乐观；
      更严的对账用 v2 roofline 行的 floor（合计/floor）。
    每列的求和构成紧跟在分解行下面打印。分步跑含 Event/sync 扰动，分步合计
    与干净跑的 bwd event 略有出入，作归因用。
    返回分解 dict（BENCH_JSON_OUT 结构化记录用），无数据返回 None。
    """
    stats = timing.step_stats("bigop_bwd")
    if not stats:
        return
    recompute_on = _bwd_recompute_enabled()
    gemm_buckets = (["gmm1_recompute"] if recompute_on else []) + ["dhp", "dw2", "dx", "dw1"]
    comm_buckets = ["alltoall_dispatch", "alltoall_combine"]
    gemm_ms = sum(stats.get(b, 0.0) for b in gemm_buckets)
    comm_ms = sum(stats.get(b, 0.0) for b in comm_buckets)
    other_keys = [k for k in stats if k not in set(gemm_buckets + comm_buckets)]
    other_ms = sum(stats[k] for k in other_keys)
    total_ms = gemm_ms + comm_ms + other_ms
    non_gemm_ms = comm_ms + other_ms

    def _pct(x):
        return 100.0 * x / total_ms if total_ms > 0 else float("nan")

    coef = 16.0 if recompute_on else 12.0   # RFK 系数：重算口径 16，保存口径 12
    r = seq * TOPK                            # 每 rank 均衡行带 M×topK
    flops = coef * r * FFN * HIDDEN
    tflops = flops / (gemm_ms * 1e-3) / 1e12 if gemm_ms > 0 else float("nan")
    roof_pct = 100.0 * tflops / 410.4 if gemm_ms > 0 else float("nan")
    timing.rank0_print(
        f"[seq={seq}] bigop bwd 分解: GEMM={gemm_ms:.2f}ms({_pct(gemm_ms):.1f}%) "
        f"≈{tflops:.0f} TFLOPS({roof_pct:.0f}% of roofline) | "
        f"alltoall={comm_ms:.2f}ms({_pct(comm_ms):.1f}%) | "
        f"other={other_ms:.2f}ms({_pct(other_ms):.1f}%) | "
        f"分步合计={total_ms:.2f}ms (BIGOP_TIMING口径)")
    timing.rank0_print(
        f"    求和构成: GEMM={'+'.join(gemm_buckets)} | "
        f"alltoall={'+'.join(comm_buckets)} | "
        f"other={'+'.join(other_keys)}")
    if gemm_ms <= 0 or total_ms <= 0:
        return None
    timing.rank0_print(
        f"    cube : 非cube(alltoall+other) = {gemm_ms:.2f} : {non_gemm_ms:.2f}ms "
        f"({_pct(gemm_ms):.1f}% : {_pct(non_gemm_ms):.1f}%) -> "
        f"非GEMM全吸收的理论加速上限 = 合计/GEMM ≈ {total_ms / gemm_ms:.2f}x "
        f"（前提：双方GEMM同效率、v2自身overhead≈0）")
    # 逐 GEMM 效率表（与 megamoe 的 kernel 分阶段表对称）
    _GEMM_STEP_COEF = {
        "gmm1_recompute": 4.0,   # P0-C
        "dhp": 2.0,              # P1
        "dw2": 2.0,              # P4
        "dx": 4.0,               # P2
        "dw1": 4.0,              # P3
    }
    timing.rank0_print(
        f"    ==== [bigop] bwd per-GEMM（BIGOP_TIMING 口径；含 W1/W2 转置 copy）====")
    timing.rank0_print(
        f"    {'step':<18} {'mean_ms':>8} {'TFLOPS':>8} {'%roofline':>10}")
    per_gemm = {}
    for step in gemm_buckets:
        step_ms = stats.get(step, 0.0)
        if step_ms <= 0:
            continue
        step_coef = _GEMM_STEP_COEF.get(step, 0.0)
        if step_coef > 0 and r > 0:
            step_flops = step_coef * r * FFN * HIDDEN
            step_tf = step_flops / (step_ms * 1e-3) / 1e12
            step_pct = 100.0 * step_tf / 410.4
        else:
            step_tf = float("nan")
            step_pct = float("nan")
        per_gemm[step] = {"ms": step_ms, "tflops": step_tf,
                          "pct_roofline": step_pct}
        timing.rank0_print(
            f"    {step:<18} {step_ms:>8.2f} {step_tf:>8.0f} {step_pct:>9.0f}%")
    return {
        "gemm_ms": gemm_ms, "alltoall_ms": comm_ms, "other_ms": other_ms,
        "total_ms": total_ms, "non_gemm_ms": non_gemm_ms,
        "gemm_pct": _pct(gemm_ms), "alltoall_pct": _pct(comm_ms),
        "other_pct": _pct(other_ms), "non_gemm_pct": _pct(non_gemm_ms),
        "gemm_tflops": tflops, "gemm_pct_roofline": roof_pct,
        "speedup_ceiling": total_ms / gemm_ms,
        "per_gemm": per_gemm,
        "buckets": {"gemm": gemm_buckets, "alltoall": comm_buckets,
                    "other": other_keys},
    }


def _stat_dict(st: timing.BenchStats) -> dict:
    """BenchStats -> JSON 友好的 dict（event 口径 + 离散度）。"""
    return {
        "event_ms": st.mean_ms, "p50_ms": st.p50_ms,
        "min_ms": st.min_ms, "max_ms": st.max_ms, "n": st.n,
    }


def _maybe_dump_json(records: list) -> None:
    """BENCH_JSON_OUT=<path> 时 rank0 落盘结构化结果（plot_speed_matrix.py 的输入）。

    在 finally 里调用：中途异常也尽量保留已完成 seq 的部分数据。
    """
    out_path = os.environ.get("BENCH_JSON_OUT", "")
    if not out_path or not records:
        return
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    if rank != 0:
        return
    payload = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ep": int(os.environ["WORLD_SIZE"]) if os.environ.get("WORLD_SIZE") else None,
            "hidden": HIDDEN, "ffn": FFN, "experts": NUM_EXPERTS, "topk": TOPK,
            "bwd_mode": BWD_MODE, "bwd_rfk": BWD_RFK,
            "bwd_recompute": _bwd_recompute_enabled(),
            "warmup": int(os.environ.get("BENCH_WARMUP", str(WARMUP))),
            "iters": int(os.environ.get("BENCH_ITERS", str(ITERS))),
            "local_mem_bytes": LOCAL_MEM_BYTES,
        },
        "results": records,
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    timing.rank0_print(f"[json] 结构化结果已写入 {path}")


def main():
    from torch_npu.contrib import transfer_to_npu  # noqa: F401

    import time as _time

    rank = int(os.environ.get("RANK", "0"))
    t0 = _time.monotonic()

    def _stage(name):
        now = _time.monotonic()
        print(f"[rank{rank}] [{now - t0:8.3f}s] {name}", flush=True)
        return now

    ip_port = os.environ.get("ASCEND_DEEPEP_SHMEM_IP_PORT", DEFAULT_IP_PORT)
    os.environ.setdefault("SHMEM_UID_SESSION_ID", ip_port.removeprefix("tcp://"))
    seqs = [int(s) for s in os.environ.get("BENCH_SEQS", "128").split(",")]
    warmup = int(os.environ.get("BENCH_WARMUP", str(WARMUP)))
    iters = int(os.environ.get("BENCH_ITERS", str(ITERS)))
    _validate_shape()
    _print_capacity(seqs)
    _bwd_mode = ("重计算口径 16 RFK（对齐 MegaMoe bf16 backward 的 P0-C/P1 重算）"
                 if _bwd_recompute_enabled()
                 else "保存中间量旧口径 12 RFK（不重算，仅对比用）")
    timing.rank0_print(
        f"[bwd] bigop 反向口径: BIGOP_BWD_RECOMPUTE="
        f"{os.environ.get('BIGOP_BWD_RECOMPUTE', '1')}（{_bwd_mode}）")

    t0 = _stage("setup: dist init + fake MoELayer ...")
    ctx = BenchContext(0, 0, None, ip_port, 0)
    setup(ctx)
    t0 = _stage(f"building megamoe (SHMEM {ip_port}, {LOCAL_MEM_BYTES / 1024**3:.1f} GiB window) ...")
    shmem, v2 = build_v2(ctx)
    t0 = _stage("megamoe ready")

    # ---- bigop v1 热身（与 validate 完全一致）----
    # validate 在跑 v2 反向之前先跑一次 bigop v1 的 forward+backward，
    # 完成 HCCL alltoall 组网初始化。bench 脚本此前缺这步，直接建 shmem
    # 跑 fused 反向卡死（launch kernel failure / hang）。
    # 这里用固定 seq=4096 热身一次，不进计时。
    t0 = _stage("bigop v1 HCCL warmup (fwd+bwd, not timed) ...")
    HEATUP_SEQ = 4096
    x_h = torch.randn(HEATUP_SEQ, 1, ctx.K, device="npu", dtype=torch.bfloat16)
    grad_y_h = torch.randn_like(x_h)
    xg = x_h.detach().clone().requires_grad_(True)
    ref_h = BigOp.apply(xg, ctx.moe_layer)
    ref_h.backward(grad_y_h)
    torch.npu.synchronize()
    t0 = _stage(f"HCCL warmup done (seq={HEATUP_SEQ})")

    timing.reset()  # BIGOP_TIMING=1 时清空累计（bigop_fwd/bigop_bwd 分步表），关闭时 no-op
    records: list = []  # 每 seq 一条结构化结果（BENCH_JSON_OUT 时落盘，见 _maybe_dump_json）
    try:
        for seq in seqs:
            if seq > 16384:
                timing.rank0_print(f"[skip] seq={seq} >16384 超算子上限")
                continue
            x = torch.randn(seq, 1, ctx.K, device="npu", dtype=torch.bfloat16)
            grad_y = torch.randn_like(x)

            # ---- v2 三组闭包（每轮重建 x 叶子，backward 后自然释放）----
            # MEGAMOE_TRACE=1 时 bwd 建图带 holder：backward_with_timer 把
            # AscendTimerV2 buffer 写进 holder["backward_timer"]（每轮覆盖，
            # bench 结束后取最后一轮做分阶段归因）
            _v2_diag = MEGAMOE_TRACE or timing.env_enabled("V2_LAYOUT_TIMING")
            v2_fwd_holder = ({
                "timing_tag": "v2_fwd_only",
                "_skip_grad_edge_capture": True,
            } if _v2_diag else None)
            if MEGAMOE_TRACE:
                v2_fwd_holder["capture_forward_timer"] = True
            v2_bwd_holder = ({
                "timing_tag": "v2_bwd_build",
                "_skip_grad_edge_capture": True,
            } if _v2_diag else None)
            if MEGAMOE_TRACE:
                v2_bwd_holder["capture_backward_timer"] = True
            v2_fwd_bwd_holder = ({
                "timing_tag": "v2_fwd_bwd",
                "_skip_grad_edge_capture": True,
            } if _v2_diag else None)

            def v2_fwd():
                xf = x.detach().clone().requires_grad_(True)
                v2(xf, ctx.moe_layer, diagnostic_holder=v2_fwd_holder)

            # 仅反向：每轮先 forward 建图（bench_once 的 build_fn，不计时），
            # Event 只圈 backward。fused backward 每次原地清零 forward_workspace
            # （save 的张量），不能 retain_graph 复用同图，故每轮重建。
            def v2_bwd_build():
                xb = x.detach().clone().requires_grad_(True)
                yb, _aux = v2(xb, ctx.moe_layer, diagnostic_holder=v2_bwd_holder)
                return yb

            def v2_bwd(yb):
                yb.backward(grad_y)

            def v2_fwd_bwd():
                xb = x.detach().clone().requires_grad_(True)
                yb, _aux = v2(
                    xb, ctx.moe_layer, diagnostic_holder=v2_fwd_bwd_holder)
                yb.backward(grad_y)

            # ---- bigop 三组闭包 ----
            def bigop_fwd():
                xf = x.detach().clone().requires_grad_(True)
                BigOp.apply(xf, ctx.moe_layer)

            def bigop_bwd_build():
                xb = x.detach().clone().requires_grad_(True)
                return BigOp.apply(xb, ctx.moe_layer)

            def bigop_bwd(out):
                out.backward(grad_y)

            def bigop_fwd_bwd():
                xb = x.detach().clone().requires_grad_(True)
                out = BigOp.apply(xb, ctx.moe_layer)
                out.backward(grad_y)

            st_f_v2 = bench_once(
                v2_fwd, warmup, iters, clear_timing_tag="v2_fwd_only")
            t0 = _stage(f"megamoe fwd done ({st_f_v2.mean_ms:.2f}ms)")
            st_b_v2 = bench_once(
                v2_bwd, warmup, iters, build_fn=v2_bwd_build,
                clear_timing_tag="v2_bwd_build")
            t0 = _stage(f"megamoe bwd done ({st_b_v2.mean_ms:.2f}ms)")
            st_fb_v2 = bench_once(
                v2_fwd_bwd, warmup, iters,
                clear_timing_tag="v2_fwd_bwd")
            t0 = _stage(f"megamoe fwd+bwd done ({st_fb_v2.mean_ms:.2f}ms)")
            st_f_bp = bench_once(bigop_fwd, warmup, iters)
            t0 = _stage(f"bigop fwd done ({st_f_bp.mean_ms:.2f}ms)")
            st_b_bp = bench_once(bigop_bwd, warmup, iters, build_fn=bigop_bwd_build)
            t0 = _stage(f"bigop bwd done ({st_b_bp.mean_ms:.2f}ms)")
            st_fb_bp = bench_once(bigop_fwd_bwd, warmup, iters)
            t0 = _stage(f"bigop fwd+bwd done ({st_fb_bp.mean_ms:.2f}ms)")

            def _row(tag, st_mm, st_bp):
                """单行报告：event 口径 speedup。"""
                ev_su = st_bp.mean_ms / max(st_mm.mean_ms, 1e-9)
                timing.rank0_print(
                    f"[seq={seq}] {tag:<8} "
                    f"event megamoe={st_mm.mean_ms:8.3f} "
                    f"bigop={st_bp.mean_ms:8.3f} "
                    f"(speedup={ev_su:5.2f}x)")

            _row("fwd", st_f_v2, st_f_bp)
            _row("bwd", st_b_v2, st_b_bp)
            _row("fwd+bwd", st_fb_v2, st_fb_bp)

            # ---- 结构化记录（先入列表，下方各块回填；BENCH_JSON_OUT 落盘）----
            seq_rec = {
                "seq": seq,
                "megamoe": {"fwd": _stat_dict(st_f_v2), "bwd": _stat_dict(st_b_v2),
                            "fwd_bwd": _stat_dict(st_fb_v2)},
                "bigop": {"fwd": _stat_dict(st_f_bp), "bwd": _stat_dict(st_b_bp),
                          "fwd_bwd": _stat_dict(st_fb_bp)},
                "speedup": {
                    "fwd": st_f_bp.mean_ms / max(st_f_v2.mean_ms, 1e-9),
                    "bwd": st_b_bp.mean_ms / max(st_b_v2.mean_ms, 1e-9),
                    "fwd_bwd": st_fb_bp.mean_ms / max(st_fb_v2.mean_ms, 1e-9),
                },
                "megamoe_roofline": None,
                "bigop_decomposition": None,
                "megamoe_kernel_stages": None,
            }
            records.append(seq_rec)

            # v2 反向 roofline 检查（410.4 TFLOPS = 432 峰值×0.95，docs/backward_perf_analysis.md）：
            # floor = BWD_RFK 理论最短耗时（纸面 FLOPs ÷ 410.4）。recompute 模式 16 RFK
            # （含 P0-C 重算），saved 模式 12 RFK（gate/up 由前向保存）。实际只可能
            # >= floor；event - floor = overhead。
            # >=95% 即 cube 已贴 roofline，GEMM 无优化空间。
            _r = seq * TOPK
            _flops_bwd = BWD_RFK * _r * FFN * HIDDEN
            if st_b_v2.mean_ms > 0:
                _floor_ms = _flops_bwd / 410.4e12 * 1e3
                _v2_tf = _flops_bwd / (st_b_v2.mean_ms * 1e-3) / 1e12
                timing.rank0_print(
                    f"[seq={seq}] megamoe bwd roofline [{BWD_MODE}]: "
                    f"event={st_b_v2.mean_ms:.2f}ms, "
                    f"{BWD_RFK:.0f}RFK floor={_floor_ms:.2f}ms"
                    f" -> 实际≈{_v2_tf:.0f} TFLOPS"
                    f"({100.0 * _v2_tf / 410.4:.0f}% of roofline)；"
                    f"event-floor={st_b_v2.mean_ms - _floor_ms:.2f}ms overhead")
                seq_rec["megamoe_roofline"] = {
                    "event_ms": st_b_v2.mean_ms, "floor_ms": _floor_ms,
                    "tflops": _v2_tf,
                    "pct_roofline": 100.0 * _v2_tf / 410.4,
                    "overhead_ms": st_b_v2.mean_ms - _floor_ms,
                }
            timing.rank0_print(
                f"[seq={seq}] detail megamoe fwd n={st_f_v2.n} mean={st_f_v2.mean_ms:.3f} "
                f"p50={st_f_v2.p50_ms:.3f} min={st_f_v2.min_ms:.3f} max={st_f_v2.max_ms:.3f}")
            timing.rank0_print(
                f"[seq={seq}] detail megamoe bwd n={st_b_v2.n} mean={st_b_v2.mean_ms:.3f} "
                f"p50={st_b_v2.p50_ms:.3f} min={st_b_v2.min_ms:.3f} max={st_b_v2.max_ms:.3f}")

            # MEGAMOE_TRACE=1：v2 kernel 分阶段表（holder 里是最后一轮 backward
            # 的 timer buffer；buffer 全零 = kernel 未带 --enable-timer 构建）
            if MEGAMOE_TRACE:
                _ft = v2_fwd_holder.get("forward_timer")
                if _ft is None:
                    timing.rank0_print(
                        "[megamoe] 未能捕获 forward timer，跳过 forward kernel 对账")
                else:
                    from megamoe_timer import summarize_forward
                    _fwd_stages = summarize_forward(
                        _ft,
                        rank_print=timing.rank0_print,
                        dump_csv=str(_out_path(
                            f"{_case_tag()}_rank_{ctx.rank}_forward_time.csv")))
                    _fwd_kernel_us = (
                        _fwd_stages.get("kernel_us") if _fwd_stages else None)
                    if _fwd_kernel_us and _fwd_kernel_us > 0:
                        _fwd_kernel_ms = _fwd_kernel_us / 1000.0
                        _fused_ms = timing.step_stats(
                            "v2_fwd_only").get("fused_forward")
                        message = (
                            f"[seq={seq}] megamoe fwd outside-kernel: "
                            f"event_mean={st_f_v2.mean_ms:.3f}ms "
                            f"kernel_last={_fwd_kernel_ms:.3f}ms "
                            f"event_mean_minus_kernel_last="
                            f"{st_f_v2.mean_ms - _fwd_kernel_ms:.3f}ms")
                        if _fused_ms is not None:
                            message += (
                                f" fused_forward_mean={_fused_ms:.3f}ms "
                                f"fused_mean_minus_kernel_last="
                                f"{_fused_ms - _fwd_kernel_ms:.3f}ms")
                        message += (
                            " exact_single_sample=PASS"
                            if st_f_v2.n == 1
                            else " exact_single_sample=NO(use iters=1 for exact pairing)")
                        timing.rank0_print(message)

                _mt = v2_bwd_holder.get("backward_timer")
                if _mt is None:
                    timing.rank0_print(
                        "[megamoe] 未能捕获 backward timer（fused_op 不支持 "
                        "diagnostic_holder，或本轮无 backward），跳过 kernel 分阶段表")
                else:
                    from megamoe_timer import summarize_backward
                    # CSV 带 case 前缀（含 shape 参数）：run_speed_matrix 多组时互不覆盖
                    _raw_csv = _out_path(f"{_case_tag()}_rank_{ctx.rank}_backward_time.csv")
                    _stages = summarize_backward(
                        _mt, r=seq * TOPK, k=HIDDEN, f=FFN,
                        rank_print=timing.rank0_print,
                        dump_csv=str(_raw_csv))
                    if _stages:
                        seq_rec["megamoe_kernel_stages"] = _stages

                    # GEMM-only roofline：分母 = KERNEL timer（kernel 入口→出口），
                    # 与 bigop 分解行的 GEMM% 同口径（都不含 host 开销），可直接
                    # 对比。event 行的分母含 workspace 分配/autograd/launch，两行
                    # 的差 = host 开销。
                    _kernel_us = _stages.get("kernel_us") if _stages else None
                    if _kernel_us and _kernel_us > 0:
                        _r2 = seq * TOPK
                        _fl = BWD_RFK * _r2 * FFN * HIDDEN
                        _k_ms = _kernel_us / 1000.0
                        _k_tf = _fl / (_k_ms * 1e-3) / 1e12
                        _k_floor = _fl / 410.4e12 * 1e3
                        timing.rank0_print(
                            f"[seq={seq}] megamoe bwd GEMM roofline [{BWD_MODE}]: "
                            f"kernel={_k_ms:.2f}ms (timer), "
                            f"{BWD_RFK:.0f}RFK floor={_k_floor:.2f}ms "
                            f"-> ≈{_k_tf:.0f} TFLOPS"
                            f"({100.0 * _k_tf / 410.4:.0f}% of roofline)"
                            f" ← GEMM 口径，与 bigop GEMM% 可直接对比")
                        if seq_rec["megamoe_roofline"]:
                            seq_rec["megamoe_roofline"]["kernel_ms"] = _k_ms
                            seq_rec["megamoe_roofline"]["kernel_pct_roofline"] = (
                                100.0 * _k_tf / 410.4)
                            # potential speedup：消掉 host / 消掉 host+kernel wait
                            # 各能到多少——直接回答"修 host 值不值"
                            _bp_ms = st_b_bp.mean_ms
                            _mm_ms = st_b_v2.mean_ms
                            timing.rank0_print(
                                f"[seq={seq}] megamoe bwd outside-kernel: "
                                f"event_mean={_mm_ms:.3f}ms "
                                f"kernel_last={_k_ms:.3f}ms "
                                f"event_mean_minus_kernel_last="
                                f"{_mm_ms - _k_ms:.3f}ms "
                                + ("exact_single_sample=PASS"
                                   if st_b_v2.n == 1 else
                                   "exact_single_sample=NO(use iters=1 for exact pairing)"))
                            _su_now = _bp_ms / max(_mm_ms, 1e-9)
                            _su_no_host = _bp_ms / max(_k_ms, 1e-9)
                            _su_ideal = _bp_ms / max(_k_floor, 1e-9)
                            timing.rank0_print(
                                f"[seq={seq}] speedup potential: "
                                f"current={_su_now:.2f}x, "
                                f"if host eliminated={_su_no_host:.2f}x, "
                                f"ideal (host+wait=0)={_su_ideal:.2f}x")
                            seq_rec["megamoe_roofline"]["speedup_no_host"] = _su_no_host
                            seq_rec["megamoe_roofline"]["speedup_ideal"] = _su_ideal

                    # Kernel trace：数据采集阶段只落盘 raw CSV（上方 dump_csv），
                    # trace 生成挪到 shmem.destroy() 之后（见 _generate_traces）——
                    # 在循环内生成时 rank 0 的 parse/JSON 写入会阻塞 SHMEM 的
                    # free_mem barrier（其他 rank 先到 destroy 的 barrier 超时）。

            # BIGOP_TIMING=1 打印 BigOp 分步；V2_LAYOUT_TIMING=1 单独打印
            # v2 router / weight layout transform / fused forward。每个 seq 后清空。
            if (timing.timing_enabled()
                    or timing.env_enabled("V2_LAYOUT_TIMING")):
                timing.report()
                if timing.timing_enabled():
                    _dec = _report_bwd_decomposition(seq)
                    if _dec:
                        seq_rec["bigop_decomposition"] = _dec
                timing.reset()
    finally:
        # 异常退出时仍打印已收集的 BigOp/V2 分步数据；默认关闭时 registry 为空。
        timing.report()
        _maybe_dump_json(records)
        v2.destroy()
        shmem.destroy()
    dist.destroy_process_group()

    # ---- Kernel trace（收尾后生成：纯文件 I/O，不依赖 SHMEM/dist）----
    # 挪到 shmem.destroy() 之后的原因：在循环内生成时，rank 0 的 CSV parse +
    # JSON 写入耗时（8 rank × 几秒）会让其他 rank 在 destroy 的 group barrier
    # 处超时（"free mem barrier failed"）。此处所有进程已完成清理，本进程
    # 独立扫盘生成，无跨进程协调。
    if MEGAMOE_TRACE and int(os.environ.get("LOCAL_RANK", "0")) == 0:
        try:
            _scan_dir = Path(BENCH_OUT_DIR) if BENCH_OUT_DIR else Path(".")
            print(f"[megamoe] generating kernel traces from {_scan_dir} ...",
                  flush=True)
            _mm_dir = Path(__file__).parent.parent / "megamoe_bf16_test"
            sys.path.insert(0, str(_mm_dir))
            from parse_backward_timer import parse_rank as parse_bwd_rank
            from parse_timer_csv import parse_rank as parse_fwd_rank
            from trace_parser import generate_rank_events

            def _gen_traces(glob_pat, phase, parse_fn, parse_args=()):
                csvs = sorted(_scan_dir.glob(glob_pat))
                by_case = {}
                for c in csvs:
                    # 文件名格式: s{seq}_K{K}_F{F}_E{E}_t{topk}_rank_{R}_{phase}_time.csv
                    # case_tag = "_rank_" 之前的部分
                    stem = c.stem  # 去掉 .csv
                    rank_marker = "_rank_"
                    idx = stem.find(rank_marker)
                    if idx < 0:
                        continue
                    case_tag = stem[:idx]
                    rank_str = stem[idx + len(rank_marker):].split("_")[0]
                    try:
                        r = int(rank_str)
                    except ValueError:
                        continue
                    by_case.setdefault(case_tag, []).append((c, r))
                for case_tag, case_csvs in by_case.items():
                    all_events = []
                    merged_ranks = []
                    for r_raw, r in case_csvs:
                        r_parsed = r_raw.with_name(r_raw.stem + "_parsed.csv")
                        parse_fn(r_raw, r_parsed, *parse_args)
                        r_events = generate_rank_events(r_parsed, r, phase)
                        suffix = f"_{phase}_time.csv"
                        r_trace = r_raw.with_name(
                            r_raw.name.replace(suffix, f"_{phase}_trace.json"))
                        with open(r_trace, "w", encoding="utf-8") as fh:
                            json.dump(r_events, fh, indent=2,
                                      ensure_ascii=False)
                        all_events.extend(r_events)
                        merged_ranks.append(r)
                    if all_events:
                        merged = _scan_dir / f"{case_tag}_{phase}_all_rank_trace.json"
                        with open(merged, "w", encoding="utf-8") as fh:
                            json.dump(all_events, fh, indent=2,
                                      ensure_ascii=False)
                        print(f"[megamoe] {phase} trace: {merged.name}"
                              f"（合并 {len(merged_ranks)} 个 rank；"
                              "Chrome Trace 格式，拖到 ui.perfetto.dev）",
                              flush=True)

            _gen_traces("s*_K*_F*_E*_t*_rank_*_backward_time.csv", "backward",
                        parse_bwd_rank, parse_args=(32,))
            _gen_traces("s*_K*_F*_E*_t*_rank_*_forward_time.csv", "forward",
                        parse_fwd_rank)
        except Exception as _exc:
            import traceback
            print(f"[megamoe] kernel trace 生成失败({_exc})，"
                  f"raw CSV 已保留，可手动: python "
                  f"test/megamoe_bf16_test/parse_backward_timer.py ...",
                  flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
