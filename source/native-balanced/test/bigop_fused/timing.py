"""timing: bigop 用 NPU 计时工具（可复用、纯计时，编排逻辑交给调用方）。

三种计时方式（对应常用三模版）:
  1. bench_wall(fn)                       —— CPU 墙钟: perf_counter + torch.npu.synchronize()
                                            （含 launch/同步开销，端到端视角）
  2. bench_event(fn)                      —— 设备侧纯 NPU 耗时: torch.npu.Event(enable_timing=True)
                                            elapsed_time 返回 ms；recorded_time 官方文档为微秒
                                            (beta)，/1000.0 换算并交叉校验（模版里 "/1_000_000
                                            纳秒" 的注释是错的）
  3. profile_fn({"name": fn}) / profile_block(dir) —— torch_npu.profiler 采集
                                            （CPU+NPU activities, Level1, tensorboard_trace_handler 导出）

前向每步打点（已嵌入 BigOp.forward，其他前向亦可复用）:
  StepTimer.tick(name) 在当前流 record Event（不 sync、不打断流水线）；finish() 统一 sync
  后相邻 Event 做差得每步 ms。开关: 环境变量 BIGOP_TIMING=1（默认关，_NullTimer 零开销）。
  get_step_timer(tag) 的每次 finish 自动累计进全局 registry，调用方在合适时机 report() 打印。
  BigOpV2 的布局转换诊断可独立用 V2_LAYOUT_TIMING=1 开启，不影响 BIGOP_TIMING。

约定: 仅 import torch / torch_npu / 标准库；多 rank 下仅 rank0 打印（RANK/LOCAL_RANK）。
用法: 各 run_*.sh 已 export PYTHONPATH=$repo_root，直接 `import timing`。
"""
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import torch
import torch_npu


# ---------- 统计与输出基建 ----------

def rank0_print(*args):
    """多 rank 下仅 rank0 打印（torchrun 注入 RANK/LOCAL_RANK；缺失视为单进程 rank0）。"""
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        rank = int(rank)
    except ValueError:
        rank = 0
    if rank == 0:
        print(*args, flush=True)


@dataclass
class BenchStats:
    """一批计时样本（单位 ms）的统计视图；样本保留在 samples 里供自行分析。
    """

    samples: List[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def mean_ms(self) -> float:
        return sum(self.samples) / self.n if self.samples else float("nan")

    @property
    def min_ms(self) -> float:
        return min(self.samples) if self.samples else float("nan")

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else float("nan")

    @property
    def p50_ms(self) -> float:
        if not self.samples:
            return float("nan")
        xs = sorted(self.samples)
        m = len(xs) // 2
        return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

    def __str__(self):
        return (f"n={self.n} mean={self.mean_ms:.3f}ms p50={self.p50_ms:.3f}ms "
                f"min={self.min_ms:.3f}ms max={self.max_ms:.3f}ms")


def stats_from_ms(samples) -> BenchStats:
    return BenchStats(list(samples))


# ---------- 方式一: CPU 墙钟（perf_counter + synchronize） ----------

def bench_wall(fn: Callable, warmup: int = 10, iters: int = 1) -> BenchStats:
    """模版一: 预热丢弃冷启动（图编译/显存分配），逐次 sync 计时，含 launch 与同步开销。

    iters=1 时即模版"跑一次报一个毫秒数"；iters>1 时每次迭代单独计时取统计。
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(iters):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return stats_from_ms(samples)


# ---------- 方式二: Event 设备侧纯 NPU 耗时 ----------

def bench_event(fn: Callable, warmup: int = 10, iters: int = 1,
                cross_check: bool = True) -> BenchStats:
    """模版二: torch.npu.Event(enable_timing=True) 测纯 NPU 耗时（不含 CPU launch 开销）。

    统计以 elapsed_time(start, end) 为准（官方文档: 返回 ms，与 CUDA 语义一致）。
    交叉校验: recorded_time() 官方文档为微秒（beta 接口），ms = 差值 / 1000.0；
    注意常见模版里 "/ 1_000_000.0（纳秒）" 的注释是错的。两法均值偏差 >2% 时打 warning。
    """
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    pairs = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        pairs.append((start, end))
    torch.npu.synchronize()  # Event 只需最后统一同步，无须逐次阻塞
    samples = [s.elapsed_time(e) for s, e in pairs]
    if cross_check:
        alt = []
        for s, e in pairs:
            try:
                t_end, t_start = e.recorded_time(), s.recorded_time()
                if isinstance(t_end, (int, float)) and isinstance(t_start, (int, float)) \
                        and t_end >= t_start:
                    alt.append((t_end - t_start) / 1000.0)  # 微秒 -> ms
            except Exception:
                pass  # 部分 torch_npu 版本无此 beta 接口，跳过校验
        if alt and samples:
            m_elapsed = sum(samples) / len(samples)
            m_recorded = sum(alt) / len(alt)
            if abs(m_recorded - m_elapsed) > 0.02 * m_elapsed:
                rank0_print(f"[timing] WARNING: elapsed_time={m_elapsed:.3f}ms 与 "
                            f"recorded_time差/1000={m_recorded:.3f}ms 偏差>2%，请核对单位语义")
    return stats_from_ms(samples)


def bench_event_synced(fn, warmup: int = 0, iters: int = 1) -> BenchStats:
    """设备侧纯 NPU 耗时，每轮迭代前后 torch.npu.synchronize()。

    与 bench_event 的区别：深队列异步对 SHMEM/EP 算子（MegaMoeBF16Train）
    会触发 launch failure / hang，逐次同步强制每轮完全结束再进下一轮，
    隔离迭代间 SHMEM workspace 状态。sync 在计时窗口外，不影响 Event
    测得的两点间设备耗时。warmup/iters 默认 0/1（单轮，最小化 SHMEM 交互）。
    """
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    pairs = []
    for _ in range(iters):
        torch.npu.synchronize()
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        pairs.append((start, end))
    samples = [s.elapsed_time(e) for s, e in pairs]
    return stats_from_ms(samples)


def bench_event_with_build(build_fn, fn, warmup: int = 10, iters: int = 1,
                           cross_check: bool = True) -> BenchStats:
    """设备侧纯 NPU 耗时；每次迭代先 build_fn() 建图（不计时），再 fn() 计时。

    适用：backward 不能 retain_graph 复用同一张图（前向保存的张量会被 C++ 反向
    原地修改，第 2 次 backward 触发 PyTorch inplace 版本校验失败），需每迭代
    重建前向图；Event 只圈住 fn() 部分，build 的开销不计入耗时。

    build_fn() 返回本次要 backward 的输出；fn(out) 执行 backward。warmup 同样
    build+fn 对齐真实迭代，但不采集样本。
    """
    for _ in range(warmup):
        out = build_fn()
        fn(out)
    torch.npu.synchronize()
    pairs = []
    for _ in range(iters):
        # 逐次同步：SHMEM/EP 算子的 forward 会写 progress、backward 会原地清零，
        # 无同步深队列下多轮迭代会争用同一块 workspace 导致 launch 失败。
        # 同步只隔离迭代，不影响 Event 测得的两点间设备耗时。
        torch.npu.synchronize()
        out = build_fn()  # 建图不计时
        torch.npu.synchronize()  # 前向的 SHMEM 通信完成后再计 backward
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn(out)
        end.record()
        torch.npu.synchronize()  # backward 完全结束再进下一轮，隔离 SHMEM 状态
        pairs.append((start, end))
    torch.npu.synchronize()  # Event 只需最后统一同步，无须逐次阻塞
    samples = [s.elapsed_time(e) for s, e in pairs]
    if cross_check:
        alt = []
        for s, e in pairs:
            try:
                t_end, t_start = e.recorded_time(), s.recorded_time()
                if isinstance(t_end, (int, float)) and isinstance(t_start, (int, float)) \
                        and t_end >= t_start:
                    alt.append((t_end - t_start) / 1000.0)  # 微秒 -> ms
            except Exception:
                pass  # 部分 torch_npu 版本无此 beta 接口，跳过校验
        if alt and samples:
            m_elapsed = sum(samples) / len(samples)
            m_recorded = sum(alt) / len(alt)
            if abs(m_recorded - m_elapsed) > 0.02 * m_elapsed:
                rank0_print(f"[timing] WARNING: elapsed_time={m_elapsed:.3f}ms 与 "
                            f"recorded_time差/1000={m_recorded:.3f}ms 偏差>2%，请核对单位语义")
    return stats_from_ms(samples)


# ---------- 方式三: torch_npu.profiler 采集 ----------

@contextmanager
def profile_block(out_dir: str = "./bench_trace", record_shapes: bool = True,
                  skip_first: int = 0):
    """模版三上下文: torch_npu.profiler 采集块（CPU+NPU, Level1, trace 导出到 out_dir）。

    schedule(wait=0, warmup=0, active=1, repeat=1) 单块采集一次；repeat>1 会在同一目录生成
    多份数据需手动按时间戳拆分（官方文档建议 repeat=1）。动态 shape 场景可传 skip_first=10
    跳过前若干轮保证数据稳定（官方建议）。record_shapes 需开启 CPU activity 才生效（已开）。
    """
    prof = torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU],
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=1, skip_first=skip_first),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
        record_shapes=record_shapes,  # 采集 torch op 的 input shape/type
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
    )
    with prof:
        yield prof
        prof.step()  # 结束 active=1 的采集窗口并触发 trace 导出


def profile_fn(fns: Dict[str, Callable], out_dir: str = "./bench_trace",
               record_shapes: bool = True, skip_first: int = 0):
    """模版三便捷入口: 同一采集块内按序执行多个命名函数（如 ref 版与融合版同块对比）。"""
    rank0_print(f"[timing] profiler 采集 {list(fns)} -> {out_dir}")
    with profile_block(out_dir, record_shapes=record_shapes, skip_first=skip_first):
        for fn in fns.values():
            fn()
    rank0_print(f"[timing] trace 已导出到 {out_dir}/（tensorboard / MindStudio Insight 解析）")


# ---------- 每步打点（嵌入前向用，非阻塞） ----------

def env_enabled(name: str) -> bool:
    """Return whether an optional diagnostic environment switch is enabled."""
    return os.environ.get(name, "").lower() in ("1", "true", "on", "yes")


def timing_enabled() -> bool:
    """BIGOP_TIMING=1/true/on 开启每步打点；默认关闭（_NullTimer 零开销）。"""
    return env_enabled("BIGOP_TIMING")


class StepTimer:
    """单次调用内的每步打点: tick() 在当前流 record Event（不 sync 不打断流水线），
    finish() 统一 sync 后相邻 Event 做差 -> {step_name: ms}，并累计进全局 registry。

    耗时归属到"结束打点"的步骤名（首个 tick 是基准点不产生耗时）。注意相邻做差得到的是
    "该步对设备流的贡献时长"——若某步主要是 CPU 侧工作（流空闲等待 launch），该段空闲
    也计入该步；调试视角下这正是每步的端到端贡献。
    """

    def __init__(self, sink: dict = None):
        self._events = []  # [(step_name, Event)]
        self._sink = sink  # 全局 registry 条目（dict）或 None

    def tick(self, name: str):
        ev = torch.npu.Event(enable_timing=True)
        ev.record()
        self._events.append((name, ev))

    def finish(self) -> Dict[str, float]:
        if not self._events:
            return {}
        torch.npu.synchronize()
        res = {}
        for (name_a, ev_a), (name_b, ev_b) in zip(self._events, self._events[1:]):
            res[name_b] = ev_a.elapsed_time(ev_b)  # ms
        self._events = []
        if self._sink is not None:
            samples = self._sink["samples"]
            for step, ms in res.items():
                if step not in samples:
                    self._sink["order"].append(step)
                    samples[step] = []
                samples[step].append(ms)
        return res


class _NullTimer:
    """关闭态 no-op: tick/finish 均空操作，嵌入代码无须分支判断。"""

    def tick(self, name: str):
        pass

    def finish(self) -> Dict[str, float]:
        return {}


_NULL_TIMER = _NullTimer()
_REGISTRY = {}  # tag -> {"order": [step...], "samples": {step: [ms...]}}


def get_step_timer(tag: str, enabled=None):
    """取一个绑定 tag 的打点计时器（BIGOP_TIMING 关闭时返回 no-op 单例）。

    每次调用返回新 StepTimer 但共享 tag 的累计 registry，跨多次 forward 自动聚合。
    enabled 可为独立诊断显式覆盖默认 BIGOP_TIMING 开关。
    """
    if enabled is None:
        enabled = timing_enabled()
    if not enabled:
        return _NULL_TIMER
    return StepTimer(sink=_REGISTRY.setdefault(tag, {"order": [], "samples": {}}))


def reset():
    """清空全局累计（新一轮测试前调用）。"""
    _REGISTRY.clear()


def clear_tag(tag: str):
    """只清一个诊断 tag，保留同轮其他阶段已经收集的样本。"""
    _REGISTRY.pop(tag, None)


def report():
    """打印所有 tag 的每步累计统计（跨多次 finish 聚合，含占总耗时百分比；rank0 视角）。"""
    for tag, entry in _REGISTRY.items():
        rank0_print(f"==== [timing] 每步耗时: {tag} ====")
        total = sum(sum(v) for v in entry["samples"].values())
        for step in entry["order"]:
            st = stats_from_ms(entry["samples"][step])
            pct = 100.0 * sum(entry["samples"][step]) / total if total else 0.0
            rank0_print(f"  {step:<18} {st}  ({pct:.1f}%)")


def step_stats(tag: str) -> Dict[str, float]:
    """取某个 tag 的每步均值（ms）；无累计数据时返回 {}。

    供调用方在 report() 之外做归因计算（如 bwd 的 GEMM/通信分解行）。
    """
    entry = _REGISTRY.get(tag)
    if not entry:
        return {}
    return {step: sum(v) / len(v) for step, v in entry["samples"].items() if v}
