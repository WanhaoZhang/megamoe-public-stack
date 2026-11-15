"""Online ready-set selection, independent of parameters and execution streams."""
from dataclasses import dataclass
from itertools import combinations
from math import ceil, isfinite


@dataclass(frozen=True, order=True)
class TaskKey:
    """Logical training invocation; optimizer step is also its weight version."""
    step: int
    microbatch: int
    layer: int
    invocation: int = 0


@dataclass(frozen=True)
class ReadyTask:
    """Observed ready task only; rank_cost excludes unknown future work."""
    key: TaskKey
    rank_cost: tuple[float, ...]
    deferred: int = 0

    def __post_init__(self):
        if not self.rank_cost or any(not isfinite(cost) or cost < 0 for cost in self.rank_cost):
            raise ValueError('Rank costs must be finite and nonnegative')
        if self.deferred < 0:
            raise ValueError('Negative deferral count')


@dataclass(frozen=True)
class WindowPlan:
    """Immutable selection; existing per-task execution plans remain authoritative."""
    selected: tuple[TaskKey, ...]
    deferred: tuple[TaskKey, ...]
    utilization: float | None
    complementarity: float | None


def _scores(tasks):
    ranks = len(tasks[0].rank_cost)
    aggregate = [sum(task.rank_cost[rank] for task in tasks) for rank in range(ranks)]
    peak = max(aggregate)
    serial = sum(max(task.rank_cost) for task in tasks)
    return (sum(aggregate) / (ranks * peak) if peak else 1.0,
            1.0 - peak / serial if serial else 0.0)


def select_window(tasks: tuple[ReadyTask, ...], policy: str, window_size: int) -> WindowPlan:
    """Select FIFO, same-layer fixed groups or EasyBalance-style MaxUtil.

    MaxUtil uses min-runset ceil(W/2), forced admission after two deferrals,
    and deterministic work/identity tie breaks. Scores are diagnostic linear
    compute estimates, not predicted end-to-end training durations.
    """
    if policy not in ('fifo', 'fixed_group', 'easybalance', 'easybalance_upstream') or window_size not in (1, 2, 4):
        raise ValueError('Unsupported policy or window size')
    if not tasks:
        return WindowPlan((), (), 1.0, 0.0)
    arrival = tasks
    tasks = tuple(sorted(tasks, key=lambda task: task.key))
    if len({task.key for task in tasks}) != len(tasks):
        raise ValueError('Duplicate ready task')
    if len({task.key.step for task in tasks}) != 1 or len({len(task.rank_cost) for task in tasks}) != 1:
        raise ValueError('Mixed weight versions or rank counts')
    if len(tasks) > window_size:
        raise ValueError('Ready set exceeds admitted window')
    forced = {task.key for task in tasks if task.deferred >= 2}
    if policy == 'fifo':
        minimum = min(len(tasks),ceil(window_size/2))
        chosen = [task for task in arrival if task.key in forced]
        for task in arrival:
            if len(chosen)>=minimum:break
            if task.key not in forced:chosen.append(task)
        selected = tuple(sorted(chosen,key=lambda task:task.key))
    elif policy == 'fixed_group':
        layer = next((task.key.layer for task in tasks if task.key in forced), tasks[0].key.layer)
        selected = tuple(task for task in tasks if task.key.layer == layer or task.key in forced)
    elif policy=='easybalance_upstream':
        minimum=min(len(tasks),ceil(window_size/2))
        # Author code enumerates numeric masks in ready-arrival order, keeps
        # the first maximizing mask, then appends forced tasks. Execution
        # identity order remains canonical for the training collective graph.
        candidates=[]
        for mask in range(1,1<<len(arrival)):
            subset=tuple(task for index,task in enumerate(arrival) if mask&(1<<index))
            selected_keys={task.key for task in subset}|forced
            if len(subset)>=minimum and len({task.key.layer for task in tasks if task.key in selected_keys})<=2:
                candidates.append(subset)
        if not candidates:raise ValueError('No legal upstream-style runset in the admitted layer domain')
        subset=max(candidates,key=lambda value:_scores(value)[0])
        selected_keys={task.key for task in subset}|forced
        selected=tuple(task for task in tasks if task.key in selected_keys)
    else:
        minimum = min(len(tasks), ceil(window_size / 2))
        candidates = (subset for size in range(minimum, len(tasks)+1)
                      for subset in combinations(tasks, size)
                      if forced.issubset({task.key for task in subset}) and
                      len({task.key.layer for task in subset})<=2)
        selected = max(candidates, key=lambda subset: (_scores(subset)[0],
                       sum(sum(task.rank_cost) for task in subset),
                       tuple(tuple(-value for value in (task.key.step, task.key.microbatch,
                             task.key.layer, task.key.invocation)) for task in subset)))
    keys = tuple(task.key for task in selected)
    utilization, complementarity = _scores(selected)
    return WindowPlan(keys, tuple(task.key for task in tasks if task.key not in keys),
                      utilization, complementarity)


def compile_waves(keys:tuple[TaskKey,...],tokens:int,*,independent:bool=False
                  )->tuple[tuple[tuple[int,int,int],...],...]:
    """Coexecute the entire legal runset, preserving static Native row count.

    Every task contributes one chunk to every wave. Cyclic chunk lengths make
    even three-task groups total exactly S rows, preserving each task's order.
    Attention is already complete and its sequence is never cut here.
    """
    if not keys or len(keys)>4 or len(set(keys))!=len(keys) or tokens%32:
        raise ValueError('Invalid ready runset/token alignment')
    if independent:return tuple(((index,0,tokens),) for index in range(len(keys)))
    if len({key.layer for key in keys})>2:
        raise ValueError('A coexecuted runset exceeds the 64-home expert domain')
    count=len(keys)
    size=(tokens//(count*32))*32
    if not size:raise ValueError('Not enough rows to partition a runset')
    lengths=(size,)*(count-1)+(tokens-size*(count-1),)
    boundaries=[]
    for task in range(count):
        cursor=0;offsets=[0]
        for chunk in range(count):
            cursor+=lengths[(chunk+task)%count]
            offsets.append(cursor)
        boundaries.append(offsets)
    return tuple(tuple((index,boundaries[index][wave],boundaries[index][wave+1])
        for index in range(count)) for wave in range(count))
