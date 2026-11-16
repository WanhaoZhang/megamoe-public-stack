"""Explicit host-only diagnostic: no new device Events, tensor reads or syncs.

Timestamps are same-host monotonic_ns function envelopes, not device completion.
Records stay in CPU memory until training ends; traced wall times are diagnostic.
"""
import functools
import gc
import json
from pathlib import Path
import sys
import time


def install(out: Path, rank: int):
    from ascend_deepep.fsdp2_bridge import NativeFSDP2Pool
    from ascend_deepep.plan_replay import invocation as plan_invocation
    from ascend_deepep.unified_planner import CommonOnlinePlanner
    from lifetime_device import LifetimeDevice
    from fsdp_turbo.distributed.fine_grained_fully_shard.param_group import FSDPParamGroup
    from mindspeed_llm.fsdp2.optim.optimizer import FunctionalFusedAdamW
    import torch.distributed as dist

    rows = []
    contexts = []
    update = [None]
    sequence = [0]
    gc_starts = {}

    def identity():
        request = contexts[-1] if contexts else plan_invocation()
        if request is None:
            return None, False
        key, recompute = request
        return tuple(key), bool(recompute)

    def save(name, start, finish, key, recompute, seq, failed):
        rows.append(dict(rank=rank, sequence=seq, name=name, update=update[0],
                         invocation=None if key is None else list(key), recompute=recompute,
                         host_begin_ns=start, host_return_ns=finish, failed=failed))

    def call(fn, name, args, kwargs, *, start=None, key=None, recompute=None):
        # First operation at wrapper entry is the host clock read. In particular,
        # no stream query/Event creation occurs before this timestamp.
        start = time.monotonic_ns() if start is None else start
        seq = sequence[0]
        sequence[0] += 1
        if key is None:
            key, observed_recompute = identity()
            recompute = observed_recompute if recompute is None else recompute
        failed = False
        try:
            return fn(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            finish = time.monotonic_ns()
            save(name, start, finish, key, bool(recompute), seq, failed)

    def wrapped(fn, name):
        @functools.wraps(fn)
        def invoke(*args, **kwargs):
            start = time.monotonic_ns()
            return call(fn, name, args, kwargs, start=start)
        return invoke

    class OperatorBoundary:
        def __init__(self, native):
            self.native = native

        def __getattr__(self, name):
            value = getattr(self.native, name)
            if name not in ('compute_train_replica', 'backward_replica', 'backward_replica_redispatch', 'backward_replica_redispatch_with_timer'):
                return value
            phase = 'forward' if name == 'compute_train_replica' else 'backward'
            return wrapped(value, 'operator.'+phase)

    original_forward = NativeFSDP2Pool.forward
    original_backward = NativeFSDP2Pool.backward

    def forward(pool, *args, **kwargs):
        start = time.monotonic_ns()
        if not isinstance(pool.module.op, OperatorBoundary):
            pool.module.op = OperatorBoundary(pool.module.op)
        request = plan_invocation()
        if request is None:
            return call(original_forward, 'MegaMoE.forward', (pool, *args), kwargs, start=start)
        contexts.append(request)
        try:
            return call(original_forward, 'MegaMoE.forward', (pool, *args), kwargs, start=start)
        finally:
            contexts.pop()

    def backward(pool, *args, **kwargs):
        start = time.monotonic_ns()
        placement = args[5] if len(args) > 5 else kwargs['placement']
        # data_ptr reads host metadata only; it does not materialize tensor data.
        matches = [key for key, ticket in pool._plan_bodies.items()
                   if placement[0].data_ptr() == ticket.logical_ids.data_ptr()]
        if len(matches) != 1:
            raise RuntimeError('Host diagnostic backward has no unique saved plan')
        contexts.append((matches[0], False))
        try:
            return call(original_backward, 'MegaMoE.backward', (pool, *args), kwargs, start=start)
        finally:
            contexts.pop()

    NativeFSDP2Pool.forward = forward
    NativeFSDP2Pool.backward = backward
    NativeFSDP2Pool._publish = wrapped(NativeFSDP2Pool._publish, 'owner.publish')
    CommonOnlinePlanner._compile = wrapped(CommonOnlinePlanner._compile, 'planner.compile')
    for method in ('begin', 'prepare', 'commit', 'gradients', '_drain', '_add_owner',
                   'flush', 'window_boundary', 'finish'):
        setattr(LifetimeDevice, method, wrapped(getattr(LifetimeDevice, method), 'lifetime.'+method))
    for method in ('unshard', 'wait_for_unshard', 'post_backward', 'finalize_backward'):
        if hasattr(FSDPParamGroup, method):
            setattr(FSDPParamGroup, method, wrapped(getattr(FSDPParamGroup, method), 'FSDP.'+method))
    dist.batch_isend_irecv = wrapped(dist.batch_isend_irecv, 'P2P.submit')
    dist.all_gather_object = wrapped(dist.all_gather_object, 'collective.all_gather_object')
    FunctionalFusedAdamW.step = wrapped(FunctionalFusedAdamW.step, 'optimizer.AdamW')
    main = sys.modules['__main__']
    main.clip_grad_norm = wrapped(main.clip_grad_norm, 'optimizer.global_clip')
    original_set_step = main.set_step

    def set_step(step):
        update[0] = step
        return original_set_step(step)

    main.set_step = set_step

    def gc_event(phase, info):
        stamp = time.monotonic_ns()
        generation = info['generation']
        if phase == 'start':
            gc_starts[generation] = (stamp, identity(), sequence[0])
            sequence[0] += 1
        elif generation in gc_starts:
            began, (key, recompute), seq = gc_starts.pop(generation)
            save('python.gc'+str(generation), began, stamp, key, recompute, seq, False)

    gc.callbacks.append(gc_event)

    def dump():
        gc.callbacks.remove(gc_event)
        with (out/f'host-boundary-rank{rank}.jsonl').open('w') as handle:
            for row in sorted(rows, key=lambda row: row['sequence']):
                handle.write(json.dumps(row)+'\n')
        (out/f'host-boundary-contract-rank{rank}.json').write_text(json.dumps(dict(
            diagnostic_only=True, records=len(rows), clock='same-host monotonic_ns',
            no_new_device_events=True, no_new_device_syncs=True, no_tensor_snapshots=True,
            dump_after_training=True, host_return_is_device_completion=False), indent=2))
    return dump
