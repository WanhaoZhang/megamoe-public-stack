"""Checkpoint frame identity and lease retirement, independent of NPU kernels.

The original graph keeps a small logical call record. Checkpoint replay binds
new native state to that same record. Only the original autograd backward owns
the replay lease; the replay's shadow Function must never own a second lease.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import itertools


_scope = ContextVar('megaflux_sft_checkpoint_scope', default=None)
_serial = itertools.count()


@dataclass
class Call:
    layer: str
    call_id: int = field(default_factory=lambda: next(_serial))
    placement: object = None
    live: object = None
    original: object = None
    completed: bool = False
    checkpointed: bool = False
    replayed: bool = False

    def bind(self, lease, replay):
        if self.completed or self.live is not None:
            raise RuntimeError('a completed/live SFT call cannot be rebound')
        if replay:
            if self.original is not None or self.replayed:
                raise RuntimeError('original forward not retired or duplicate checkpoint replay')
            self.replayed = True
            self.live = lease
        elif self.checkpointed:
            if self.original is not None:
                raise RuntimeError('duplicate original checkpoint call')
            self.original = lease
        else:
            self.live = lease

    def retire_original(self):
        if self.original is not None:
            self.original.finish_forward()
            if hasattr(self.original,'_sft_saved'):self.original._sft_saved=None
            self.original = None

    def take_backward(self):
        if self.completed or self.live is None:
            raise RuntimeError('SFT backward lacks its original/replayed native lease')
        return self.live

    def finish_backward(self):
        if self.completed or self.live is None:
            raise RuntimeError('duplicate SFT backward completion')
        self.live.finish_backward()
        if hasattr(self.live,'_sft_saved'):self.live._sft_saved=None
        self.live = None
        self.completed = True


class Frame:
    def __init__(self):
        self.calls = []
        self.original_done = False
        self.replay_done = False

    @contextmanager
    def context(self, replay=False):
        if replay and (not self.original_done or self.replay_done):
            raise RuntimeError('invalid checkpoint replay order')
        cursor = [0]
        token = _scope.set((self, replay, cursor))
        succeeded = False
        try:
            yield
            if replay and cursor[0] != len(self.calls):
                raise RuntimeError('checkpoint replay changed the MoE call count')
            succeeded = True
        finally:
            _scope.reset(token)
            if not replay:
                # Retire issued forward work even on a Python-side failure;
                # the launcher's real exit and post-gate still decide validity.
                for call in self.calls:
                    call.retire_original()
                self.original_done = succeeded
            else:
                self.replay_done = succeeded


def claim(layer):
    scope = _scope.get()
    if scope is None:
        return Call(layer), False
    frame, replay, cursor = scope
    index = cursor[0]; cursor[0] += 1
    if replay:
        if index >= len(frame.calls) or frame.calls[index].layer != layer:
            raise RuntimeError('checkpoint replay changed the MoE layer order')
        return frame.calls[index], True
    call = Call(layer, checkpointed=True)
    frame.calls.append(call)
    return call, False


def checkpoint_function(function, *args, **kwargs):
    """Use at whole-layer boundary, in place of the old recompute wrapper."""
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop
    frame = Frame()
    def contexts():
        return frame.context(False), frame.context(True)
    with set_checkpoint_early_stop(False):
        return checkpoint(function, *args, use_reentrant=False,
                          context_fn=contexts, **kwargs)
