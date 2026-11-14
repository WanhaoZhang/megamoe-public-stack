"""Opt-in diagnostic Timer capture; requires an isolated Timer-ON Full SO.

No production dispatch, planner, autograd or GEMM schedule is replaced. The
proxy exposes timer tensors from the same native replica entry points.
"""
from .megaflux_full import MegaFluxBF16FullTrain


class _TimerReplicaOp:
    def __init__(self, native, buffers):
        self.native, self.buffers = native, buffers

    def __getattr__(self, name):
        return getattr(self.native, name)

    def compute_train_replica(self, *args):
        outputs = self.native.compute_train_replica_with_timer(*args)
        self.buffers['forward'] = outputs[-1]
        return outputs[:-1]

    def backward_replica(self, *args):
        outputs = self.native.backward_replica_with_timer(*args)
        self.buffers['backward'] = outputs[-1]
        return outputs[:-1]


class MegaFluxBF16TraceTrain(MegaFluxBF16FullTrain):
    """Keep sparse readiness milestones and full per-core TimerV2 buffers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, trace=True, **kwargs)
        self.trace_timers = {}
        self.op = _TimerReplicaOp(self.op, self.trace_timers)
