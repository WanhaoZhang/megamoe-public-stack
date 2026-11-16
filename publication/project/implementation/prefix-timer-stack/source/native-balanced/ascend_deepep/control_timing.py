"""Opt-in host scopes and optional diagnostic current-stream Event intervals.

Disabled scopes do not read a clock. Default recorders do not import Torch.
Event recorders are explicitly opt-in; resolving them never synchronizes.
Nested durations are deliberately retained and must never be summed.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import math

_RECORDER = ContextVar('ascend_deepep_control_timing', default=None)


class ControlTimingRecorder:
    def __init__(self, device_events=False):
        self.records = []
        self.device_events = bool(device_events)
        self._pending_events = []
        self._torch = None
        if self.device_events:
            import torch
            self._torch = torch

    def resolve_device_timings(self):
        """After caller's final device sync, resolve without adding a sync.

        Event elapsed time spans current-stream execution/queue dependencies
        and can include host supply gaps. It is not pure kernel active time.
        """
        resolved = []
        for record, begin, end in self._pending_events:
            elapsed = float(begin.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('invalid control timing Event elapsed_time')
            resolved.append((record, elapsed))
        for record, elapsed in resolved:
            record.update(device_elapsed_ms=elapsed, device_timing_resolved=True)
        self._pending_events.clear()
        return self.records


@contextmanager
def record_control_timing(recorder):
    """Use a recorder only for the current context; restore it on exceptions."""
    if recorder is not None and not isinstance(recorder, ControlTimingRecorder):
        raise TypeError('expected ControlTimingRecorder or None')
    token = _RECORDER.set(recorder)
    try:
        yield recorder
    finally:
        _RECORDER.reset(token)


class _DisabledScope:
    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        return False


_DISABLED = _DisabledScope()


class _Scope:
    def __init__(self, recorder, phase):
        self.recorder, self.phase = recorder, str(phase)

    def __enter__(self):
        if self.recorder.device_events:
            npu = self.recorder._torch.npu
            self.begin_event = npu.Event(enable_timing=True)
            self.end_event = npu.Event(enable_timing=True)
            self.begin_stream = int(npu.current_stream().npu_stream)
            self.begin_event.record()  # default current stream, same API as timing.py
        self.start = time.monotonic_ns()
        return self

    def __exit__(self, kind, value, traceback):
        record = dict(phase=self.phase, start_ns=self.start,
            end_ns=time.monotonic_ns(), clock='host_monotonic_ns',
            semantics='host_enqueue_or_python_scope', failed=kind is not None)
        self.recorder.records.append(record)
        if self.recorder.device_events:
            npu = self.recorder._torch.npu
            end_stream = int(npu.current_stream().npu_stream)
            record.update(device_timing_resolved=False, device_stream=self.begin_stream,
                device_timing_semantics='current-stream Event interval; includes queued execution, dependency waits and host supply gaps; not pure kernel active time',
                device_timing_diagnostic_overhead=True, nested_scopes_are_not_additive=True)
            if self.begin_stream != end_stream:
                record['device_timing_error'] = 'scope changed current stream; no valid same-stream interval'
                if kind is None:
                    raise RuntimeError(record['device_timing_error'])
                return False
            self.end_event.record()
            self.recorder._pending_events.append((record,self.begin_event,self.end_event))
        return False


def control_timing_scope(phase):
    recorder = _RECORDER.get()
    return _DISABLED if recorder is None else _Scope(recorder, phase)
