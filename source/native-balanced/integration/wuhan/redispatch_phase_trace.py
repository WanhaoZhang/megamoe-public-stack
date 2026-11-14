"""Diagnostic device scopes around external eager phases, using the clock SO.

The scope enqueues two independent 96-lane probes on the caller's stream.
It never synchronizes a stream. Call resolve/save only after the complete-F/B
capture's final synchronize. These are enclosing scopes, include probe and
host-supply overhead, and are neither pure kernel time nor additive costs.
"""
from contextlib import contextmanager
import ctypes
import json
from pathlib import Path
import re


SCHEMA = 1
LANES = 96


class RedispatchPhaseTrace:
    def __init__(self, probe):
        self.probe = probe
        self._pending = []
        self._resolved = None
        self._active = 0

    @contextmanager
    def scope(self, name, invocation):
        if self._resolved is not None:
            raise RuntimeError('resolved external phase capture is immutable')
        if not name or invocation not in ('forward', 'backward'):
            raise ValueError('a phase name and forward/backward invocation are required')
        torch = self.probe.torch
        stream = torch.npu.current_stream(self.probe.device)
        before = torch.empty(LANES, dtype=torch.int64, device=self.probe.device)
        after = torch.empty(LANES, dtype=torch.int64, device=self.probe.device)
        item = dict(name=str(name), invocation=invocation,
                    device_stream=int(stream.npu_stream), failed=False)
        self._pending.append((item, before, after))
        self.probe.launch(ctypes.c_void_p(before.data_ptr()), ctypes.c_void_p(stream.npu_stream))
        self._active += 1
        try:
            yield
        except BaseException:
            item['failed'] = True
            raise
        finally:
            self._active -= 1
            end_stream = torch.npu.current_stream(self.probe.device)
            if int(end_stream.npu_stream) != item['device_stream']:
                item['failed'] = True
                raise RuntimeError('external phase changed the current stream')
            self.probe.launch(ctypes.c_void_p(after.data_ptr()), ctypes.c_void_p(end_stream.npu_stream))

    def resolve(self):
        if self._active:
            raise RuntimeError('cannot resolve an active external phase scope')
        if self._resolved is None:
            records = []
            for item, before, after in self._pending:
                records.append(dict(item,
                    before_cycles=before.detach().cpu().tolist(),
                    after_cycles=after.detach().cpu().tolist()))
            self._resolved = dict(schema=SCHEMA, probe_sha256=self.probe.library_sha256,
                records=records, clock='device_system_counter',
                includes_probe_overhead=True, scope_only=True,
                durations_are_not_additive=True, physical_core_id_recorded=False,
                no_explicit_sync_during_scopes=True,
                semantics='current-stream probe-enclosed phase; includes probe, queued dependencies and host supply gaps; not pure kernel active time')
            # Numerical validation here has no calibration dependency; export
            # additionally verifies the probe SHA and calibrated anchor range.
            validate_external_phase_capture(self._resolved)
            self._pending.clear()
        return self._resolved

    def save(self, frame):
        path = Path(frame) / 'external-phase-timing.json'
        with path.open('x', encoding='utf-8') as stream:
            json.dump(self.resolve(), stream, indent=2)
        return path


def _median_cycle(values):
    ordered = sorted(values)
    return (ordered[47] + ordered[48]) // 2


def validate_external_phase_capture(capture, calibration=None):
    if (capture.get('schema') != SCHEMA
            or not re.fullmatch('[a-f0-9]{64}', capture.get('probe_sha256', ''))
            or capture.get('clock') != 'device_system_counter'
            or capture.get('includes_probe_overhead') is not True
            or capture.get('scope_only') is not True
            or capture.get('durations_are_not_additive') is not True
            or capture.get('physical_core_id_recorded') is not False
            or not isinstance(capture.get('records'), list)):
        raise ValueError('invalid external phase capture identity/semantics')
    if calibration is not None:
        if (capture['probe_sha256'] != calibration['pre_batch']['library_sha256']
                or capture['probe_sha256'] != calibration['post_batch']['library_sha256']):
            raise ValueError('external phase clock probe SHA differs from calibration')
    for item in capture['records']:
        if (not isinstance(item.get('name'), str) or not item['name']
                or item.get('invocation') not in ('forward', 'backward')
                or type(item.get('device_stream')) is not int or item['device_stream'] < 0
                or item.get('failed') is not False):
            raise ValueError('invalid/failed external phase record')
        for field in ('before_cycles', 'after_cycles'):
            values = item.get(field)
            if (not isinstance(values, list) or len(values) != LANES
                    or any(type(value) is not int or value <= 0 for value in values)):
                raise ValueError('external phase requires positive complete 96-lane probes')
        if max(item['before_cycles']) >= min(item['after_cycles']):
            raise ValueError('external phase probe boundaries overlap or run backwards')
        if calibration is not None:
            # Every preserved lane must lie within the real calibration anchors.
            from device_clock_calibration import map_cycle
            mapping = calibration['mapping']
            map_cycle(mapping, min(item['before_cycles']))
            map_cycle(mapping, max(item['after_cycles']))
    return True


def external_phase_events(capture, calibration, rank, invocation=None):
    """Raw-cycle scope events, ready for the normal device-clock mapper."""
    validate_external_phase_capture(capture, calibration)
    if invocation is not None and invocation not in ('forward', 'backward'):
        raise ValueError('invalid external phase selection')
    events = []
    for index, item in enumerate(capture['records']):
        if invocation is not None and item['invocation'] != invocation:
            continue
        before, after = item['before_cycles'], item['after_cycles']
        events.append(dict(name='EXTERNAL_SCOPE/' + item['name'], ph='X', pid=rank,
            tid=9000 + index, cat='external_device_scope', args=dict(
                start_cycle=_median_cycle(before), end_cycle=_median_cycle(after),
                invocation=item['invocation'], phase_name=item['name'],
                source='independent_clock_probe_scope_schema1',
                probe_sha256=capture['probe_sha256'], device_stream=item['device_stream'],
                before_lane_spread_cycles=max(before)-min(before),
                after_lane_spread_cycles=max(after)-min(after),
                includes_probe_overhead=True, scope_only=True,
                durations_are_not_additive=True, pure_kernel_active_time=False,
                physical_core_id_recorded=False)))
    return events


def external_phase_metadata(events, rank):
    return [dict(ph='M', pid=rank, tid=item['tid'], name='thread_name',
                 args=dict(name='external device scope / ' + item['args']['phase_name']))
            for item in events if item.get('cat') == 'external_device_scope']
