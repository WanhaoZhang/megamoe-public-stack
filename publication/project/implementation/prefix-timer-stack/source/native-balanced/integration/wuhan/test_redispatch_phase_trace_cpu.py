"""CPU-only checks for external probe scopes, calibration and export coverage."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from redispatch_phase_trace import (RedispatchPhaseTrace,
    validate_external_phase_capture, external_phase_events)
from device_clock_calibration import map_device_events
from export_shared_fb_trace import export_shared_fb_traces
from test_clock_calibration_cpu import calibration, SharedFBExportTest


def capture():
    return dict(schema=1, probe_sha256='a'*64, clock='device_system_counter',
        includes_probe_overhead=True, scope_only=True,
        durations_are_not_additive=True, physical_core_id_recorded=False,
        records=[dict(name='SHARED_FORWARD', invocation='forward', device_stream=42,
                      failed=False, before_cycles=list(range(100,196)),
                      after_cycles=list(range(300,396))),
                 dict(name='X_REDISPATCH', invocation='backward', device_stream=42,
                      failed=False, before_cycles=list(range(4000,4096)),
                      after_cycles=list(range(5000,5096)))])


class FakeTensor:
    registry = {}
    reads = 0

    def __init__(self):
        self.values = None
        self.registry[id(self)] = self

    def data_ptr(self): return id(self)
    def detach(self): return self
    def cpu(self):
        FakeTensor.reads += 1
        return self
    def tolist(self): return self.values


class FakeProbe:
    def __init__(self):
        self.library_sha256 = 'a'*64
        self.device = 'fake'
        self.calls = []
        self.torch = SimpleNamespace(int64='int64', empty=lambda *args,**kwargs: FakeTensor(),
            npu=SimpleNamespace(current_stream=lambda device:SimpleNamespace(npu_stream=42)))

    def launch(self, output, stream):
        self.calls.append((output.value, stream.value))
        FakeTensor.registry[output.value].values = [100*len(self.calls)]*96


class ExternalPhaseTraceTest(unittest.TestCase):
    def test_async_unique_buffers_and_deferred_resolution(self):
        FakeTensor.reads = 0
        probe = FakeProbe()
        recorder = RedispatchPhaseTrace(probe)
        with recorder.scope('outer', 'backward'):
            with recorder.scope('inner', 'backward'):
                self.assertEqual(FakeTensor.reads, 0)
                with self.assertRaisesRegex(RuntimeError, 'active'):
                    recorder.resolve()
        self.assertEqual(len(probe.calls), 4)
        self.assertEqual(len({p[0] for p in probe.calls}), 4)
        self.assertEqual(FakeTensor.reads, 0)
        value = recorder.resolve()
        self.assertEqual(FakeTensor.reads, 4)
        self.assertIs(value, recorder.resolve())
        self.assertEqual(value['records'][0]['before_cycles'], [100]*96)
        self.assertEqual(value['records'][0]['after_cycles'], [400]*96)
        with self.assertRaisesRegex(RuntimeError, 'immutable'):
            with recorder.scope('late','backward'): pass

    def test_invalid_probe_data_identity_and_calibration_range_rejected(self):
        for mutate in (lambda d:d['records'][0]['before_cycles'].pop(),
                       lambda d:d['records'][0]['after_cycles'].__setitem__(0,0),
                       lambda d:d['records'][0].__setitem__('failed',True),
                       lambda d:d['records'][0].__setitem__('after_cycles',[110]*96)):
            data = capture(); mutate(data)
            with self.assertRaises(ValueError): validate_external_phase_capture(data, calibration())
        data = capture(); data['probe_sha256'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'SHA'):
            validate_external_phase_capture(data, calibration())
        data = capture(); data['records'][1]['after_cycles'] = [20000]*96
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_external_phase_capture(data, calibration())

    def test_median_cycle_mapping_and_phase_selection(self):
        data = capture()
        events = external_phase_events(data, calibration(), 0, 'forward')
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['args']['start_cycle'],147)
        self.assertEqual(events[0]['args']['end_cycle'],347)
        self.assertTrue(events[0]['args']['includes_probe_overhead'])
        self.assertFalse(events[0]['args']['pure_kernel_active_time'])
        mapped = map_device_events(events, calibration(), 10001)
        self.assertAlmostEqual(mapped[0]['dur'], .2)
        self.assertEqual(len(external_phase_events(data,calibration(),0)),2)

    def test_single_phase_and_combined_exports_include_external_device_scopes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.external-phase-') as temporary:
            root = Path(temporary); source = root/'input'
            frames = [SharedFBExportTest.write_frame(source,rank) for rank in range(2)]
            for frame in frames:
                (frame/'external-phase-timing.json').write_text(json.dumps(capture()))
            audit = export_shared_fb_traces(source,root/'out',2)
            self.assertTrue(audit['external_device_scopes_present'])
            for phase,expected in (('forward',2),('backward',2),('full_fb',4)):
                data = json.loads(Path(audit['traces']['fused/'+phase]).read_text())
                scopes = [e for e in data if e.get('cat') == 'external_device_scope']
                self.assertEqual(len(scopes),expected)
                self.assertTrue(all(e['args']['includes_probe_overhead'] for e in scopes))
                self.assertEqual(sum(e.get('name')=='thread_name' and e.get('tid',0)>=9000 for e in data),expected)
            (frames[1]/'external-phase-timing.json').unlink()
            with self.assertRaisesRegex(ValueError,'complete mode/rank'):
                export_shared_fb_traces(source,root/'missing',2)
            self.assertFalse((root/'missing').exists())


if __name__ == '__main__': unittest.main()
