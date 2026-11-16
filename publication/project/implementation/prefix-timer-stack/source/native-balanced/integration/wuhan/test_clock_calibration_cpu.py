"""No device/Torch imports: real clock math, failure boundaries and trace IO."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import benchmark_shared_expert_forward as forward
import device_clock_calibration as clock
from export_shared_fb_trace import export_shared_fb_traces


IDENTITY = dict(hostname='same-host',boot_id='test-boot',clock='CLOCK_MONOTONIC/time.monotonic_ns',single_host_only=True)


def batch(cycle,host,width=100):
    return dict(samples=[dict(cycles=[cycle]*96,host_before_ns=host,host_after_ns=host+width)],
        host_identity=IDENTITY,library_sha256='a'*64)


def calibration(rank=0,offset=0):
    return clock.make_capture_calibration(batch(1+offset,10001),batch(10001+offset,20001),rank,2)


class ClockMathTest(unittest.TestCase):
    def test_minimum_bracket_no_planner_alignment(self):
        a=batch(100,10000,width=400)
        a['samples'].append(dict(cycles=[200]*96,host_before_ns=10500,host_after_ns=10600))
        selected=clock.select_probe_sample(a['samples'])
        self.assertEqual((selected['index'],selected['bracket_ns']),(1,100))
        m=clock.build_clock_mapping(batch(1000,20000),batch(11000,40000))
        self.assertEqual(m['ns_per_cycle'],2)
        self.assertEqual(clock.map_cycle(m,6000),(30050.,50.))
        self.assertEqual(m['offset_drift_vs_nominal_ns'],10000)
        with self.assertRaisesRegex(ValueError,'outside'):
            clock.map_cycle(m,999)

    def test_common_host_origin_preserves_rank_offset(self):
        records=[calibration(0),calibration(1,100000)]
        origin=clock.validate_calibration_set(records)
        self.assertEqual(origin,10001)
        base=dict(name='event',ph='X',pid=0,tid=0,cat='forward',args=dict(start_cycle=100,end_cycle=200))
        e0=clock.map_device_events([base],records[0],origin)[0]
        base['args'].update(start_cycle=100100,end_cycle=100200)
        e1=clock.map_device_events([base],records[1],origin)[0]
        self.assertAlmostEqual(e0['ts'],e1['ts'])
        self.assertAlmostEqual(e0['args']['clock_uncertainty_ns'],50)
        # A later event remains later; its workload start is never forced to 0.
        base['args'].update(start_cycle=100200,end_cycle=100300)
        later=clock.map_device_events([base],records[1],origin)[0]
        self.assertAlmostEqual(later['ts']-e0['ts'],.1)

    def test_reject_bad_identity_reset_tamper_or_missing_lanes(self):
        bad=batch(100,1000);bad['samples'][0]['cycles'].pop()
        with self.assertRaisesRegex(ValueError,'96-lane'):
            clock.build_clock_mapping(bad,batch(200,2000))
        with self.assertRaisesRegex(ValueError,'backwards'):
            clock.build_clock_mapping(batch(200,1000),batch(100,2000))
        records=[calibration(0),calibration(1)]
        records[1]['mapping']['ns_per_cycle']=2
        with self.assertRaisesRegex(ValueError,'does not match'):
            clock.validate_calibration_set(records)
        records=[calibration(0),calibration(1)]
        records[1]['host_identity']=dict(IDENTITY,boot_id='different')
        with self.assertRaisesRegex(ValueError,'one verified host'):
            clock.validate_calibration_set(records)

    def test_spread_and_resolution_warning_preserved(self):
        a=batch(100,1000000,width=500000)
        a['samples'][0]['cycles'][-1]=300
        m=clock.build_clock_mapping(a,batch(1000100,2000000,width=500000))
        self.assertTrue(m['resolution_warning'])
        self.assertEqual(m['pre']['lane_spread_cycles'],200)
        self.assertGreater(m['pre']['uncertainty_ns'],250000)
        self.assertFalse(m['physical_core_id_recorded'])

    def test_host_scopes_are_not_device_durations(self):
        records=[dict(phase='forward.native_launch',start_ns=100,end_ns=200,
            clock='host_monotonic_ns',semantics='host_enqueue_or_python_scope')]
        events=clock.host_control_events(records,2,50)
        self.assertEqual(events[1]['ts'],.05)
        self.assertFalse(events[1]['args']['device_elapsed_time'])


class RecorderTest(unittest.TestCase):
    @staticmethod
    def module():
        path=Path(__file__).resolve().parents[2]/'ascend_deepep/control_timing.py'
        spec=importlib.util.spec_from_file_location('control_timing_cpu_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_disabled_no_clock_and_context_restored_on_exception(self):
        m=self.module()
        with patch.object(m.time,'monotonic_ns',side_effect=AssertionError('disabled read')):
            with m.control_timing_scope('off'):pass
        recorder=m.ControlTimingRecorder()
        with patch.object(m.time,'monotonic_ns',side_effect=[10,20]):
            with self.assertRaisesRegex(RuntimeError,'test'):
                with m.record_control_timing(recorder):
                    with m.control_timing_scope('failed'):
                        raise RuntimeError('test')
        self.assertTrue(recorder.records[0]['failed'])
        self.assertEqual(recorder.records[0]['end_ns'],20)
        with patch.object(m.time,'monotonic_ns',side_effect=AssertionError('context leaked')):
            with m.control_timing_scope('off-again'):pass

    def test_default_recorder_does_not_import_torch(self):
        m=self.module()
        original_import=__import__
        def guarded(name,*args,**kwargs):
            if name=='torch':raise AssertionError('default imported Torch')
            return original_import(name,*args,**kwargs)
        with patch('builtins.__import__',side_effect=guarded):
            recorder=m.ControlTimingRecorder()
            with m.record_control_timing(recorder):
                with m.control_timing_scope('host-only'):pass
            self.assertEqual(recorder.resolve_device_timings(),recorder.records)
        self.assertNotIn('device_elapsed_ms',recorder.records[0])

    def test_opt_in_events_nested_order_exception_and_resolve_without_sync(self):
        m=self.module();log=[]
        class Event:
            def __init__(self,enable_timing):
                self.index=sum(entry[0]=='new' for entry in log)
                log.append(('new',self.index,enable_timing))
            def record(self):log.append(('record',self.index))
            def elapsed_time(self,other):
                log.append(('elapsed',self.index,other.index));return 1.25
        fake=SimpleNamespace(npu=SimpleNamespace(Event=Event,
            current_stream=lambda:SimpleNamespace(npu_stream=42),
            synchronize=lambda:(_ for _ in ()).throw(AssertionError('recorder synchronized'))))
        with patch.dict('sys.modules',{'torch':fake}):
            recorder=m.ControlTimingRecorder(device_events=True)
            with self.assertRaisesRegex(RuntimeError,'body failed'):
                with m.record_control_timing(recorder):
                    with m.control_timing_scope('outer'):
                        with m.control_timing_scope('inner'):pass
                        raise RuntimeError('body failed')
            self.assertEqual([entry[1] for entry in log if entry[0]=='record'],[0,2,3,1])
            self.assertFalse(any('device_elapsed_ms' in row for row in recorder.records))
            recorder.resolve_device_timings()
            self.assertEqual([row['phase'] for row in recorder.records],['inner','outer'])
            self.assertEqual([row['device_elapsed_ms'] for row in recorder.records],[1.25,1.25])
            self.assertEqual([row['failed'] for row in recorder.records],[False,True])
            self.assertTrue(all(row['device_timing_resolved'] for row in recorder.records))
            elapsed_calls=len([row for row in log if row[0]=='elapsed'])
            recorder.resolve_device_timings()
            self.assertEqual(len([row for row in log if row[0]=='elapsed']),elapsed_calls)


class SharedFBExportTest(unittest.TestCase):
    @staticmethod
    def write_frame(root,rank,with_clock=True):
        frame=root/'fused'/f'rank{rank}'/'step0';frame.mkdir(parents=True)
        raw=np.zeros(96*2592,dtype=np.int64)
        raw[2592+2:2592+4]=(900,1000)
        for lane in (1,2):raw[lane*2592+16:lane*2592+18]=(690,850)
        np.save(frame/'forward_timer.npy',raw)
        pairs=np.zeros((4,128,2),dtype=np.int64)
        for phase,span in enumerate(((100,200),(300,400),(500,600))):pairs[phase,0]=span
        pairs[3,:2]=(700,800)
        np.save(frame/'shared_forward_timer.npy',pairs)
        backward=np.zeros_like(raw);backward[2592+2:2592+4]=(5000,5100)
        backward[38:40]=(1900,4100)  # AIC_INIT includes the shared S0 prefix
        for lane in (1,2):backward[lane*2592+40:lane*2592+42]=(1900,4100)
        np.save(frame/'backward_timer.npy',backward)
        b=np.zeros((10,128,2),dtype=np.int64)
        for phase in range(10):
            count=2 if phase in (1,3,9) else 1
            b[phase,:count]=(2000+phase*200,2100+phase*200)
        np.save(frame/'shared_backward_timer.npy',b)
        if with_clock:
            (frame/'clock-calibration.json').write_text(json.dumps(calibration(rank)))
            (frame/'host-control-timing.json').write_text(json.dumps([dict(phase='forward.plan',
                start_ns=10200,end_ns=10500,clock='host_monotonic_ns',semantics='host_enqueue_or_python_scope'),
                dict(phase='backward.native_launch',start_ns=15000,end_ns=16000,
                clock='host_monotonic_ns',semantics='host_enqueue_or_python_scope')]))
        return frame

    def test_backward_lane_mapping_keep_optional_and_conflict(self):
        b=np.zeros((10,128,2),dtype=np.int64)
        for phase in range(2,10):b[phase,:2 if phase in (3,9) else 1]=(100*phase,100*phase+10)
        events=forward.shared_backward_timer_events(b,0)
        self.assertEqual([e['tid'] for e in events if e['args']['shared_phase']==3],[1,2])
        self.assertFalse(any(e['args']['shared_phase'] in (0,1) for e in events))
        exporter=forward.load_file(Path(__file__).with_name('export_timer_trace.py'),'shared_bwd_scope_test')
        parents=[dict(name='AIC_INIT' if lane==0 else 'AIV_INIT',ph='X',pid=0,tid=lane,
            cat='backward',args=dict(start_cycle=150,end_cycle=1000,source='native_fixed_pair'))
            for lane in (0,1,2)]
        merged,count=forward.merge_shared_backward_events(parents,b,0,exporter)
        self.assertEqual(count,3)
        self.assertEqual(sum(e.get('cat')=='milestone' for e in merged),6)
        bad=copy.deepcopy(parents);bad[1]['args']['end_cycle']=905
        with self.assertRaisesRegex(ValueError,'outside native Init'):
            forward.merge_shared_backward_events(bad,b,0,exporter)
        parents.append(dict(name='AIC_DW1_0',ph='X',pid=0,tid=0,cat='backward',
            args=dict(start_cycle=200,end_cycle=205,source='native_per_expert_pair')))
        with self.assertRaisesRegex(ValueError,'overlapping'):
            forward.merge_shared_backward_events(parents,b,0,exporter)
        b[1,:2]=(100,110)
        with self.assertRaisesRegex(ValueError,'recompute pair'):
            forward.shared_backward_timer_events(b,0)

    def test_calibrated_and_explicit_old_export(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.clock-cpu-') as temporary:
            root=Path(temporary);source=root/'input'
            frames=[self.write_frame(source,rank) for rank in range(2)]
            audit=export_shared_fb_traces(source,root/'aligned',2)
            self.assertTrue(audit['clock_aligned'])
            events=json.loads(Path(audit['traces']['fused/backward']).read_text())
            self.assertEqual(sum(e.get('name')=='SHARED_BWD/MERGE_SHARED_DX' for e in events),4)
            self.assertFalse(any(e.get('name','').startswith('HOST/') for e in events))
            self.assertFalse(any('WAIT' in e.get('name','') for e in events))
            forward_events=json.loads(Path(audit['traces']['fused/forward']).read_text())
            self.assertFalse(any(e.get('name','').startswith('HOST/') for e in forward_events))
            self.assertFalse(any(e.get('cat')=='shared_backward' for e in forward_events))
            combined=json.loads(Path(audit['recommended_traces']['fused']).read_text())
            self.assertEqual(sum(e.get('name')=='HOST/forward.plan' for e in combined),2)
            self.assertEqual(sum(e.get('name')=='HOST/backward.native_launch' for e in combined),2)
            self.assertTrue(any(e.get('cat')=='shared_forward' for e in combined))
            self.assertTrue(any(e.get('cat')=='shared_backward' for e in combined))
            fwd_end=max(e['ts']+e.get('dur',0) for e in forward_events if e.get('ph')!='M')
            combined_end=max(e['ts']+e.get('dur',0) for e in combined if e.get('ph')!='M')
            self.assertLess(fwd_end,combined_end)
            (frames[1]/'clock-calibration.json').unlink()
            with self.assertRaisesRegex(ValueError,'missing actual'):
                export_shared_fb_traces(source,root/'refuse',2)
            self.assertFalse((root/'refuse').exists())
            old=export_shared_fb_traces(source,root/'old',2,allow_uncalibrated=True)
            self.assertFalse(old['clock_aligned'])
            self.assertIn('explicitly uncalibrated',old['clock'])
            old_combined=json.loads(Path(old['recommended_traces']['fused']).read_text())
            self.assertFalse(any(e.get('name','').startswith('HOST/') for e in old_combined))

    def test_combined_rejects_cross_phase_execution_overlap(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.clock-cpu-') as temporary:
            root=Path(temporary);source=root/'input'
            frames=[self.write_frame(source,rank) for rank in range(2)]
            raw=np.load(frames[0]/'backward_timer.npy')
            # Individually valid Bwd gather, but overlaps Fwd SwiGLU lane1.
            raw[2592+2:2592+4]=(350,360)
            np.save(frames[0]/'backward_timer.npy',raw)
            with self.assertRaisesRegex(ValueError,'overlapping'):
                export_shared_fb_traces(source,root/'refuse-combined',2)

    def test_new_detail_and_complete_tiles_export_compute_with_no_wait(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.clock-detail-cpu-') as temporary:
            root=Path(temporary);source=root/'input'
            metadata=forward.shared_forward_tile_metadata(dict(schema=2,tile_enabled=True,
                tokens=128,hidden=256,ffn=256))
            for rank in range(2):
                frame=self.write_frame(source,rank)
                raw=np.load(frame/'forward_timer.npy')
                for lane in range(64):
                    track=3*(lane//2)+1+lane%2
                    raw[track*2592+16:track*2592+18]=(690,850)
                np.save(frame/'forward_timer.npy',raw)
                summary=np.zeros((4,128,2),dtype=np.int64);summary[3,:64]=(700,800)
                np.save(frame/'shared_forward_timer.npy',summary)
                detail=np.zeros((2,128,2),dtype=np.int64)
                detail[0,:64]=(810,825);detail[1,:64]=(830,840)
                np.save(frame/'forward_detail_timer.npy',detail)
                np.save(frame/'shared_forward_tile_timer.npy',np.array([[100,200],[300,400],[500,600]],dtype=np.int64))
                (frame/'shared_forward_tile_timer_metadata.json').write_text(json.dumps(metadata))
            audit=export_shared_fb_traces(source,root/'new',2)
            events=json.loads(Path(audit['traces']['fused/forward']).read_text())
            self.assertEqual(sum(e.get('name')=='AIV_UNPERMUTE_EXEC' and e.get('ph')=='X' for e in events),128)
            self.assertEqual(sum(e.get('cat')=='enclosing_scope' and e.get('ph')=='X' for e in events),128)
            for name in ['SHARED/GMM1_GATE_UP_TILE','SHARED/SWIGLU_TILE','SHARED/GMM2_DOWN_TILE']:
                self.assertEqual(sum(e.get('name')==name for e in events),2)
            self.assertFalse(any(e.get('cat')=='wait' for e in events))
            self.assertFalse(any(e.get('name')=='AIV_UNPERMUTE_TAIL_EP_SYNC' for e in events))
            # Metadata is mandatory; zeroed old summaries alone are never
            # treated as evidence that the shared computation did not run.
            (source/'fused/rank1/step0/shared_forward_tile_timer_metadata.json').unlink()
            with self.assertRaisesRegex(ValueError,'incomplete shared tile'):
                export_shared_fb_traces(source,root/'missing-metadata',2)


if __name__=='__main__':unittest.main()
