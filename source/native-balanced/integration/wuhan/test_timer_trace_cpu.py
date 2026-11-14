"""CPU-only contracts for the reference-style TimerV2 export (no package import)."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
TRACE = load('offline_trace', HERE/'export_timer_trace.py')
CSV = load('csv_trace', HERE.parents[1]/'test/megamoe_bf16_test/trace_parser.py')
INSTALL = load('trace_install', HERE/'install_timer_trace_defaults.py')


def raw_pair(path, phase, start, end):
    values = np.zeros(TRACE.LANES*TRACE.STRIDE, dtype=np.int64)
    # AIC GMM1 expert0 / AIV gather: independently recorded native intervals.
    index = 2*(10+6*64) if phase == 'forward' else TRACE.STRIDE+2
    values[index:index+2] = [start, end]
    np.save(path/f'{phase}_timer.npy', values, allow_pickle=False)


class TraceContracts(unittest.TestCase):
    def test_owner_experts_keep_their_projection_expert_and_lane(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame=Path(temporary)
            raw_pair(frame,'backward',1000,2000)
            base=TRACE.load_base()
            d=base.summary_layout(4,2,2,7168,3072,20480,'backward',version=8)
            old=base.summary_layout(4,2,2,7168,3072,20480,'backward')
            self.assertEqual(d['extra_ranges'][43][0],old['total'])
            header=np.zeros(16+96*8,dtype=np.int64)
            header[:10]=[base.MAGIC,4,2,2,7168,3072,20480,7,d['total'],8]
            header[16+2*8]=4
            order=[(1,0),(1,1),(0,0),(0,1)]
            indices=[old['total']+(projection*2+expert)*96+2 for projection,expert in order]
            pairs=[[3000+1000*i,3800+1000*i,2,1,-1,-1,0,0] for i in range(4)]
            np.save(frame/'backward_stage_header.npy',header)
            np.save(frame/'backward_stage_indices.npy',np.array(indices,dtype=np.int64))
            np.save(frame/'backward_stage_pairs.npy',np.array(pairs,dtype=np.int64))
            events,coverage=TRACE.collect_frame(frame,0,'backward','backward')
            starts=sorted((e for e in events if e['name'].startswith('OWNER_EXPERT_PHASE') and
                           e['name'].endswith(':BEGIN')),key=lambda e:e['args']['start_cycle'])
            self.assertEqual(coverage['observed_kind_43'],4)
            self.assertEqual([(e['args']['projection'],e['args']['physical_expert']) for e in starts],order)
            self.assertTrue(all(e['tid']==2 for e in starts))

    def test_owner_suboperations_stay_serial_and_scopes_are_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            frame=Path(temporary)
            raw_pair(frame,'backward',3000,4000)
            base=TRACE.load_base();d=base.layout(2,1,1,256,256,128,'backward',version=3)
            header=np.zeros(16+96*8,dtype=np.int64)
            header[:10]=[base.MAGIC,2,1,1,256,256,128,7,d['total'],3]
            records=[
                (d['op_base'],10000,11000),
                (d['op_base']+1,12000,13000),
                (d['op_base']+d['ops'],20000,21000),
                (d['op_base']+d['ops']+1,22000,23000),
                (d['phase_base'],9000,24000),
                (d['phase_base']+128,8500,24500),
                (d['extra_ranges'][20][0],9500,23500),
            ]
            header[16+8]=len(records)
            np.save(frame/'backward_stage_header.npy',header)
            np.save(frame/'backward_stage_indices.npy',np.array([i for i,s,t in records],dtype=np.int64))
            np.save(frame/'backward_stage_pairs.npy',np.array([[s,t,1,1,0,0,0,0] for i,s,t in records],dtype=np.int64))
            events,coverage=TRACE.collect_frame(frame,0,'backward','backward')
            loads=[e for e in events if e['args'].get('kind')==8]
            self.assertEqual([(e['args']['start_cycle'],e['args']['end_cycle']) for e in loads],[(10000,11000),(20000,21000)])
            scopes=[e for e in events if e['args'].get('kind') in TRACE.SCOPES]
            self.assertEqual(len(scopes),6)
            self.assertTrue(all(e['cat']=='milestone' and e['args']['scope_boundary'] for e in scopes))
            self.assertEqual(coverage['observed_kind_8'],2)
            TRACE.validate_serial_lanes(events)

    def test_overlapping_work_is_rejected_without_retiming(self):
        events=[dict(tid=1,name='load',cat='work',args=dict(start_cycle=10,end_cycle=20)),
                dict(tid=1,name='sum',cat='work',args=dict(start_cycle=15,end_cycle=25))]
        with self.assertRaisesRegex(ValueError,'overlapping execution intervals'):
            TRACE.validate_serial_lanes(events)
        self.assertEqual(events[1]['args']['start_cycle'],15)

    def test_nested_wait_scope_does_not_create_parallel_work(self):
        events=[dict(tid=1,name='OWNER_LOAD',cat='work',args=dict(start_cycle=10,end_cycle=20)),
                dict(tid=1,name='OWNER_TILE_WAIT',cat='wait',args=dict(start_cycle=5,end_cycle=25,kind=19))]
        fixed,count=TRACE.mark_nested_waits(events)
        self.assertEqual(count,1)
        self.assertEqual(len(fixed),3)
        self.assertEqual(fixed[0]['args'],events[0]['args'])
        self.assertTrue(all(e['cat']=='milestone' for e in fixed[1:]))
        TRACE.validate_serial_lanes(fixed)

    def test_two_phase_outputs_and_recompute_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rank = root/'input/rank0'
            fwd = rank/'forward'; back = rank/'recompute'
            fwd.mkdir(parents=True); back.mkdir()
            raw_pair(fwd, 'forward', 1000, 2000)
            raw_pair(back, 'forward', 100000, 102000)
            raw_pair(back, 'backward', 103000, 106000)
            before = {p: p.read_bytes() for p in rank.rglob('*.npy')}
            result = TRACE.export_traces(root/'input', root/'out', expected_ranks=[0])
            self.assertEqual(len(list((root/'out').glob('*.json'))), 3)  # two traces + audit
            for phase, path in result['traces'].items():
                events = json.loads(Path(path).read_text())
                names = [e for e in events if e['name'] == 'thread_name']
                self.assertEqual([e['tid'] for e in names], list(range(96)))
                self.assertEqual(names[0]['args']['name'], 'group0/AIC0')
                self.assertEqual(names[95]['args']['name'], 'group31/AIV1')
                work = [e for e in events if e['ph'] != 'M']
                if phase == 'backward':
                    self.assertEqual([e['args']['start_cycle'] for e in work], [100000, 103000])
                    self.assertTrue(work[0]['name'].startswith('RECOMPUTE/'))
                    self.assertEqual(work[1]['ts'], 3.0)
                else:
                    self.assertEqual(work[0]['dur'], 1.0)
            self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))
            with self.assertRaises(FileExistsError):
                TRACE.export_traces(root/'input', root/'out')

    def test_partial_rank_collection_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); (root/'rank0').mkdir()
            with self.assertRaisesRegex(ValueError, 'incomplete ranks'):
                TRACE.export_traces(root, root/'out', expected_ranks=[0,1])
            self.assertFalse((root/'out').exists())

    def test_unknown_schema_rejected_before_trace_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); frame = root/'rank0'; frame.mkdir()
            raw_pair(frame, 'forward', 1000, 2000)
            raw_pair(frame, 'backward', 3000, 4000)
            header = np.zeros(16+96*8, dtype=np.int64)
            header[0] = TRACE.load_base().MAGIC; header[9] = 99
            np.save(frame/'forward_stage_header.npy', header)
            np.save(frame/'forward_stage_indices.npy', np.array([],dtype=np.int64))
            np.save(frame/'forward_stage_pairs.npy', np.zeros((0,8),dtype=np.int64))
            with self.assertRaisesRegex(ValueError, 'unsupported TimerV2 stage schema'):
                TRACE.export_traces(root, root/'out')
            self.assertFalse(list((root/'out').glob('*trace.json')))

    def test_csv_custom_events_are_not_filtered_or_given_extra_tracks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for phase, name in [('forward','npu'),('backward','backward')]:
                with (root/f'rank_0_{name}_time_parsed.csv').open('w',newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['group_id','core_type','sub_id',
                        'WEIGHT_COPY_start_cycle','WEIGHT_COPY_end_cycle',
                        'OWNER_FP32_SUM_start_cycle','OWNER_FP32_SUM_end_cycle'])
                    writer.writerow([31,'AIV',1,1000,2000,3000,5000])
                CSV.generate_trace_files(root,0,0,root/'out',phase=phase)
            files = sorted((root/'out').glob('*.json'))
            self.assertEqual(len(files),2)
            for file in files:
                events = json.loads(file.read_text())
                self.assertEqual(len([e for e in events if e['name']=='thread_name']),96)
                work = [e for e in events if e['ph']=='X']
                self.assertEqual({e['name'] for e in work},{'WEIGHT_COPY','OWNER_FP32_SUM'})
                self.assertEqual({e['tid'] for e in work},{95})


if __name__ == '__main__': unittest.main()
