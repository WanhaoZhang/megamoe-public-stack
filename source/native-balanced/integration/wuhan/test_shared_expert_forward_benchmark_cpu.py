"""CPU CLI, route/load, pairing and shared Timer-coordinate checks."""
import json
from contextlib import redirect_stderr
import io
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import benchmark_shared_expert_forward as benchmark


class SharedForwardBenchmarkTest(unittest.TestCase):
    def test_off_defaults_and_diagnostic_on_constraints(self):
        args = benchmark.parse_args(['--out', 'unused'])
        self.assertEqual((args.mode, args.warmup, args.measure), ('compare', 10, 30))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(['--out', 'unused', '--timer', '1'])
        args = benchmark.parse_args(['--out', 'unused', '--timer', '1', '--reference', 'off'])
        self.assertEqual((args.warmup, args.measure), (0, 0))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(['--out', 'unused', '--timer', '1', '--reference', 'off', '--measure', '1'])

    def test_unique_routes_and_rank_skew_load(self):
        expected = [45880, 6552, 6552, 6552]
        for route in ('balanced', 'random', 'rank_skew'):
            rows, quotas = benchmark.routing_ids(8192, 4, 2, 2, route, 6., 2, 42, 0)
            self.assertTrue(all(len(row) == len(set(row)) == 2 and min(row) >= 0 and max(row) < 8 for row in rows))
            self.assertEqual((rows, quotas), benchmark.routing_ids(8192, 4, 2, 2, route, 6., 2, 42, 0))
            counts = [sum(expert // 2 == owner for row in rows for expert in row) * 4 for owner in range(4)]
            if route == 'balanced':
                self.assertEqual(counts, [16384] * 4)
            elif route == 'rank_skew':
                self.assertEqual(quotas, [5735, 819, 819, 819])
                self.assertEqual(counts, expected)
        self.assertNotEqual(benchmark.routing_ids(16, 4, 2, 2, 'random', 6., 2, 42, 0)[0],
                            benchmark.routing_ids(16, 4, 2, 2, 'random', 6., 2, 42, 1)[0])

    def test_balanced_random_topk_can_exceed_home_without_rank_skew_constraints(self):
        for route in ('balanced', 'random'):
            args = benchmark.parse_args(['--out', 'unused', '--route', route, '--topk', '6', '--capacity', '49152'])
            benchmark.validate_benchmark_shape(args, 8)
            rows, _ = benchmark.routing_ids(32, 4, 2, 6, route, 6., 2, 42, 0)
            self.assertTrue(all(len(row) == len(set(row)) == 6 for row in rows))
        args.route = 'rank_skew'
        with self.assertRaisesRegex(ValueError, 'rank_skew requires'):
            benchmark.validate_benchmark_shape(args, 8)
        args.route, args.topk = 'random', 9
        with self.assertRaisesRegex(ValueError, 'unique TopK'):
            benchmark.validate_benchmark_shape(args, 8)

    def test_ab_ba_and_per_iteration_rank_max_before_quantiles(self):
        self.assertEqual(benchmark.alternating_order(0, ('framework', 'fused')), ('framework', 'fused'))
        self.assertEqual(benchmark.alternating_order(1, ('framework', 'fused')), ('fused', 'framework'))
        receipts = []
        for rank in range(2):
            samples = []
            for mode, values in (('framework', (1, 9) if rank == 0 else (10, 2)), ('fused', (2, 8) if rank == 0 else (9, 1))):
                samples.extend(dict(mode=mode, iteration=i, forward_ms=value, max_memory_allocated=100 + rank)
                               for i, value in enumerate(values))
            receipts.append(dict(measured_modes=['framework', 'fused'], measure=2, samples=samples))
        result = benchmark.summarize_ranks(receipts)
        self.assertEqual(result['framework']['per_iteration_rank_max_ms'], [10, 9])
        self.assertEqual(result['framework']['p50_ms'], 9.5)
        self.assertEqual(result['fused']['p50_ms'], 8.5)
        self.assertAlmostEqual(result['fused_vs_framework_pct'], (8.5 / 9.5 - 1) * 100)

    @staticmethod
    def shared_pairs():
        pairs = np.zeros((4, 128, 2), dtype=np.int64)
        pairs[0, 0] = (100, 200)
        pairs[1, 0] = (300, 400)
        pairs[2, 0] = (500, 600)
        pairs[3, 0] = (700, 800)
        pairs[3, 1] = (700, 800)
        return pairs

    def test_shared_timer_maps_cube_vector_lanes_and_rejects_partial_data(self):
        pairs = self.shared_pairs()
        events = benchmark.shared_timer_events(pairs, rank=3, include_waits=True)
        self.assertEqual([(event['name'], event['tid']) for event in events], [
            ('SHARED/GMM1_GATE_UP', 0), ('SHARED/SWIGLU', 1), ('SHARED/GMM2_DOWN', 0),
            ('SHARED_OUTPUT_WAIT', 1), ('SHARED_OUTPUT_WAIT', 2)])
        self.assertEqual(len(benchmark.shared_timer_events(pairs, 3)), 3)
        pairs[1, 0] = 0
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            benchmark.shared_timer_events(pairs, 3)

    def test_unpermute_envelopes_preserve_exact_duration_on_scope_tracks(self):
        exporter = benchmark.load_file(Path(benchmark.__file__).with_name('export_timer_trace.py'),
                                       'shared_forward_test_exporter')
        native = [dict(name='AIV_UNPERMUTE', ph='X', pid=0, tid=lane, cat='forward',
                       args=dict(start_cycle=690, end_cycle=850, source='native_fixed_pair'))
                  for lane in (1, 2)]
        events, count = benchmark.merge_shared_timer_events(native, self.shared_pairs(), 0, exporter, True)
        self.assertEqual(count, 2)
        scopes=[event for event in events if event.get('cat')=='enclosing_scope']
        self.assertEqual([(event['tid'],event['ph']) for event in scopes],[(101,'X'),(102,'X')])
        self.assertTrue(all((e['args']['start_cycle'],e['args']['end_cycle'])==(690,850) for e in scopes))
        self.assertTrue(all(e['args']['duration_is_not_pure_unpermute'] for e in scopes))
        self.assertEqual(exporter.validate_serial_lanes(events),5)
        self.assertEqual(len(benchmark.forward_scope_metadata(events,0)),4)
        # A synthetic compute overlap is not explained away as a wait/scope.
        native += [dict(name='AIC_GMM1_0', ph='X', pid=0, tid=0, cat='forward',
                        args=dict(start_cycle=150, end_cycle=180, source='native_dynamic_pair'))]
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            benchmark.merge_shared_timer_events(native, self.shared_pairs(), 0, exporter, True)
        native[0]['args']['end_cycle'] = 750
        with self.assertRaisesRegex(ValueError, 'outside'):
            benchmark.merge_shared_timer_events(native, self.shared_pairs(), 0, exporter, True)

    def test_detail_no_wait_keeps_pure_unpermute_and_preserves_scope(self):
        exporter=benchmark.load_file(Path(__file__).with_name('export_timer_trace.py'),'detail_no_wait_exporter')
        parents=[dict(name='AIV_UNPERMUTE',ph='X',pid=0,tid=lane,cat='forward',
            args=dict(start_cycle=690,end_cycle=850,source='native_fixed_pair')) for lane in (1,2)]
        detail=np.zeros((2,128,2),dtype=np.int64)
        detail[0,:2]=(810,825);detail[1,:2]=(830,840)
        events,count=benchmark.merge_shared_timer_events(parents,self.shared_pairs(),0,exporter,detail=detail)
        self.assertEqual(count,2)
        pure=[e for e in events if e['name']=='AIV_UNPERMUTE_EXEC']
        self.assertEqual([e['tid'] for e in pure],[1,2])
        self.assertTrue(all(e['args']['start_cycle']==810 and e['args']['end_cycle']==825 for e in pure))
        self.assertFalse(any(e['cat']=='wait' for e in events))
        self.assertEqual(sum(e['cat']=='enclosing_scope' for e in events),2)
        visible,_=benchmark.merge_shared_timer_events(parents,self.shared_pairs(),0,exporter,True,detail)
        self.assertEqual(sum(e['name']=='AIV_UNPERMUTE_TAIL_EP_SYNC' for e in visible),2)
        self.assertEqual(sum(e['name']=='SHARED_OUTPUT_WAIT' for e in visible),2)
        exporter.validate_serial_lanes(visible)
        detail[0,0]=(790,825)
        with self.assertRaisesRegex(ValueError,'overlaps shared output wait'):
            benchmark.merge_shared_timer_events(parents,self.shared_pairs(),0,exporter,detail=detail)
        detail[0,0]=(810,825);detail[1,0]=(820,840)
        with self.assertRaisesRegex(ValueError,'scope or order'):
            benchmark.merge_shared_timer_events(parents,self.shared_pairs(),0,exporter,detail=detail)
        detail[1,0]=(830,0)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            benchmark.merge_shared_timer_events(parents,self.shared_pairs(),0,exporter,detail=detail)

    @staticmethod
    def tile_fixture():
        geometry=dict(schema=2,tile_enabled=True,tokens=4096,hidden=512,ffn=512)
        metadata=benchmark.shared_forward_tile_metadata(geometry)
        pairs=np.zeros((metadata['tile_pairs'],2),dtype=np.int64)
        for section,delta in [('gate_up',0),('swiglu',4),('down',8)]:
            spec=metadata['sections'][section]
            for task in range(spec['pair_count']):
                start=100+(task//32)*20+delta
                pairs[spec['pair_offset']+task]=(start,start+3)
        return geometry,metadata,pairs

    def test_tile_task_geometry_owner_and_incomplete_pair_rejected(self):
        _,metadata,pairs=self.tile_fixture()
        events=benchmark.shared_forward_tile_events(pairs,metadata,rank=2)
        self.assertEqual(len(events),192)
        selected=[e for e in events if e['args']['section']=='swiglu' and e['args']['task']==63][0]
        self.assertEqual((selected['tid'],selected['args']['owner_group']),(94,31))
        selected=[e for e in events if e['args']['section']=='gate_up' and e['args']['task']==32][0]
        self.assertEqual((selected['tid'],selected['args']['row_tile'],selected['args']['n_tile']),(0,16,0))
        exporter=benchmark.load_file(Path(__file__).with_name('export_timer_trace.py'),'tile_serial_test')
        exporter.validate_serial_lanes(events)
        with self.assertRaisesRegex(ValueError,'complete task set'):
            benchmark.shared_forward_tile_events(pairs[:-1],metadata,2)
        bad=dict(metadata,gate_tasks=65)
        with self.assertRaisesRegex(ValueError,'disagrees'):
            benchmark.shared_forward_tile_events(pairs,bad,2)
        bad_pairs=pairs.copy();bad_pairs[0,1]=0
        with self.assertRaisesRegex(ValueError,'incomplete'):
            benchmark.shared_forward_tile_events(bad_pairs,metadata,2)
        bad_pairs=pairs.copy();bad_pairs[32]=bad_pairs[0]
        with self.assertRaisesRegex(ValueError,'overlapping'):
            exporter.validate_serial_lanes(benchmark.shared_forward_tile_events(bad_pairs,metadata,2))

    def test_tile_capture_allows_zero_legacy_summary_only_with_complete_files(self):
        _,metadata,tiles=self.tile_fixture()
        summary=np.zeros((4,128,2),dtype=np.int64);summary[3,:64]=(2000,2010)
        detail=np.zeros((2,128,2),dtype=np.int64);detail[0,:64]=(2020,2030);detail[1,:64]=(2040,2050)
        parents=[dict(name='AIV_UNPERMUTE',ph='X',pid=0,tid=3*(lane//2)+1+lane%2,cat='forward',
            args=dict(start_cycle=1990,end_cycle=2060,source='native_fixed_pair')) for lane in range(64)]
        exporter=benchmark.load_file(Path(__file__).with_name('export_timer_trace.py'),'tile_merge_test')
        with self.assertRaisesRegex(ValueError,'incomplete shared AIC/AIV'):
            benchmark.merge_shared_timer_events(parents,summary,0,exporter,detail=detail)
        events,count=benchmark.merge_shared_timer_events(parents,summary,0,exporter,detail=detail,tiles=tiles,tile_metadata=metadata)
        self.assertEqual(count,64)
        self.assertEqual(sum(e['cat']=='shared_forward_tile' for e in events),192)
        self.assertEqual(sum(e['name']=='AIV_UNPERMUTE_EXEC' for e in events),64)
        self.assertFalse(any(e['cat']=='wait' for e in events))
        with self.assertRaisesRegex(ValueError,'requires its forward detail'):
            benchmark.merge_shared_timer_events(parents,summary,0,exporter,tiles=tiles,tile_metadata=metadata)

    def test_backing_schema_split_preserves_legacy_prefix_and_exact_lengths(self):
        geometry,metadata,tiles=self.tile_fixture()
        raw=np.arange(1536+2*len(tiles),dtype=np.int64)
        prefix,detail,recovered,record=benchmark.split_shared_forward_timer(raw,2,geometry)
        self.assertEqual(prefix.shape,(4,128,2));self.assertEqual(detail.shape,(2,128,2))
        np.testing.assert_array_equal(prefix.reshape(-1),raw[:1024])
        np.testing.assert_array_equal(detail.reshape(-1),raw[1024:1536])
        np.testing.assert_array_equal(recovered.reshape(-1),raw[1536:])
        self.assertEqual(record,metadata)
        with self.assertRaisesRegex(ValueError,'backing size'):
            benchmark.split_shared_forward_timer(raw[:-1],2,geometry)
        no_tiles=dict(geometry,tile_enabled=False)
        self.assertIsNone(benchmark.split_shared_forward_timer(raw[:1536],2,no_tiles)[2])
        legacy=self.shared_pairs()
        self.assertIs(benchmark.split_shared_forward_timer(legacy,1)[0],legacy)
        with self.assertRaisesRegex(ValueError,'explicit geometry'):
            benchmark.split_shared_forward_timer(raw,2)

    def test_save_schema2_writes_independent_detail_and_tile_metadata(self):
        class Tensor:
            def __init__(self,value):self.value=value
            def __getitem__(self,key):return Tensor(self.value[key])
            def detach(self):return self
            def cpu(self):return self
            def numpy(self):return self.value
        geometry,metadata,tiles=self.tile_fixture()
        raw=np.zeros(1536+tiles.size,dtype=np.int64)
        prefix=raw[:1024].reshape(4,128,2);prefix[3,:64]=(2000,2010)
        detail=raw[1024:1536].reshape(2,128,2);detail[0,:64]=(2020,2030);detail[1,:64]=(2040,2050)
        raw[1536:]=tiles.reshape(-1)
        planner=np.zeros(20+32*5*8,dtype=np.int64)
        planner[20:].reshape(32,5,8)[:,:,:2]=(10,20)
        module=SimpleNamespace(collect_timers=lambda:{},trace_timers={'forward':Tensor(np.zeros(96*2592,dtype=np.int64))},
            _worker_native=SimpleNamespace(megaflux_shared_forward_timer_schema=lambda:2))
        pool=SimpleNamespace(module=module,policy=SimpleNamespace(timer_backing=Tensor(planner)),
            last_shared_timer=Tensor(raw),last_shared_native_timer=module.trace_timers['forward'],last_shared_timer_geometry=geometry)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='.shared-save-cpu-') as tmp:
            frame=Path(tmp)/'capture';benchmark.save_timers(pool,frame,'fused')
            self.assertEqual(np.load(frame/'shared_forward_timer.npy').shape,(4,128,2))
            self.assertEqual(np.load(frame/'forward_detail_timer.npy').shape,(2,128,2))
            np.testing.assert_array_equal(np.load(frame/'shared_forward_tile_timer.npy'),tiles)
            self.assertEqual(json.loads((frame/'shared_forward_tile_timer_metadata.json').read_text()),metadata)

    def test_forward_only_export_preserves_actual_native_and_shared_events(self):
        workspace = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=workspace, prefix='.shared-forward-cpu-') as temporary:
            root = Path(temporary).resolve()
            self.assertEqual(root.parent, workspace)
            for rank in range(2):
                frame = root / 'fused' / f'rank{rank}' / 'step0'
                frame.mkdir(parents=True)
                native = np.zeros(96 * 2592, dtype=np.int64)
                native[2592 + 2:2592 + 4] = (900, 1000)
                for lane in (1, 2):
                    native[lane * 2592 + 16:lane * 2592 + 18] = (690, 850)
                np.save(frame / 'forward_timer.npy', native, allow_pickle=False)
                np.save(frame / 'shared_forward_timer.npy', self.shared_pairs(), allow_pickle=False)
            args = benchmark.parse_args(['--out', str(root), '--export-only', '--world-size', '2'])
            audit = benchmark.export_forward(args)
            self.assertTrue(audit['pass_'])
            self.assertEqual(audit['duration_overlap_count'], 0)
            events = json.loads((root / 'forward-traces/all_2_rank_fused_forward_trace.json').read_text())
            self.assertEqual(len([event for event in events if event.get('name') == 'SHARED/GMM1_GATE_UP']), 2)
            self.assertFalse(any('backward' in path.name for path in root.rglob('*.json')))
            (root / 'fused/rank1/step0/shared_forward_timer.npy').unlink()
            args.trace_out = root / 'other-output'
            with self.assertRaises(FileNotFoundError):
                benchmark.export_forward(args)


if __name__ == '__main__':
    unittest.main()
