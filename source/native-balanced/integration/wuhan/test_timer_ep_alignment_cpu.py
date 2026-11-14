"""Exercise actual NPY decoding/export; EP anchors are logical, not calibrated."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('timer_ep_alignment_exporter', HERE/'export_timer_trace.py')
EX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EX)


def frame(folder, rank=0):
    folder.mkdir(parents=True, exist_ok=True)
    base = 1_000_000 + rank*10_000_000  # Deliberately unrelated device origins.
    forward = np.zeros(EX.LANES*EX.STRIDE, dtype=np.int64)
    forward[EX.STRIDE+2:EX.STRIDE+4] = [base+700, base+800]
    index = 2*(10+EX.FORWARD_DYNAMIC.index('AIC_GMM1')*64)
    forward[index:index+2] = [base+1000, base+1200]
    np.save(folder/'forward_timer.npy', forward, allow_pickle=False)
    planner = np.zeros((32, 5, 8), dtype=np.int64)
    planner[0, 0, :2] = [base+10, base+20]
    planner[0, 2, :2] = [base+300, base+350]
    planner[1, 2, :2] = [base+200, base+220]  # Not the fixed worker0 anchor.
    np.save(folder/'planner_pairs.npy', planner, allow_pickle=False)
    backward = np.zeros(EX.LANES*EX.STRIDE, dtype=np.int64)
    for lane, name, start, end in (
        (0, 'AIC_INIT', 5000, 5050),
        (1, 'AIV_GATHER', 5100, 5200),
        (1, 'AIV_XSYNC_GATHER', 5200, 5600),
        (1, 'AIV_RESTORE_TABLE', 5700, 5800),
        (4, 'AIV_XSYNC_GATHER', 5250, 5300),
    ):
        index = lane*EX.STRIDE+2*EX.BACKWARD_FIXED.index(name)
        backward[index:index+2] = [base+start, base+end]
    np.save(folder/'backward_timer.npy', backward, allow_pickle=False)
    return base


class EPAlignmentContracts(unittest.TestCase):
    def test_fixed_source_anchors_survive_hidden_waits(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            base = frame(folder)
            self.assertEqual(EX.barrier_anchor(folder, 'forward')[0], base+300)
            self.assertEqual(EX.barrier_anchor(folder, 'backward')[0], base+5600)
            events, _ = EX.collect_frame(folder, 0, 'backward', 'backward', False)
            self.assertFalse(any('XSYNC' in event['name'] for event in events))
            self.assertEqual(EX.barrier_anchor(folder, 'backward')[0], base+5600)

    def test_missing_worker0_does_not_fall_back_to_another_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame(folder)
            path = folder/'planner_pairs.npy'
            pairs = np.load(path)
            pairs[0, 2, :2] = 0
            np.save(path, pairs)
            with self.assertRaisesRegex(ValueError, 'anchor'):
                EX.barrier_anchor(folder, 'forward')
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                EX.barrier_anchor(folder, 'forward')

    def test_invalid_or_truncated_anchor_sources_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            frame(folder)
            for pair in ((0, 1), (200, 199), (-1, 200)):
                with self.subTest(pair=pair):
                    values = np.zeros((32, 5, 8), dtype=np.int64)
                    values[0, 2, :2] = pair
                    np.save(folder/'planner_pairs.npy', values)
                    with self.assertRaises(ValueError):
                        EX.barrier_anchor(folder, 'forward')
            np.save(folder/'planner_pairs.npy', np.zeros((31, 5, 8), dtype=np.int64))
            with self.assertRaises(ValueError):
                EX.barrier_anchor(folder, 'forward')
            raw = np.load(folder/'backward_timer.npy')
            pos = EX.STRIDE+2*EX.BACKWARD_FIXED.index('AIV_XSYNC_GATHER')
            for pair in ((0, 0), (0, 1), (200, 199)):
                with self.subTest(backward_pair=pair):
                    raw[pos:pos+2] = pair  # Other lanes retain valid sync records.
                    np.save(folder/'backward_timer.npy', raw)
                    with self.assertRaises(ValueError):
                        EX.barrier_anchor(folder, 'backward')
            np.save(folder/'backward_timer.npy', raw[:-1])
            with self.assertRaises(ValueError):
                EX.barrier_anchor(folder, 'backward')

    def test_complete_ep4_groups_share_only_a_logical_origin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bases = {rank: frame(root/'input'/f'rank{rank}', rank) for rank in range(8)}
            before = {path: path.read_bytes() for path in (root/'input').rglob('*.npy')}
            result = EX.export_traces(root/'input', root/'out', expected_ranks=range(8),
                                      time_origin='ep-barrier', ep_size=4)
            self.assertFalse(result['cross_rank_clock_aligned'])
            self.assertFalse(result['cross_ep_group_aligned'])
            self.assertEqual(result['ep_size_sources'], ['explicit_unverified'])
            self.assertFalse(result['epoch_consistency_checked'])
            for phase, filename in result['traces'].items():
                rows = json.loads(Path(filename).read_text(encoding='utf-8'))
                labels = [event for event in rows if event['name'] == 'process_labels']
                self.assertEqual(len(labels), 8)
                for event in labels:
                    rank, args = event['pid'], event['args']
                    self.assertEqual(args['ep_group'], list(range(rank//4*4, rank//4*4+4)))
                    self.assertEqual(args['alignment'], 'ep-barrier-anchor')
                    self.assertFalse(args['cross_rank_clock_aligned'])
                    self.assertFalse(args['cross_ep_group_aligned'])
                work = [event for event in rows if event['ph'] != 'M']
                self.assertTrue(any(event['ts'] < 0 for event in work))
                if phase == 'backward':
                    self.assertFalse(any('XSYNC' in event['name'] for event in work))
                for event in work:
                    origin = bases[event['pid']] + (300 if phase == 'forward' else 5600)
                    self.assertEqual(event['ts'], (event['args']['start_cycle']-origin)/1000)
                    self.assertEqual(event['dur'], (event['args']['end_cycle']-event['args']['start_cycle'])/1000)
            self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))

    def test_partial_noncontiguous_and_invalid_ep_sizes_are_rejected(self):
        for ranks, ep_size in (((0, 1, 2), 4), ((0, 2), 2), ((0, 1), None),
                               ((0, 1), 1), ((0, 1), 3)):
            with self.subTest(ranks=ranks, ep_size=ep_size), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for rank in ranks:
                    (root/'input'/f'rank{rank}').mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, 'complete contiguous EP groups'):
                    EX.export_traces(root/'input', root/'out', time_origin='ep-barrier', ep_size=ep_size)
                self.assertFalse((root/'out').exists())

    def test_legacy_default_matches_explicit_rank_local_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame(root/'input'/'rank0')
            default = EX.export_traces(root/'input', root/'default')
            explicit = EX.export_traces(root/'input', root/'explicit', time_origin='rank-local')
            self.assertEqual(default['time_origin'], 'rank-local')
            for phase in ('forward', 'backward'):
                rows = json.loads(Path(default['traces'][phase]).read_text(encoding='utf-8'))
                other = json.loads(Path(explicit['traces'][phase]).read_text(encoding='utf-8'))
                self.assertEqual(rows, other)
                self.assertEqual(min(event['ts'] for event in rows if event['ph'] != 'M'), 0)
                self.assertFalse(any(event['name'] == 'process_labels' for event in rows))

    def test_header_epochs_must_agree_only_within_the_same_phase_and_ep_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epochs = {}

            def header(rank, phase, epoch):
                folder = root/f'rank{rank}'
                folder.mkdir(exist_ok=True)
                values = np.zeros(8, dtype=np.int64)
                values[1], values[7] = 4, epoch
                np.save(folder/f'{phase}_stage_header.npy', values)
                return folder

            a = EX.validate_ep_alignment_sources(header(0, 'forward', 31), 'forward', 0, 4, epochs)
            self.assertTrue(a['epoch_checked'])
            self.assertEqual(a['ep_size_source'], 'stage_header')
            EX.validate_ep_alignment_sources(header(1, 'forward', 31), 'forward', 1, 4, epochs)
            with self.assertRaisesRegex(ValueError, 'epoch'):
                EX.validate_ep_alignment_sources(header(1, 'forward', 32), 'forward', 1, 4, epochs)
            EX.validate_ep_alignment_sources(header(4, 'forward', 99), 'forward', 4, 4, epochs)
            EX.validate_ep_alignment_sources(header(0, 'backward', 35), 'backward', 0, 4, epochs)

    def test_stage_ep_and_receipt_ep_are_checked_and_missing_evidence_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            receipt = folder/'rank0.json'
            result = EX.validate_ep_alignment_sources(folder, 'forward', 0, 4, {}, receipt)
            self.assertEqual(result['ep_size_source'], 'explicit_unverified')
            self.assertFalse(result['epoch_checked'])
            receipt.write_text(json.dumps({'config': {'ep': 4}}), encoding='utf-8')
            result = EX.validate_ep_alignment_sources(folder, 'forward', 0, 4, {}, receipt)
            self.assertEqual(result['ep_size_source'], 'receipt')
            self.assertFalse(result['epoch_checked'])
            values = np.zeros(8, dtype=np.int64)
            values[1], values[7] = 4, 31
            np.save(folder/'forward_stage_header.npy', values)
            result = EX.validate_ep_alignment_sources(folder, 'forward', 0, 4, {}, receipt)
            self.assertEqual(result['ep_size_source'], 'stage_header+receipt')
            values[1] = 8
            np.save(folder/'forward_stage_header.npy', values)
            with self.assertRaisesRegex(ValueError, 'stage header EP'):
                EX.validate_ep_alignment_sources(folder, 'forward', 0, 4, {}, receipt)
            values[1] = 4
            np.save(folder/'forward_stage_header.npy', values)
            receipt.write_text(json.dumps({'config': {'ep': 8}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'receipt config.ep'):
                EX.validate_ep_alignment_sources(folder, 'forward', 0, 4, {}, receipt)


if __name__ == '__main__':
    unittest.main()
