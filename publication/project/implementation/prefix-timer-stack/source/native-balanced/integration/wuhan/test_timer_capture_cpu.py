"""CPU-only contracts: portable routing, schema6 coordinates and paired receipts."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from timer_capture_config import validate, rank_skew_ids, window_bytes
from verify_timer_capture import verify


def worker_layout():
    package = '_timer_layout_cpu'
    module = types.ModuleType(package)
    module.__path__ = [str(HERE.parents[1]/'ascend_deepep')]
    sys.modules[package] = module
    return __import__(package+'.megaflux_worker_stage_layout', fromlist=['layout'])


class CaptureContracts(unittest.TestCase):
    def test_rank_skew_unique_quota_and_reproducible(self):
        from collections import Counter
        ids, quotas = rank_skew_ids(8192,4,2,2,6.,2)
        self.assertEqual(len(ids),8192)
        self.assertEqual((ids,quotas),rank_skew_ids(8192,4,2,2,6.,2))
        self.assertTrue(all(len(set(row))==2 and min(row)>=0 and max(row)<8 for row in ids))
        counts=Counter(e//2 for row in ids for e in row)
        self.assertLess(abs(max(counts.values())/min(counts.values())-1-6),.02)
        self.assertEqual(sum(quotas),8192)

    def test_invalid_topology_and_capacity_fail_before_device(self):
        args=[8,4,8192,7168,3072,8,2,2,20480,6.,2]
        validate(*args)
        for index,value in [(0,16),(1,8),(8,8192),(3,7167),(9,float('nan'))]:
            changed=args.copy();changed[index]=value
            with self.assertRaises(ValueError):validate(*changed)

    def test_window_matches_recorded_rho6_size(self):
        self.assertEqual(window_bytes(8192,7168,2,4,4,20480),589299712)

    def test_source_coordinates_have_disjoint_ranges_and_decode(self):
        stage=worker_layout()
        for ep in (2,4,8):
            for phase in ('forward','backward'):
                d=stage.layout(ep,2,2,7168,3072,20480,phase)
                self.assertEqual(d['version'],6)
                ranges=list(d['extra_ranges'].values())
                for start,count in ranges:
                    self.assertGreater(count,0)
                    self.assertLessEqual(start+count,d['total'])
                for i,(start,count) in enumerate(ranges):
                    for other,size in ranges[i+1:]:
                        self.assertTrue(start+count<=other or other+size<=start)
                for kind in ((35,36,37) if phase=='forward' else (38,39)):
                    start,count=d['extra_ranges'][kind]
                    first=stage.decode(d,phase,start)
                    last=stage.decode(d,phase,start+count-1)
                    self.assertEqual(first[0],kind);self.assertEqual(last[0],kind)
                    self.assertEqual((first[1],first[5],first[6]),(0,0,0))
                    self.assertEqual((last[1],last[5],last[6]),(3,ep-1,63))
                    if kind==36:self.assertEqual(last[3],79)

    def make_receipts(self, root):
        for kind,mode in [('control',0),('timer',1)]:
            (root/kind).mkdir()
            row=dict(pass_=True,rank=0,world_size=1,timer=mode,config={'ep':2},
                     inputs={'x':'same'},outputs={k:'same' for k in ('y','dx','dp','dw1','dw2')},copies=[[-1],[-1]],plan_tensors={'copies':'same'},
                     checked_iterations=1,five_outputs_bitwise_reference=bool(mode),
                     producer_counts={'forward':{'35':1,'37':1},'backward':{'38':1,'39':1}})
            (root/kind/'rank0.json').write_text(json.dumps(row))
            (root/kind/'iterations-rank0.jsonl').write_text(json.dumps(dict(pass_=True,outputs=row['outputs']))+'\n')
        (root/'kernel-trace').mkdir()
        audit=dict(pass_=True,ranks=[0],duration_overlap_count=0)
        (root/'kernel-trace/trace-export-audit.json').write_text(json.dumps(audit))
        for phase in ('forward','backward'):
            (root/'kernel-trace'/f'all_1_rank_{phase}_trace.json').write_text('[]')

    def test_verify_rejects_reference_and_iteration_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.make_receipts(root)
            self.assertTrue(verify(root,1)['pass_'])
            path=root/'timer/rank0.json';before=json.loads(path.read_text())
            changed=copy.deepcopy(before);changed['inputs']['x']='other'
            path.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError):verify(root,1)
            path.write_text(json.dumps(before))
            changed=copy.deepcopy(before)
            changed['saved_gate_up']=dict(requested='mxfp8',resolved='mxfp8')
            path.write_text(json.dumps(changed))
            with self.assertRaises(AssertionError):verify(root,1)
            path.write_text(json.dumps(before))
            (root/'timer/iterations-rank0.jsonl').write_text('')
            with self.assertRaises(AssertionError):verify(root,1)


if __name__=='__main__':unittest.main()
