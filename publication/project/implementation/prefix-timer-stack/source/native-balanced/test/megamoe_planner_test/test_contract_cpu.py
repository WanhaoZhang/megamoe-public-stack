import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import check_pair, summarize_samples, validate_case, validate_receipts
from gate_and_run import inspect_smi


def receipts():
    return [dict(rank=i,ep=8,world=8,pass_=True,case={'m':16},backend='s9',mode='performance',
                 scope='online',code_identity='sha',inputs={'x':str(i)},plan_digest='plan',
                 samples_ms=[1+i,9-i]) for i in range(8)]


class AcceptanceTests(unittest.TestCase):
    def test_portable_launch_paths_and_output_names(self):
        import shlex
        from types import SimpleNamespace
        from launch_config import shell_command, validate_output_name
        args=SimpleNamespace(cann_env='/my env/cann/set_env.sh')
        command=shell_command(args,['python','/my source/run.py','--name','a b'])
        self.assertEqual(shlex.split(command),['source',args.cann_env,'&&','exec',
                                               'python','/my source/run.py','--name','a b'])
        validate_output_name('delivery/random-1')
        for name in ('../escape','/absolute','a/../../b','a;cmd',''):
            with self.assertRaises(ValueError):validate_output_name(name)

    def test_controlled_load_and_random_routes(self):
        import numpy as np
        from inputs import scenario_ids,calibrate
        c=dict(ep=8,experts=32,slots=4,tokens=4096,topk=6,seed=42,route='controlled')
        for ratio in (1.5,2.):
            c['load_ratio']=ratio
            arrays=[scenario_ids(c,r) for r in range(8)]
            for ids in arrays:
                self.assertFalse(np.any(np.diff(np.sort(ids,axis=1),axis=1)==0))
            counts=np.bincount(np.stack(arrays).ravel(),minlength=32).reshape(8,4).sum(1)
            self.assertEqual(counts.max()/counts.mean(),ratio)
            self.assertEqual(calibrate(c,arrays[0])['owner_rank_rows'],counts.tolist())
        c['route']='random'
        arrays=[scenario_ids(c,r) for r in range(8)]
        self.assertEqual(len({a.tobytes() for a in arrays}),8)
        self.assertFalse(any(np.any(np.diff(np.sort(a,axis=1),axis=1)==0) for a in arrays))
        counts=np.bincount(np.stack(arrays).ravel(),minlength=32).reshape(8,4).sum(1)
        self.assertGreater(counts.max()/counts.mean(),1)
        self.assertLess(counts.max()/counts.mean(),1.1)

    def test_missing_rank(self):
        with self.assertRaises(ValueError):validate_receipts(receipts()[:-1])

    def test_mixed_rank_policy_rejected(self):
        for key in ('worker_variant','routing_variant','execution'):
            r=receipts();r[3][key]='different'
            with self.assertRaises(ValueError):validate_receipts(r)

    def test_duplicate_rank(self):
        r=receipts();r[-1]['rank']=0
        with self.assertRaises(ValueError):validate_receipts(r)

    def test_world8_ep4_rejected(self):
        r=receipts();r[0]['ep']=4
        with self.assertRaises(ValueError):validate_receipts(r)

    def test_changed_inputs_rejected(self):
        a=receipts();b=copy.deepcopy(a);b[3]['inputs']['x']='changed'
        with self.assertRaises(ValueError):check_pair(a,b)

    def test_changed_source_rejected(self):
        a=receipts();b=copy.deepcopy(a)
        for r in b:r['code_identity']='new'
        with self.assertRaises(ValueError):check_pair(a,b)

    def test_changed_plan_rejected(self):
        a=receipts();b=copy.deepcopy(a);b[5]['plan_digest']='new'
        with self.assertRaises(ValueError):check_pair(a,b,require_plan=True)

    def test_max_before_median(self):
        result=summarize_samples(receipts())
        self.assertEqual(result['samples_ms'],[8,9])
        self.assertEqual(result['median_ms'],8.5)

    def test_truncated_samples_rejected(self):
        r=receipts();r[3]['samples_ms'].pop()
        with self.assertRaises(ValueError):summarize_samples(r)

    def test_nonfinite_sample_rejected(self):
        r=receipts();r[0]['samples_ms'][0]=float('nan')
        with self.assertRaises(ValueError):summarize_samples(r)

    def test_reference_requires_vmm_alignment(self):
        c=dict(ep=8,experts=16,slots=2,tokens=2049,hidden=512,ffn=512,topk=2,
               capacity=36864,route='hot',planner='moonep',scope='online',rho=0.)
        with self.assertRaises(ValueError):validate_case(c)
        c['ffn']=1024;validate_case(c)

    def test_health_gate_rejects_warning_busy_or_partial(self):
        good='\n'.join(f'| {i} | Ascend950DT | OK | 390 |\n'
            '| | | NA | 0 0 / 0 4788 / 98304 |\n'
            f'| No running processes found in NPU {i} |' for i in range(8))
        self.assertTrue(inspect_smi(good))
        self.assertFalse(inspect_smi(good.replace('OK','Warning',1)))
        self.assertTrue(inspect_smi(good.replace('OK','Warning',1),True))
        self.assertTrue(inspect_smi(good.replace('OK','Alarm',1),True))
        self.assertFalse(inspect_smi(good.replace('OK','Offline',1),True))
        self.assertFalse(inspect_smi(good.replace('NPU 7','NPU 6')))
        self.assertFalse(inspect_smi(good.replace('NA | 0','NA | 12',1)))
        self.assertFalse(inspect_smi(good.replace('4788','38839',1)))
        self.assertFalse(inspect_smi(good.replace('4788','38839',1),True))
        self.assertFalse(inspect_smi(''))

    def test_wrong_physical_expert_rejected(self):
        import numpy as np
        from common import audit_placement
        c=dict(experts=16,slots=2,tokens=1,topk=2)
        ids=np.array([[0,1]]);copies=np.full((8,2),-1)
        counts=np.array([8,8]+[0]*14)
        audit_placement(c,ids,ids,copies,counts,1,counts)
        with self.assertRaises(ValueError):
            audit_placement(c,ids,np.array([[1,0]]),copies,counts,1,counts)

    def test_zero_output_cannot_pass_small_absolute_error(self):
        import numpy as np
        from compare_results import metric
        expected=np.full(128,1e-5,dtype=np.float32)
        for forward in (True,False):
            self.assertFalse(metric(np.zeros_like(expected),expected,forward)['pass_'])
            self.assertTrue(metric(expected,expected,forward)['pass_'])

    def test_balanced_random_is_unique_balanced_and_rank_distinct(self):
        import numpy as np
        from inputs import balanced_random_ids,calibrate
        c=dict(ep=8,experts=16,slots=2,tokens=4096,topk=6,seed=42,route='balanced_random')
        arrays=[balanced_random_ids(c,r) for r in range(8)]
        for ids in arrays:
            self.assertTrue(np.all(np.bincount(ids.ravel(),minlength=16)==1536))
            self.assertFalse(np.any(np.diff(np.sort(ids,axis=1),axis=1)==0))
        self.assertEqual(len({v.tobytes() for v in arrays}),8)
        self.assertTrue(np.array_equal(arrays[0],balanced_random_ids(c,0)))
        result=calibrate(c,arrays[3])
        self.assertEqual(result['owner_rank_rows'],[24576]*8)
        self.assertEqual(result['suggested_capacity'],27776)

    def test_probe_nested_exclusive_does_not_double_count(self):
        from stage_probe import StageProbe
        ticks=iter(range(20));p=StageProbe(lambda:None,clock=lambda:next(ticks))
        p.call('outer',lambda:p.call('inner',lambda:7))
        inner,outer=p.spans
        self.assertEqual(inner['parent'],'outer')
        self.assertAlmostEqual(outer['inclusive_ms'],outer['exclusive_ms']+inner['inclusive_ms'])
        self.assertIsNone(outer['parent'])


if __name__=='__main__':unittest.main()
