import json
from pathlib import Path
import tempfile
import unittest
from summarize_early_dw1_perf import summarize


class SummaryTests(unittest.TestCase):
    def fixture(self,root,fb=0.95):
        for i,v in enumerate(['c0','c0','c0','c1','c1','c0','c0','c1','c1','c0']):
            d=root/'evidence'/f'perf-{i:02d}-{v}';d.mkdir(parents=True)
            (d/'acceptance.json').write_text(json.dumps(dict(eligible=True)))
            values=dict(backward=[.95 if v=='c1' else 1.]*30,
                        fb=[fb if v=='c1' else 1.]*30)
            for rank in range(8):
                record=dict(pass_=True,rank=rank,warmup=10,repeat=30,
                    storage=dict(y='same'),final_storage=dict(y='same'),config=dict(ep=4),
                    samples_ms=values,rank_max_samples_ms=values)
                (d/f'rank{rank}.json').write_text(json.dumps(record))

    def test_stable_pair_improvement_passes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            result=summarize(root)
            self.assertTrue(result['performance_pass'])
            self.assertAlmostEqual(result['gains']['backward'][0],.05)

    def test_fb_regression_rejects_even_with_faster_backward(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root,1.03)
            self.assertFalse(summarize(root)['performance_pass'])

    def test_failed_gate_cannot_supply_performance(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            (root/'evidence/perf-09-c0/acceptance.json').write_text('{"eligible":false}')
            with self.assertRaises(AssertionError):summarize(root)


if __name__=='__main__':unittest.main()
