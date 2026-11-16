"""CPU-only audit of the read-only test mask against the existing layout.

No torch/NPU import. This checks that the immutability gate exempts exactly
the existing progress suffix across metadata alignment/tail boundaries.
"""
import ast
from pathlib import Path
import unittest

from shared_expert_backward_reference import routed_progress_range, require_active_replica


class ReadonlyMaskContract(unittest.TestCase):
    def test_replica_coverage_cannot_leak_between_cases_or_epochs(self):
        self.assertTrue(require_active_replica('hot', True, 0, 0))
        with self.assertRaisesRegex(AssertionError, 'rotate.*EP group=0'):
            require_active_replica('rotate', False, 0, 0)
        self.assertTrue(require_active_replica('rotate', True, 1, 0))
        with self.assertRaisesRegex(AssertionError, 'iteration=1'):
            require_active_replica('rotate', False, 1, 1)
        self.assertFalse(require_active_replica('balanced', False, 0, 0))

    def test_matches_existing_saved_state_parser(self):
        source = Path(__file__).resolve().parents[2] / 'test/megamoe_bf16_test/test_mega_moe_backward.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                    node.name in ('_align_up', 'training_saved_state_layout')]
        self.assertEqual(len(selected), 2)
        namespace = {}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)
        old_parser = namespace['training_saved_state_layout']
        for ep in (2, 4, 8):
            for physical in (4, 8, 32):
                for m in (1, 127, 128, 129, 255, 257, 16384):
                    with self.subTest(ep=ep, physical=physical, m=m):
                        existing = old_parser(m, 512, 2, ep, physical, 32768, 1024, 'bf16')
                        begin, end = routed_progress_range(m, 2, ep, physical)
                        self.assertEqual(begin, existing['progress'])
                        self.assertEqual(end, existing['legacy_bytes'])
                        self.assertEqual(end - begin, 1024)
                        self.assertLessEqual(end, existing['gate'])
                        self.assertLess(end, existing['total_bytes'])


if __name__ == '__main__':
    unittest.main()
