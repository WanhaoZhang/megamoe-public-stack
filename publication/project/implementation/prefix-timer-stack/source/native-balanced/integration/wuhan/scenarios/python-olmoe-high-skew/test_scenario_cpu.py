"""Frozen-case reconstruction and CLI guard checks; NumPy only, no NPU."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('python_case_generator', ROOT/'generate.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)
loader_spec = importlib.util.spec_from_file_location('scenario_loader', ROOT.parents[1]/'scenario_capture_inputs.py')
loader = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(loader)


class FrozenCaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get('SCENARIO_TEST_TMPDIR'))
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)

    def test_exact_files_and_all_eight_rank_ids(self):
        expected = generator.read_json(ROOT/'expected-inputs.json')
        first = generator.write_routes(self.out/'first')
        second = generator.write_routes(self.out/'second')
        self.assertEqual(first, second)
        for rank in range(8):
            item = first['files'][rank % 4]
            self.assertEqual(item['raw_tensor_sha256'], expected[str(rank)]['tensors']['logical_ids']['sha256'])
            self.assertEqual((self.out/'first'/item['file']).read_bytes(),
                             (self.out/'second'/item['file']).read_bytes())
        self.assertEqual(first['counts'], [12051,32768,3495,572,5497,1175,8107,1871])
        self.assertEqual(first['initial_rank_routes'], [44819,4067,6672,9978])
        self.assertEqual(first['target_kappa'], 2.481662403792143)

    def test_existing_output_preserved(self):
        target = self.out/'existing'
        target.mkdir()
        marker = target/'important.txt'
        marker.write_text('keep')
        with self.assertRaises(FileExistsError):
            generator.write_routes(target)
        self.assertEqual(marker.read_text(), 'keep')
        self.assertEqual(list(target.iterdir()), [marker])

    def test_changed_source_histogram_rejected(self):
        source = generator.read_json(ROOT/'source-profile.json')
        source['counts'][0] += 1
        (self.out/'source-profile.json').write_text(json.dumps(source))
        (self.out/'case.json').write_bytes((ROOT/'case.json').read_bytes())
        old = generator.ROOT
        try:
            generator.ROOT = self.out
            with self.assertRaisesRegex(ValueError, 'source histogram'):
                generator.make_routes()
        finally:
            generator.ROOT = old

    def test_cli_invalid_rank_rejected_before_writes(self):
        for ranks in (['8'], ['0', '0']):
            result = subprocess.run([sys.executable, str(ROOT/'generate.py'), '--out',
                                     str(self.out/'bad'), '--ranks', *ranks], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('unique global ranks', result.stderr)
            self.assertFalse((self.out/'bad').exists())

    def test_capture_sources_mutually_exclusive_before_torch_import(self):
        entry = ROOT.parents[1]/'capture_kernel_timer.py'
        result = subprocess.run([sys.executable, str(entry), '--out', str(self.out/'capture'),
                                 '--timer', '0', '--capture', 'old', '--scenario-inputs', 'new'],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn('mutually exclusive', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_loader_rejects_wrong_world_and_global_rank_before_torch(self):
        for rank, world in ((0, 4), (-1, 8), (8, 8)):
            with self.assertRaisesRegex(ValueError, 'world8'):
                loader.load_scenario_inputs(self.out, rank, world)

    def test_loader_rejects_corrupted_file_before_deserialization(self):
        case = generator.read_json(ROOT/'case.json')
        (self.out/'rank0-inputs.pt').write_bytes(b'corrupted file')
        (self.out/'rank0-inputs.json').write_text(json.dumps(dict(
            kind=case['kind'], name=case['name'], file_sha256='0'*64)))
        with self.assertRaisesRegex(ValueError, 'file SHA'):
            loader.load_scenario_inputs(self.out, 0, 8)

    def test_benchmark_requires_reference_before_torch_import(self):
        entry = ROOT.parents[1]/'replay_captured_layer.py'
        result = subprocess.run([sys.executable, str(entry), '--out', str(self.out/'benchmark'),
                                 '--restore', 'panel', '--scenario-inputs', 'new'],
                                capture_output=True, text=True)
        # This distributed entry preserves its existing BaseException -> _exit(1) wrapper.
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('requires --reference', result.stderr)
        self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
