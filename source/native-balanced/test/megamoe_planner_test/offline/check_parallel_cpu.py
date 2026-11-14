"""Compare the full parallel report with the unchanged serial checker on CPU fixtures."""
import json
import os
from pathlib import Path
import tempfile
import sys

os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
import parallel_precision as parallel
import compare_results as serial
from common import OUTPUTS, sha_file


def main():
    import torch
    templates = {backend: [dict(rank=rank, ep=8, world=8, pass_=True,
        case={'fixture': True}, backend=backend, mode='precision', scope='online',
        code_identity='fixture', inputs={'x': str(rank)}, plan_digest='fixture',
        dependency_digest='fixture', native={}, outputs={}) for rank in range(8)]
        for backend in ('original', 's9')}
    with tempfile.TemporaryDirectory(prefix='offline-check-') as directory:
        root = Path(directory)
        folders = [root / backend for backend in templates]
        for f in folders:
            f.mkdir()
        records = {str(f): json.loads(json.dumps(templates[f.name])) for f in folders}
        original_reader, parallel_reader = serial.read_run, parallel.read_run
        serial.read_run = parallel.read_run = lambda folder: records[str(folder)]
        try:
            for failing in (False, True):
                for f in folders:
                    for rank in range(8):
                        tensors = {name: torch.arange(24, dtype=torch.float32).reshape(4, 6) / 20
                                   for name in OUTPUTS}
                        if f.name == 's9':
                            tensors['dw1'][0, 0] += 0.001
                            if failing and rank == 3:
                                tensors['dx'][0, 0] = float('nan')
                        path = f / f'rank{rank}.pt'
                        torch.save(tensors, path)
                        records[str(f)][rank]['artifact_sha256'] = sha_file(path)
                expected = serial.precision(folders)
                actual = parallel.precision(folders, workers=4)
                assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
                assert actual['pass_'] == (not failing)
            path = folders[1] / 'rank0.pt'
            with path.open('ab') as stream:
                stream.write(b'tampered')
            try:
                parallel.precision(folders, workers=4)
            except ValueError as error:
                assert 'modified precision artifact' in str(error)
            else:
                raise AssertionError('tampered artifact accepted')
        finally:
            serial.read_run, parallel.read_run = original_reader, parallel_reader
    print(json.dumps(dict(pass_=True, checks=['full-report-equality-pass',
          'full-report-equality-nonfinite-rejection', 'artifact-tamper-rejection'],
          checker_sha256=sha_file(Path(parallel.__file__)))))


if __name__ == '__main__':
    main()
