"""CPU-only rank-parallel form of the existing precision predicate.

Kept outside the device runner's source set. Its own receipt pins this file and
the original predicate; no device execution, tolerance, or timing is changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import sys

for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from common import OUTPUTS, check_pair, sha_file
from compare_results import metric, read_run


def rank_metrics(job):
    import torch
    backend, rank, paths, hashes = job
    for path, expected in zip(paths, hashes):
        if sha_file(path) != expected:
            raise ValueError('modified precision artifact: ' + str(path))
    expected, actual = [torch.load(p, map_location='cpu', weights_only=True) for p in paths]
    values = {name: metric(actual[name].float().numpy(), expected[name].float().numpy(), name == 'y')
              for name in OUTPUTS}
    return backend, dict(rank=rank, outputs=values)


def precision(directories, workers=4, baseline=None):
    if not 1 <= workers <= 4:
        raise ValueError('use one to four CPU workers')
    runs = {str(Path(d).resolve()): (Path(d), read_run(d)) for d in directories}
    if len(runs) != len(directories):
        raise ValueError('duplicate run directory')
    by_backend = {records[0]['backend']: (folder, records) for folder, records in runs.values()}
    if len(by_backend) != len(runs) or 'original' not in by_backend:
        raise ValueError('precision requires original and distinct backends')
    base_folder, base = by_backend['original']
    jobs = []
    for backend, (folder, records) in by_backend.items():
        check_pair(base, records)
        if any(r['mode'] != 'precision' for r in records):
            raise ValueError('not a precision run')
        for rank, (x, y) in enumerate(zip(base, records)):
            if x['dependency_digest'] != y['dependency_digest']:
                raise ValueError('external dependency mismatch')
            jobs.append((backend, rank,
                         [base_folder / f'rank{rank}.pt', folder / f'rank{rank}.pt'],
                         [x['artifact_sha256'], y['artifact_sha256']]))
    if 'local-serial' in by_backend and 's9' in by_backend:
        check_pair(by_backend['local-serial'][1], by_backend['s9'][1], require_plan=True)
    if baseline is not None:
        candidate = by_backend['s9'][1]
        previous = read_run(baseline)
        check_pair(previous, candidate, require_plan=True)
        for x, y in zip(previous, candidate):
            if x['dependency_digest'] != y['dependency_digest'] or x['outputs'] != y['outputs']:
                raise ValueError('candidate is not the same-plan bitwise baseline')
    details = {backend: [] for backend in by_backend}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for backend, row in pool.map(rank_metrics, jobs):
            details[backend].append(row)
    passed = all(m['pass_'] for ranks in details.values() for r in ranks for m in r['outputs'].values())
    result = dict(pass_=passed, case=base[0]['case'], code_identity=base[0]['code_identity'],
                  dependency_digest=base[0]['dependency_digest'], backends=list(by_backend),
                  input_hashes={str(r['rank']): r['inputs'] for r in base},
                  native_by_backend={b: {str(r['rank']): r['native'] for r in records}
                                     for b, (_, records) in by_backend.items()},
                  output_hashes_by_backend={b: {str(r['rank']): r['outputs'] for r in records}
                                           for b, (_, records) in by_backend.items()},
                  execution_by_backend={b: records[0].get('execution') for b, (_, records) in by_backend.items()},
                  details=details, tolerance_source='integration/wuhan/test_bridge_npu.py:metrics',
                  additional_relative_l2_limit=.05, scope=base[0]['scope'])
    if baseline is not None:
        result.update(same_plan_baseline_bitwise=True, baseline_directory=str(baseline))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))[-args.workers:]
        os.sched_setaffinity(0, cpus)
    else:
        cpus = None
    result = precision(args.runs, args.workers, args.baseline)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    receipt = dict(workers=args.workers, cpus=cpus,
                   checker_sha256=sha_file(Path(__file__)),
                   predicate_sha256=sha_file(BENCH / 'compare_results.py'),
                   cpu_only=True, report_sha256=sha_file(args.out))
    with args.out.with_suffix('.checker.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(dict(pass_=result['pass_'], report=str(args.out))))
    if not result['pass_']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
