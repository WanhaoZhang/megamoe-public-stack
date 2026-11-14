"""Reconstruct the frozen histogram-derived case on CPU; never opens an NPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
KIND = 'NEW_PYTHON_TRAINING_HISTOGRAM_DERIVED'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def kappa(counts):
    values = np.asarray(counts, dtype=np.float64)
    return float(len(values) * np.dot(values, values) / values.sum()**2)


def make_routes():
    """Original CDF/capping/rounding/RNG order, including expert permutation."""
    case, source = read_json(ROOT/'case.json'), read_json(ROOT/'source-profile.json')
    rho = np.sort(np.asarray(source['counts'], dtype=np.float64))[::-1]
    if len(rho) != 64 or rho.sum() != 262144 or abs(kappa(rho)-case['source_kappa']) > 1e-12:
        raise ValueError('source histogram differs from the frozen sampling event')
    rho /= rho.sum()
    experts, ep, tokens, topk = case['E'], case['ep'], case['M'], case['TopK']
    rows, total = ep*tokens, ep*tokens*topk
    weights = np.diff(np.interp(np.linspace(0, 1, experts+1),
                               np.linspace(0, 1, 65), np.r_[0, np.cumsum(rho)]))
    quotas = weights*total
    for _ in range(experts):
        over = quotas > rows
        if not over.any():
            break
        excess = (quotas[over]-rows).sum()
        quotas[over] = rows
        free = quotas < rows
        quotas[free] += excess*weights[free]/weights[free].sum()
    counts = np.floor(quotas).astype(np.int64)
    order = np.lexsort((np.arange(experts), -(quotas-counts)))
    counts[order[:total-int(counts.sum())]] += 1
    rng = np.random.default_rng(case['route_seed'])
    assignment = rng.permutation(experts)
    shuffled = np.empty(experts, np.int64)
    shuffled[assignment] = counts
    ids = np.full((rows, topk), -1, np.int32)
    positions = np.zeros(rows, np.int32)
    cursor = 0
    for expert in range(experts):
        selected = (np.arange(shuffled[expert], dtype=np.int64)+cursor) % rows
        ids[selected, positions[selected]] = expert
        positions[selected] += 1
        cursor += len(selected)
    ids = ids[rng.permutation(rows)]
    if (not np.all(positions == topk)
            or not np.all(np.diff(np.sort(ids, axis=1), axis=1) > 0)
            or not np.array_equal(np.bincount(ids.ravel(), minlength=experts), case['counts'])
            or abs(kappa(case['counts'])-case['target_kappa']) > 1e-12):
        raise ValueError('routing conservation/uniqueness/frozen histogram mismatch')
    return case, ids


def write_routes(out):
    case, ids = make_routes()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for rank, frozen in enumerate(case['exact_route_files']):
        value = ids[rank*case['M']:(rank+1)*case['M']]
        path = out/f'rank{rank}-ids.npy'
        np.save(path, value, allow_pickle=False)
        actual = dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path),
                      raw_tensor_sha256=hashlib.sha256(value.tobytes()).hexdigest())
        if any(actual[key] != frozen[key] for key in actual):
            raise ValueError('route file/tensor fingerprint mismatch: '+path.name)
        files.append(actual)
    record = dict(kind=KIND, name=case['name'], numpy=np.__version__,
                  counts=case['counts'], target_kappa=kappa(case['counts']),
                  initial_rank_routes=case['initial_rank_routes'], files=files,
                  source_profile_sha256=file_sha(ROOT/'source-profile.json'))
    (out/'routes.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record


def write_tensors(out, ranks):
    # Lazy import: route reconstruction requires only NumPy, no model or torch_npu.
    import torch
    sys.path.insert(0, str(ROOT.parents[1]))
    from timer_capture_config import window_bytes
    from scenario_capture_inputs import load_scenario_inputs, tensor_sha
    case, ids = make_routes()
    expected = read_json(ROOT/'expected-inputs.json')
    if torch.__version__ != '2.10.0+cpu':
        raise ValueError('historical dense RNG qualified with torch 2.10.0+cpu; use that runtime')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for rank in ranks:
        generator = torch.Generator(device='cpu').manual_seed(42+rank)
        def dense(shape, scale):
            return (torch.randn(shape, generator=generator, dtype=torch.float32)*scale).to(torch.bfloat16)
        home, m, h, f, k = (case[key] for key in ('home', 'M', 'H', 'F', 'TopK'))
        values = dict(x=dense((m, h), .2), dy=dense((m, h), .1),
                      prob=torch.full((m, k), .5, dtype=torch.float32),
                      w1=dense((home, 2*f, h), .01), w2=dense((home, h, f), .01),
                      logical_ids=torch.from_numpy(ids[(rank%case['ep'])*m:(rank%case['ep']+1)*m].copy()))
        config = dict(rank=rank, ep_rank=rank%case['ep'], world_size=case['world'],
                      ep=case['ep'], tokens=m, hidden=h, ffn=f, home=home,
                      slots=case['slots'], topk=k, capacity=case['capacity'], limit=case['limit'])
        config['window_bytes'] = window_bytes(m, h, k, case['ep'], home+case['slots'], case['capacity'])
        fingerprints = {key: dict(shape=list(value.shape), dtype=str(value.dtype),
                                 sha256=tensor_sha(value)) for key, value in values.items()}
        if fingerprints != expected[str(rank)]['tensors']:
            raise ValueError(f'rank{rank} dense input differs from historical tensor SHA; do not relax checks')
        target = out/f'rank{rank}-inputs.pt'
        torch.save(dict(config=config, tensors=values), target)
        manifest = dict(kind=KIND, name=case['name'], config=config,
                        file_sha256=file_sha(target), tensors=fingerprints,
                        torch=torch.__version__, numpy=np.__version__,
                        historical_outputs=expected[str(rank)]['historical_outputs'],
                        historical_outputs_scope='reference only; requires identical placement and numerical path')
        target.with_suffix('.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
        # Validate exactly the reader used by the Timer entry, including the serialized payload.
        load_scenario_inputs(out, rank, case['world'])
        print(f'rank{rank}: six historical input tensor SHA verified', flush=True)
        del values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='new output directory outside Git')
    parser.add_argument('--with-tensors', action='store_true', help='also generate ~4GB CPU inputs for eight ranks')
    parser.add_argument('--ranks', nargs='+', type=int, default=list(range(8)), help='dense input ranks (default 0..7)')
    args = parser.parse_args()
    if len(set(args.ranks)) != len(args.ranks) or any(rank not in range(8) for rank in args.ranks):
        parser.error('unique global ranks 0..7 required')
    if args.out.exists():
        parser.error('output exists; choose a new directory')
    result = write_routes(args.out)
    print(json.dumps(result, indent=2))
    if args.with_tensors:
        write_tensors(args.out/'inputs', args.ranks)


if __name__ == '__main__':
    main()
