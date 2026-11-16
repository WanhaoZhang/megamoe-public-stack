"""Calibrate official routing on CPU and preserve its histogram under permutations."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from generate_topk_routing import generate_topk_routing


def prepare(task: tuple) -> dict:
    destination, target, tokens, topk, experts, ranks, seed = task
    torch.set_num_threads(4)
    tolerance = max(0.02 * target, 0.005)
    def sample(sigma: float) -> tuple[float, torch.Tensor, torch.Tensor]:
        routes = []
        counts = torch.zeros(experts, dtype=torch.int64)
        for rank in range(ranks):
            ids, histogram = generate_topk_routing(tokens, topk, experts, ranks,
                                                  sigma, 'cpu', seed, rank=rank)
            routes.append(ids)
            counts += histogram
        violation = float(counts.max()) / (float(counts.sum()) / experts) - 1.0
        return violation, torch.stack(routes), counts
    low, high = 1e-5, 10.0
    best = None
    for iteration in range(60):
        sigma = (low * high) ** 0.5
        actual, routes, counts = sample(sigma)
        if best is None or abs(actual - target) < abs(best[0] - target):
            best = actual, routes, counts, sigma
        if abs(actual - target) <= tolerance:
            break
        if actual < target:
            low = sigma
        else:
            high = sigma
    actual, routes, counts, sigma = best
    if abs(actual - target) > tolerance:
        raise RuntimeError(f'Unable to calibrate target={target}: actual={actual}')
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    route_path = output / f'maxvio-{target:g}.npy'
    np.save(route_path, routes.numpy(), allow_pickle=False)
    result = dict(target=target, realized=actual, sigma=sigma, tolerance=tolerance,
                  shape=list(routes.shape), counts=counts.tolist(), seed=seed,
                  iterations=iteration + 1,
                  sha256=hashlib.sha256(route_path.read_bytes()).hexdigest(),
                  variation='Shared expert permutation and per-rank token permutation per layer/step; histogram values and MaxVio preserved')
    (output / f'maxvio-{target:g}.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ('target', 'realized', 'sigma', 'iterations')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=16384)
    parser.add_argument('--experts', type=int, default=256)
    parser.add_argument('--topk', type=int, default=8)
    parser.add_argument('--ranks', type=int, default=8)
    parser.add_argument('--seed',type=int,default=1234)
    parser.add_argument('--targets',default='0.2,1,5,10,15,20')
    args = parser.parse_args()
    jobs = [(str(args.out), value, args.tokens, args.topk, args.experts, args.ranks,args.seed)
            for value in map(float,args.targets.split(','))]
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(prepare, jobs))
    (args.out / 'manifest.json').write_text(json.dumps(results, indent=2))
