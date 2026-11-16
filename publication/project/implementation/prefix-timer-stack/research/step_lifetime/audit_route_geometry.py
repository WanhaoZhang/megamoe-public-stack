"""Offline histogram audit of the frozen RouteProvider permutations, never a planner."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    path = ROOT / 'routes/maxvio-10.npy'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = json.loads((ROOT / 'evidence/lifetime-a3-main-adam3/source-identity.json').read_text())
    if digest != identity['routes/maxvio-10.npy']:
        raise ValueError('Route bank differs from the measured main shape')
    base = np.load(path, allow_pickle=False)
    if base.shape != (8, 16384, 8):
        raise ValueError('This audit covers only the frozen EP8/S16K/Top8 family')
    counts = np.bincount(base.ravel(), minlength=256)
    records, summary = [], {}
    for pattern in ('same', 'alternating', 'random'):
        violations, complementarity, reuse = [], [], []
        for step in range(25):
            for layer in range(4):
                loads, hotsets = [], []
                for microbatch in range(4):
                    counter = step * 4 + layer
                    if pattern == 'random':
                        counter = counter * 4 + microbatch
                    permutation = np.random.default_rng(42 + 100003 * counter).permutation(256)
                    if pattern == 'alternating':
                        permutation = (permutation + microbatch * 32) % 256
                    mapped = np.empty_like(counts)
                    mapped[permutation] = counts
                    rank_load = mapped.reshape(8, 32).sum(1)
                    loads.append(rank_load)
                    hotsets.append(set(np.argsort(-mapped, kind='stable')[:16].tolist()))
                    violation = float(rank_load.max() / rank_load.mean() - 1)
                    violations.append(violation)
                    records.append(dict(pattern=pattern, step=step, layer=layer, microbatch=microbatch,
                                        home_rank_load=rank_load.tolist(), home_rank_maxvio=violation))
                for first in (0, 2):
                    left, right = loads[first:first + 2]
                    complementarity.append(float(1 - (left + right).max() / (left.max() + right.max())))
                    first_hot, second_hot = hotsets[first:first + 2]
                    reuse.append(len(first_hot & second_hot) / len(first_hot | second_hot))
        summary[pattern] = dict(home_rank_maxvio_mean=float(np.mean(violations)),
                                home_rank_maxvio_max=float(np.max(violations)),
                                ideal_same_layer_pair_compute_saving=float(np.mean(complementarity)),
                                top16_hot_identity_jaccard=float(np.mean(reuse)))
    result = dict(role='offline_route_geometry_audit_not_online_score_or_full_step_bound',
                  bank_sha256=digest, expert_maxvio=float(counts.max() / counts.mean() - 1),
                  summary=summary, records=records)
    (ROOT / 'diagnostics/route-opportunity.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    main()
