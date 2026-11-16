"""Audit explicit paired run directories; never infer acceptance from logs alone."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def load_run(directory, steps, warmup, ranks=None):
    launch = read(directory / 'launch.json')
    acceptance = read(directory / 'acceptance.json')
    assert acceptance['eligible'] and acceptance['run_exit'] == 0 and acceptance['post_gate']
    assert read(directory / 'pre.json')['pass_'] and read(directory / 'post.json')['pass_']
    assert int((directory / 'run.exit').read_text()) == 0
    assert launch['steps'] == steps and steps > warmup >= 0
    ranks = list(range(8) if ranks is None else ranks)
    assert ranks and len(set(ranks)) == len(ranks)
    by_rank, initial = [], []
    for rank in ranks:
        values = [json.loads(line) for line in (directory / 'steps' / f'rank{rank}.jsonl').read_text().splitlines()]
        assert [r['step'] for r in values] == list(range(1, steps + 1))
        assert all(r['rank'] == rank and r['wall_ms'] > 0 and
                   all(math.isfinite(r[k]) for k in ('wall_ms', 'loss', 'lm_loss', 'mtp_loss', 'aux_loss', 'grad_norm'))
                   for r in values)
        assert all(r['boundary'] == 'before_fetch_to_optimizer_zero_grad_device_complete' for r in values)
        by_rank.append(values)
        fingerprint = directory / 'steps' / f'initial-rank{rank}.json'
        initial.append(read(fingerprint)['parameters'] if fingerprint.exists() else None)
    text = (directory / 'run.log').read_text(errors='replace')
    for failure in ('Skipping optimizer and scheduler', 'Invalid raw ', 'FloatingPointError', 'Traceback (most recent'):
        assert failure not in text, failure
    maxima = [max(rows[s]['wall_ms'] for rows in by_rank) for s in range(warmup, steps)]
    ordered = sorted(maxima)
    origin_path = directory/'origin.json'
    origin = read(origin_path) if origin_path.exists() else None
    if origin:
        for name, expected in origin['file_hashes'].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, ('changed evidence', name)
    return dict(path=str(directory), launch=launch, rank_max_wall_ms=maxima,
                node=origin['node'] if origin else None,
                timing_eligible=not launch.get('diagnostic_capture', False),
                median_ms=statistics.median(maxima), p90_ms=ordered[math.ceil(.9 * len(ordered)) - 1],
                initial=initial, rows=by_rank)


def compare_runs(run, reference):
    if reference is None:
        return dict(initial_shards_identical=False, initial_hashes_available=False,
                    data_fingerprints_available=False, data_identical=None,
                    max_local_lm_relative_error_vs_A=None, max_mean_lm_relative_error_vs_A=None,
                    max_reduced_lm_relative_error_vs_A=None, same_node=False,
                    same_training_config=False, speedup_vs_A=None)
    same_node = bool(run.get('node') and run.get('node') == reference.get('node'))
    keys = ('rho', 'activation', 'layers', 'experts', 'topk', 'sequence', 'ep', 'expert_fsdp', 'steps')
    same_config = (all(run['launch'].get(k) == reference['launch'].get(k) for k in keys) and
                   run['launch'].get('world_size', 8) == reference['launch'].get('world_size', 8) and
                   len(run['rows']) == len(reference['rows']) == run['launch'].get('world_size', 8))
    fingerprinted = all(v is not None for v in run['initial'] + reference['initial'])
    matching = fingerprinted and run['initial'] == reference['initial']
    pairs = [(row, base) for rank, ref in zip(run['rows'], reference['rows'])
             for row, base in zip(rank, ref)]
    data_available = all(a.get('batch_fingerprints') and b.get('batch_fingerprints') for a, b in pairs)
    data_matching = (all(a['batch_fingerprints'] == b['batch_fingerprints'] for a, b in pairs)
                     if data_available else None)
    relative = lambda actual, expected: abs(actual - expected) / max(abs(expected), 1e-12)
    local_error = max(relative(a['lm_loss'], b['lm_loss']) for a, b in pairs)
    mean_error = max(relative(statistics.mean(r[s]['lm_loss'] for r in run['rows']),
                              statistics.mean(r[s]['lm_loss'] for r in reference['rows']))
                     for s in range(len(run['rows'][0])))
    reduced = all('reduced_losses' in a and 'reduced_losses' in b for a, b in pairs)
    reduced_error = (max(relative(a['reduced_losses']['lm_loss'], b['reduced_losses']['lm_loss'])
                         for a, b in pairs) if reduced else None)
    return dict(initial_shards_identical=matching, initial_hashes_available=fingerprinted,
                same_node=same_node, same_training_config=same_config,
                data_fingerprints_available=data_available, data_identical=data_matching,
                max_local_lm_relative_error_vs_A=local_error,
                max_mean_lm_relative_error_vs_A=mean_error,
                max_reduced_lm_relative_error_vs_A=reduced_error,
                speedup_vs_A=reference['median_ms'] / run['median_ms']
                if same_node and same_config and matching and data_matching and run.get('timing_eligible', True)
                and reference.get('timing_eligible', True) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--warmup', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    runs = [load_run(p, args.steps, args.warmup) for p in args.runs]
    groups = {}
    for run in runs:
        p = run['launch']
        key = (p['rho'], p['activation'], p['layers'], p['experts'], p['topk'], p['sequence'])
        methods = groups.setdefault(key, {})
        assert p['method'] not in methods, 'use one process order per analysis'
        methods[p['method']] = run
    summary = []
    for key, methods in groups.items():
        a = methods.get('A')
        for method, run in methods.items():
            summary.append(dict(config=list(key), method=method, path=run['path'],
                                median_rank_max_wall_ms=run['median_ms'], p90_rank_max_wall_ms=run['p90_ms'],
                                **compare_runs(run, a),
                                timing_only_without_precision_acceptance=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(steps=args.steps, warmup=args.warmup, results=summary), indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
