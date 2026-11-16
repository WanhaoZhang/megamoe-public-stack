"""Audit matched 200-step runs and export per-step/segment stability evidence."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from summarize_compact import peak_hbm
from summarize_training import compare_runs, load_run


def analyze(runs):
    methods = {r['launch']['method']: r for r in runs}
    if len(methods) != len(runs) or 'A' not in methods:
        raise ValueError('one run per method and a same-campaign A are required')
    output, steps = [], []
    for method, run in methods.items():
        comparison = compare_runs(run, methods['A'])
        if comparison['speedup_vs_A'] is None:
            raise ValueError('unpaired node/config/initial/batch: '+method)
        segments = []
        for first, last in ((11, 50), (51, 100), (101, 150), (151, 200)):
            walls = [max(rank[s]['wall_ms'] for rank in run['rows']) for s in range(first-1, last)]
            segments.append(dict(first=first, last=last, median_ms=statistics.median(walls),
                p90_ms=sorted(walls)[math.ceil(.9*len(walls))-1]))
        for s in range(200):
            row, reference = [r[s] for r in run['rows']], [r[s] for r in methods['A']['rows']]
            relative = lambda a,b: abs(a-b)/max(abs(b), 1e-12)
            steps.append(dict(method=method, step=s+1,
                rank_max_wall_ms=max(r['wall_ms'] for r in row),
                mean_lm_loss=statistics.mean(r['lm_loss'] for r in row),
                max_local_lm_relative_error_vs_A=max(relative(a['lm_loss'], b['lm_loss'])
                    for a,b in zip(row,reference)),
                max_grad_relative_error_vs_A=max(relative(a['grad_norm'], b['grad_norm'])
                    for a,b in zip(row,reference)),
                grad_norm_max=max(r['grad_norm'] for r in row),
                allocated_gib_max=max(r['allocated_bytes'] for r in row)/(1024**3),
                reserved_gib_max=max(r['reserved_bytes'] for r in row)/(1024**3),
                peak_allocated_gib_max=max(r['peak_allocated_bytes'] for r in row)/(1024**3),
                peak_reserved_gib_max=max(r['peak_reserved_bytes'] for r in row)/(1024**3)))
        output.append(dict(method=method, median_ms=run['median_ms'], p90_ms=run['p90_ms'],
            segments=segments, peak_hbm_mb=peak_hbm(run['path']), **comparison,
            scope='200-step controlled routing; not convergence or natural SFT'))
    return dict(steps=200, warmup=10, rows=output), steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result, rows = analyze([load_run(path, 200, 10) for path in a.runs])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    with a.out.with_suffix('.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
