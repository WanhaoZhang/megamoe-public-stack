"""Aggregate only complete and qualified controlled EP8 training runs."""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import importlib.util
import os
import sys
plot_python = Path(__file__).resolve().parents[1] / 'dependencies/plot-env/bin/python'
if importlib.util.find_spec('matplotlib') is None and plot_python.exists():
    os.execv(str(plot_python), [str(plot_python), __file__, *sys.argv[1:]])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--prefix', default='formal')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    output_root = args.output_root or args.root
    output_root.mkdir(parents=True,exist_ok=True)
    figures, tables = output_root / 'figures', output_root / 'tables'
    (output_root / 'analysis-environment.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,numpy=np.__version__,matplotlib=matplotlib.__version__),indent=2))
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    qualified, excluded, detailed, identities = [], [], [], {}
    for directory in sorted((args.root / 'evidence').glob(f'{args.prefix}-L*')):
        status_path = directory / 'status.json'
        if not status_path.exists():
            excluded.append(dict(run=directory.name, reason='missing final status'))
            continue
        status = json.loads(status_path.read_text())
        if not status.get('qualified'):
            excluded.append(dict(run=directory.name, reason='nonzero exit or missing rank completions'))
            continue
        qualification_path = directory / 'qualification-checks.json'
        if not qualification_path.exists() or not json.loads(qualification_path.read_text()).get('passed'):
            excluded.append(dict(run=directory.name, reason='missing successful numerical/runtime qualification'))
            continue
        parts = directory.name.split('-')
        layers = int(parts[1][1:])
        maxvio, backend, repeat = float(parts[2][3:]), parts[3], int(parts[4][1:])
        ranks = [list(map(json.loads, (directory / f'steps-rank{rank}.jsonl').read_text().splitlines())) for rank in range(8)]
        if any(len(rows) != 25 or [row['step'] for row in rows] != list(range(25)) for rows in ranks):
            excluded.append(dict(run=directory.name, reason='incomplete or duplicate step sequence'))
            continue
        for rank in range(8):
            logical = json.loads((directory / f'initial-rank{rank}.json').read_text())
            routes = json.loads((directory / f'route-identity-rank{rank}.json').read_text())
            inputs = json.loads((directory / f'input-identity-rank{rank}.json').read_text())
            key = (layers, maxvio, rank)
            value = dict(logical=logical, routes=routes, inputs=inputs)
            if key in identities and identities[key] != value:
                raise RuntimeError(f'Unequal initial conditions: {directory.name} rank{rank}')
            identities[key] = value
        steps = np.max([[row['milliseconds'] for row in rank] for rank in ranks], axis=0)
        for step, value in enumerate(steps):
            detailed.append(dict(run=directory.name, backend=backend, maxvio=maxvio, repeat=repeat,
                                 layers=layers, step=step, warmup=int(step < 5), max_rank_ms=value))
        qualified.append(dict(run=directory.name, backend=backend, maxvio=maxvio, repeat=repeat, layers=layers,
                              mean_ms=float(steps[5:].mean()), median_ms=float(np.median(steps[5:])),
                              p95_ms=float(np.quantile(steps[5:], .95)),
                              driver_peak_GiB=status['peak_driver_MiB'] / 1024,
                              allocator_peak_GiB=max(row['peak_allocated_bytes'] for rank in ranks for row in rank) / 2**30))
    (tables / 'excluded.json').write_text(json.dumps(excluded, indent=2))
    if not qualified:
        (tables / 'aggregation-status.json').write_text(json.dumps(dict(qualified_runs=0, figures_generated=False)))
        print('No qualified timing data; no performance curves generated')
        return
    if len({row['layers'] for row in qualified}) != 1:
        raise RuntimeError('Different model depths must never share a comparison')
    save_csv(tables / 'runs.csv', qualified)
    save_csv(tables / 'steps.csv', detailed)
    groups = defaultdict(list)
    for row in qualified:
        groups[row['backend'], row['maxvio']].append(row)
    baseline = np.mean([r['mean_ms'] for r in groups['moonep', .2]])
    if not np.isfinite(baseline):
        raise RuntimeError('Missing common normalization denominator')
    rows = []
    for (backend, maxvio), records in sorted(groups.items()):
        moon = groups.get(('moonep', maxvio), [])
        moon_ms = np.mean([r['mean_ms'] for r in moon]) if moon else np.nan
        mean = np.mean([r['mean_ms'] for r in records])
        rows.append(dict(backend=backend, maxvio=maxvio, realized_maxvio=json.loads((args.root/'routes'/f'maxvio-{maxvio:g}.json').read_text())['realized'], repeats=len(records), mean_ms=mean,
                         std_ms=np.std([r['mean_ms'] for r in records], ddof=1) if len(records) > 1 else 0.,
                         normalized=mean / baseline, speedup_vs_moonep=moon_ms / mean,
                         driver_peak_GiB=max(r['driver_peak_GiB'] for r in records),
                         mean_driver_peak_GiB=np.mean([r['driver_peak_GiB'] for r in records])))
    save_csv(tables / 'summary.csv', rows)
    specifications = [
        ('absolute-step-time', 'mean_ms', 'Full training step (ms)'),
        ('normalized-step-time', 'normalized', 'Step time / MoonEP at MaxVio=0.2'),
        ('speedup-vs-moonep', 'speedup_vs_moonep', 'MoonEP time / backend time'),
        ('whole-card-hbm', 'driver_peak_GiB', 'Peak whole-card HBM (GiB)')]
    for filename, metric, ylabel in specifications:
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for backend, color, label in [('moonep', '#e98112', 'Ascend MoonEP'),
                                      ('balanced', '#266bc4', 'Native balanced'),
                                      ('shared', '#20a077', 'Native balanced shared')]:
            values = sorted([row for row in rows if row['backend'] == backend], key=lambda r: r['maxvio'])
            if values:
                ax.plot([r['maxvio'] for r in values], [r[metric] for r in values], 'o-', color=color, label=label)
                if metric == 'mean_ms':
                    ax.errorbar([r['maxvio'] for r in values], [r[metric] for r in values],
                                yerr=[r['std_ms'] for r in values], fmt='none', color=color, capsize=3)
        ax.set_xscale('log')
        ax.set_xticks([.2, 1, 5, 10, 15, 20], ['0.2', '1', '5', '10', '15', '20'])
        ax.set_xlabel('Target MaxVio')
        ax.set_ylabel(ylabel)
        layers = qualified[0]['layers']
        ax.set_title(f'Ascend 950DT x8 | L{layers}, EP8 E256 Top8 H7168 F2048 S16384/rank')
        ax.grid(alpha=.25)
        ax.legend()
        if metric in ('normalized', 'speedup_vs_moonep'):
            ax.axhline(1., color='gray', linestyle=':', linewidth=1)
        fig.text(.5, .01, 'Common crop: vocab32768, mHC2, Q-heads32; FP32 Adam / BF16 experts; measured data only.',
                 ha='center', fontsize=8)
        repeats=sorted({row['repeats'] for row in rows})
        fig.text(.5, .035, f'25 steps/process; 5 warmup + 20 measured; max over 8 ranks; repeats/point={repeats}'+('; bars=repeat SD' if metric=='mean_ms' else ''),
                 ha='center', fontsize=8)
        fig.tight_layout(rect=[0,.075,1,1])
        for extension in ('png', 'svg', 'pdf'):
            fig.savefig(figures / f'{filename}.{extension}', dpi=180)
        plt.close(fig)
    (tables / 'aggregation-status.json').write_text(json.dumps(dict(
        qualified_runs=len(qualified), figures_generated=True, normalization_baseline_ms=baseline,
        normalization='mean of qualified MoonEP repeat means at target MaxVio 0.2',
        aggregation='mean over repeat means of 20 per-step maximum rank wall times'), indent=2))
    print(json.dumps(dict(qualified=len(qualified), excluded=len(excluded), baseline=baseline)))

if __name__ == '__main__':
    main()
