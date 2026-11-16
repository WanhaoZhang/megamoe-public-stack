"""Audit the preregistered online screen; never mix offline replay into scores."""
from pathlib import Path
import csv
import hashlib
import json
import math
import statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'evidence'
OUT = ROOT / 'figures'
LABELS = ('legacy_w1', 'legacy_w2', 'fixed_native', 'fifo_native', 'easy_home', 'easy_independent')
PATTERNS = ('same', 'alternating', 'random')
WORKER = '46772ebc9fe69442e786dd4e9bd6892bb79bfe6e3944069df05bfafaf684712e'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def task_loads(counts, pattern, key):
    """Offline-only reconstruction, never supplied to the online controller."""
    step, microbatch, layer = key
    counter = step * 4 + layer
    if pattern == 'random':
        counter = counter * 4 + microbatch
    permutation = np.random.default_rng(42 + 100003 * counter).permutation(256)
    if pattern == 'alternating':
        permutation = (permutation + microbatch * 32) % 256
    return np.bincount(permutation // 32, weights=counts, minlength=8).astype(np.int64)


def reduce():
    manifest = read(ROOT / 'screening-results.json')
    frozen = read(ROOT / 'stack/configs/runtime-freeze.json')['binaries']
    frozen.update(read(ROOT / 'stack/configs/temporal-scheduler-freeze.json')['binaries'])
    # Validate bank statistics against actual fixture rows, rather than trusting a plotted target.
    bank_path = ROOT / 'diagnostics/screening-maxvio-10.npy'
    bank_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    base = np.load(bank_path, allow_pickle=False)
    if base.shape != (8, 16384, 8):
        raise RuntimeError('Unexpected screening bank shape')
    counts = np.bincount(base.reshape(-1), minlength=256)
    if counts.tolist() != read(ROOT / 'stack/routes/maxvio-10.json')['counts']:
        raise RuntimeError('Fixture counts disagree with source data')
    rows, step_rows, group_rows, identity_checks = [], [], [], {}
    failed_attempts=[attempt for job in manifest['jobs'] for attempt in job.get('failed_attempts', [])]
    failed_attempts.extend(dict(tag=job['tag'], classification='unclassified_execution_failure',
                                receipt=job.get('receipt', {}))
                           for job in manifest['jobs'] if job['status']=='failed')
    reference = {}
    for job in manifest['jobs']:
        if job['status'] != 'completed':
            continue
        folder = DATA / job['tag']
        if read(folder / 'source-identity.json').get('routes/maxvio-10.npy') != bank_sha:
            raise RuntimeError('Offline audit bank differs from the guarded runtime bank')
        status = read(folder / 'status.json')
        if not (status.get('qualified') and status.get('exclusive_devices') and
                status.get('source_unchanged') and status.get('rank_completions') == 8 and
                status['peak_driver_MiB'] <= 90 * 1024):
            raise RuntimeError('Unqualified screen run: ' + job['tag'])
        config = read(folder / 'effective-config.json')
        if config.get('uses_future_derived_schedule') or config.get('offline_diagnostic_only'):
            raise RuntimeError('Offline data cannot enter the online screen')
        if (config['layers'], config['tokens'], config['steps'], config['microbatches'],
                config['receive_capacity'], config['temporal_pattern'], config['grad_sync']) != (
                4, 16384, 25, 4, 262144, job['pattern'], True):
            raise RuntimeError('Screen recipe mismatch')
        expected_prepass = job['policy'] in ('easy_home', 'easy_independent')
        if config.get('routing_prepass_counted') != expected_prepass:
            raise RuntimeError('Baseline observation-cost contract mismatch')
        rank_steps = []
        for rank in range(8):
            workers = []
            for path, value in read(folder / f'libraries-rank{rank}.json')['libraries'].items():
                if '/source/native-balanced/' not in path:
                    continue
                relative = 'source/native-balanced/' + path.split('/source/native-balanced/', 1)[1]
                if frozen.get(relative) != value['sha256']:
                    raise RuntimeError('Actual loaded SO mismatch: ' + relative)
                if '/worker-variants/' in path:
                    workers.append(value['sha256'])
            if workers != [WORKER]:
                raise RuntimeError('Expected exactly the frozen non-Timer performance worker')
            updates = lines(folder / f'steps-rank{rank}.jsonl')
            if [row['step'] for row in updates] != list(range(25)):
                raise RuntimeError('Incomplete optimizer updates')
            if any(not math.isfinite(row[key]) for row in updates
                   for key in ('milliseconds', 'loss', 'grad_norm')):
                raise RuntimeError('Nonfinite performance data')
            calls = read(folder / f'route-calls-rank{rank}.json')
            if len(calls) != 400 or set(calls.values()) != {3 if expected_prepass else 2}:
                raise RuntimeError('Unexpected route/recompute call multiplicity')
            rank_steps.append(updates)
        identity = {name: [read(folder / f'{name}-rank{rank}.json') for rank in range(8)]
                    for name in ('initial', 'input-identity', 'route-identity')}
        pattern = job['pattern']
        if pattern not in reference:
            reference[pattern] = (identity, config['model'])
        equal = {name: identity[name] == reference[pattern][0][name] for name in identity}
        equal['model'] = config['model'] == reference[pattern][1]
        if not all(equal.values()):
            raise RuntimeError('Logical experiment identities differ within pattern')
        identity_checks[job['tag']] = equal
        times = [max(rank_steps[rank][step]['milliseconds'] for rank in range(8)) / 1000
                 for step in range(25)]
        if min(times) <= 0:
            raise RuntimeError('Invalid wall duration')
        step_rows.extend(dict(pattern=pattern, label=job['label'], tag=job['tag'], step=step,
                              warmup=step < 5, max_rank_wall_seconds=value)
                         for step, value in enumerate(times))
        groups = []
        group_file = folder / 'temporal-groups-rank0.jsonl'
        if group_file.exists():
            groups = lines(group_file)
            canonical = [{key: row[key] for key in ('step', 'group', 'keys', 'ready_keys')} for row in groups]
            for rank in range(1, 8):
                other = lines(folder / f'temporal-groups-rank{rank}.jsonl')
                if [{key: row[key] for key in ('step', 'group', 'keys', 'ready_keys')} for row in other] != canonical:
                    raise RuntimeError('Rank task schedule disagreement')
            for group in groups:
                selected_loads = [task_loads(counts, pattern, key) for key in group['keys']]
                recorded = group.get('loads')
                unknown = recorded is None or all(value is None for value in recorded)
                if unknown == expected_prepass:
                    raise RuntimeError('Online load-observation provenance disagrees with policy')
                if not unknown and not np.array_equal(recorded, selected_loads):
                    raise RuntimeError('Recorded real loads differ from offline reconstruction')
                separate = sum(int(load.max()) for load in selected_loads)
                combined = int(np.sum(selected_loads, axis=0).max())
                group_rows.append(dict(pattern=pattern, label=job['label'], step=group['step'],
                    group=group['group'], selected_tasks=len(selected_loads),
                    distinct_layers=len({key[2] for key in group['keys']}),
                    native_waves=len(group['native_waves']),
                    declared_home_expert_planes=sum(32*len(wave['domain_layers'])
                                                    for wave in group['native_waves']),
                    offline_linear_complementarity=1 - combined / separate,
                    observation_available_online=expected_prepass,
                    cost_and_dependency_ignored=True))
        measured_groups = [row for row in group_rows if row['pattern'] == pattern and
                           row['label'] == job['label'] and row['step'] >= 5]
        rows.append(dict(pattern=pattern, label=job['label'], policy=job['policy'], tag=job['tag'],
            window=job['window'], routing_probe=expected_prepass,
            mean_wall_seconds=statistics.mean(times[5:]), median_wall_seconds=statistics.median(times[5:]),
            sampled_peak_gib=status['peak_driver_MiB'] / 1024, independent_repeats=1, measured_updates=20,
            failed_startup_attempts=len(job.get('failed_attempts', [])),
            groups=len(measured_groups), multi_task_groups=sum(row['selected_tasks'] > 1 for row in measured_groups),
            mixed_layer_groups=sum(row['distinct_layers'] > 1 for row in measured_groups),
            declared_home_expert_planes_per_update=(sum(row['declared_home_expert_planes']
                for row in measured_groups)/20 if measured_groups else 16*32),
            offline_linear_complementarity_mean=(statistics.mean(row['offline_linear_complementarity']
                 for row in measured_groups) if measured_groups else None),
            normalized_mean_to_native_w2=None, speedup_vs_native_w2=None,
            actual_loaded_binaries_checked=True, formal=False))
    for row in rows:
        baseline = next((candidate for candidate in rows if candidate['pattern'] == row['pattern'] and
                         candidate['label'] == 'legacy_w2'), None)
        if baseline:
            row['normalized_mean_to_native_w2'] = row['mean_wall_seconds'] / baseline['mean_wall_seconds']
            row['speedup_vs_native_w2'] = baseline['mean_wall_seconds'] / row['mean_wall_seconds']
    OUT.mkdir(exist_ok=True)
    write_csv(OUT / 'screening-runs.csv', rows)
    write_csv(OUT / 'screening-steps.csv', step_rows)
    write_csv(OUT / 'screening-groups-offline-audit.csv', group_rows)
    write_csv(OUT / 'screening-failed-attempts.csv', [dict(tag=attempt['tag'],
        classification=attempt['classification'],phase=attempt.get('phase'),updates=attempt.get('updates'),
        execution_exit=attempt.get('receipt',{}).get('exit'),
        source_unchanged=attempt.get('receipt',{}).get('source_unchanged'),
        exclusive_devices=attempt.get('receipt',{}).get('exclusive_devices'),
        root_cause_not_established=attempt.get('root_cause_not_established',True)) for attempt in failed_attempts])
    result = dict(completed=len(rows), planned=18, complete=len(rows) == 18, runs=rows,
                  job_states={job['tag']: job['status'] for job in manifest['jobs']},
                  identities_equal=identity_checks, actual_fixture_expert_maxvio=float(counts.max()/counts.mean()-1),
                  failed_attempts=failed_attempts,
                  independent_repeats=1, formal=False, stable_speedup_claim=False,
                  joint_strategy_included=False, afore_cache_included=False,
                  group_load_audit_role='offline diagnostic; never an input to online task selection')
    result['declared_home_plane_metric_role']='sum of per-wave home-domain sizes; structural capacity, not measured traffic or FLOPs'
    (ROOT / 'screening-summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    if rows:
        for metric, ylabel, stem in (
                ('normalized_mean_to_native_w2', 'Mean update / Native W2 mean', 'screening-online-step-times'),
                ('sampled_peak_gib', 'Sampled peak whole-card HBM (GiB)', 'screening-online-hbm')):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
            for axis, pattern in zip(axes, PATTERNS):
                current = {row['label']: row for row in rows if row['pattern'] == pattern}
                current_jobs = {job['label']: job for job in manifest['jobs'] if job['pattern'] == pattern}
                for index, label in enumerate(LABELS):
                    row = current.get(label)
                    if row is None or row[metric] is None:
                        state=current_jobs[label]['status']
                        note = 'await Native W2' if row is not None else 'failed' if state=='failed' else 'pending'
                        retained_failure=bool(current_jobs[label].get('failed_attempts'))
                        if retained_failure and row is None:note='startup failed; retry1 '+state+'*'
                        axis.text(index, .03, note, rotation=90, ha='center', transform=axis.get_xaxis_transform(),
                                  fontsize=8, color='crimson' if state=='failed' or retained_failure else 'black')
                        continue
                    color = '#d58b2a' if row['routing_probe'] else '#3578a2'
                    axis.bar(index, row[metric], color=color)
                    annotation = f"{row['mean_wall_seconds']:.3f}s" if metric.startswith('normalized') else f"{row[metric]:.2f}"
                    if row['failed_startup_attempts']:annotation+='\nretry1*'
                    axis.text(index, row[metric], annotation, ha='center', va='bottom', fontsize=8, rotation=20)
                axis.set_xticks(range(6), [label.replace('_', '\n') for label in LABELS], fontsize=8)
                axis.set_xlim(-.6, 5.6)
                axis.set_title(pattern)
                axis.grid(axis='y', alpha=.2)
                axis.set_ylim(0, None)
                axis.margins(y=.18)
            axes[0].set_ylabel(ylabel)
            if metric.startswith('normalized'):
                for axis in axes:
                    axis.axhline(1, color='gray', linestyle='--', alpha=.5)
            fig.suptitle(f'Online screening {len(rows)}/18 | MaxVio10 | L4 EP8 GA4 | one process per point\n'
                         '5 warmup + 20 measured updates; blue: no routing probe, orange: online probe; no stability claim', fontsize=10)
            fig.tight_layout()
            for suffix in ('png', 'svg', 'pdf'):
                fig.savefig(OUT / f'{stem}.{suffix}', dpi=180, bbox_inches='tight')
            plt.close(fig)
    print(json.dumps(dict(completed=len(rows), states=result['job_states'], runs=rows)))


if __name__ == '__main__':
    reduce()
