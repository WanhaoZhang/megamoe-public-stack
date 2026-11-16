"""Summarize explicit same-node ABCD/DCBA short-run pairs against A."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from summarize_training import compare_runs, load_run


def summarize(paths, steps=40, warmup=10):
    runs = [load_run(Path(p), steps, warmup) for p in paths]
    groups = {}
    for run in runs:
        p = run['launch']
        assert run.get('node'), 'seal execution-host provenance before formal aggregation'
        # Evidence names are fast-rhoN-METHOD-{f,r}-aN. Never mix host projects.
        parts = Path(run['path']).name.split('-')
        order = parts[-2]
        assert order in ('f', 'r'), 'explicit process-order evidence name required'
        key = (str(Path(run['path']).parent.resolve()), p['rho'], p['activation'], order)
        methods = groups.setdefault(key, {})
        assert p['method'] not in methods, 'duplicate method/order'
        methods[p['method']] = run
    rows = []
    for (project, rho, activation, order), methods in sorted(groups.items()):
        for method, run in sorted(methods.items()):
            a, d = methods.get('A'), methods.get('D')
            pair = compare_runs(run, a)
            peak = lambda key: max(row.get(key, 0) for rank in run['rows'] for row in rank)
            hbm = []
            f = Path(run['path'])/'hbm-samples.jsonl'
            if f.exists():
                for line in f.read_text().splitlines():
                    record = json.loads(line)
                    if record['rc'] == 0:
                        assert len(record['hbm']) == 8, 'partial HBM observation'
                        hbm.extend(v['used_mb'] for v in record['hbm'].values())
            pair_d = compare_runs(run, d) if d else None
            eligible_d = pair_d and pair_d['speedup_vs_A'] is not None
            rows.append(dict(project=project, rho=rho, activation=activation, order=order,
                node=run['node'],
                method=method, median_ms=run['median_ms'], p90_ms=run['p90_ms'],
                speedup_vs_A=pair['speedup_vs_A'],
                step_change_pct_vs_D=100*(run['median_ms']/d['median_ms']-1) if eligible_d else None,
                peak_allocated_gib=peak('peak_allocated_bytes')/2**30,
                peak_reserved_gib=peak('peak_reserved_bytes')/2**30,
                peak_hbm_mb=max(hbm) if hbm else None,
                initial_match_A=pair['initial_shards_identical'],
                batch_match_A=pair['data_identical'],
                same_node_A=pair['same_node'], same_config_A=pair['same_training_config'],
                initial_match_D=pair_d['initial_shards_identical'] if pair_d else False,
                batch_match_D=pair_d['data_identical'] if pair_d else False,
                max_reduced_lm_relative_error_vs_D=pair_d['max_reduced_lm_relative_error_vs_A'] if pair_d else None,
                max_local_lm_relative_error_vs_A=pair['max_local_lm_relative_error_vs_A'],
                max_reduced_lm_relative_error_vs_A=pair['max_reduced_lm_relative_error_vs_A'],
                path=run['path']))
    combined = []
    configs = sorted({(r['project'], r['rho'], r['activation'], r['method']) for r in rows})
    for project, rho, activation, method in configs:
        selected = [r for r in rows if (r['project'],r['rho'],r['activation'],r['method']) ==
                    (project,rho,activation,method)]
        valid = len(selected) == 2 and {r['order'] for r in selected} == {'f','r'}
        corresponding = [run for run in runs if run['path'] in {r['path'] for r in selected}]
        if valid:
            keys = ('rho','activation','layers','experts','topk','sequence','ep','expert_fsdp',
                    'steps','method','restore','publish_sync','worker_variant',
                    'native_slots','recv_capacity_override','control_tape_device','data_no_shared_storage')
            signature = lambda r: (r['node'], {k:r['launch'].get(k) for k in keys},
                sorted(r['launch'].get('worker_hashes', {}).values()),
                [arg for arg in r['launch'].get('command', []) if arg.endswith('/runtime_bootstrap.py')])
            assert signature(corresponding[0]) == signature(corresponding[1]), 'different f/r implementation or node'
        item = dict(project=project, rho=rho, activation=activation, method=method,
                    paired_orders_complete=valid)
        for key in ('median_ms',):
            values = [r[key] for r in selected]
            item[key] = statistics.median(values) if valid and all(v is not None for v in values) else None
        if valid:
            reference_rows = lambda name: [r for r in rows if (r['project'],r['rho'],r['activation'],r['method']) == (project,rho,activation,name)]
            aa, dd = reference_rows('A'), reference_rows('D')
            item['speedup_vs_A'] = (statistics.median(r['median_ms'] for r in aa)/item['median_ms']
                if len(aa)==2 and all(r['speedup_vs_A'] is not None for r in selected) else None)
            item['step_change_pct_vs_D'] = (100*(item['median_ms']/statistics.median(r['median_ms'] for r in dd)-1)
                if len(dd)==2 and all(r['step_change_pct_vs_D'] is not None for r in selected) else None)
        else:
            item['speedup_vs_A'] = item['step_change_pct_vs_D'] = None
        item['aggregation'] = 'ratio of median process times; process ratios retained separately'
        item['max_peak_hbm_mb'] = max((r['peak_hbm_mb'] for r in selected if r['peak_hbm_mb'] is not None), default=None)
        item['all_initial_and_batch_match_A'] = all(r['initial_match_A'] and r['batch_match_A'] for r in selected)
        item['short_run_only'] = True
        combined.append(item)
    return dict(steps=steps, warmup=warmup, process_rows=rows, combined=combined)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', type=Path, required=True)
    p.add_argument('--steps', type=int, default=40)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    result = summarize(a.runs, a.steps, a.warmup)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    with a.out.with_suffix('.csv').open('w', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=list(result['process_rows'][0]))
        w.writeheader()
        w.writerows(result['process_rows'])
    print(json.dumps(result['combined'], indent=2))


if __name__ == '__main__':
    main()
